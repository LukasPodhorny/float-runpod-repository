import os
import sys
import subprocess
import uuid
import boto3
import runpod
import tempfile
import shutil
import traceback
import glob
from urllib.parse import urlparse
from botocore.client import Config

# ---------- Cloudflare R2 CONFIG ----------
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_BUCKET = os.getenv("R2_BUCKET")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")


# ---------- HELPERS ----------
def download_file(url, destination_dir="/tmp"):
    os.makedirs(destination_dir, exist_ok=True)
    parsed = urlparse(url)
    basename = os.path.basename(parsed.path) or str(uuid.uuid4())
    filename = os.path.join(destination_dir, basename)
    print(f"[download] {url} -> {filename}", flush=True)
    try:
        subprocess.check_call(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "3",
                "--retry-delay",
                "2",
                "-o",
                filename,
                url,
            ],
            close_fds=False,
        )
        print(f"[download] success: {filename}", flush=True)
        return filename
    except subprocess.CalledProcessError as e:
        if os.path.exists(filename):
            os.remove(filename)
        print(f"[download] curl failed: {e}", flush=True)
        raise


def upload_to_r2(local_path, object_name, expires_in=3600):
    if not all([R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
        raise Exception("Missing one or more R2 environment variables.")
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )
    print(f"[upload] uploading {local_path} to {R2_BUCKET}/{object_name}", flush=True)
    s3.upload_file(local_path, R2_BUCKET, object_name)
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": R2_BUCKET, "Key": object_name},
        ExpiresIn=expires_in,
    )
    print(f"[upload] presigned url created", flush=True)
    return url


# ---------- RUNPOD HANDLER ----------
def handler(event):
    workdir = None
    try:
        inp = event.get("input", {})
        face_url = inp["face_url"]
        audio_url = inp["audio_url"]
        emotion = inp.get("emotion", "neutral")
        seed = inp.get("seed", 0)
        a_cfg_scale = inp.get("a_cfg_scale", 2)
        e_cfg_scale = inp.get("e_cfg_scale", 1)

        ckpt_path = "./checkpoints/float.pth"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        workdir = tempfile.mkdtemp(prefix="float_job_")
        print(f"[job] workdir: {workdir}", flush=True)

        # Download inputs
        face_path = download_file(face_url, workdir)
        audio_path = download_file(audio_url, workdir)

        # Expected output file
        final_output = os.path.join(workdir, "result.mp4")

        print("STARTING INFERENCE")
        print("FACE PATH: " + face_path)
        print("AUDIO PATH: " + audio_path)
        # Run FLOAT inference
        inference_cmd = [
            sys.executable,
            "generate.py",
            "--ref_path",
            face_path,
            "--aud_path",
            audio_path,
            "--emo",
            str(emotion),
            "--seed",
            str(seed),
            "--a_cfg_scale",
            str(a_cfg_scale),
            "--e_cfg_scale",
            str(e_cfg_scale),
            "--ckpt_path",
            ckpt_path,
            "--res_dir",
            workdir,
            "--res_video_path",
            final_output,
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        try:
            result = subprocess.run(
                inference_cmd, env=env, capture_output=True, text=True
            )
            print("[generate.py stdout]\n", result.stdout, flush=True)
            print("[generate.py stderr]\n", result.stderr, flush=True)
            result.check_returncode()  # this raises CalledProcessError if non-zero
        except subprocess.CalledProcessError as e:
            print("[generate.py stdout before error]\n", e.stdout, flush=True)
            print("[generate.py stderr before error]\n", e.stderr, flush=True)
            raise  # re-raise so your outer try/except catches it

        def print_tree(startpath, prefix=""):
            for i, name in enumerate(os.listdir(startpath)):
                path = os.path.join(startpath, name)
                connector = "├── " if i < len(os.listdir(startpath)) - 1 else "└── "
                print(prefix + connector + name)
                if os.path.isdir(path):
                    extension = "│   " if i < len(os.listdir(startpath)) - 1 else "    "
                    print_tree(path, prefix + extension)

        print("WORK DIRECTORY: " + workdir)
        print_tree(workdir)
        # Verify output
        if not os.path.exists(final_output):
            # fallback: look for any mp4 in workdir
            outputs = glob.glob(os.path.join(workdir, "**", "*.mp4"), recursive=True)
            if not outputs:
                print("ONLY FILE WITHOUT AUDIO FOUND, TRY INTALLING FFMPEG!")
                outputs = glob.glob("/tmp/**/*.mp4", recursive=True)
            if not outputs:
                return {"error": "No output file generated by inference."}
            final_output = max(outputs, key=os.path.getmtime)

        print(f"[run] found output: {final_output}", flush=True)

        # Upload to R2
        object_name = f"float_outputs/{uuid.uuid4()}.mp4"
        presigned_url = upload_to_r2(final_output, object_name)
        return {"output_url": presigned_url}

    except Exception as e:
        tb = traceback.format_exc()
        print("[handler] exception:", tb, flush=True)
        return {
            "error_type": str(type(e)),
            "error_message": str(e),
            "traceback": tb,
        }

    finally:
        if workdir:
            try:
                shutil.rmtree(workdir)
                print(f"[job] cleaned workdir {workdir}", flush=True)
            except Exception as e:
                print(f"[job] failed to cleanup {workdir}: {e}", flush=True)


# ---------- RUNPOD ----------
if __name__ == "__main__":
    print(">>> RunPod handler starting", flush=True)
    runpod.serverless.start({"handler": handler})
