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
from concurrent.futures import ThreadPoolExecutor
import torch

# ---------- Cloudflare R2 CONFIG ----------
R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_BUCKET = os.getenv("R2_BUCKET")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")

# ---------- GLOBAL CACHE ----------
_model_loaded = False
_executor = ThreadPoolExecutor(max_workers=3)


# ---------- HELPERS ----------
def download_file(url, destination_dir="/tmp"):
    """Download file with wget (faster than curl)"""
    os.makedirs(destination_dir, exist_ok=True)
    parsed = urlparse(url)
    basename = os.path.basename(parsed.path) or str(uuid.uuid4())
    filename = os.path.join(destination_dir, basename)
    print(f"[download] {url} -> {filename}", flush=True)
    try:
        subprocess.check_call(
            ["wget", "-q", "-O", filename, url],
            close_fds=False,
        )
        print(f"[download] success: {filename}", flush=True)
        return filename
    except subprocess.CalledProcessError as e:
        if os.path.exists(filename):
            os.remove(filename)
        print(f"[download] wget failed, trying curl: {e}", flush=True)
        # Fallback to curl
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
            print(f"[download] curl success: {filename}", flush=True)
            return filename
        except subprocess.CalledProcessError as e2:
            if os.path.exists(filename):
                os.remove(filename)
            print(f"[download] curl also failed: {e2}", flush=True)
            raise


def upload_to_r2(local_path, object_name, expires_in=3600):
    """Upload file to R2 and return presigned URL"""
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


def verify_model_loaded():
    """Verify FLOAT model is loaded (done once on first request)"""
    global _model_loaded
    if not _model_loaded:
        ckpt_path = "./checkpoints/float.pth"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"[model] Checkpoint verified: {ckpt_path}", flush=True)
        print(
            f"[model] CUDA available: {torch.cuda.is_available()}",
            flush=True,
        )
        if torch.cuda.is_available():
            print(
                f"[model] GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
                flush=True,
            )
        _model_loaded = True
    return "./checkpoints/float.pth"


# ---------- RUNPOD HANDLER ----------
def handler(event):
    workdir = None
    upload_futures = []

    try:
        # Verify model is ready
        ckpt_path = verify_model_loaded()

        inp = event.get("input", {})
        items = inp.get("items", [])

        if not items:
            return {"error": "No items provided in input"}

        workdir = tempfile.mkdtemp(prefix="float_job_")
        print(f"[job] workdir: {workdir}", flush=True)

        results = {"items": []}

        # Process each item (avatar image + audio URLs)
        for item_idx, item in enumerate(items):
            avatar_image_url = item.get("avatar_image")
            audio_urls = item.get("audio_urls", [])
            emotion = item.get("emotion", "neutral")
            seed = item.get("seed", 0)
            a_cfg_scale = item.get("a_cfg_scale", 2)
            e_cfg_scale = item.get("e_cfg_scale", 1)

            if not avatar_image_url:
                results["items"].append(
                    {"error": "Missing avatar_image", "video_urls": []}
                )
                continue

            if not audio_urls:
                results["items"].append(
                    {"error": "No audio_urls provided", "video_urls": []}
                )
                continue

            print(
                f"[job] processing item {item_idx}: {len(audio_urls)} videos",
                flush=True,
            )

            # Download avatar image ONCE per item
            # Check if it's a local path (from network volume) or URL
            if avatar_image_url.startswith("http://") or avatar_image_url.startswith(
                "https://"
            ):
                face_path = download_file(avatar_image_url, workdir)
            else:
                # It's a local path, use directly
                face_path = avatar_image_url
                print(f"[job] using local avatar image: {face_path}", flush=True)

            video_urls = []
            item_upload_futures = []

            # Process each audio for this avatar
            for audio_idx, audio_url in enumerate(audio_urls):
                try:
                    print(
                        f"[job] processing audio {audio_idx}: {audio_url[:80]}...",
                        flush=True,
                    )

                    # Download audio
                    # Check if it's a local path or URL
                    if audio_url.startswith("http://") or audio_url.startswith(
                        "https://"
                    ):
                        audio_path = download_file(audio_url, workdir)
                    else:
                        audio_path = audio_url
                        print(f"[job] using local audio: {audio_path}", flush=True)

                    # Output filename
                    output_filename = (
                        f"output_{item_idx}_{audio_idx}_{uuid.uuid4().hex[:8]}.mp4"
                    )
                    final_output = os.path.join(workdir, output_filename)

                    print(
                        f"[inference] starting FLOAT generation for audio {audio_idx}...",
                        flush=True,
                    )

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
                    result = subprocess.run(
                        inference_cmd,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 minute timeout per video
                    )

                    print(f"[generate.py stdout] {result.stdout}", flush=True)
                    if result.stderr:
                        print(f"[generate.py stderr] {result.stderr}", flush=True)

                    result.check_returncode()

                    # Verify output exists
                    if not os.path.exists(final_output):
                        # Fallback: look for any mp4 in workdir
                        outputs = glob.glob(
                            os.path.join(workdir, "**", "*.mp4"), recursive=True
                        )
                        if not outputs:
                            print(
                                f"[job] no output found for audio {audio_idx}",
                                flush=True,
                            )
                            video_urls.append(None)
                            item_upload_futures.append(None)
                            continue
                        final_output = max(outputs, key=os.path.getmtime)

                    print(f"[job] generated: {final_output}", flush=True)

                    # Upload to R2 in parallel (non-blocking)
                    object_name = f"float_outputs/{uuid.uuid4()}.mp4"
                    upload_future = _executor.submit(
                        upload_to_r2, final_output, object_name
                    )
                    item_upload_futures.append(upload_future)
                    video_urls.append("pending")  # Placeholder

                except subprocess.TimeoutExpired:
                    print(f"[job] audio {audio_idx} timed out", flush=True)
                    video_urls.append(None)
                    item_upload_futures.append(None)
                except Exception as e:
                    print(f"[job] audio {audio_idx} failed: {e}", flush=True)
                    traceback.print_exc()
                    video_urls.append(None)
                    item_upload_futures.append(None)

            # Wait for all uploads for this item to complete
            print(
                f"[job] waiting for {len(item_upload_futures)} uploads to complete...",
                flush=True,
            )
            for idx, future in enumerate(item_upload_futures):
                if future is not None:
                    try:
                        presigned_url = future.result(timeout=60)
                        video_urls[idx] = presigned_url
                    except Exception as e:
                        print(f"[upload] failed for audio {idx}: {e}", flush=True)
                        video_urls[idx] = None

            results["items"].append({"video_urls": video_urls})

        return results

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
    print(">>> RunPod FLOAT handler starting", flush=True)
    runpod.serverless.start({"handler": handler})
