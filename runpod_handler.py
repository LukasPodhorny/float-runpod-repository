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

    try:
        # Verify model is ready
        ckpt_path = verify_model_loaded()

        inp = event.get("input", {})
        avatars_config = inp.get("avatars", {})
        dialogues = inp.get("dialogues", [])

        if not avatars_config:
            return {"error": "No avatars provided in input"}

        if not dialogues:
            return {"error": "No dialogues provided in input"}

        workdir = tempfile.mkdtemp(prefix="float_job_")
        print(f"[job] workdir: {workdir}", flush=True)

        # Download and cache all avatar images
        avatar_paths = {}
        for avatar_name, avatar_config in avatars_config.items():
            avatar_image_url = avatar_config.get("avatar_image")
            if not avatar_image_url:
                raise ValueError(f"Missing avatar_image for avatar: {avatar_name}")

            # Check if it's a local path or URL
            if avatar_image_url.startswith("http://") or avatar_image_url.startswith(
                "https://"
            ):
                face_path = download_file(avatar_image_url, workdir)
            else:
                face_path = avatar_image_url
                print(f"[job] using local avatar image: {face_path}", flush=True)

            avatar_paths[avatar_name] = {
                "face_path": face_path,
                "seed": avatar_config.get("seed", 0),
                "a_cfg_scale": avatar_config.get("a_cfg_scale", 2),
                "e_cfg_scale": avatar_config.get("e_cfg_scale", 1),
            }

        print(f"[job] loaded {len(avatar_paths)} avatars", flush=True)

        # Process all dialogues in order
        results = []
        upload_futures = []

        for dialogue_idx, dialogue in enumerate(dialogues):
            # Get custom ID or use index as fallback
            dialogue_id = dialogue.get("id", dialogue_idx)

            try:
                avatar_name = dialogue.get("avatar")
                audio_url = dialogue.get("audio_url")
                emotion = dialogue.get("emotion", "neutral")

                if not avatar_name or not audio_url:
                    print(
                        f"[job] dialogue {dialogue_idx} (id={dialogue_id}): missing avatar or audio_url",
                        flush=True,
                    )
                    results.append({"id": dialogue_id, "video_url": None})
                    upload_futures.append(None)
                    continue

                if avatar_name not in avatar_paths:
                    print(
                        f"[job] dialogue {dialogue_idx} (id={dialogue_id}): unknown avatar '{avatar_name}'",
                        flush=True,
                    )
                    results.append({"id": dialogue_id, "video_url": None})
                    upload_futures.append(None)
                    continue

                print(
                    f"[job] dialogue {dialogue_idx} (id={dialogue_id}, avatar={avatar_name}, emotion={emotion}): {audio_url[:80]}...",
                    flush=True,
                )

                avatar_info = avatar_paths[avatar_name]

                # Download audio
                if audio_url.startswith("http://") or audio_url.startswith("https://"):
                    audio_path = download_file(audio_url, workdir)
                else:
                    audio_path = audio_url
                    print(f"[job] using local audio: {audio_path}", flush=True)

                # Output filename
                output_filename = f"output_{dialogue_idx}_{uuid.uuid4().hex[:8]}.mp4"
                final_output = os.path.join(workdir, output_filename)

                print(
                    f"[inference] starting FLOAT generation for dialogue {dialogue_idx} (id={dialogue_id})...",
                    flush=True,
                )

                # Run FLOAT inference
                inference_cmd = [
                    sys.executable,
                    "generate.py",
                    "--ref_path",
                    avatar_info["face_path"],
                    "--aud_path",
                    audio_path,
                    "--emo",
                    str(emotion),
                    "--seed",
                    str(avatar_info["seed"]),
                    "--a_cfg_scale",
                    str(avatar_info["a_cfg_scale"]),
                    "--e_cfg_scale",
                    str(avatar_info["e_cfg_scale"]),
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
                            f"[job] no output found for dialogue {dialogue_idx} (id={dialogue_id})",
                            flush=True,
                        )
                        results.append({"id": dialogue_id, "video_url": None})
                        upload_futures.append(None)
                        continue
                    final_output = max(outputs, key=os.path.getmtime)

                print(f"[job] generated: {final_output}", flush=True)

                # Upload to R2 in parallel
                object_name = f"float_outputs/{uuid.uuid4()}.mp4"
                upload_future = _executor.submit(
                    upload_to_r2, final_output, object_name
                )
                upload_futures.append(upload_future)
                results.append({"id": dialogue_id, "video_url": "pending"})

            except subprocess.TimeoutExpired:
                print(
                    f"[job] dialogue {dialogue_idx} (id={dialogue_id}) timed out",
                    flush=True,
                )
                results.append({"id": dialogue_id, "video_url": None})
                upload_futures.append(None)
            except Exception as e:
                print(
                    f"[job] dialogue {dialogue_idx} (id={dialogue_id}) failed: {e}",
                    flush=True,
                )
                traceback.print_exc()
                results.append({"id": dialogue_id, "video_url": None})
                upload_futures.append(None)

        # Wait for all uploads to complete
        print(
            f"[job] waiting for {len(upload_futures)} uploads to complete...",
            flush=True,
        )
        for idx, future in enumerate(upload_futures):
            if future is not None:
                try:
                    presigned_url = future.result(timeout=60)
                    results[idx]["video_url"] = presigned_url
                except Exception as e:
                    print(f"[upload] failed for dialogue {idx}: {e}", flush=True)
                    results[idx]["video_url"] = None

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
