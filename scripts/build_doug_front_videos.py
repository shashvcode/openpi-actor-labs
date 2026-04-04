"""Build per-session front-camera (csi0) videos from chunk tars in S3.

Encodes each chunk immediately to avoid holding all frames on disk.
"""

import os
import shutil
import subprocess
import sys

AWS_ENV = {
    **os.environ,
    "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
    "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
    "AWS_REGION": "us-west-1",
}
SRC_BUCKET = "actor-labeler-videos"
DST_BUCKET = "actorgemma-raw-data"
FPS = 15
TMP_DIR = "/tmp/doug_build"


def s3_cmd(args, **kwargs):
    return subprocess.run(
        ["aws", "s3", *args, "--region", "us-west-1"],
        env=AWS_ENV, capture_output=True, text=True, **kwargs,
    )


def list_sessions():
    r = s3_cmd(["ls", f"s3://{SRC_BUCKET}/doug-data/"])
    sessions = []
    for line in r.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("PRE") and "_session/" in line:
            sessions.append(line.split()[-1].rstrip("/"))
    return sorted(sessions)


def list_chunks(session):
    r = s3_cmd(["ls", f"s3://{SRC_BUCKET}/doug-data/{session}/chunks/"])
    chunks = []
    for line in r.stdout.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) >= 4 and parts[-1].startswith("chunk_") and parts[-1].endswith(".tar"):
            chunks.append(parts[-1])
    return sorted(chunks)


def extract_and_encode_chunk(session, chunk_name, chunk_video_path):
    """Stream tar from S3, extract csi0 frames, encode to video, clean up."""
    frames_dir = os.path.join(TMP_DIR, "chunk_frames")
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    s3_path = f"s3://{SRC_BUCKET}/doug-data/{session}/chunks/{chunk_name}"
    s3_proc = subprocess.Popen(
        ["aws", "s3", "cp", s3_path, "-", "--region", "us-west-1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=AWS_ENV,
    )
    tar_proc = subprocess.Popen(
        ["tar", "xf", "-", "--include=*/frames/csi0/*.jpg", "-C", frames_dir, "--strip-components=3"],
        stdin=s3_proc.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    s3_proc.stdout.close()
    tar_proc.wait()
    s3_proc.wait()

    frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    if not frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
        return 0

    for i, f in enumerate(frames):
        os.rename(os.path.join(frames_dir, f), os.path.join(frames_dir, f"{i:06d}.jpg"))

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(FPS),
                "-i", os.path.join(frames_dir, "%06d.jpg"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-crf", "23", "-preset", "fast",
                chunk_video_path,
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"    ffmpeg failed on {chunk_name} (exit {e.returncode}), skipping", flush=True)
        shutil.rmtree(frames_dir, ignore_errors=True)
        return 0

    count = len(frames)
    shutil.rmtree(frames_dir, ignore_errors=True)
    return count


def concat_videos(video_paths, output_path):
    """Concatenate chunk videos using ffmpeg concat demuxer."""
    list_file = os.path.join(TMP_DIR, "concat.txt")
    with open(list_file, "w") as f:
        for vp in video_paths:
            f.write(f"file '{vp}'\n")

    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            output_path,
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )
    os.remove(list_file)


def process_session(session):
    session_id = session.replace("_session", "")
    out_name = f"doug_{session_id}.mp4"
    print(f"\n=== {session} ===", flush=True)

    chunks = list_chunks(session)
    if not chunks:
        print("  No chunks, skipping", flush=True)
        return

    print(f"  {len(chunks)} chunks", flush=True)

    chunk_vids_dir = os.path.join(TMP_DIR, "chunk_vids")
    if os.path.exists(chunk_vids_dir):
        shutil.rmtree(chunk_vids_dir)
    os.makedirs(chunk_vids_dir, exist_ok=True)

    total_frames = 0
    chunk_videos = []
    for i, chunk_name in enumerate(chunks):
        chunk_vid = os.path.join(chunk_vids_dir, f"{i:04d}.mp4")
        count = extract_and_encode_chunk(session, chunk_name, chunk_vid)
        if count > 0:
            chunk_videos.append(chunk_vid)
        total_frames += count

        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            print(f"  Processed {i+1}/{len(chunks)} chunks ({total_frames} frames)", flush=True)

    if not chunk_videos:
        print("  No frames extracted, skipping", flush=True)
        shutil.rmtree(chunk_vids_dir, ignore_errors=True)
        return

    out_path = os.path.join(TMP_DIR, out_name)
    if len(chunk_videos) == 1:
        os.rename(chunk_videos[0], out_path)
    else:
        print(f"  Concatenating {len(chunk_videos)} chunk videos...", flush=True)
        concat_videos(chunk_videos, out_path)

    size_mb = os.path.getsize(out_path) / 1048576
    duration_min = total_frames / FPS / 60
    print(f"  Result: {size_mb:.1f} MB, ~{duration_min:.1f} min", flush=True)

    print(f"  Uploading to s3://{DST_BUCKET}/{out_name}...", flush=True)
    s3_cmd(["cp", out_path, f"s3://{DST_BUCKET}/{out_name}"])

    shutil.rmtree(chunk_vids_dir, ignore_errors=True)
    os.remove(out_path)
    print(f"  Done: {out_name}", flush=True)


def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    sessions = list_sessions()
    print(f"Found {len(sessions)} sessions", flush=True)

    start_from = sys.argv[1] if len(sys.argv) > 1 else None
    for session in sessions:
        if start_from and session < start_from:
            continue
        process_session(session)

    print("\n=== ALL SESSIONS DONE ===", flush=True)


if __name__ == "__main__":
    main()
