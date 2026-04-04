"""Combine per-chunk JPEG frames into one video per sensor per session.
Downloads tar.zst chunks from S3, extracts frames, encodes with ffmpeg, uploads."""

import os, sys, subprocess, tempfile, shutil, json, tarfile
from pathlib import Path

BUCKET = "actor-labeler-videos"
PREFIX = "doug-data"
SENSOR = "csi0"
FPS = 15

SESSIONS = [
    "2026-02-24T11-13-49_session",
    "2026-02-24T14-58-05_session",
    "2026-02-25T08-04-07_session",
]

def run(cmd, **kw):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(f"  CMD FAILED: {cmd}\n  {r.stderr.strip()}", flush=True)
    return r

def list_chunks(session):
    r = run(f"aws s3 ls s3://{BUCKET}/{PREFIX}/{session}/chunks/ 2>/dev/null")
    chunks = []
    for line in r.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        name = parts[-1]
        if name.endswith(".tar.zst"):
            size = int(parts[2])
            chunks.append((name, size))
    return sorted(chunks)

def process_session(session, workdir):
    print(f"\n{'='*60}", flush=True)
    print(f"SESSION: {session}", flush=True)
    print(f"{'='*60}", flush=True)

    chunks = list_chunks(session)
    print(f"  Found {len(chunks)} tar.zst chunks", flush=True)

    frames_dir = workdir / "frames"
    frames_dir.mkdir(exist_ok=True)

    global_frame = 0

    for chunk_name, chunk_size in chunks:
        chunk_id = chunk_name.replace(".tar.zst", "")
        s3_path = f"s3://{BUCKET}/{PREFIX}/{session}/chunks/{chunk_name}"

        print(f"\n  [{chunk_id}] Downloading ({chunk_size/(1024*1024):.0f} MB)...", flush=True)
        local_zst = workdir / chunk_name
        r = run(f"aws s3 cp '{s3_path}' '{local_zst}' --quiet")
        if r.returncode != 0:
            print(f"  SKIP: download failed", flush=True)
            continue

        local_tar = workdir / f"{chunk_id}.tar"
        print(f"  [{chunk_id}] Decompressing...", flush=True)
        r = run(f"zstd -d '{local_zst}' -o '{local_tar}' --rm 2>&1")
        if r.returncode != 0:
            print(f"  SKIP: decompression failed (likely truncated)", flush=True)
            local_zst.unlink(missing_ok=True)
            continue

        print(f"  [{chunk_id}] Extracting {SENSOR} frames...", flush=True)
        try:
            with tarfile.open(local_tar) as tf:
                sensor_members = [
                    m for m in tf.getmembers()
                    if f"frames/{SENSOR}/" in m.name
                    and m.name.endswith(".jpg")
                    and m.size > 0
                ]
                sensor_members.sort(key=lambda m: m.name)

                for m in sensor_members:
                    f_out = frames_dir / f"{global_frame:08d}.jpg"
                    with tf.extractfile(m) as src:
                        f_out.write_bytes(src.read())
                    global_frame += 1

            print(f"  [{chunk_id}] Extracted {len(sensor_members)} frames (total: {global_frame})", flush=True)
        except Exception as e:
            print(f"  SKIP: tar extraction failed: {e}", flush=True)

        local_tar.unlink(missing_ok=True)

    if global_frame == 0:
        print(f"  NO FRAMES — skipping video encode", flush=True)
        return None

    duration = global_frame / FPS
    print(f"\n  Total frames: {global_frame}, duration: {duration:.1f}s", flush=True)

    session_id = session.replace("_session", "")
    output_mp4 = workdir / f"{session_id}_{SENSOR}.mp4"

    print(f"  Encoding video at {FPS} fps...", flush=True)
    r = run(
        f"ffmpeg -y -framerate {FPS} -i '{frames_dir}/%08d.jpg' "
        f"-c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p "
        f"'{output_mp4}' 2>&1 | tail -3"
    )
    if not output_mp4.exists():
        print(f"  ENCODE FAILED", flush=True)
        return None

    size_mb = output_mp4.stat().st_size / (1024 * 1024)
    print(f"  Video: {output_mp4.name} ({size_mb:.1f} MB)", flush=True)

    s3_dest = f"s3://{BUCKET}/{PREFIX}/{session}/videos/{SENSOR}.mp4"
    print(f"  Uploading to {s3_dest}...", flush=True)
    r = run(f"aws s3 cp '{output_mp4}' '{s3_dest}' --quiet")
    if r.returncode == 0:
        print(f"  UPLOADED", flush=True)
    else:
        print(f"  UPLOAD FAILED", flush=True)

    # Cleanup
    shutil.rmtree(frames_dir)
    output_mp4.unlink(missing_ok=True)
    return s3_dest

def main():
    for session in SESSIONS:
        workdir = Path(tempfile.mkdtemp(prefix="chunks2vid_"))
        try:
            process_session(session, workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    print("\nDONE.", flush=True)

if __name__ == "__main__":
    main()
