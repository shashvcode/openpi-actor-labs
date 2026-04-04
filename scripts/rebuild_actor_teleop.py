"""Rebuild actor_teleop from pi_teleopv2v3, skipping corrupted episodes.

Same fixes as build_actor_teleop.py (USB trim, task fix, contiguous index)
plus the ability to skip specific episodes and renumber the rest.

Usage:
    python scripts/rebuild_actor_teleop.py --skip 2
    python scripts/rebuild_actor_teleop.py --skip 2 --dry-run
    python scripts/rebuild_actor_teleop.py --skip 2 5 10    # skip multiple
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

SOURCE_REPO = "verm11/pi_teleopv2v3"
TARGET_REPO = "verm11/actor_teleop"
TRIM_FRAMES = 6
BATCH_SIZE = 50

TASK_FIX = {
    "old_task": "Scoop up packing peanuts from large pool and put in to small pool",
    "new_task_index": 2,
    "new_task": "Scoop packing peanuts from large pool and dump into small pool",
}

PARQUET_SCHEMA = pa.schema([
    ("observation.state", pa.list_(pa.float32())),
    ("action", pa.list_(pa.float32())),
    ("observation.images.csi_0_imx219", pa.struct([
        ("bytes", pa.binary()),
        ("path", pa.string()),
    ])),
    ("observation.images.usb_0", pa.struct([
        ("bytes", pa.binary()),
        ("path", pa.string()),
    ])),
    ("timestamp", pa.float32()),
    ("frame_index", pa.int64()),
    ("episode_index", pa.int64()),
    ("index", pa.int64()),
    ("task_index", pa.int64()),
])


def load_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"]:
        if not candidate.exists():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        break
    return os.environ.get("HF_TOKEN")


def build_episode_path(info: dict, episode_idx: int) -> str:
    template = info["data_path"]
    chunk = episode_idx // 1000
    return template.format(
        episode_index=episode_idx,
        file_index=episode_idx,
        episode_chunk=chunk,
        chunk_index=chunk,
    )


def process_episode(
    table: pa.Table,
    new_episode_idx: int,
    global_index: int,
    trim: int,
    fixed_task_index: int | None,
) -> tuple[pa.Table, int]:
    L = len(table)
    new_len = L - trim
    if new_len < 10:
        raise ValueError(f"Episode too short ({L} frames) to trim {trim}")

    csi_trimmed = table.column("observation.images.csi_0_imx219").slice(0, new_len)
    usb_trimmed = table.column("observation.images.usb_0").slice(trim, new_len)
    state_trimmed = table.column("observation.state").slice(0, new_len)
    action_trimmed = table.column("action").slice(0, new_len)

    timestamps = pa.array([i * 0.1 for i in range(new_len)], type=pa.float32())
    frame_indices = pa.array(range(new_len), type=pa.int64())
    episode_indices = pa.array([new_episode_idx] * new_len, type=pa.int64())
    global_indices = pa.array(range(global_index, global_index + new_len), type=pa.int64())

    task_idx = fixed_task_index if fixed_task_index is not None else table.column("task_index")[0].as_py()
    task_indices = pa.array([task_idx] * new_len, type=pa.int64())

    new_table = pa.table(
        {
            "observation.state": state_trimmed,
            "action": action_trimmed,
            "observation.images.csi_0_imx219": csi_trimmed,
            "observation.images.usb_0": usb_trimmed,
            "timestamp": timestamps,
            "frame_index": frame_indices,
            "episode_index": episode_indices,
            "index": global_indices,
            "task_index": task_indices,
        },
        schema=PARQUET_SCHEMA,
    )
    return new_table, new_len


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild actor_teleop, skipping corrupted episodes")
    parser.add_argument("--skip", type=int, nargs="+", required=True, help="Source episode indices to skip")
    parser.add_argument("--trim-frames", type=int, default=TRIM_FRAMES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="Process first batch only, skip upload")
    parser.add_argument("--target-repo", default=TARGET_REPO)
    args = parser.parse_args()

    token = load_hf_token()
    if not token:
        raise RuntimeError("HF_TOKEN not found in environment or .env")

    api = HfApi(token=token)
    trim = args.trim_frames
    skip_set = set(args.skip)
    batch_size = args.batch_size

    print(f"Source:     {SOURCE_REPO}")
    print(f"Target:     {args.target_repo}")
    print(f"Trim:       {trim} frames ({trim * 100}ms)")
    print(f"Skipping:   episodes {sorted(skip_set)}")
    print(f"Batch size: {batch_size}")
    print()

    info_path = hf_hub_download(SOURCE_REPO, "meta/info.json", repo_type="dataset", token=token)
    episodes_path = hf_hub_download(SOURCE_REPO, "meta/episodes.jsonl", repo_type="dataset", token=token)
    tasks_path = hf_hub_download(SOURCE_REPO, "meta/tasks.jsonl", repo_type="dataset", token=token)

    with open(info_path) as f:
        info = json.load(f)
    with open(episodes_path) as f:
        all_episodes_meta = [json.loads(line) for line in f]
    with open(tasks_path) as f:
        tasks_meta = [json.loads(line) for line in f]

    # Filter out skipped episodes
    kept_episodes = [e for e in all_episodes_meta if e["episode_index"] not in skip_set]
    skipped = [e for e in all_episodes_meta if e["episode_index"] in skip_set]

    print(f"Source episodes: {len(all_episodes_meta)}")
    print(f"Skipping {len(skipped)} episodes:")
    for e in skipped:
        print(f"  ep {e['episode_index']}: {e['length']} frames, task: \"{e['task'][:60]}\"")
    print(f"Keeping: {len(kept_episodes)} episodes\n")

    # Pre-compute new metadata with renumbered indices
    new_episodes_meta = []
    global_index_starts = []
    source_indices = []
    global_index = 0

    for new_idx, ep_meta in enumerate(kept_episodes):
        new_len = ep_meta["length"] - trim
        new_task = ep_meta["task"]
        new_task_index = ep_meta["task_index"]
        if ep_meta["task"] == TASK_FIX["old_task"]:
            new_task_index = TASK_FIX["new_task_index"]
            new_task = TASK_FIX["new_task"]

        source_indices.append(ep_meta["episode_index"])
        global_index_starts.append(global_index)
        new_episodes_meta.append({
            "episode_index": new_idx,
            "length": new_len,
            "task_index": new_task_index,
            "task": new_task,
        })
        global_index += new_len

    new_total_episodes = len(new_episodes_meta)
    new_total_frames = global_index

    print(f"New dataset: {new_total_episodes} episodes, {new_total_frames} frames")
    print(f"Renumbered: source ep {source_indices[0]}->0, {source_indices[1]}->1, {source_indices[2]}->2, ...")
    if len(source_indices) > 5:
        print(f"            source ep {source_indices[-1]}->{new_total_episodes - 1}")
    print()

    new_info = {
        "codebase_version": info["codebase_version"],
        "robot_type": info["robot_type"],
        "fps": info["fps"],
        "total_episodes": new_total_episodes,
        "total_frames": new_total_frames,
        "data_path": info["data_path"],
        "features": info["features"],
        "source_datasets": info.get("source_datasets", {}),
    }

    work_dir = Path(tempfile.mkdtemp(prefix="actor_teleop_rebuild_"))

    if not args.dry_run:
        api.create_repo(args.target_repo, repo_type="dataset", exist_ok=True)

    # Upload meta first
    meta_dir = work_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_dir / "info.json", "w") as f:
        json.dump(new_info, f, indent=2)
        f.write("\n")
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in new_episodes_meta:
            f.write(json.dumps(ep) + "\n")
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task in tasks_meta:
            f.write(json.dumps(task) + "\n")

    if not args.dry_run:
        print("Uploading meta/ ...")
        api.upload_folder(
            repo_id=args.target_repo,
            folder_path=str(meta_dir),
            path_in_repo="meta",
            repo_type="dataset",
            commit_message="Update metadata (removed corrupted episodes, renumbered)",
        )
        print("  meta/ uploaded.\n")

    # Process in batches
    total = len(kept_episodes)
    if args.dry_run:
        total = min(batch_size, total)

    num_batches = (total + batch_size - 1) // batch_size
    print(f"Processing {total} episodes in {num_batches} batches...\n")

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)

        data_dir = work_dir / "data" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)

        for i in range(start, end):
            src_ep_idx = source_indices[i]
            new_ep_idx = i

            src_path = hf_hub_download(
                SOURCE_REPO,
                build_episode_path(info, src_ep_idx),
                repo_type="dataset",
                token=token,
            )
            table = pq.read_table(src_path)

            fixed_task_index = None
            ep_meta = kept_episodes[i]
            if ep_meta["task"] == TASK_FIX["old_task"]:
                fixed_task_index = TASK_FIX["new_task_index"]

            new_table, new_len = process_episode(
                table, new_ep_idx, global_index_starts[i], trim, fixed_task_index
            )

            out_file = data_dir / f"episode_{new_ep_idx:06d}.parquet"
            pq.write_table(new_table, out_file)

        batch_label = f"[batch {batch_idx + 1}/{num_batches}] new eps {start}-{end - 1}"
        print(f"  {batch_label}: processed")

        if not args.dry_run:
            api.upload_folder(
                repo_id=args.target_repo,
                folder_path=str(data_dir),
                path_in_repo="data/chunk-000",
                repo_type="dataset",
                commit_message=f"Add episodes {start}-{end - 1} (rebuilt)",
            )
            print(f"  {batch_label}: uploaded, cleaning up...")
            for f in data_dir.iterdir():
                f.unlink()
        else:
            print(f"  {batch_label}: dry-run, files at {data_dir}")

    # Delete old episode files that no longer exist (if target had more episodes before)
    if not args.dry_run:
        old_info_path = hf_hub_download(args.target_repo, "meta/info.json", repo_type="dataset", token=token)
        with open(old_info_path) as f:
            old_target_info = json.load(f)
        old_count = old_target_info.get("total_episodes", 0)
        if old_count > new_total_episodes:
            stale_files = [
                f"data/chunk-000/episode_{i:06d}.parquet"
                for i in range(new_total_episodes, old_count)
            ]
            if stale_files:
                print(f"\nDeleting {len(stale_files)} stale episode files from repo...")
                api.delete_files(
                    repo_id=args.target_repo,
                    repo_type="dataset",
                    paths=stale_files,
                    commit_message=f"Remove stale episodes {new_total_episodes}-{old_count - 1}",
                )

    print(f"\nDone.")
    print(f"  Episodes:        {new_total_episodes}")
    print(f"  Total frames:    {new_total_frames}")
    print(f"  Skipped:         {sorted(skip_set)}")

    if args.dry_run:
        print(f"\nDRY RUN complete. Files at: {work_dir}")
    else:
        shutil.rmtree(work_dir)
        print(f"\nUpload complete: https://huggingface.co/datasets/{args.target_repo}")


if __name__ == "__main__":
    main()
