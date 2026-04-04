"""Strip extra USB cameras from a LeRobot excavator dataset.

This rewrites every parquet file in-place on Hugging Face so that only:
  - observation.images.csi_0_imx219
  - observation.images.usb_0

remain in the dataset schema. It also updates meta/info.json accordingly.

Usage:
    python examples/excavator/strip_v2_4cam_to_2cam.py --dry-run
    python examples/excavator/strip_v2_4cam_to_2cam.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download, snapshot_download


REPO_ID = "verm11/excavator_v2_4cam"
KEEP_IMAGE_COLUMNS = {
    "observation.images.csi_0_imx219",
    "observation.images.usb_0",
}


def load_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    script_path = Path(__file__).resolve()
    for parent in [script_path.parent, *script_path.parents]:
        env_path = parent / ".env"
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
        break

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not found in environment or .env")
    return token


def keep_columns(column_names: list[str]) -> list[str]:
    kept: list[str] = []
    for name in column_names:
        if not name.startswith("observation.images."):
            kept.append(name)
        elif name in KEEP_IMAGE_COLUMNS:
            kept.append(name)
    return kept


def update_info_json(info: dict) -> dict:
    features = info.get("features", {})
    info["features"] = {k: v for k, v in features.items() if not k.startswith("observation.images.")}
    for key in KEEP_IMAGE_COLUMNS:
        if key in features:
            info["features"][key] = features[key]
    return info


def rewrite_one_parquet(local_in: Path, repo_path: str, output_root: Path) -> str:
    table = pq.read_table(local_in)
    desired_columns = keep_columns(table.column_names)
    trimmed = table.select(desired_columns)

    local_out = output_root / repo_path
    local_out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(trimmed, local_out, compression="snappy")
    return repo_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without uploading")
    args = parser.parse_args()

    token = load_hf_token()
    api = HfApi(token=token)
    files = api.list_repo_files(REPO_ID, repo_type="dataset")

    parquet_files = sorted(f for f in files if f.endswith(".parquet"))
    meta_files = [f for f in files if f.startswith("meta/")]

    if "meta/info.json" not in meta_files:
        raise RuntimeError("meta/info.json not found in dataset")

    info_path = hf_hub_download(REPO_ID, "meta/info.json", repo_type="dataset", token=token)
    with open(info_path) as f:
        info = json.load(f)

    old_image_columns = [k for k in info.get("features", {}) if k.startswith("observation.images.")]
    new_info = update_info_json(info)
    new_image_columns = [k for k in new_info.get("features", {}) if k.startswith("observation.images.")]

    print(f"Repo: {REPO_ID}")
    print(f"Parquet files: {len(parquet_files)}")
    print(f"Old image columns: {old_image_columns}")
    print(f"New image columns: {new_image_columns}")

    if args.dry_run:
        sample_path = hf_hub_download(REPO_ID, parquet_files[0], repo_type="dataset", token=token)
        sample_table = pq.read_table(sample_path)
        print(f"Sample parquet: {parquet_files[0]}")
        print(f"Sample old columns: {sample_table.column_names}")
        print(f"Sample new columns: {keep_columns(sample_table.column_names)}")
        print("Dry run complete. No uploads performed.")
        return

    source_dir = Path("tmp/strip_v2_4cam_to_2cam_source")
    upload_dir = Path("tmp/strip_v2_4cam_to_2cam_upload")
    for tmp_dir in (source_dir, upload_dir):
        if not tmp_dir.exists():
            continue
        for path in sorted(tmp_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    source_dir.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path("tmp/strip_v2_4cam_to_2cam_cache")
    if cache_dir.exists():
        for path in sorted(cache_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset snapshot ...", flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        token=token,
        local_dir=source_dir,
        cache_dir=cache_dir,
        allow_patterns=["data/**/*.parquet", "meta/info.json"],
        max_workers=8,
    )

    info_out = upload_dir / "meta" / "info.json"
    info_out.parent.mkdir(parents=True, exist_ok=True)
    with open(info_out, "w") as f:
        json.dump(new_info, f, indent=2)

    print("Rewriting parquet files locally ...", flush=True)
    for idx, repo_path in enumerate(parquet_files, start=1):
        local_in = source_dir / repo_path
        rewrite_one_parquet(local_in, repo_path, upload_dir)
        print(f"[{idx}/{len(parquet_files)}] Rewrote {repo_path}", flush=True)

    print("Uploading rewritten files in one batch commit ...", flush=True)
    api.upload_folder(
        folder_path=str(upload_dir),
        path_in_repo="",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Remove usb_1 and usb_2 from dataset schema",
    )

    print("Done. Dataset updated to 2-camera schema.", flush=True)


if __name__ == "__main__":
    main()
