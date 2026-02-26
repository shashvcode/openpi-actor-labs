"""Re-compress large episode images to ~100KB each (matching the small episodes).

The model resizes everything to 224x224 anyway, so high-res JPEG is wasted space.
This brings the dataset from ~17 GB to ~4 GB.
"""

import io
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

DATA_DIR = pathlib.Path("./so100_merged/data/chunk-000")
TARGET_QUALITY = 50
SIZE_THRESHOLD = 50_000_000  # 50 MB - only recompress files larger than this


def recompress_image_bytes(img_bytes: bytes, quality: int = TARGET_QUALITY) -> bytes:
    img = Image.open(io.BytesIO(img_bytes))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def process_episode(fpath: pathlib.Path) -> tuple[int, float, float]:
    """Returns (num_frames, old_size_mb, new_size_mb)."""
    old_size = fpath.stat().st_size / 1e6

    pf = pq.ParquetFile(fpath)
    new_batches = []

    for batch in pf.iter_batches(batch_size=100):
        scenes = batch.column("observation.images.scene").to_pylist()
        wrists = batch.column("observation.images.wrist").to_pylist()

        new_scenes = []
        new_wrists = []
        for s, w in zip(scenes, wrists):
            new_scenes.append({
                "bytes": recompress_image_bytes(s["bytes"]),
                "path": s["path"],
            })
            new_wrists.append({
                "bytes": recompress_image_bytes(w["bytes"]),
                "path": w["path"],
            })

        img_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
        batch = batch.set_column(
            batch.schema.get_field_index("observation.images.scene"),
            "observation.images.scene",
            pa.array(new_scenes, type=img_type),
        )
        batch = batch.set_column(
            batch.schema.get_field_index("observation.images.wrist"),
            "observation.images.wrist",
            pa.array(new_wrists, type=img_type),
        )
        new_batches.append(batch)

    table = pa.Table.from_batches(new_batches)
    pq.write_table(table, fpath)
    new_size = fpath.stat().st_size / 1e6
    return len(table), old_size, new_size


def main():
    files = sorted(DATA_DIR.glob("episode_*.parquet"))
    large_files = [f for f in files if f.stat().st_size > SIZE_THRESHOLD]

    print(f"Found {len(large_files)} episodes > 50 MB to recompress")
    total_old = 0
    total_new = 0

    for i, f in enumerate(large_files):
        nframes, old_mb, new_mb = process_episode(f)
        total_old += old_mb
        total_new += new_mb
        print(f"  [{i+1}/{len(large_files)}] {f.name}: {old_mb:.0f} MB -> {new_mb:.0f} MB ({nframes} frames)")

    print(f"\nSaved: {total_old - total_new:.0f} MB ({total_old:.0f} -> {total_new:.0f} MB)")

    all_size = sum(f.stat().st_size for f in files) / 1e9
    print(f"Total dataset size: {all_size:.1f} GB")


if __name__ == "__main__":
    main()
