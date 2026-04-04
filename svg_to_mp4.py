import asyncio
from playwright.async_api import async_playwright
import subprocess
import os
import tempfile
import shutil

SVG_PATH = os.path.join(os.path.dirname(__file__), "actor_animated.svg")
OUT_PATH = os.path.join(os.path.dirname(__file__), "actor_animated.mp4")

TOTAL_DURATION_MS = 35000  # 35s: initial build + several slow blink cycles
FRAME_INTERVAL_MS = 16    # ~60fps
WIDTH, HEIGHT = 800, 350   # match SVG viewBox exactly

async def main():
    tmp_dir = tempfile.mkdtemp()

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=2,  # capture at 1600x700 for crisp output
        )
        svg_abs = os.path.abspath(SVG_PATH)
        html = f"""<!DOCTYPE html><html><head><style>
        *{{margin:0;padding:0}}body{{background:#e8e4dc;overflow:hidden}}
        </style></head><body>
        <img src="file://{svg_abs}" width="{WIDTH}" height="{HEIGHT}">
        </body></html>"""
        html_path = os.path.join(os.path.dirname(__file__), "_tmp_render.html")
        with open(html_path, "w") as f:
            f.write(html)
        await page.goto(f"file://{os.path.abspath(html_path)}")

        num_frames = TOTAL_DURATION_MS // FRAME_INTERVAL_MS

        for i in range(num_frames):
            path = os.path.join(tmp_dir, f"frame_{i:05d}.png")
            await page.screenshot(path=path, type="png")
            await page.wait_for_timeout(FRAME_INTERVAL_MS)

            if (i + 1) % 120 == 0:
                print(f"  {(i+1)*FRAME_INTERVAL_MS/1000:.1f}s / {TOTAL_DURATION_MS/1000:.0f}s")

        await browser.close()

    print(f"Encoding MP4 from {num_frames} frames at 60fps...")
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", "60",
        "-i", os.path.join(tmp_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "slow",
        "-movflags", "+faststart",
        "-vf", "scale=1600:700",
        OUT_PATH,
    ], check=True, capture_output=True)

    shutil.rmtree(tmp_dir)
    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"Saved: {OUT_PATH} ({size_mb:.2f} MB)")

asyncio.run(main())
