import asyncio
from playwright.async_api import async_playwright
from PIL import Image
import io
import os

SVG_PATH = os.path.join(os.path.dirname(__file__), "actor_animated.svg")
OUT_PATH = os.path.join(os.path.dirname(__file__), "actor_animated.gif")

TOTAL_DURATION_MS = 7000  # capture ~7s (full build + a couple blinks)
FRAME_INTERVAL_MS = 33   # ~30fps
SCALE = 2                # retina-quality

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": 800 * SCALE, "height": 350 * SCALE},
            device_scale_factor=1,
        )
        file_url = f"file://{os.path.abspath(SVG_PATH)}"
        await page.goto(file_url)
        await page.set_viewport_size({"width": 800, "height": 350})

        frames = []
        num_frames = TOTAL_DURATION_MS // FRAME_INTERVAL_MS

        for i in range(num_frames):
            png_bytes = await page.screenshot(type="png")
            img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            # Convert to RGB with white bg for GIF compat
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            composite = Image.alpha_composite(bg, img)
            frames.append(composite.convert("RGB"))
            await page.wait_for_timeout(FRAME_INTERVAL_MS)

            if (i + 1) % 30 == 0:
                print(f"  captured {i+1}/{num_frames} frames...")

        await browser.close()

    print(f"Assembling GIF from {len(frames)} frames...")
    frames[0].save(
        OUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_INTERVAL_MS,
        loop=0,
        optimize=True,
    )
    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"Saved: {OUT_PATH} ({size_mb:.1f} MB)")

asyncio.run(main())
