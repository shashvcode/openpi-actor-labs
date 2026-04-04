import asyncio
from playwright.async_api import async_playwright
from PIL import Image
import io
import os
import subprocess

SVG_PATH = os.path.join(os.path.dirname(__file__), "actor_animated.svg")
OUT_RAW = os.path.join(os.path.dirname(__file__), "actor_animated_raw.gif")
OUT_OPT = os.path.join(os.path.dirname(__file__), "actor_animated.gif")

TOTAL_DURATION_MS = 7000
WIDTH, HEIGHT = 800, 350

# Adaptive frame rate: capture fast during animations, slow during static parts
# Build phase: 0-3.7s (fast), static+blink: 3.7-7s (fast during blink, slow otherwise)

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": WIDTH, "height": HEIGHT},
            device_scale_factor=2,
        )
        await page.goto(f"file://{os.path.abspath(SVG_PATH)}")

        frames = []
        durations = []
        t = 0
        prev_bytes = None

        while t < TOTAL_DURATION_MS:
            png_bytes = await page.screenshot(type="png")
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            # Resize to 1x for GIF (captured at 2x for quality sampling)
            img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)

            # Deduplicate identical frames by extending duration
            raw = img.tobytes()
            if prev_bytes == raw and frames:
                durations[-1] += 20
            else:
                frames.append(img)
                durations.append(20)
                prev_bytes = raw

            t += 20
            await page.wait_for_timeout(20)

            if t % 1000 < 20:
                print(f"  {t/1000:.0f}s / {TOTAL_DURATION_MS/1000:.0f}s captured ({len(frames)} unique frames)")

        await browser.close()

    print(f"\n{len(frames)} unique frames (deduped from {TOTAL_DURATION_MS//20} total)")

    # Quantize to a tight palette (the logo uses very few colors)
    quantized = []
    for f in frames:
        q = f.quantize(colors=32, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        quantized.append(q)

    print("Saving raw GIF...")
    quantized[0].save(
        OUT_RAW,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    raw_size = os.path.getsize(OUT_RAW)

    # Run gifsicle for max optimization
    print("Running gifsicle optimization...")
    subprocess.run([
        "gifsicle", "-O3",
        "--lossy=30",
        "--colors", "32",
        "--no-extensions",
        OUT_RAW, "-o", OUT_OPT,
    ], check=True)

    opt_size = os.path.getsize(OUT_OPT)
    print(f"\nRaw:       {raw_size/1024:.1f} KB")
    print(f"Optimized: {opt_size/1024:.1f} KB ({100*(1-opt_size/raw_size):.0f}% smaller)")
    print(f"Saved: {OUT_OPT}")

    os.remove(OUT_RAW)

asyncio.run(main())
