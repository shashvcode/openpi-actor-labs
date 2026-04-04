import subprocess
import os

DIR = os.path.dirname(__file__)
POP_MP3 = os.path.join(DIR, "pop.mp3")
VIDEO = os.path.join(DIR, "actor_animated.mp4")
OUTPUT = os.path.join(DIR, "actor_animated_sound.mp4")
CLICK_WAV = os.path.join(DIR, "_click.wav")
AUDIO_MIX = os.path.join(DIR, "_mixed_audio.wav")

VIDEO_DURATION = 35.0

# Timing from SVG
POP_TIME = 1.0        # arc emergence
TEXT_START = 4.2       # text type-out begins
TEXT_DUR = 1.3         # text type-out duration
TEXT = "ACTOR LABS"    # 10 characters
CLICK_INTERVAL = TEXT_DUR / len(TEXT)  # ~0.13s per character

# 1. Generate a short keyboard click sound (30ms percussive noise burst)
subprocess.run([
    "ffmpeg", "-y",
    "-f", "lavfi", "-i",
    "anoisesrc=d=0.03:c=white:a=0.4,highpass=f=2000,lowpass=f=8000,afade=t=out:st=0.01:d=0.02",
    "-ar", "44100", "-ac", "1",
    CLICK_WAV,
], check=True, capture_output=True)
print("Generated click sound")

# 2. Build ffmpeg filter to place all sounds on a timeline
# Create a silent base track for the full duration
inputs = [
    "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={VIDEO_DURATION}",  # [0] silence
    "-i", POP_MP3,      # [1] pop
    "-i", CLICK_WAV,    # [2] click
]

# Build the complex filter
filter_parts = []

# Pop sound at arc emergence (with volume boost)
# Video=0, silence=1, pop=2, click=3
filter_parts.append(f"[2:a]adelay={int(POP_TIME*1000)}|{int(POP_TIME*1000)},volume=1.5[pop]")

# Generate click instances for each character
click_labels = []
for i, char in enumerate(TEXT):
    t_ms = int((TEXT_START + i * CLICK_INTERVAL) * 1000)
    label = f"click{i}"
    vol = 0.8 if char == " " else 1.2
    filter_parts.append(f"[3:a]adelay={t_ms}|{t_ms},volume={vol}[{label}]")
    click_labels.append(f"[{label}]")

# Mix all together
all_inputs = "[1:a][pop]" + "".join(click_labels)
n_streams = 2 + len(click_labels)
filter_parts.append(f"{all_inputs}amix=inputs={n_streams}:duration=first:dropout_transition=0,volume={n_streams}[out]")

filter_complex = ";".join(filter_parts)

# 3. Combine video + audio
cmd = [
    "ffmpeg", "-y",
    "-i", VIDEO,
    *inputs,
    "-filter_complex", filter_complex,
    "-map", "0:v",
    "-map", "[out]",
    "-c:v", "copy",
    "-c:a", "aac", "-b:a", "128k",
    "-shortest",
    "-movflags", "+faststart",
    OUTPUT,
]

print("Mixing audio...")
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    raise RuntimeError("ffmpeg failed")

# Clean up
os.remove(CLICK_WAV)

size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
print(f"Saved: {OUTPUT} ({size_mb:.2f} MB)")
