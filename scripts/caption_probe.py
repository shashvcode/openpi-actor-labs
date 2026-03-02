"""Probe what PaliGemma sees from your robot's cameras.

Feed live camera frames (scene + wrist) to PaliGemma and ask it
questions in natural language. Runs locally on your Mac (CPU) or
on your GPU pod (CUDA).

Usage:
    python scripts/caption_probe.py --scene-cam 0 --wrist-cam 1
    python scripts/caption_probe.py --image /path/to/image.png
"""

import argparse
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor


def load_model(device: str):
    model_id = "google/paligemma-3b-mix-224"
    print(f"Loading {model_id} on {device} ...")
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    ).to(device).eval()
    print("Model loaded.")
    return model, processor


def caption_image(model, processor, image: Image.Image, prompt: str, device: str) -> str:
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)
    decoded = processor.decode(output[0], skip_special_tokens=True)
    if prompt in decoded:
        decoded = decoded[len(prompt):].strip()
    return decoded


def grab_frame(cap, name: str) -> np.ndarray | None:
    ret, frame = cap.read()
    if not ret:
        print(f"  [{name}] Failed to grab frame")
        return None
    frame = cv2.resize(frame, (640, 480))
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def run_live(args, model, processor, device):
    scene_cap = cv2.VideoCapture(args.scene_cam)
    wrist_cap = cv2.VideoCapture(args.wrist_cam)

    if not scene_cap.isOpened():
        print(f"ERROR: Cannot open scene camera (index {args.scene_cam})")
        return
    if not wrist_cap.isOpened():
        print(f"ERROR: Cannot open wrist camera (index {args.wrist_cam})")
        return

    print(f"\nScene camera: index {args.scene_cam}")
    print(f"Wrist camera: index {args.wrist_cam}")
    print("\n--- Interactive mode ---")
    print("Type a question/prompt and press Enter.")
    print("Prefix with 'w:' to ask about the wrist camera (default is scene).")
    print("Type 'both:' to ask both cameras the same question.")
    print("Type 'q' to quit.\n")

    default_prompts = [
        "Describe this image in detail.",
        "What objects do you see?",
        "Where is the bottle?",
        "Is there a robot arm in this image?",
    ]
    print("Example prompts:")
    for p in default_prompts:
        print(f"  - {p}")
    print()

    while True:
        try:
            user_input = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "q":
            break

        scene_frame = grab_frame(scene_cap, "scene")
        wrist_frame = grab_frame(wrist_cap, "wrist")

        if user_input.lower().startswith("both:"):
            prompt = user_input[5:].strip()
            targets = []
            if scene_frame is not None:
                targets.append(("SCENE", scene_frame))
            if wrist_frame is not None:
                targets.append(("WRIST", wrist_frame))
        elif user_input.lower().startswith("w:"):
            prompt = user_input[2:].strip()
            targets = [("WRIST", wrist_frame)] if wrist_frame is not None else []
        else:
            prompt = user_input
            targets = [("SCENE", scene_frame)] if scene_frame is not None else []

        if not targets:
            print("  No valid frames to process.")
            continue

        for cam_name, frame in targets:
            pil_img = Image.fromarray(frame)
            print(f"  [{cam_name}] Thinking...")
            answer = caption_image(model, processor, pil_img, prompt, device)
            print(f"  [{cam_name}] {answer}\n")

    scene_cap.release()
    wrist_cap.release()


def run_static(args, model, processor, device):
    img = Image.open(args.image).convert("RGB")
    print(f"\nLoaded image: {args.image} ({img.size[0]}x{img.size[1]})")
    print("Type a question/prompt and press Enter. Type 'q' to quit.\n")

    while True:
        try:
            prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() == "q":
            break
        print("  Thinking...")
        answer = caption_image(model, processor, img, prompt, device)
        print(f"  {answer}\n")


def main():
    parser = argparse.ArgumentParser(description="Probe PaliGemma's understanding of robot camera views")
    parser.add_argument("--scene-cam", type=int, default=0, help="Scene camera index")
    parser.add_argument("--wrist-cam", type=int, default=1, help="Wrist camera index")
    parser.add_argument("--image", type=str, default=None, help="Path to a static image (skips live cameras)")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, cuda, mps (auto-detected)")
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    print(f"Device: {args.device}")
    model, processor = load_model(args.device)

    if args.image:
        run_static(args, model, processor, args.device)
    else:
        run_live(args, model, processor, args.device)

    print("Done.")


if __name__ == "__main__":
    main()
