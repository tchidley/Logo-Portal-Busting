"""
EDITABLE VIDEO EFFECTS SCRIPT
============================

This script uses the supplied MP4 as its visual source and gives you editable
controls for the overall look and timing. It does not contain the original
Gemini scene as separate 3D objects/layers; it processes the rendered frames.

Requirements:
    pip install opencv-python numpy

Usage:
    1. Put this .py file and gemini_generated_video_a1057997.mp4 in the same folder.
    2. Open the folder in Visual Studio Code.
    3. Edit the SETTINGS section below.
    4. Run: python editable_energy_video.py

The default settings preserve the source image as closely as possible.
"""

from pathlib import Path
import cv2
import numpy as np

# ============================================================
# SETTINGS - CHANGE THESE VALUES
# ============================================================

INPUT_VIDEO = "logo_portal_busting.mp4"
OUTPUT_VIDEO = "editable_energy_video.mp4"

# Output quality / size. Use None to keep the original dimensions.
OUTPUT_WIDTH = None
OUTPUT_HEIGHT = None
OUTPUT_FPS = None                 # None = use source FPS

# Overall image adjustments
BRIGHTNESS = 0                    # -100 to +100
CONTRAST = 1.00                   # 1.0 = unchanged; try 1.1
SATURATION = 1.00                 # 1.0 = unchanged
GAMMA = 1.00                      # 1.0 = unchanged

# Glow enhancement
GLOW_ENABLED = False
GLOW_STRENGTH = 0.35              # 0.0 to 1.0
GLOW_BLUR_SIZE = 31               # odd number; larger = softer glow
GLOW_THRESHOLD = 185               # 0-255; higher targets brighter details

# Purple colour treatment
PURPLE_ENHANCEMENT = 0.00          # 0.0 = off; try 0.10 to 0.30
PURPLE_TINT = (1.00, 0.85, 1.00)  # BGR multipliers

# Framing
ZOOM = 1.00                        # 1.0 = original; 1.05 = slight zoom
SHIFT_X = 0                        # pixels; positive moves image right
SHIFT_Y = 0                        # pixels; positive moves image down

# Optional horizontal energy streaks


# Optional fade
FADE_IN_SECONDS = 0.0
FADE_OUT_SECONDS = 0.0

# ============================================================
# PROCESSING FUNCTIONS
# ============================================================

def odd(value):
    value = max(3, int(value))
    return value if value % 2 else value + 1


def adjust_framing(frame):
    """Apply zoom and image shifts without changing output dimensions."""
    h, w = frame.shape[:2]
    if abs(ZOOM - 1.0) < 1e-6 and SHIFT_X == 0 and SHIFT_Y == 0:
        return frame

    new_w = max(1, int(w / ZOOM))
    new_h = max(1, int(h / ZOOM))
    x1 = max(0, (w - new_w) // 2)
    y1 = max(0, (h - new_h) // 2)
    crop = frame[y1:y1 + new_h, x1:x1 + new_w]
    result = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LANCZOS4)

    matrix = np.float32([[1, 0, SHIFT_X], [0, 1, SHIFT_Y]])
    return cv2.warpAffine(
        result, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


def adjust_colour(frame):
    """Brightness, contrast, saturation, gamma and optional purple treatment."""
    result = frame.astype(np.float32)

    result = (result - 127.5) * CONTRAST + 127.5 + BRIGHTNESS
    result = np.clip(result, 0, 255).astype(np.uint8)

    if abs(SATURATION - 1.0) > 1e-6:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= SATURATION
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if abs(GAMMA - 1.0) > 1e-6:
        inv_gamma = 1.0 / max(0.01, GAMMA)
        lut = np.array([
            np.clip(((i / 255.0) ** inv_gamma) * 255, 0, 255)
            for i in range(256)
        ], dtype=np.uint8)
        result = cv2.LUT(result, lut)

    if PURPLE_ENHANCEMENT > 0:
        # Add a restrained magenta/purple bias while retaining the original detail.
        tint = result.astype(np.float32)
        tint *= np.array(PURPLE_TINT, dtype=np.float32).reshape(1, 1, 3)
        result = cv2.addWeighted(
            result, 1.0 - PURPLE_ENHANCEMENT,
            np.clip(tint, 0, 255).astype(np.uint8),
            PURPLE_ENHANCEMENT, 0
        )

    return result


def add_glow(frame):
    """Extract bright pixels, blur them, and blend the glow back into the frame."""
    if not GLOW_ENABLED or GLOW_STRENGTH <= 0:
        return frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, GLOW_THRESHOLD, 255, cv2.THRESH_BINARY)
    bright = cv2.bitwise_and(frame, frame, mask=mask)

    k = odd(GLOW_BLUR_SIZE)
    glow = cv2.GaussianBlur(bright, (k, k), 0)
    return cv2.addWeighted(frame, 1.0, glow, GLOW_STRENGTH, 0)



def apply_fades(frame, seconds, duration):
    """Apply optional fade-in and fade-out."""
    factor = 1.0
    if FADE_IN_SECONDS > 0:
        factor = min(factor, seconds / FADE_IN_SECONDS)
    if FADE_OUT_SECONDS > 0:
        remaining = duration - seconds
        factor = min(factor, max(0.0, remaining / FADE_OUT_SECONDS))
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def process_video():
    input_path = Path(__file__).resolve().parent / INPUT_VIDEO
    output_path = Path(__file__).resolve().parent / OUTPUT_VIDEO

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cannot find input video: {input_path}\n"
            "Put the MP4 in the same folder as this Python script, "
            "or change INPUT_VIDEO at the top."
        )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    fps = OUTPUT_FPS or source_fps
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = OUTPUT_WIDTH or source_w
    out_h = OUTPUT_HEIGHT or source_h
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / source_fps if source_fps else 0

    # MP4V is widely supported by OpenCV. If needed, convert the result to H.264
    # using a video editor or FFmpeg.
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(out_w), int(out_h))
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output MP4.")

    frame_index = 0
    print(f"Reading: {input_path.name}")
    print(f"Source: {source_w}x{source_h} at {source_fps:.2f} FPS")
    print("Rendering...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        seconds = frame_index / source_fps

        frame = adjust_framing(frame)
        frame = adjust_colour(frame)
        frame = add_glow(frame)
        frame = apply_fades(frame, seconds, duration)

    



        # Remove unwanted horizontal effects from final 7 frames
        if frame_index >= frame_count - 7:
            frame = cv2.medianBlur(frame, 5)

        if (out_w, out_h) != (source_w, source_h):
            frame = cv2.resize(
                frame, (int(out_w), int(out_h)),
                interpolation=cv2.INTER_LANCZOS4
            )

        writer.write(frame)
        frame_index += 1

        if frame_index % 48 == 0:
            print(f"  Processed {frame_index}/{frame_count} frames")

    cap.release()
    writer.release()
    print(f"Finished: {output_path}")


if __name__ == "__main__":
    process_video()
