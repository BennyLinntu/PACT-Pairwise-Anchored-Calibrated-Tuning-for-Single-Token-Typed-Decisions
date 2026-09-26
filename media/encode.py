"""Turn the screencast frames into a 2x-speed MP4 plus a short GIF preview."""
import bisect
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FRAMES = Path(sys.argv[1])
OUT_MP4 = Path(sys.argv[2])
OUT_GIF = Path(sys.argv[3])
SPEED, FPS, VIDEO_SECONDS = 2.0, 30, 180.0
LEAD = 1.0  # real seconds shown before "Let PACT play" is pressed

meta = json.loads((FRAMES / "meta.json").read_text(encoding="utf-8"))
stamps = meta["frames"]
W, H = meta["viewport"]
t_click = meta["click_wall"]
t_start = t_click - LEAD
n_out = int(VIDEO_SECONDS * FPS)
if stamps[-1] < t_start + (n_out - 1) * SPEED / FPS:
    print(f"warning: recording ends at {stamps[-1] - t_start:.1f}s real")

FONT_DIR = Path("C:/Windows/Fonts")
bold = ImageFont.truetype(str(FONT_DIR / "segoeuib.ttf"), 22)
regular = ImageFont.truetype(str(FONT_DIR / "segoeui.ttf"), 15)
small = ImageFont.truetype(str(FONT_DIR / "segoeui.ttf"), 13)
clock_font = ImageFont.truetype(str(FONT_DIR / "segoeuib.ttf"), 30)
TEAL, AMBER, INK, MUTED = (14, 140, 134), (238, 138, 46), (230, 233, 238), (150, 158, 170)
BOX = (166, 712, 440, 1060)  # empty area of the board panel, below the playfield


def overlay(img, real_t):
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle(BOX, radius=10, fill=(18, 21, 28, 235), outline=(14, 140, 134, 255), width=2)
    x, y = BOX[0] + 16, BOX[1] + 14
    d.text((x, y), "PACT-RL", font=bold, fill=TEAL)
    y += 34
    for line in ("REINFORCE-tuned LoRA adapter", "on Qwen3.5-9B (4-bit)",
                 "one forward pass per piece", "1x RTX 5060 Ti 16 GB"):
        d.text((x, y), line, font=regular, fill=INK)
        y += 22
    y += 14
    d.rounded_rectangle((x, y, x + 118, y + 34), radius=8, fill=(238, 138, 46, 255))
    d.text((x + 13, y + 5), "2x SPEED", font=bold, fill=(20, 20, 24))
    y += 52
    d.text((x, y), "real time", font=small, fill=MUTED)
    t = max(0.0, real_t)
    d.text((x, y + 16), f"{int(t // 60):02d}:{int(t % 60):02d}", font=clock_font, fill=INK)
    y += 62
    d.text((x, y), "no scripted moves: the engine lists", font=small, fill=MUTED)
    d.text((x, y + 17), "legal placements, the model picks one", font=small, fill=MUTED)
    return img


cmd_writer = imageio_ffmpeg.write_frames(
    str(OUT_MP4), (W, H), fps=FPS, codec="libx264", quality=None,
    output_params=["-crf", "24", "-preset", "slow", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart"], macro_block_size=8)
cmd_writer.send(None)
cache_idx, cache_img = None, None
for i in range(n_out):
    real = t_start + i * SPEED / FPS
    j = max(0, bisect.bisect_right(stamps, real) - 1)
    if j != cache_idx:
        cache_img = Image.open(FRAMES / f"{j:06d}.jpg").convert("RGB")
        if cache_img.size != (W, H):
            cache_img = cache_img.resize((W, H))
        cache_idx = j
    frame = overlay(cache_img.copy(), real - t_click)
    cmd_writer.send(np.asarray(frame).tobytes())
cmd_writer.close()
print("wrote", OUT_MP4, OUT_MP4.stat().st_size / 1e6, "MB")

# GIF preview: 12 s of the video from 0:40, 10 fps, 520 px wide, palette-optimised.
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
vf = "fps=10,scale=520:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
subprocess.run([ffmpeg, "-y", "-ss", "40", "-t", "12", "-i", str(OUT_MP4), "-vf", vf,
                "-loop", "0", str(OUT_GIF)], check=True, capture_output=True)
print("wrote", OUT_GIF, OUT_GIF.stat().st_size / 1e6, "MB")
