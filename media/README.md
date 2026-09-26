# PACT-RL gameplay recording

| File | What |
| --- | --- |
| `pact_rl_tetris_2x.mp4` | 3:00 video at **2× speed** (6 min of real play), 1280×1184, H.264 |
| `pact_rl_tetris_preview.gif` | 14-second excerpt used in the top-level README |
| `record.py` | drives the real app in headless Chrome and captures every repaint |
| `encode.py` | assembles the frames at 2× speed and adds the label panel |

**What is shown.** The RL-tuned adapter (`RL/runs/tetris-rl/latest`, REINFORCE on
top of the shipped PACT adapter) playing the unmodified app in `desktop/app/`,
served by `tetris/server.py` with its default 4-bit config on one 16 GB
consumer GPU (RTX 5060 Ti). One forward pass per piece; the engine lists the
legal placements and the model chooses one. Nothing is scripted or edited: one
continuous game from an empty board, recorded end to end. In the recorded six
minutes the model cleared more than 130 lines and reached level 13 without
topping out, while gravity speeds up with every level. The clock in the label
panel shows real (not video) time.

**Reproduce.**

```bash
# 1. serve the RL-tuned adapter (repo root)
PACT_ADAPTER_DIR=RL/runs/tetris-rl/latest python tetris/server.py

# 2. in a second shell: record 362 s of play, then encode (needs Chrome installed)
python -m pip install playwright imageio-ffmpeg pillow
python media/record.py 362 frames
python media/encode.py frames media/pact_rl_tetris_2x.mp4 media/pact_rl_tetris_preview.gif
```

`encode.py` uses the Segoe UI fonts from `C:/Windows/Fonts`; on another OS point
`FONT_DIR` at any TrueType font directory.

This is a qualitative demonstration, not an evaluation: see Appendix C of
[`paper/main.pdf`](../paper/main.pdf) for the numbers behind the RL run.
