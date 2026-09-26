"""Record the real Tetris app while the RL-tuned PACT adapter plays.

Drives headless Chrome via Playwright, captures frames with the DevTools
screencast (frames arrive whenever the page repaints, each with a timestamp),
and writes them to disk for encode.py.
"""
import asyncio
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8848/index_en.html"
W, H = 1280, 1184
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 362.0
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "frames")
OUT.mkdir(parents=True, exist_ok=True)


def health():
    with urllib.request.urlopen("http://127.0.0.1:8848/health", timeout=5) as r:
        return json.loads(r.read())


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True)
        page = await browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        await page.goto(URL)
        await page.wait_for_function("!document.getElementById('ai-btn').disabled", timeout=60000)

        # Warm-up: the first AI move loads the 9B model onto the GPU.
        await page.click("#ai-btn")
        t = time.time()
        while not health().get("model_loaded"):
            await asyncio.sleep(2)
            if time.time() - t > 600:
                raise RuntimeError("model did not load")
        await page.wait_for_timeout(8000)
        print(f"[record] model warm after {time.time() - t:.0f}s", flush=True)

        # Fresh game, AI on, recording on.
        await page.click("#new-game-btn")          # resets the board and turns AI off
        await page.wait_for_timeout(400)
        cdp = await page.context.new_cdp_session(page)
        frames = []

        def on_frame(evt):
            idx = len(frames)
            (OUT / f"{idx:06d}.jpg").write_bytes(base64.b64decode(evt["data"]))
            frames.append(evt["metadata"]["timestamp"])
            asyncio.ensure_future(cdp.send("Page.screencastFrameAck", {"sessionId": evt["sessionId"]}))

        cdp.on("Page.screencastFrame", on_frame)
        await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 92,
                                                 "maxWidth": W, "maxHeight": H, "everyNthFrame": 1})
        await page.wait_for_timeout(500)
        start = time.time()
        await page.click("#ai-btn")
        start_ts = time.time()
        print("[record] AI playing; recording", SECONDS, "s", flush=True)
        stats = []
        while time.time() - start_ts < SECONDS:
            await asyncio.sleep(10)
            s = await page.evaluate("""() => ({score: document.getElementById('score').textContent,
                lines: document.getElementById('lines').textContent,
                level: document.getElementById('level').textContent,
                status: document.getElementById('ai-status').textContent,
                session: window.pactSession})""")
            s["t"] = round(time.time() - start_ts, 1)
            stats.append(s)
            print("[record]", json.dumps(s), flush=True)
        await cdp.send("Page.stopScreencast")
        await page.wait_for_timeout(500)
        meta = {"frames": frames, "click_wall": start_ts, "start_wall": start,
                "viewport": [W, H], "seconds": SECONDS, "stats": stats}
        (OUT / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        print(f"[record] {len(frames)} frames", flush=True)
        await browser.close()


asyncio.run(main())
