#!/usr/bin/env python
"""Local backend for tetris/: serves the static game and one AI endpoint.

The model was fine-tuned to answer typed schema questions: given a context
and a short list of described choices, pick the single best-fitting one. So
that's exactly what each move is. The frontend (engine.js:placementStats)
works out every legal placement for the current piece *and* simulates the
consequence of each one -- lines cleared, holes left behind, resulting
height, skyline bumpiness -- because spatial simulation is the engine's job,
not something a classification model can be expected to do from a raw
board. This backend turns those already-computed consequences into plain-
language choice descriptions and asks the model to pick one. The model can
still pick a *bad* option (nothing stops it preferring a tall wobbly stack),
but it can never pick an *illegal* one, and it's judging real outcomes
instead of guessing at board geometry.

Run from the repo root:

    python tetris/server.py
    # then open http://127.0.0.1:8848/index_en.html (or index_zh.html)

By default this serves the shipped adapter (results/model) via the 4-bit
single-GPU config. To serve a different adapter (e.g. an RL-fine-tuned
checkpoint from pact/train_tetris_rl.py) and/or a different hardware config
(e.g. configs/server_2x40gb_rl.json for a 2-GPU BF16 box), set:

    PACT_ADAPTER_DIR=runs/tetris-rl/latest PACT_CONFIG=configs/server_2x40gb_rl.json \\
        python tetris/server.py

Both are relative to this bundle's root if not absolute. If you started the
server, played a move, and it doesn't seem to reflect your RL training at
all -- check these two first; the server has no way to know which adapter
you meant without being told.
"""
import os
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))

CONFIG_PATH = os.environ.get("PACT_CONFIG", "configs/server_9b_24gb_qlora.json")
ADAPTER_DIR = os.environ.get("PACT_ADAPTER_DIR", "results/model")

from flask import Flask, jsonify, request, send_from_directory  # noqa: E402

app = Flask(__name__, static_folder=None)
# The frontend lives in desktop/app/ -- it's also the source the Electron
# build packages, so there's one copy of the game, not two.
STATIC_DIR = BUNDLE / "desktop" / "app"


@app.after_request
def allow_local_origins(response):
    # The packaged desktop app loads pages from file://, which fetch() sends
    # as a "null" origin -- allow it (and any localhost dev server) to reach
    # this purely-local API. Not a general-purpose CORS policy: this server
    # is meant to be bound to 127.0.0.1 only (see __main__ below).
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

_scorer = None  # loaded lazily, once, on first /ai_move call


def _resolve(path_str):
    path = Path(path_str)
    return path if path.is_absolute() else BUNDLE / path


def get_scorer():
    global _scorer
    if _scorer is None:
        config_path = _resolve(CONFIG_PATH)
        adapter_dir = _resolve(ADAPTER_DIR)
        print(f"[server] loading PACT model: config={config_path} adapter={adapter_dir} "
              "(first AI move will be slow: ~10-20s)...")
        from pact.config import PactConfig
        from pact.model import load_tokenizer, _load_base, LogitReader
        import torch

        cfg = PactConfig.load(config_path)
        tokenizer = load_tokenizer(cfg)
        base, _ = _load_base(cfg, device="cuda" if torch.cuda.is_available() else "cpu",
                              dtype=torch.bfloat16, log=print)
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
        _scorer = {
            "model": model, "tokenizer": tokenizer, "reader": LogitReader(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "max_length": cfg.data.max_length,
        }
        print("[server] model ready.")
    return _scorer


ROTATION_LABEL = {0: "no rotation", 1: "rotated 90° clockwise",
                   2: "rotated 180°", 3: "rotated 270° clockwise"}


def build_context(payload):
    heights = payload.get("column_heights", [])
    heights_text = ", ".join(f"col{i}={h}" for i, h in enumerate(heights))
    hold = payload.get("hold") or "none"
    return (
        "You are placing falling pieces in a game of Tetris, 10 columns wide, "
        "20 rows tall. Below is a numbered list of every legal way to place the "
        "current piece. Each option already has its consequences worked out for "
        "you two moves deep: what placing it does right now (rows completed, "
        "holes created, stack height, skyline evenness), and then, assuming the "
        "*next* piece (already known) is then placed in its own best spot, what "
        "the board looks like after that too. The lookahead is what matters most "
        "-- a move that looks fine by itself can still be a trap if it wrecks "
        "every good option for the piece right after it.\n\n"
        f"Column heights before this move (0 = empty, higher = taller): {heights_text}\n"
        f"Current piece: {payload.get('current')}\n"
        f"Next piece: {payload.get('next')}\n"
        f"Held piece: {hold}\n"
        f"Lines cleared so far this game: {payload.get('lines', 0)}\n\n"
        "Pick the placement with the best two-move trade-off. If any option is "
        "marked BEST CHOICE (it clears a row immediately), pick it -- clearing "
        "rows is worth far more than any other consideration here. Otherwise: "
        "creating holes is bad (a trapped hole usually can't be fixed later), a "
        "low flat stack survives longer than a tall or jagged one, and a move "
        "that leaves the next piece nowhere legal to land ends the game -- avoid "
        "those even if the immediate result looks fine."
    )


def describe_candidate(c):
    lines = c["lines_cleared"]
    clears = "clears no rows" if lines == 0 else (
        "clears 1 row" if lines == 1 else f"clears {lines} rows")
    holes = "no new holes" if c["holes"] == 0 else (
        "1 new hole" if c["holes"] == 1 else f"{c['holes']} new holes")
    rotation_label = ROTATION_LABEL.get(c["rotation"], f"rotation {c['rotation']}")
    prefix = "BEST CHOICE, clears a row now: " if lines > 0 else ""
    now = (f"{prefix}{rotation_label}, column {c['col']}: {clears}, {holes}, "
           f"stack height {c['max_height']}/20, bumpiness {c['bumpiness']}")
    if c.get("topout_next"):
        return now + ". WARNING: leaves no legal spot for the next piece -- this ends the game."
    follow_lines = c["lines_cleared"] + c["followup_lines_cleared"]
    follow_clears = "no rows cleared in total" if follow_lines == 0 else (
        "1 row cleared in total" if follow_lines == 1 else f"{follow_lines} rows cleared in total")
    follow_holes = "0 holes left" if c["followup_holes"] == 0 else (
        "1 hole left" if c["followup_holes"] == 1 else f"{c['followup_holes']} holes left")
    return (f"{now}. Then, with the best spot for the next piece too: "
            f"{follow_clears} across both pieces, {follow_holes} on the board, "
            f"height {c['followup_max_height']}/20, bumpiness {c['followup_bumpiness']}.")


def build_schema(candidates):
    choices = [str(i) for i in range(len(candidates))]
    return {
        "placement": {
            "type": "enum",
            "description": ("The best placement for the current piece, given its "
                            "worked-out consequences."),
            "choices": choices,
            "choice_descriptions": {str(i): describe_candidate(c)
                                     for i, c in enumerate(candidates)},
        },
    }


@app.post("/ai_move")
def ai_move():
    payload = request.get_json(force=True)
    candidates = payload.get("candidates") or []
    if not candidates:
        return jsonify({"index": 0})
    scorer = get_scorer()
    context = build_context(payload)
    schema = build_schema(candidates)

    from nimble.scoring.parallel_schema import prepare_prompts, validate_schema

    validate_schema(schema)
    prompt = prepare_prompts(scorer["tokenizer"], context, schema, scorer["max_length"])

    import torch

    index = 0  # the single field, "placement"
    ids = torch.tensor([list(prompt.full_ids[index])], device=scorer["device"])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        if scorer["device"] == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = scorer["reader"](scorer["model"], ids, mask).float()[0]
        else:
            logits = scorer["reader"](scorer["model"], ids, mask).float()[0]
    candidate_logits = [float(logits[code]) for code in prompt.candidate_ids[index]]
    best = max(range(len(candidate_logits)), key=lambda i: candidate_logits[i])

    return jsonify({"index": int(schema["placement"]["choices"][best])})


@app.get("/health")
def health():
    return jsonify({"ok": True, "model_loaded": _scorer is not None,
                    "config": str(_resolve(CONFIG_PATH)), "adapter_dir": str(_resolve(ADAPTER_DIR))})


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index_en.html")


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


if __name__ == "__main__":
    print("[server] tetris app: http://127.0.0.1:8848/index_en.html "
          "(or index_zh.html)")
    print(f"[server] will serve config={_resolve(CONFIG_PATH)} adapter={_resolve(ADAPTER_DIR)} "
          "(override with PACT_CONFIG / PACT_ADAPTER_DIR env vars) -- "
          "check http://127.0.0.1:8848/health once running to confirm.")
    app.run(host="127.0.0.1", port=8848, debug=False)
