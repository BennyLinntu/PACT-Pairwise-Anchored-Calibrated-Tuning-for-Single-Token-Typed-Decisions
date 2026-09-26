# PACT plays Tetris

A from-scratch Tetris (`engine.js`, no libraries) in two full UI languages,
playable by hand or handed to the shipped PACT adapter. The game itself lives
in [`../desktop/app`](../desktop/app) — that's the single copy `server.py`
serves and the [desktop app](../desktop/README.md) packages; nothing here
duplicates it.

```bash
python tetris/server.py
# open http://127.0.0.1:8848/index_en.html  (or index_zh.html)
```

Want a double-click Windows/Mac app instead of a browser tab? See
[`../desktop/README.md`](../desktop/README.md) — manual play works there with
no Python at all; this server is only needed for AI play.

`server.py` serves the static files and one endpoint, `/ai_move`. The model
was fine-tuned to answer schema questions (pick the best-fitting choice from
a short list), not to play games, so each new piece is framed as exactly that
kind of question: a single enum field, "which placement", over a short list
of legal placements the engine already worked out for it — one forward pass
per piece. The engine (not the model) enforces Tetris legality; every
candidate offered is already a real legal move, so the model can only ever
choose badly, never illegally.

**Two-ply lookahead, done by the engine, not the model.** A purely greedy
one-piece choice reliably leaves a stray hole that blocks the last cell of an
otherwise-complete row. So for every current-piece placement, `engine.js`
also searches every placement of the *known* next piece on top of it and
keeps the best two-piece outcome — that's what ranks and filters the
candidates the model gets to choose from (`ai_client.js`, weights from the
well-known Dellacherie/Lee heuristic). Any candidate that clears a row
immediately is always kept and flagged `BEST CHOICE` in its description,
regardless of how it ranks otherwise — clearing a row is worth far more than
the heuristic's continuous score can convey on its own, and it's cheap
insurance against the model not reliably picking the actual top-ranked
option (it doesn't, reliably — logged and confirmed).

**Board balance.** Bumpiness (the standard Dellacherie/Lee term) only prices
the *seam* between a tall region and an empty one, once -- nothing about
piling on further on either side changes that seam, so it gave no reason to
stop favouring whichever side already had structure, and games would
routinely finish with entire columns untouched. `heightVariance` (ours, not
part of the original four) prices the spread across *every* column instead
of just neighbours, and is what actually rewards using the whole board width
over piling higher where it's already tall.

**Expect uneven, not masterful, play.** This is a business-document
classifier applied out of distribution, not a specialized game-playing
model — it does clear lines and chase real scores now, but don't expect
Tetris-grade strategy. See the note in-app.

First AI move takes ~10-20s (loading the 4-bit model onto the GPU); every
move after that is one ~0.3-0.5s forward pass plus a client-side lookahead
search (a few hundred to ~1,000 cheap board simulations, comfortably under
100ms).

Manual controls: arrows to move/rotate, `Z` to rotate the other way, space to
hard-drop, `C` to hold, `P` to pause.

## Making it actually reliable: RL fine-tuning

The candidate list handed to the model is already heuristically pre-filtered
to good options, but logging which index the model actually picked shows it
doesn't reliably pick the best one on offer -- it's a business-document
classifier, never trained to weigh five short paragraphs of Tetris outcomes
against each other. `pact/train_tetris_rl.py` closes that gap by training it
to, with REINFORCE (policy-gradient RL), instead of replacing it with the
heuristic outright:

```bash
# measure real per-step time on your hardware first, 1 step, no side effects
python pact/train_tetris_rl.py --config configs/server_2x40gb_rl.json \
    --max-steps 1 --batch-size 1

# a real run
python pact/train_tetris_rl.py --config configs/server_2x40gb_rl.json \
    --max-steps 2000 --batch-size 16 --eval-every 50 --save-every 50
```

`configs/server_2x40gb_rl.json` shards the model BF16 across both GPUs via
`device_map: auto` (80 GB combined comfortably fits 9B BF16 + LoRA + rollout
activations, no need for 4-bit here). `pact/tetris_env.py` is a pure-Python
port of `engine.js`'s simulation core (piece shapes, placement search, the
two-ply lookahead, the same heightVariance-augmented ranking), so training
states are generated without a browser; the actual prompt is built by
importing `tetris/server.py`'s `build_context` / `build_schema` /
`describe_candidate` directly, so training and serving can never drift apart.

Each move is treated as an independent contextual bandit, not a long
episode: the lookahead heuristic already scores every candidate at a state,
not just the chosen one, so the state's own mean candidate score is an
exact, zero-cost baseline -- no value network or GAE needed. Reward for the
move actually taken is `(its heuristic score) − (that baseline)`; loss is
`-log_prob(action) × reward`, plus a small entropy bonus (`--entropy-weight`)
against premature convergence. Full details in the script's own docstring
(`python pact/train_tetris_rl.py --help`).

Tested end-to-end on a single 16 GB card (4-bit, `configs/server_9b_24gb_qlora.json`):
training steps, checkpointing, and `--resume-from` all verified working.
Ctrl+C-safe (finishes the in-flight move, then saves). Progress is written to
`<output-dir>/train_log.jsonl` (every step) and `eval_log.jsonl` (every
`--eval-every` steps: mean score/lines over `--eval-games` *greedy* -- not
sampled -- games, i.e. what the served app would actually do).

### A completed run, and how to play against it

`RL/` in the repo root has a real 1,574-step run on 2×40GB GPUs, checkpoint
and all (`RL/runs/tetris-rl/`). `top_pick_rate` -- how often the sampled move
matched the heuristic's own top-ranked candidate -- rose from 0.19 at step 1
to 0.875 by step 16, and its 100-step average stayed between 0.88 and 0.99
from step 200 to the end of training.
Greedy eval under a fixed 400-piece cap stayed flat at ~17,000-18,000 mean
score the whole run -- because the cap is what ends the game, not skill; one
uncapped eval game at the final checkpoint ran to 1,500 pieces, 598 lines,
score 65,700 before we stopped it. Full numbers: `RL/runs/tetris-rl/eval_log.jsonl`,
`train_log.jsonl`, `eval_only_log.jsonl`.

To play against that checkpoint instead of the original shipped adapter,
point the server at it (paths are relative to the repo root):

```bash
PACT_ADAPTER_DIR=RL/runs/tetris-rl/latest PACT_CONFIG=configs/server_2x40gb_rl.json \
    python tetris/server.py
```

Check `curl localhost:8848/health` once it's up -- it echoes back the resolved
`adapter_dir` so you can confirm the RL-trained one actually loaded before
trusting anything you see in the game. Without these two environment
variables the server defaults to `results/model`, the original adapter
evaluated in the paper -- deliberately: that default is what makes the
"shipped adapter" in the rest of this repo an unambiguous, fixed artifact.
