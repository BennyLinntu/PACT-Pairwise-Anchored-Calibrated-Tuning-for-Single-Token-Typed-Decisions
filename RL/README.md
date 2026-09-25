# PACT Tetris RL — self-contained

Everything needed to RL-fine-tune the shipped PACT adapter on the Tetris
move-choice task is in this folder. Copy `RL/` anywhere — a server, a
container — and nothing outside it is read.

**This copy ships one completed run**, `runs/tetris-rl/` — `eval_log.jsonl`,
`train_log.jsonl`, `eval_only_log.jsonl`, and only the `latest/` checkpoint
(the adapter weights + config needed to load and serve it, no optimizer
state). The 31 intermediate `step-<N>/` checkpoints a fresh run produces
along the way aren't included here to keep the repo a reasonable size — a
new run regenerates them locally under `runs/<a-new-name>/`, or ask if you
want a specific intermediate step from this run.

## What this does

The [`tetris/` app](../tetris/README.md) hands the PACT model a short list of
candidate Tetris placements, each already described in plain language
(rows cleared, holes, height, bumpiness — worked out by a two-ply lookahead
search, not by the model). Logging which one the model actually picked shows
it doesn't reliably pick the best option on offer — it's a business-document
classifier, never trained to weigh outcomes like these against each other.

This trains it to, with REINFORCE (policy-gradient RL), rather than
replacing it with the heuristic outright — the model keeps making the final
call; RL is how it gets better at making that call correctly. Full design
rationale is in `pact/train_tetris_rl.py`'s own docstring.

## Quickstart

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

python run_all_rl.py
```

That's the whole thing. `run_all_rl.py` runs a real 2-step timing probe on
your actual hardware first (reads real per-step seconds back out of its own
log, not a guess), picks a batch size aimed at ~90s/gradient-step from that,
then launches the full run (2000 steps by default) using
`configs/server_2x40gb_rl.json` — BF16, the model sharded across both GPUs
via `device_map: auto` (needs 2 GPUs with 40GB+ each; for a single smaller
GPU use `--config configs/server_9b_24gb_qlora.json` instead, 4-bit, and
`pip install bitsandbytes`).

Progress:
- Console + `runs/tetris-rl/train_log.jsonl` — every gradient step (loss,
  how often the model agreed with the heuristic's top pick, mean advantage).
- `runs/tetris-rl/eval_log.jsonl` — every `--eval-every` steps (default 50):
  mean score/lines over `--eval-games` (default 5) *greedy* games, i.e. what
  the served app actually does with argmax, not the exploration-sampled
  training policy. This is the number that actually tells you whether it's
  getting better.
- Checkpoints: `runs/tetris-rl/step-<N>/` (kept forever) and
  `runs/tetris-rl/latest/` (overwritten each save) — both are a normal PEFT
  adapter dir, loadable by `PactScorer` or `tetris/server.py` exactly like
  the original shipped adapter.

Ctrl+C is safe at any point — it finishes the in-flight move, saves a
checkpoint, and exits. Resume with:

```bash
python run_all_rl.py --skip-probe --batch-size <N> \
    --resume-from runs/tetris-rl/latest --max-steps <higher than before>
```

(`--skip-probe --batch-size N` because you already know your hardware's
number from the first run; `<N>` is whatever `run_all_rl.py` printed then.)

## Using the trained adapter

Once you're happy with `eval_log.jsonl`'s trend, point the real app at it
with environment variables — **`server.py` has no idea which adapter you
mean otherwise, and silently serves the original shipped one** (this bit
people during testing: the server starts fine and answers moves fine either
way, so nothing *looks* wrong even when it's not using your training run):

```bash
PACT_ADAPTER_DIR=runs/tetris-rl/latest PACT_CONFIG=configs/server_2x40gb_rl.json \
    python tetris/server.py
```

Both paths are relative to this bundle's root if not absolute; `PACT_CONFIG`
only matters if you want the 2-GPU BF16 config instead of the default 4-bit
single-GPU one. Check `curl localhost:8848/health` once it's up — it echoes
back the resolved `config` and `adapter_dir` so you can confirm which one
actually loaded before trusting anything you see in the game.

## What's in this folder

```
run_all_rl.py           the one-command entry point described above
requirements.txt
configs/
  server_2x40gb_rl.json      2 GPUs, 40GB+ each, BF16 (the default)
  server_9b_24gb_qlora.json  1 GPU, 24GB+, 4-bit (fallback)
pact/
  train_tetris_rl.py         the actual training script (full docstring: --help)
  tetris_env.py              pure-Python port of the game engine, for offline rollouts
  config.py, model.py, ...   the rest of pact/ is carried along for import
                             compatibility; only the two files above and
                             config.py/model.py are actually exercised here
nimble/                      vendored upstream modules the prompt contract
                             depends on (same copy the main project uses)
tetris/server.py             only for its build_context / build_schema /
                             describe_candidate — training reuses these
                             directly so the training prompt is byte-identical
                             to what the served app actually sends
results/model/               the starting point: the shipped PACT adapter
```
