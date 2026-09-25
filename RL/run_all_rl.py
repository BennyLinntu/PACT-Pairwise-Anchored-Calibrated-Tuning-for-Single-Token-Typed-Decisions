#!/usr/bin/env python
"""One command, two phases:

    python run_all_rl.py

Phase 1 runs a real 2-step timing probe (batch size 1) and reads the actual
per-step time back out of its own log -- not a guess, not a formula applied
to a single noisy combined measurement -- and uses that to pick a batch size
for phase 2 that keeps a full gradient step to roughly a target duration
(default ~90s; a bigger batch means a smoother but slower-per-step gradient,
a smaller one the opposite -- see --target-step-seconds).

Phase 2 launches the real run with that batch size via pact/train_tetris_rl.py
(see that file's own --help for every flag and the full design rationale).
Ctrl+C during phase 2 is forwarded and handled there (finishes the in-flight
move, saves, exits cleanly).

Skip the probe and pick your own numbers directly:

    python run_all_rl.py --skip-probe --batch-size 16 --max-steps 2000

Resume a previous run:

    python run_all_rl.py --skip-probe --batch-size 16 \\
        --resume-from runs/tetris-rl/latest --max-steps 4000
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "pact" / "train_tetris_rl.py"


def run(args):
    print("$ " + " ".join(args), flush=True)
    return subprocess.run(args, cwd=str(HERE)).returncode


def probe_batch_size(python, config, target_step_seconds, min_batch, max_batch):
    probe_dir = HERE / "runs" / "_probe"
    rc = run([python, str(SCRIPT), "--config", config,
             "--max-steps", "2", "--batch-size", "1",
             "--eval-every", "0", "--save-every", "0",
             "--output-dir", str(probe_dir)])
    if rc != 0:
        print("\nProbe failed -- see the error above. Not starting the full run.", file=sys.stderr)
        sys.exit(rc)

    log_path = probe_dir / "train_log.jsonl"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        print(f"\nProbe produced no log records ({log_path}) -- can't size the batch. "
              "Pass --skip-probe --batch-size N to bypass this.", file=sys.stderr)
        sys.exit(1)
    # Step 1 can carry one-time warmup cost (first-ever CUDA kernel launch of
    # a given shape, etc); step 2 is the more honest steady-state number.
    per_move_seconds = records[-1]["step_seconds"]
    print(f"\nprobe: step 1 = {records[0]['step_seconds']:.1f}s/move, "
          f"step 2 = {per_move_seconds:.1f}s/move (using step 2 as the estimate)")

    batch_size = round(target_step_seconds / max(per_move_seconds, 0.1))
    batch_size = max(min_batch, min(max_batch, batch_size))
    print(f"picking --batch-size {batch_size} "
          f"(~{batch_size * per_move_seconds:.0f}s/step at this speed)\n")
    return batch_size


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(HERE / "configs" / "server_2x40gb_rl.json"),
                        help="Falls back to configs/server_9b_24gb_qlora.json (4-bit, single "
                             "GPU) if you don't have 2x40GB+.")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--output-dir", default=str(HERE / "runs" / "tetris-rl"))
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--skip-probe", action="store_true",
                        help="go straight to the full run; requires --batch-size")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="force this batch size (implies --skip-probe if the probe would "
                             "otherwise run); required when --skip-probe is set")
    parser.add_argument("--target-step-seconds", type=float, default=90.0,
                        help="what the probe aims a full gradient step at when picking a batch "
                             "size (default ~90s/step: frequent enough logging to watch "
                             "progress, large enough batch for a less noisy gradient)")
    parser.add_argument("--min-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.skip_probe and args.batch_size is None:
        print("--skip-probe requires --batch-size N", file=sys.stderr)
        sys.exit(2)

    python = sys.executable

    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        print("=== phase 1: timing probe ===", flush=True)
        batch_size = probe_batch_size(python, args.config, args.target_step_seconds,
                                      args.min_batch_size, args.max_batch_size)

    print("=== phase 2: full run ===", flush=True)
    full_args = [python, str(SCRIPT), "--config", args.config,
                "--max-steps", str(args.max_steps), "--batch-size", str(batch_size),
                "--eval-every", str(args.eval_every), "--eval-games", str(args.eval_games),
                "--save-every", str(args.save_every), "--output-dir", str(args.output_dir)]
    if args.resume_from:
        full_args += ["--resume-from", args.resume_from]
    sys.exit(run(full_args))


if __name__ == "__main__":
    main()
