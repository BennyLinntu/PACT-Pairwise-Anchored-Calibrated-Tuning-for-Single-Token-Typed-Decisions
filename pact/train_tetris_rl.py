#!/usr/bin/env python
"""REINFORCE fine-tuning of the PACT LoRA adapter on the Tetris move-choice
task -- the mechanism behind "let PACT decide, but teach it to reliably pick
the heuristic's recommended move" (as opposed to swapping the model out for
the heuristic directly, which the tetris/ app deliberately does not do).

Framed as a single-step contextual bandit, not full episodic RL: each move's
reward is known immediately, because the two-ply lookahead heuristic
(pact/tetris_env.py, mirroring desktop/app/ai_client.js exactly) already
scores every candidate at that state, not just the one chosen. That gives an
exact, zero-variance-cost baseline for free -- the state's own mean candidate
score -- which is why plain REINFORCE is enough here: no value network, no
GAE, no PPO clipping. Rollouts are on-policy: the model's own sampled choice
during self-play is what gets trained on.

Reuses tetris/server.py's build_context / describe_candidate / build_schema
verbatim, so the training prompt is byte-identical to the serving prompt --
train/serve skew would otherwise quietly defeat the whole point. Nothing in
tetris/, desktop/app/, or the existing supervised pact/ training pipeline is
imported in a way that mutates it; this is a new, additive entry point.

    # measure real throughput first, on whatever GPU(s) you have -- one
    # gradient step, one example, no side effects:
    python pact/train_tetris_rl.py --config configs/server_2x40gb_rl.json \\
        --max-steps 1 --batch-size 1

    # a real run, once you've seen that timing and picked a batch size that
    # fits your VRAM (see --batch-size docstring below):
    python pact/train_tetris_rl.py --config configs/server_2x40gb_rl.json \\
        --max-steps 2000 --batch-size 16 --eval-every 50 --save-every 50

Safe to Ctrl+C at any point: nothing is written until a checkpoint boundary,
and SIGINT during training triggers one last save before exiting. Resume with
--resume-from <checkpoint dir>.
"""
import argparse
import json
import random
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(BUNDLE / "tetris"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from pact.tetris_env import TetrisGame, placements_for, simulate_placement, rank_score, column_heights  # noqa: E402
from pact.config import PactConfig  # noqa: E402
from pact.model import load_tokenizer, _load_base, LogitReader  # noqa: E402

MAX_CANDIDATES = 6          # must match desktop/app/ai_client.js's MAX_CANDIDATES
TOPOUT_SCORE = -1e6         # must match desktop/app/ai_client.js's TOPOUT_SCORE


def build_candidates(game):
    """Mirrors ai_client.js:requestAiMove's two-ply search, forced line-clear
    priority, and MAX_CANDIDATES cap exactly."""
    placements1 = placements_for(game.grid, game.current)
    if not placements1:
        return []
    next_type = game.next_queue[0]
    scored = []
    for p1 in placements1:
        sim1 = simulate_placement(game.grid, game.current, p1)
        placements2 = placements_for(sim1["grid"], next_type)
        best_score, best_follow = TOPOUT_SCORE, None
        for p2 in placements2:
            sim2 = simulate_placement(sim1["grid"], next_type, p2)
            combined = dict(sim2)
            combined["linesCleared"] = sim1["linesCleared"] + sim2["linesCleared"]
            s = rank_score(combined)
            if s > best_score:
                best_score, best_follow = s, sim2
        scored.append({"rotation": p1["rotation"], "col": p1["col"], "row": p1["row"],
                       "immediate": sim1, "lookaheadScore": best_score, "bestFollowUp": best_follow})
    scored.sort(key=lambda c: -c["lookaheadScore"])
    clearers = [c for c in scored if c["immediate"]["linesCleared"] > 0]
    rest = [c for c in scored if c["immediate"]["linesCleared"] == 0]
    return (clearers + rest)[:MAX_CANDIDATES]


def to_payload(game, candidates):
    """Builds the same JSON shape ai_client.js sends to /ai_move."""
    payload = {
        "column_heights": column_heights(game.grid),
        "current": game.current, "next": game.next_queue[0], "hold": None,
        "lines": game.lines,
    }
    payload_candidates = []
    for c in candidates:
        bf = c["bestFollowUp"]
        payload_candidates.append({
            "rotation": c["rotation"], "col": c["col"],
            "lines_cleared": c["immediate"]["linesCleared"], "holes": c["immediate"]["holes"],
            "max_height": c["immediate"]["maxHeight"], "bumpiness": c["immediate"]["bumpiness"],
            "topout_next": bf is None,
            "followup_lines_cleared": bf["linesCleared"] if bf else None,
            "followup_holes": bf["holes"] if bf else None,
            "followup_max_height": bf["maxHeight"] if bf else None,
            "followup_bumpiness": bf["bumpiness"] if bf else None,
        })
    return payload, payload_candidates


def build_prompt(server, tokenizer, max_length, game, candidates):
    from nimble.scoring.parallel_schema import prepare_prompts, validate_schema

    payload, payload_candidates = to_payload(game, candidates)
    context = server.build_context(payload)
    schema = server.build_schema(payload_candidates)
    validate_schema(schema)
    return prepare_prompts(tokenizer, context, schema, max_length)


def candidate_logits_for(model, reader, device, prompt, grad_enabled):
    ids = torch.tensor([list(prompt.full_ids[0])], device=device)
    mask = torch.ones_like(ids)
    # autocast is independent of grad tracking: always want bf16 compute on
    # CUDA, but only want gradients during training rollouts, not eval.
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()
    grad_ctx = nullcontext() if grad_enabled else torch.no_grad()
    with autocast_ctx, grad_ctx:
        logits = reader(model, ids, mask)[0]
    return torch.stack([logits[code] for code in prompt.candidate_ids[0]]).float()


def rollout_one_move(game, model, reader, tokenizer, server, max_length, device, temperature):
    """On-policy sample: forward pass, sample an action from the model's own
    distribution over the current candidates, apply it, return the pieces
    needed for a REINFORCE update. Returns None only if the game just ended
    with no legal placement at all (freshly reset, so this shouldn't recur)."""
    if game.game_over:
        game.reset()
    candidates = build_candidates(game)
    if not candidates:
        game.reset()
        return None

    prompt = build_prompt(server, tokenizer, max_length, game, candidates)
    candidate_logits = candidate_logits_for(model, reader, device, prompt, grad_enabled=True)
    probs = F.softmax(candidate_logits / temperature, dim=0)
    dist = torch.distributions.Categorical(probs=probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()

    scores = [c["lookaheadScore"] for c in candidates]
    baseline = sum(scores) / len(scores)
    advantage = scores[action.item()] - baseline

    chosen = candidates[action.item()]
    game.apply({"rotation": chosen["rotation"], "col": chosen["col"], "row": chosen["row"]})
    return {"log_prob": log_prob, "entropy": entropy, "advantage": advantage,
            "action": action.item(), "n_candidates": len(candidates),
            "picked_top": action.item() == 0}


@torch.no_grad()
def play_greedy_game(game, model, reader, tokenizer, server, max_length, device, max_pieces):
    """Argmax play (no sampling) for evaluation -- matches what the served
    app actually does. Returns (score, lines, pieces_placed, topped_out)."""
    game.reset()
    for _ in range(max_pieces):
        candidates = build_candidates(game)
        if not candidates:
            break
        prompt = build_prompt(server, tokenizer, max_length, game, candidates)
        candidate_logits = candidate_logits_for(model, reader, device, prompt, grad_enabled=False)
        best = int(torch.argmax(candidate_logits).item())
        chosen = candidates[best]
        game.apply({"rotation": chosen["rotation"], "col": chosen["col"], "row": chosen["row"]})
        if game.game_over:
            break
    return game.score, game.lines, game.pieces_placed, game.game_over


def evaluate(games_to_play, model, reader, tokenizer, server, max_length, device, seed, max_pieces=400):
    model.eval()
    rng = random.Random(seed)
    scores, lines_list, pieces_list = [], [], []
    game = TetrisGame(rng)
    for _ in range(games_to_play):
        score, lines, pieces, _ = play_greedy_game(game, model, reader, tokenizer, server,
                                                    max_length, device, max_pieces)
        scores.append(score)
        lines_list.append(lines)
        pieces_list.append(pieces)
    model.train()
    n = len(scores)
    return {"games": n, "mean_score": sum(scores) / n, "mean_lines": sum(lines_list) / n,
            "mean_pieces": sum(pieces_list) / n, "max_score": max(scores)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def save_checkpoint(model, optimizer, output_dir, step):
    out = Path(output_dir) / f"step-{step}"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, out / "optim.pt")
    latest = Path(output_dir) / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(latest))
    torch.save({"step": step, "optimizer": optimizer.state_dict()}, latest / "optim.pt")
    log(f"saved checkpoint: {out} (and latest/)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(BUNDLE / "configs" / "server_9b_24gb_qlora.json"),
                        help="Model-loading config. Use configs/server_2x40gb_rl.json for a "
                             "2-GPU 40GB+ box (BF16, device_map=auto, no quantization).")
    parser.add_argument("--adapter-dir", default=str(BUNDLE / "results" / "model"),
                        help="Starting adapter (the shipped PACT adapter by default).")
    parser.add_argument("--resume-from", default=None,
                        help="A checkpoint dir from a previous run of this script, to continue "
                             "from instead of --adapter-dir (also restores optimizer state and "
                             "step count).")
    parser.add_argument("--max-steps", type=int, default=1, help="gradient steps to run")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="moves per gradient step. Backward runs immediately after each "
                             "move (proper grad accumulation), so peak memory is ~1 move's "
                             "activations regardless of batch size -- raise this for a smoother, "
                             "less noisy gradient, not because of memory. Start small, watch "
                             "nvidia-smi, and raise it once you've seen one step complete.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01,
                        help="Small bonus for action-distribution entropy, standard REINFORCE "
                             "stabiliser -- keeps the policy from collapsing onto one candidate "
                             "before it's had enough signal to know that's actually best.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature during rollout (exploration). Greedy eval "
                             "always uses argmax regardless of this.")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=5)
    parser.add_argument("--eval-max-pieces", type=int, default=400,
                        help="Piece cap per evaluation game. The in-training eval defaults to a "
                             "low cap to keep periodic checks cheap; if a checkpoint is hitting "
                             "this cap every time (mean_pieces == the cap in eval_log.jsonl), "
                             "that says nothing about how much further it could actually go -- "
                             "rerun with --eval-only and a much higher cap to find out.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training entirely: load --adapter-dir (or --resume-from) and "
                             "just run --eval-games games at --eval-max-pieces, then exit. Use "
                             "this to answer \"how long can it actually survive\" for a finished "
                             "checkpoint, e.g. --adapter-dir runs/tetris-rl/latest "
                             "--eval-max-pieces 20000 --eval-games 3.")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--output-dir", default=str(BUNDLE / ".cache" / "runs" / "tetris-rl"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    eval_path = output_dir / "eval_log.jsonl"

    import server  # tetris/server.py's build_context / describe_candidate / build_schema

    t0 = time.time()
    log(f"loading config from {args.config} ...")
    cfg = PactConfig.load(args.config)
    tokenizer = load_tokenizer(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading base model ({cfg.model.model_id}, dtype={cfg.model.dtype}, "
        f"device_map={cfg.model.device_map}, 4bit={cfg.model.load_in_4bit}) ...")
    base, _ = _load_base(cfg, device=device, dtype=torch.bfloat16, log=log)

    if cfg.model.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        # Required for QLoRA training (not just inference): casts norms to
        # fp32 and wires up gradient flow through the frozen quantized base.
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=cfg.model.gradient_checkpointing)
    elif cfg.model.device_map == "single" and device == "cuda":
        base = base.to("cuda")

    from peft import PeftModel

    start_step = 0
    adapter_source = args.adapter_dir
    if args.resume_from:
        adapter_source = args.resume_from
        state_path = Path(args.resume_from) / "optim.pt"
        if state_path.exists():
            start_step = torch.load(state_path, map_location="cpu")["step"]
        log(f"resuming from {adapter_source} at step {start_step}")

    if args.eval_only:
        model = PeftModel.from_pretrained(base, str(adapter_source), is_trainable=False)
        model.eval()
        log(f"model ready in {time.time()-t0:.1f}s (eval-only, adapter={adapter_source})")
        reader = LogitReader()
        eval_t0 = time.time()
        result = evaluate(args.eval_games, model, reader, tokenizer, server, cfg.data.max_length,
                          device, seed=args.seed, max_pieces=args.eval_max_pieces)
        result["eval_seconds"] = time.time() - eval_t0
        log(f"eval-only, {args.eval_games} game(s), cap {args.eval_max_pieces} pieces/game: "
            f"mean_score={result['mean_score']:.0f} mean_lines={result['mean_lines']:.1f} "
            f"mean_pieces={result['mean_pieces']:.0f} max_score={result['max_score']:.0f} "
            f"({result['eval_seconds']:.0f}s)")
        if result["mean_pieces"] >= args.eval_max_pieces:
            log(f"  every game hit the {args.eval_max_pieces}-piece cap without topping out -- "
                f"the real ceiling is still unknown; rerun with a higher --eval-max-pieces.")
        output_dir.mkdir(parents=True, exist_ok=True)
        append_jsonl(output_dir / "eval_only_log.jsonl", result)
        return

    model = PeftModel.from_pretrained(base, str(adapter_source), is_trainable=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f"model ready in {time.time()-t0:.1f}s; {sum(p.numel() for p in trainable):,} trainable params")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    if args.resume_from and (Path(args.resume_from) / "optim.pt").exists():
        optimizer.load_state_dict(torch.load(Path(args.resume_from) / "optim.pt",
                                              map_location="cpu")["optimizer"])

    def lr_at(step):
        if step >= args.warmup_steps:
            return args.lr
        return args.lr * step / max(1, args.warmup_steps)

    reader = LogitReader()
    rng = random.Random(args.seed)
    game = TetrisGame(rng)

    stop_requested = {"flag": False}

    def handle_sigint(signum, frame):
        log("Ctrl+C received -- finishing the current move, then saving and exiting.")
        stop_requested["flag"] = True

    signal.signal(signal.SIGINT, handle_sigint)

    log(f"starting at step {start_step}, target {args.max_steps} step(s), "
        f"batch size {args.batch_size}, eval every {args.eval_every}, save every {args.save_every}")

    step = start_step
    last_saved_step = -1
    try:
        while step < args.max_steps and not stop_requested["flag"]:
            step += 1
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)

            step_t0 = time.time()
            optimizer.zero_grad()
            completed, top_picks, advantages, loss_sum = 0, 0, [], 0.0
            for _ in range(args.batch_size):
                if stop_requested["flag"]:
                    break
                result = rollout_one_move(game, model, reader, tokenizer, server,
                                          cfg.data.max_length, device, args.temperature)
                if result is None:
                    continue
                loss = -(result["log_prob"] * result["advantage"]) / args.batch_size
                loss = loss - args.entropy_weight * result["entropy"] / args.batch_size
                loss.backward()
                loss_sum += loss.item()
                completed += 1
                top_picks += int(result["picked_top"])
                advantages.append(result["advantage"])

            if completed == 0:
                step -= 1
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            step_dt = time.time() - step_t0

            record = {"step": step, "loss": loss_sum, "grad_norm": float(grad_norm),
                     "lr": lr_at(step), "top_pick_rate": top_picks / completed,
                     "mean_advantage": sum(advantages) / len(advantages),
                     "score": game.score, "lines": game.lines, "step_seconds": step_dt}
            append_jsonl(log_path, record)
            log(f"step {step}/{args.max_steps}: loss={loss_sum:.4f} grad_norm={grad_norm:.3f} "
                f"top_pick_rate={record['top_pick_rate']:.2f} "
                f"mean_advantage={record['mean_advantage']:+.3f} "
                f"({step_dt:.1f}s, {step_dt/completed:.1f}s/move)")

            if args.eval_every and step % args.eval_every == 0:
                eval_t0 = time.time()
                result = evaluate(args.eval_games, model, reader, tokenizer, server,
                                  cfg.data.max_length, device, seed=args.seed + 10_000 + step,
                                  max_pieces=args.eval_max_pieces)
                result["step"] = step
                result["eval_seconds"] = time.time() - eval_t0
                append_jsonl(eval_path, result)
                log(f"  eval @ step {step}: mean_score={result['mean_score']:.0f} "
                    f"mean_lines={result['mean_lines']:.2f} max_score={result['max_score']:.0f} "
                    f"({result['eval_seconds']:.0f}s for {args.eval_games} games)")

            if args.save_every and step % args.save_every == 0:
                save_checkpoint(model, optimizer, output_dir, step)
                last_saved_step = step

    finally:
        if step > start_step and step != last_saved_step:
            save_checkpoint(model, optimizer, output_dir, step)
        log(f"stopped at step {step}. total wall clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
