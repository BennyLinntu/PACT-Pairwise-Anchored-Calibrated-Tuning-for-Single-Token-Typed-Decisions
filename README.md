# PACT — Pairwise-Anchored Calibrated Tuning

A fine-tuning recipe that replaces **only the training objective, the batching
and the probability head** of Nimble, a single-token typed-decision model. The
prompt, the one-token answer codes and the saved adapter format are
unchanged, so a PACT adapter loads anywhere a Nimble adapter loads. The full
write-up is in [`paper/main.pdf`](paper/main.pdf) ([source](paper/main.tex),
[Markdown version](paper/paper.md)).

As a live, playable demonstration that the same serving mechanism works as a
general typed-choice interface — not just on the document benchmark it was
trained for — this repo also ships **a from-scratch Tetris the adapter can
play**, plus a further reinforcement-learning fine-tune that measurably
improves it at that game. See [Play Tetris](#play-tetris-no-setup) below.

---

## Play Tetris (no setup)

**Windows:** run [`desktop/dist/PACT Tetris 1.0.0.exe`](desktop/dist) (portable,
no install) or `PACT Tetris Setup 1.0.0.exe` (installer). Manual play works
completely standalone, no Python or GPU needed — arrows to move, `Z` to rotate
the other way, space to hard-drop, `C` to hold, `P` to pause.

**macOS / rebuilding it yourself / enabling AI play** (a GPU and the base
model are needed for the model to actually choose moves): see
[`desktop/README.md`](desktop/README.md) and [`tetris/README.md`](tetris/README.md).
Short version for AI play, from the repo root:

```bash
python -m pip install -r requirements.txt
python download_all.py                 # pulls the base model into .cache/
python tetris/server.py                # http://127.0.0.1:8848
```

then open the desktop app (or `desktop/app/index_en.html` directly) — the
"Let PACT play" button lights up once the server answers.

---

## Why these four ideas

Nimble curates data in *contrastive pairs* — two nearly identical contexts that
differ in one fact, which flips the answer — and then trains on them with plain
cross-entropy, one example at a time. Three things are left on the table.

**1. The pairing is never used by the loss.** Cross-entropy sees two unrelated
examples. PACT puts both members in the same micro-batch and adds a
*difference-in-differences* margin:

```
d = [ z_a(y_a) − z_a(y_b) ] + [ z_b(y_b) − z_b(y_a) ]      L_CF = softplus(m − d)
```

Any logit offset the model attaches to the shared scenario — the topic, the
policy, the wording — cancels in `d`. The only way to reduce `L_CF` is to
respond to the single edited fact, which is exactly the thing the data was
curated to teach and the thing a family-level shortcut cannot fake.

**2. The answers are letters, and letters carry position bias.** Each choice is
rendered as `A`, `B`, `C`… and the model reads one token. Relabelling the same
choices must not change the answer. PACT encodes each example a second time
under a random relabelling and penalises the Jensen–Shannon divergence between
the two distributions *after* mapping both back to canonical choice order
(`L_PC`). At test time it can also average over K relabellings; the gap between
K=1 and K=4 is a direct read-out of the remaining bias.

**3. The verified "evidence removed" contexts are thrown away.** The curation
certificate proves, per pair, that deleting either focus sentence makes the
decisive fact *unknown* given all remaining text. Nimble discards those
contexts because they have no label. PACT rebuilds them
(`pact/ablation.py`, 2,676/2,676 reconstructable) and uses them as a
*constraint* rather than a target: with the evidence gone, the two outcomes the
fact discriminated must be indistinguishable,

```
L_NR = ( z°(y_a) − z°(y_b) )²
```

while the remaining choices stay free to be ruled out by the rest of the
context. This is supervision for *evidence necessity* — it penalises guessing
the right answer from style, topic or length.

**4. Score fields are ordinal, and cross-entropy cannot see order.** Being one
rubric level off should cost less than three. PACT adds a squared
earth-mover cost between the predicted and the reference level CDFs (`L_EMD`),
which is what the "expected score" the serving code already computes actually
needs.

On top of the objective, two smaller changes address the README's own caveats:
a **contextual temperature** `T = softplus(a + b·log C + c·log(L/1000))` fitted
on an inner split (Nimble ships no calibration at all), and **LoRA+ / rsLoRA**
scaling, which costs nothing and stabilises a 16-rank adapter on 2.7k examples.

Total objective:

```
L = L_CE + λ_cf·L_CF + λ_pc·L_PC + λ_nr·L_NR + λ_emd·L_EMD
```

with the regularisers ramped in linearly over the first 10 % of steps.

---

## What changes, precisely

| | Nimble (published) | PACT |
| --- | --- | --- |
| Prompt, codes, contract | `prepare_prompts`, A–Z, `schema_config.json` | identical (reused, not reimplemented) |
| Loss | candidate cross-entropy | + pair margin, permutation consistency, necessity, ordinal EMD |
| Batching | i.i.d. examples | pairs kept together, length-bucketed groups |
| Views per example | 1 | 1 + permuted + (per pair) one evidence-ablated |
| Selection | fixed epoch count | best inner-validation NLL, families held out of training |
| Probabilities | raw softmax, temperature 1 | contextual temperature fitted on an inner split |
| Decoding | single pass | single pass, or permutation-ensembled |
| Adapter | LoRA r16 | LoRA r16 + rsLoRA + LoRA+ (optional DoRA, 4-bit base) |

The frozen `data/eval.jsonl` is used for **nothing** but the final score: the
validation and calibration splits are whole source families carved out of
`data/train.jsonl`.

---

## Reproducing the research

```bash
# 1. once: dependencies (install the torch build your driver supports first)
python -m pip install -r requirements.txt

# 2. once: the pinned base weights, into ./.cache
python download_all.py

# 3. the sweep
CUDA_VISIBLE_DEVICES=0,1 nohup python train_all.py > train.log 2>&1 &
tail -f train.log
```

Step 3 needs no arguments and ends by packaging everything into `results/`
(already included in this repo from the run behind the paper):

```
results/
  SUMMARY.md              open this first: headline table, what shipped, where things are
  model/                  the shipped adapter + adapter.pth + calibration.json + schema_config.json
  tables/                 results_tables.md / .tex / results_aggregate.json
  figures/                reliability.png, risk_coverage.png, permutation.png,
                          runs_comparison.png, training.png, figure_data.json
  runs/<run>/seed-<n>/    each run's results.json, train_report.json, dev_history.json
  holdout_rows.json       the shipped model's per-example holdout predictions
  all_results.json        every run's summaries in one file
  baselines.json          the untouched base model, same code path
```

**Which model ships.** The seed of `pact_full` with the best *inner-validation*
NLL — never the best holdout score. Selecting on the frozen holdout would turn
the only clean number in the project into a training signal. Change the rule
with `release.primary_run` / `release.selection` if you want a different one.

**`.pth`.** `results/model/adapter.pth` is the trained LoRA tensors as a plain
`torch.save` state dict, next to the PEFT directory that `PactScorer` and
Nimble's own loaders read. If you also want base+LoRA merged into one set of
weights (~18 GB, loaded and merged on CPU), set
`"release": {"merge_full_model": true}` and it lands in `results/model_merged/`.

### How the two GPUs are used

The plan is a sweep of **14 independent run × seed jobs**, so the work is split
*by job*, not by tensor: one worker process per visible GPU, each holding a
complete model and running one job at a time, with no gradient traffic between
them. Two GPUs run two jobs at once and stay busy until the tail of the sweep.

* progress: the parent logs every dispatch and completion to `train.log`;
  each job also writes its own `worker.log`.
* a failed job is recorded in `failures.json` and does not stop the others.
* rerunning the command resumes: jobs with a `results.json` are skipped
  (`--force` redoes them).
* `--devices 0` pins one card; `--no-parallel` runs everything in this process.
* if you ever want a *single* job to span both cards, set
  `"model": {"device_map": "auto"}` — useful only for one large job.

**Memory.** Per GPU: 9B BF16 weights ≈ 16.8 GiB, activations with gradient
checkpointing ≈ 6.6 GiB at 10 rows × 2,048 tokens, LoRA state negligible —
roughly 24 GiB estimated peak against 47.8 GiB, printed at run start and
measured into `train_report.json`. If a card ever does OOM, lower
`optim.groups_per_batch` to 1 and raise `optim.grad_accum` to 4, or set
`optim.perm_view_prob` / `optim.necessity_prob` to 0.5.

**Cost.** One epoch is 1,338 pairs ≈ 4,000 row-forwards (vs 2,676 for the
baseline, because of the extra views): roughly 1.5× the baseline's step cost.
The default recipe is 2 epochs with best-checkpoint selection, 14 jobs, two at
a time.

### Useful flags

```bash
python train_all.py --only-run pact_full ce_only   # a subset of runs
python train_all.py --only-seed 17                 # a subset of seeds
python train_all.py --stages evaluate report        # re-score adapters you have
python train_all.py --skip-existing-train           # reuse the adapter on disk
python train_all.py --dry-run                       # print the plan and stop
```

### On a laptop first (no GPU)

```bash
python train_all.py --stages preflight
python -m unittest discover -s tests -t .
```

`preflight` verifies the frozen data hashes, the 1,338 flipped pairs, the
train/holdout family separation and the ablation coverage, and touches neither
the network nor a GPU. The test suite (44 tests) exercises the
prompt/permutation round trip, the batching, every loss term, the metrics, the
calibrator, the bundle's self-containment, the GPU job scheduler, the
packaging stage (model selection, `adapter.pth`, figures, `SUMMARY.md`) and
three steps of the real training loop against a tiny stand-in model — all on
CPU in a few seconds. Building the view cache locally needs only the tokenizer:

```bash
python download_all.py --tokenizer-only
python train_all.py --stages build
```

### Other configurations

| Config | For |
| --- | --- |
| `configs/server_2x48gb.json` | **the default**: 2 × 48 GB, BF16, full sweep |
| `configs/server_9b_h100.json` | the same recipe, separate output directory |
| `configs/server_9b_24gb_qlora.json` | a single 24 GB card, 4-bit NF4 base |
| `configs/server_2x40gb_rl.json` | 2 × 40 GB, BF16 — used by the Tetris RL fine-tune below |
| `configs/smoke_cpu.json` | CPU plumbing check: 135M model, 24 pairs, 12 holdout examples — numbers are meaningless |

### Optional: widen the domains

Nimble's README notes the model is trained on ten synthetic domains and should
not be expected to generalise far. To test whether a small, down-weighted share
of human-labelled data helps:

```bash
python download_all.py --public boolq vitaminc --public-limit 3000
```

then add to a config:

```json
"public_mix": [{"path": ".cache/public/boolq.jsonl", "weight": 0.3}]
```

These rows have no counterfactual partner, so they feed the cross-entropy term
only, and should be reported separately from the main comparison.

### Serving the result

```python
from pact.inference import PactScorer

scorer = PactScorer("results/model", ensemble=1)
result = scorer.score("The payment service is down for all customers.", schema)
print(result["output"], result["fields"]["priority"]["scores"])
```

`ensemble=1` with no `calibration.json` is bit-for-bit the Nimble serving path,
so any difference comes from the adapter, not the harness. The adapter also
loads directly in `nimble.training.schema_train score` and in the MLX/CUDA
scorers after merging.

---

## Beyond the paper: Tetris, and teaching it to play well

| | What | Needs |
| --- | --- | --- |
| [`tetris/`](tetris/) | Browser-playable Tetris; the shipped adapter is an optional AI player (`python tetris/server.py`) | a GPU, for AI play only |
| [`desktop/`](desktop/) | The same game packaged as a Windows/Mac app — manual play, no Python needed | nothing, prebuilt exes included |
| [`RL/`](RL/) | A self-contained REINFORCE fine-tune that makes the adapter a measurably better Tetris player, and a real completed 1,574-step run | a GPU, to retrain; nothing, to read the results |

The Tetris app hands the same `PactScorer` contract a short list of legal
piece placements, each already described in plain language by a two-ply
lookahead search the game engine runs (not the model) — rows cleared, holes,
height, bumpiness. This is exactly the "read a schema question, answer one
token" interface the whole training pipeline above produces, aimed at a
domain (a game state) it never saw in training. The shipped adapter, asked
this way, does not reliably pick the option the search itself ranks best —
expected, since it's a document classifier, not a game-player. `RL/` closes
that gap with policy-gradient fine-tuning on the same adapter, same serving
path, no architecture change; a completed run and its trained checkpoint are
included. Full story, numbers and how to point the live app at the
RL-trained checkpoint: [`tetris/README.md`](tetris/README.md) and
[`RL/README.md`](RL/README.md); the same material is also written up as an
appendix in [`paper/main.pdf`](paper/main.pdf).

---

## Files

```
README.md                  this file
train_all.py                one entry point: preflight → build → train → calibrate → evaluate → report
download_all.py              base weights / tokenizer / optional public data, all into .cache
requirements.txt
run.sh                       the nohup one-liner
configs/                     run plans (server_2x48gb.json is the default)
data/                        the frozen train.jsonl / eval.jsonl / manifest.json
nimble/                      byte-identical upstream modules the prompt contract depends on
pact/
  config.py                  typed config, dotted overrides, validation
  data.py                    pair building, splits, view encoding and caching
  ablation.py                 rebuilds the evidence-removed contexts from the certificates
  canon.py                    canonical choice space and permutations
  losses.py                   L_CE, L_CF, L_PC, L_NR, L_EMD
  sampler.py                  pair-preserving length-bucketed groups, collation
  model.py                    LoRA / QLoRA / LoRA+ / rsLoRA, last-token logit reader
  trainer.py                   the training loop
  calibrate.py                 contextual temperature
  evaluate.py                  single-pass and permutation-ensembled scoring
  metrics.py                   accuracy, NLL, Brier, ECE, AURC, pair accuracy, permutation TV
  inference.py                 PactScorer, a drop-in serving path
  report.py                    Markdown + LaTeX tables from the run outputs
  figures.py                   the five charts, with a JSON fallback
  release.py                   picks the model to ship and packages results/
  public_data.py               optional public-corpus converters
  tetris_env.py                pure-Python port of the Tetris engine, for RL rollouts
  train_tetris_rl.py           the Tetris RL fine-tune (REINFORCE)
tests/                        44 offline tests, CPU only
paper/                        main.pdf / main.tex / paper.md, tables and figures
results/                      the shipped model + every run's tables, figures and predictions
tetris/                       browser-playable Tetris + server.py (AI play backend)
desktop/                      the same game as a Windows/Mac app, prebuilt exes in desktop/dist/
RL/                           self-contained Tetris RL fine-tune: script, config, a completed run
```

The vendored `nimble/` and `data/` are exact copies of the checkout; a test
fails if they ever drift, so a run can never be scored against a prompt
different from the one it was trained with.

## Honest limitations

* No numbers are claimed without an ablation behind them. Each objective term
  is motivated and tested for correctness, but whether it helps *this* model
  on *these* 324 examples is answered by the runs in `results/`, not asserted.
* The holdout is 324 synthetic, model-labelled examples from six source
  families. It is narrow, and `L_NR` and `L_CF` both derive from the same
  synthetic certificates as the labels, so a certificate error propagates.
* `L_NR` assumes the certificate's necessity check is correct for the pair. It
  is applied to one ablated view per pair, not to both.
* Permutation ensembling multiplies serving cost by K; K=1 is the default for
  the headline numbers precisely so the comparison stays fair.
* Job-level GPU parallelism keeps both cards busy across a sweep, but a single
  job still uses a single card.
* The Tetris RL result (see above) is one training run with no seeds and no
  statistical test — a qualitative demonstration, not evidence about the
  paper's holdout claims in either direction.
