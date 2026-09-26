# PACT results

Generated 2026-09-23 03:16 UTC from `C:\Users\Administrator\Desktop\PACT\.cache\runs\pact-9b`.

* runs with a holdout score: **14** of 14 recorded
* base model: `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
* holdout: the frozen `data/eval.jsonl`, used for scoring only
* figures drawn: 5

## Headline (single-pass decoding, mean ± sd over seeds)

| Run | Accuracy | Pair accuracy | ECE | NLL |
| --- | ---: | ---: | ---: | ---: |
| base_model | 0.6636 | 0.3580 | 0.2456 | 1.0944 |
| nimble_recipe | 0.8519 ± 0.0214 | 0.7099 ± 0.0385 | 0.0947 ± 0.0338 | 0.4486 ± 0.1099 |
| ce_only | 0.8220 ± 0.0600 | 0.7016 ± 0.0621 | 0.1426 ± 0.0642 | 0.7598 ± 0.2957 |
| pact_full | 0.8457 ± 0.0283 | 0.7325 ± 0.0618 | 0.1075 ± 0.0172 | 0.5595 ± 0.0880 |
| no_cf | 0.8426 | 0.7160 | 0.1107 | 0.4908 |
| no_emd | 0.8426 | 0.7037 | 0.0578 | 0.3879 |
| no_nr | 0.8086 | 0.6481 | 0.1033 | 0.5451 |
| no_pcr | 0.8765 | 0.7963 | 0.0749 | 0.5130 |
| no_perm_views | 0.8488 | 0.7407 | 0.0878 | 0.5102 |

## Shipped model

* run **pact_full**, seed **18** — chosen by lowest inner-validation NLL (inner NLL 0.1768)
* weights: `model/` (PEFT adapter, `model/adapter.pth` as a plain state dict)
* temperature: `model/calibration.json` (applied automatically by `PactScorer`)
* contract: `model/schema_config.json` — records the base model, the revision and the SHA-256 of the scoring prompt this adapter was trained against

```python
from pact.inference import PactScorer

scorer = PactScorer('results/model', ensemble=1)
print(scorer.score('The payment service is down for all customers.', schema))
```

Its own holdout numbers:

* accuracy 0.8210 (266/324)
* pair accuracy 0.6728
* NLL 0.5285, Brier 0.2579, ECE 0.0915

## What is in this folder

| Path | What |
| --- | --- |
| `SUMMARY.md` | this file |
| `model/` | the shipped adapter, its calibrator and its contract |
| `tables/results_tables.md` | every run, every metric, Markdown |
| `tables/results_tables.tex` | the same as LaTeX, for the paper |
| `tables/results_aggregate.json` | the same as JSON |
| `figures/*.png` | reliability, selective risk, position bias, run comparison, training |
| `figures/figure_data.json` | the numbers behind the figures |
| `runs/<run>/seed-<n>/results.json` | one run's full record, including its config |
| `holdout_rows.json` | the shipped model's per-example holdout predictions |
| `all_results.json` | every run's summaries in one file |

## Reading the ablations

The ablation runs are a test, not a search. `paper/main.pdf` (Sec. V-E) states in advance what each one should do if its term is working: compare against that, not against the best cell in the table. Seed-to-seed spread is in the ± column.

## Caveats

* The holdout is 324 synthetic, model-labelled examples from six source families.
* The shipped model was chosen on inner validation, so its holdout number is an honest estimate; the per-run table is not a leaderboard to pick from.
