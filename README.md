# GPT-5.2 Reasoning Effort Ablation

This repository asks one research question:

> Does increasing GPT-5.2 reasoning effort materially improve diagnosis accuracy, and is the gain worth the token/latency cost?

**Main finding (N=897 paired cases):** diagnosis accuracy rises from `0.639` (`none`) to `0.688` (`high`), but each step adds substantial token and latency cost. Pairwise exact McNemar tests show statistically significant gains for `none vs low`, `none vs medium`, `none vs high`, and `low vs high`.

**Benchmark caveat:** this is a case-report-heavy, rare-disease-skewed dataset. Treat this as a controlled ablation study, not a general-population clinical benchmark.

## Main Result

From `reports/summary_metrics.json` and `reports/pairwise_stats.json`:

| Variant | N | Accuracy | 95% CI | Avg total tokens | Avg latency (s) |
|---|---:|---:|---:|---:|---:|
| none | 897 | 0.639 | [0.607, 0.670] | 613.61 | 2.608 |
| low | 897 | 0.664 | [0.633, 0.695] | 782.13 | 5.549 |
| medium | 897 | 0.673 | [0.642, 0.703] | 935.39 | 10.807 |
| high | 897 | 0.688 | [0.657, 0.717] | 1088.05 | 13.567 |

Pairwise exact McNemar p-values:

- `none vs low`: `0.04326826` (discordant: 48 vs 71)
- `none vs medium`: `0.00572686` (discordant: 44 vs 75)
- `none vs high`: `0.00016024` (discordant: 44 vs 88)
- `low vs high`: `0.03751423` (discordant: 36 vs 57)

p-values are unadjusted exact McNemar unless otherwise stated.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
# add OPENAI_API_KEY=...
```

Run a low-cost smoke test before launching the full benchmark:

```bash
gpt52-ablation run --variants none high --limit 10
gpt52-ablation grade --variants none high
gpt52-ablation report --discordant-limit 10
```

Run a full study:

```bash
gpt52-ablation run --variants none low medium high
gpt52-ablation grade --variants none low medium high
gpt52-ablation report
```

The `report` command is the one-command publish step for analysis artifacts.

## Expected Runtime and Cost

- Runtime and cost scale approximately linearly with case count.
- Reported latency in this repo ranges from roughly `2.6s` (`none`) to `13.6s` (`high`) per case.
- Reported average tokens range from roughly `614` (`none`) to `1088` (`high`) per case.
- For rough budget planning, multiply per-case metrics by your case count and number of variants.
- Run the smoke test first to validate credentials, API quota, and end-to-end pipeline behavior.

## Reproducibility

This study is deterministic from saved `results/` and `scores/` files into `reports/` outputs.

For a reproducible public release snapshot, record:

- repository commit SHA
- Python version
- package versions (`pip freeze` or lockfile)
- dataset identifier and split (`zou-lab/MedCaseReasoning`, `test`)
- exact command sequence used for `run`, `grade`, and `report`
- run date/time window

Recommended release workflow:

1. run the 4-variant benchmark (`none`, `low`, `medium`, `high`)
2. generate report artifacts with `gpt52-ablation report`
3. publish the commit SHA and the generated CSV/JSON report files used in your post

## Project Policies

- contribution guidelines: `CONTRIBUTING.md`
- security reporting: `SECURITY.md`
- community behavior expectations: `CODE_OF_CONDUCT.md`

## Publishable Artifacts

`gpt52-ablation report` writes deterministic outputs under `reports/`:

- `summary_metrics.csv` and `summary_metrics.json`
- `pairwise_stats.csv` and `pairwise_stats.json`
- `cost_latency_tradeoffs.csv` and `cost_latency_tradeoffs.json`
- `final_report.md`
- `discordant_none_vs_high.json` (manual audit helper)

These files are designed to be quoted directly in README/blog/LinkedIn posts.

Artifact policy:

- tracked source code should stay clean and reproducible
- generated study outputs are reproducible and can be regenerated locally
- if you want immutable publication snapshots, tag a release and attach the exact generated report artifacts used in the write-up

## Discordant Case Audit Helper

Export reviewable paired disagreements (default: `none` vs `high`):

```bash
gpt52-ablation export-discordant --a none --b high --limit 30
```

Each exported row includes:

- case ID
- gold diagnosis
- both predictions and correctness labels
- visible rationale summaries
- grader diagnosis/reasoning explanations

## Method Snapshot

- Evaluated model: `gpt-5.2`
- Variants: `none`, `low`, `medium`, `high`
- Grader model (fixed): `gpt-4.1`
- Dataset: `zou-lab/MedCaseReasoning` (`test` split)
- Scoring: diagnosis correctness (`0/1`) and reasoning alignment (`0-4`)
- Reported statistics:
  - per-variant diagnosis accuracy + 95% confidence interval
  - paired exact McNemar tests
  - token/latency tradeoff and incremental gain tables

## Limitations

- **Rare-disease skew:** `MedCaseReasoning` is not representative of everyday case mix.
- **Judge-model grading:** labels depend on GPT-4.1 grader behavior, even with a fixed rubric.
- **Visible-rationale scoring:** reasoning is graded only from model-visible rationale output, not hidden chain-of-thought.
- **`xhigh` exclusion:** `xhigh` was run exploratorily but is excluded from the primary public 4-variant analysis (`none`, `low`, `medium`, `high`) due to its coverage/cost profile and reporting scope.
