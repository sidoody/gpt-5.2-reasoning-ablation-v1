# GPT-5.2 Reasoning Effort Ablation

This repository is a reproducible study of how GPT-5.2 diagnosis performance and resource usage vary across four reasoning-effort settings: `none`, `low`, `medium`, and `high`.

The project is organized as a complete evaluation package:

- committed model outputs in `results/`
- committed grader outputs in `scores/`
- reproducible analysis artifacts in `reports/`
- CLI workflows for running, grading, and reporting

## Study question

How do reasoning-effort settings trade off diagnosis accuracy against token and latency cost on the benchmark cases used in this repo?

## Method overview

- **Model under evaluation:** `gpt-5.2`
- **Variants:** `none`, `low`, `medium`, `high`
- **Dataset:** `zou-lab/MedCaseReasoning` (`test`)
- **Grader model:** `gpt-4.1` (fixed across variants)
- **Primary outcome:** diagnosis correctness (`0/1`)
- **Paired comparison design:** each pairwise test uses shared case IDs only

Reported statistics include:

- per-variant accuracy with 95% Wilson intervals
- all-pairs exact McNemar tests
- Holm-adjusted p-values across the full pair set
- deployment-oriented effect measures (additional correct cases per 1,000, extra tokens, extra latency, efficiency ratios)

## Reproducible reporting

`results/` and `scores/` are the source-of-truth study outputs.
`reports/` is generated from those outputs.

Regenerate all report artifacts with:

```bash
pip install -e .
gpt52-ablation report
```

The report command validates inputs and rebuilds `reports/` from scratch.

## Generated report artifacts

Running `gpt52-ablation report` writes:

- `reports/variant_summary.json` and `reports/variant_summary.csv`
- `reports/pairwise_matrix.json` and `reports/pairwise_matrix.csv`
- `reports/deployment_views.json` and `reports/deployment_views.csv`
- `reports/efficiency_frontier.json` and `reports/efficiency_frontier.csv`
- `reports/validation_summary.json`
- `reports/pairwise_mcnemar_p_values.svg`
- `reports/final_report.md`
- `reports/discordant_case_exports/*.json`

## Common workflows

Run inference:

```bash
gpt52-ablation run --variants none low medium high
```

Grade saved runs:

```bash
gpt52-ablation grade --variants none low medium high
```

Generate reports from committed outputs:

```bash
gpt52-ablation report
```

Export discordant cases for qualitative review:

```bash
gpt52-ablation export-discordant --a none --b high --limit 30
```

## Interpretation scope

This repository supports conclusions about:

- measured accuracy differences on this benchmark
- paired-case comparisons between observed variants
- resource tradeoffs (tokens and latency) per configuration

This repository does not establish:

- real-world prevalence estimates
- standalone clinical safety claims
- replacement of clinician judgment
- broad generalization beyond this benchmark without additional validation

## Repository structure

- `src/gpt_5_2_reasoning_ablation/runner.py`: inference pipeline
- `src/gpt_5_2_reasoning_ablation/grading.py`: grading pipeline
- `src/gpt_5_2_reasoning_ablation/reporting.py`: report generation
- `results/`: saved model outputs
- `scores/`: saved grader outputs
- `reports/`: generated analysis and reporting artifacts
- `tests/`: automated tests for CLI, schemas, analysis, and reporting
