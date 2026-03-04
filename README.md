# GPT-5.2 Reasoning Effort Ablation

**Does GPT-5.2 get better at clinical diagnosis when you ask it to think harder? Yes. Is it worth the cost? Depends.**

On 897 paired clinical cases, turning on `low` reasoning captures about half the maximum accuracy gain at a quarter of the cost. Going all the way to `high` gets you the best accuracy, but at 5x the latency and nearly 2x the tokens.

This repo contains the full evaluation: raw model outputs, grading data, and a deterministic reporting pipeline you can rerun without touching inference.

## The numbers

| Setting | Accuracy | Tokens (avg) | Latency (avg) |
|---------|----------|-------------|---------------|
| `none` | 63.9% | 614 | 2.6s |
| `low` | 66.4% | 782 | 5.5s |
| `medium` | 67.3% | 935 | 10.8s |
| `high` | 68.8% | 1,088 | 13.6s |

After Holm-corrected McNemar tests across all pairs:
- **`none` vs `high`**: p < .001 (significant)
- **`none` vs `medium`**: p = .029 (significant)
- **`none` vs `low`**: p = .15 (not significant after correction)
- Adjacent steps (`low` vs `medium`, `medium` vs `high`): not significant after correction

The step from `none` to `low` is the best deal in the curve. After that, you're paying more per accuracy point.

## What this study does

- Evaluates GPT-5.2 (`gpt-5.2`) on the [MedCaseReasoning](https://huggingface.co/datasets/zou-lab/MedCaseReasoning) benchmark (test split)
- Tests four reasoning effort levels: `none`, `low`, `medium`, `high`
- Uses GPT-4.1 as a fixed grader across all variants (binary diagnosis correctness)
- Reports paired statistics (McNemar exact test, Holm correction, Wilson CIs)
- Tracks tokens, reasoning tokens, and latency per variant

## What this study does not do

- Validate on general-population clinical data (this benchmark skews complex/rare)
- Include physician-adjudicated grading (planned for a future version)
- Test across model families (Claude/Gemini comparison is next)
- Make clinical deployment or safety claims

This is a benchmark study. It tells you how the reasoning effort knob moves accuracy on a specific set of hard diagnostic cases. Real deployment decisions need more than this.

## Design choices

**Paired design.** Every case runs through all four variants. McNemar's test is the right choice for paired binary outcomes and it's more powerful than comparing independent samples.

**Immutable outputs.** Raw model responses (`results/`) and grading scores (`scores/`) are committed and never modified by the reporting pipeline. The analysis reads from those files and regenerates everything else. If you want to audit the numbers, the source of truth is right there.

**All-pairs comparison.** Earlier versions only compared adjacent steps. That was too narrow. In practice you're deciding between `none` and `high`, not just between `medium` and `high`. The current analysis covers every pair.

**Fixed grader.** GPT-4.1 grades all variants with the same rubric. This isolates the effect of reasoning effort from grader variability. The tradeoff is that absolute accuracy depends on the grader model's judgment.

## Quick start

Regenerate all reports from committed outputs (no inference, no grading):

```bash
pip install -e .
gpt52-ablation report
```

Start with these files:
- `reports/final_report.md` for the full writeup
- `reports/pairwise_matrix.csv` for significance testing
- `reports/deployment_views.csv` for cost/accuracy tradeoffs

## Full workflows

```bash
# Run inference (requires OpenAI API key)
gpt52-ablation run --variants none low medium high

# Grade saved runs
gpt52-ablation grade --variants none low medium high

# Regenerate reports
gpt52-ablation report

# Export discordant cases for review
gpt52-ablation export-discordant --a none --b high --limit 30

# Pairwise analysis
gpt52-ablation analyze-pairs
```

## Repo structure

```
src/gpt_5_2_reasoning_ablation/
  runner.py          # inference pipeline
  grading.py         # grading pipeline
  reporting.py       # deterministic report generation
  analysis.py        # pairwise statistical analysis

results/             # committed raw model outputs (immutable)
scores/              # committed grader outputs (immutable)
reports/             # generated analysis artifacts (derived)
tests/               # automated tests
```

## Known limitations and future work

**Grader validation.** The grading pipeline uses GPT-4.1, which has not been validated against physician review on this dataset. I plan to audit 100 stratified cases with blinded physician scoring and report Cohen's kappa in a future update.

**Error analysis.** The repo already supports discordant case export but I haven't done a systematic analysis of which case types or specialties benefit most from reasoning. That would make the findings more actionable.

**Cross-model comparison.** This is GPT-5.2 only. Running the same pipeline on Claude and Gemini with identical grading would answer whether the reasoning effort tradeoff is model-specific or general.

**Benchmark skew.** MedCaseReasoning is built from published case reports and is heavily skewed toward complex, rare-disease presentations. Accuracy numbers here are likely lower than what you'd see on routine clinical questions.

## Who built this

Board-certified physician and former edtech founder, learning ML engineering by building things that matter. This project started as a multi-model benchmark, narrowed to a single clean question, and turned out more interesting for it.

If you're working on health AI at a frontier lab, I'd like to talk.

## License

MIT
