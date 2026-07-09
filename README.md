Does GPT-5.2 get better at clinical diagnosis when you ask it to think harder? 

On 897 paired clinical benchmark cases, diagnosis accuracy increased from 63.9% at `none` to 68.8% at `high`. But the cost curve was steep: average latency rose from 2.6s to 13.6s, and average total tokens rose from 614 to 1,088.

---

## The numbers

| Setting | Accuracy | Tokens (avg) | Latency (avg) |
|---|---:|---:|---:|
| `none` | 63.9% | 614 | 2.6s |
| `low` | 66.4% | 782 | 5.5s |
| `medium` | 67.3% | 935 | 10.8s |
| `high` | 68.8% | 1,088 | 13.6s |

After Holm correction across all pairwise McNemar tests:

- **`none` vs `high`**: `p = <0.001` (significant)
- **`none` vs `medium`**: `p = 0.029` (significant)
- **`none` vs `low`**: `p = 0.15` (not significant after correction)
- Adjacent steps (`low` vs `medium`, `medium` vs `high`): not significant after correction


Each increase in reasoning effort improved raw benchmark accuracy. But after multiple-comparison correction, the clearest statistical evidence is for the cumulative improvements from `none -> medium` and `none -> high`.


---

## What this benchmark measures

- Evaluates GPT-5.2 (`gpt-5.2`) on the [MedCaseReasoning](https://huggingface.co/datasets/zou-lab/MedCaseReasoning) benchmark (test split)
- Tests four reasoning effort levels: `none`, `low`, `medium`, `high`
- Uses GPT-4.1 as a fixed grader across all variants (binary diagnosis correctness)
- Reports paired statistics (exact McNemar test, Holm correction, Wilson confidence intervals)
- Tracks total tokens, reasoning tokens, and latency per variant

---

## Why GPT-4.1 as the grader

GPT-4.1 is used as the fixed judge model to keep grading consistent across all reasoning variants. It is also used in OpenAI's [HealthBench](https://openai.com/index/healthbench/) evaluation framework and has published physician-agreement results in that setting.

This does **not** eliminate grader risk. The evaluated model and grader are both OpenAI models, so shared blind spots remain possible. A physician-adjudicated audit of a stratified sample is planned for a future version to characterize where automated grading agrees and disagrees with clinical judgment.

---

## Out of scope

This repository does **not**:

- validate on general-population clinical data (this benchmark skews complex/rare)
- include physician-adjudicated grading (planned for a future version)
- test across model families (Claude/Gemini comparison is next)
- make clinical deployment or safety claims

This is a benchmark study. It tells you how the reasoning-effort knob moves accuracy on a specific set of hard diagnostic cases. 

---

## Design choices

**Paired design.** Every case runs through all four variants. McNemar's test is appropriate for paired binary outcomes and is more powerful than comparing independent samples.

**Immutable outputs.** Raw model responses (`results/`) and grading scores (`scores/`) are committed and never modified by the reporting pipeline. The analysis reads from those files and regenerates everything else.

**All-pairs comparison.** Earlier versions only compared adjacent steps. That framing is too narrow: in practice you're deciding between `none` and `high`, not just between `medium` and `high`. The current analysis covers every pair.

**Fixed grader.** GPT-4.1 grades all variants with the same rubric. This isolates the effect of reasoning effort from grader variability. The tradeoff is that absolute accuracy depends on the grader model's judgment (see above).

---

## Quick start

Regenerate all reports from committed outputs (**no inference, no grading**):

```bash
pip install -e .
gpt52-ablation report
```

Start with these files:

- **`reports/final_report.md`** - full summary report
- **`reports/pairwise_matrix.csv`** - all-pairs significance testing
- **`reports/deployment_views.csv`** - cost/accuracy tradeoffs for deployment-style choices

---

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

---

## Repo structure

```text
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

---

## Known limitations and future work

**Grader validation.** GPT-4.1 has published physician-agreement results in HealthBench, but it has not been validated against physician review on this specific dataset. A stratified physician audit is planned for a future update.

**Error analysis.** The repo supports discordant case export, but this version does not include a systematic taxonomy of when reasoning helps, fails, or regresses.

**Cross-model comparison.** This study is GPT-5.2 only. Extending the same pipeline to Claude and Gemini with identical grading would show whether the reasoning-effort tradeoff is model-specific or general.

**Benchmark skew.** MedCaseReasoning is built from published case reports and is heavily skewed toward complex, rare-disease presentations. Accuracy numbers here are likely lower than what you'd see on routine clinical questions.

**Analyze the reasoning alignment scores** The grading pipeline already captures a 0-4 reasoning quality score alongside diagnosis correctness, but this version of the study only reports the binary diagnosis outcome.

---

## License

MIT
