# Evaluation workflow

## 1. Inference

`gpt52-ablation run` loads the configured dataset (default: `zou-lab/MedCaseReasoning`, split `test`) and evaluates one or more GPT-5.2 reasoning-effort variants.

For each case, the model must return strict JSON with:

- `diagnosis` (string)
- `rationale_summary` (array of 1-8 short strings)

Each case result is saved with:

- normalized visible answer (`diagnosis`, `rationale_summary`)
- `raw_output_text` and parsed JSON payload
- optional `api_reasoning_summary` if returned by the Responses API
- latency and token usage (`input_tokens`, `output_tokens`, `reasoning_tokens`, `total_tokens`)

If inference output is incomplete due to `max_output_tokens`, the runner retries that case with a higher token limit before failing.
Inference retries use bounded doubling and clamp at a `9600` token ceiling.

## 2. Grading

`gpt52-ablation grade` reads saved inference outputs and grades each case with GPT-4.1.

This grading setup is case-report-heavy and rare-disease-skewed; it is not designed as a general-population clinical benchmark.

The grader receives:

- case prompt
- gold diagnosis
- compact gold reasoning rubric bullets (typically 3-6 atomic items, fewer when source text is sparse)
- predicted diagnosis
- full visible JSON answer
- visible rationale summary (plus `api_reasoning_summary` when available)

The grader returns:

- `diagnosis_correctness_score` (`0` or `1`) and normalized label (`incorrect` or `correct`)
- `reasoning_alignment_score` (`0` to `4`) and normalized label (`poorly aligned` / `mixed` / `mostly aligned` / `strongly aligned`)
- brief diagnosis and reasoning explanations

If grader output is incomplete due to `max_output_tokens`, grading retries that case with a higher token limit (bounded) before failing.

Scoring is frozen:

- diagnosis correctness is scored directly as `0` (incorrect) or `1` (correct)
- reasoning alignment uses a fixed `0`-`4` rubric based on decisive clues, crucial exclusions when needed, and semantic alignment rather than string matching
- missing minor paper-specific differentials should not heavily penalize otherwise strong clinical reasoning
- grading is based only on visible model output, not hidden chain-of-thought

## 3. Summaries

`gpt52-ablation summarize` aggregates completed run+grade pairs (shared case IDs only):

- diagnosis accuracy
- mean reasoning alignment score
- reasoning pass rate (`score >= 3`)
- average latency
- average total tokens
- average reasoning tokens

## 4. Pairwise analysis

`gpt52-ablation analyze-pairs` compares all unique pairs among observed reasoning-effort variants and reports:

- shared-case counts
- `lower_only_correct` and `higher_only_correct`
- McNemar chi-square statistic with continuity correction
- raw McNemar p-values and Holm-adjusted p-values across the full pairwise family
- mean reasoning-alignment delta (`lower - higher`)
- up to 10 examples where lower effort was correct and higher effort was not
