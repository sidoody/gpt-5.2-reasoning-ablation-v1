from __future__ import annotations

import itertools
import math
from pathlib import Path

from .io_utils import average, read_json, write_json
from .paths import result_path, score_path
from .schemas import GradeFile, RunFile
from .settings import SUPPORTED_REASONING_LEVELS, ModelVariant, StudySettings


def _load_run_file(settings: StudySettings, reasoning_level: str) -> RunFile | None:
    variant = ModelVariant(model=settings.model, reasoning_effort=reasoning_level)
    payload = read_json(result_path(settings, variant))
    return RunFile.model_validate(payload) if payload else None


def _load_grade_file(settings: StudySettings, reasoning_level: str) -> GradeFile | None:
    variant = ModelVariant(model=settings.model, reasoning_effort=reasoning_level)
    payload = read_json(score_path(settings, variant))
    return GradeFile.model_validate(payload) if payload else None


def summarize_runs(settings: StudySettings, write_path: str | None = None) -> list[dict]:
    rows: list[dict] = []
    for level in SUPPORTED_REASONING_LEVELS:
        run = _load_run_file(settings, level)
        grade = _load_grade_file(settings, level)
        if not run or not grade:
            continue

        shared_case_ids = sorted(set(run.cases) & set(grade.cases))
        diagnosis_scores = [grade.cases[case_id].diagnosis_correctness_score for case_id in shared_case_ids]
        reasoning_scores = [grade.cases[case_id].reasoning_alignment_score for case_id in shared_case_ids]
        reasoning_passes = [1 if grade.cases[case_id].reasoning_alignment_score >= 3 else 0 for case_id in shared_case_ids]
        latencies = [run.cases[case_id].latency_seconds for case_id in shared_case_ids]
        total_tokens = [float(run.cases[case_id].usage.get("total_tokens", 0)) for case_id in shared_case_ids]
        reasoning_tokens = [float(run.cases[case_id].usage.get("reasoning_tokens", 0)) for case_id in shared_case_ids]

        rows.append(
            {
                "variant_id": run.variant["id"],
                "reasoning_effort": level,
                "cases_scored": len(shared_case_ids),
                "diagnosis_accuracy": round(average(diagnosis_scores), 4),
                "mean_reasoning_alignment": round(average(reasoning_scores), 4),
                "reasoning_pass_rate": round(average(reasoning_passes), 4),
                "avg_latency_seconds": round(average(latencies), 3),
                "avg_total_tokens": round(average(total_tokens), 2),
                "avg_reasoning_tokens": round(average(reasoning_tokens), 2),
            }
        )

    if write_path:
        write_json(Path(write_path), rows)

    if not rows:
        print("No complete run+grade pairs found.")
        return rows

    print("\n=== Summary ===")
    for row in rows:
        print(
            f"{row['reasoning_effort']:>6} | cases={row['cases_scored']:>4} | "
            f"diag_acc={row['diagnosis_accuracy']:.3f} | "
            f"reasoning_mean={row['mean_reasoning_alignment']:.3f} | "
            f"reasoning_pass={row['reasoning_pass_rate']:.3f} | "
            f"avg_tokens={row['avg_total_tokens']:.1f} | "
            f"avg_reasoning_tokens={row['avg_reasoning_tokens']:.1f}"
        )
    return rows


def _mcnemar_statistic(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 0.0
    return ((abs(a_only - b_only) - 1) ** 2) / discordant


def _mcnemar_exact_p_value(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    min_side = min(a_only, b_only)
    tail_prob = sum(math.comb(discordant, i) for i in range(min_side + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail_prob)


def _holm_bonferroni_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    m = len(p_values)
    adjusted_sorted: list[float] = [0.0] * m
    previous = 0.0
    for rank, (_, p_value) in enumerate(indexed):
        multiplier = m - rank
        adjusted = min(1.0, p_value * multiplier)
        monotone_adjusted = max(previous, adjusted)
        adjusted_sorted[rank] = monotone_adjusted
        previous = monotone_adjusted
    adjusted_original = [0.0] * m
    for rank, (original_index, _) in enumerate(indexed):
        adjusted_original[original_index] = adjusted_sorted[rank]
    return adjusted_original


def _ordered_observed_levels(settings: StudySettings) -> list[str]:
    observed: list[str] = []
    for level in SUPPORTED_REASONING_LEVELS:
        run = _load_run_file(settings, level)
        grade = _load_grade_file(settings, level)
        if run and grade:
            observed.append(level)
    return observed


def analyze_pairs(settings: StudySettings, write_path: str | None = None) -> list[dict]:
    levels = _ordered_observed_levels(settings)
    comparisons: list[dict] = []
    raw_p_values: list[float] = []

    for variant_a, variant_b in itertools.combinations(levels, 2):
        run_a = _load_run_file(settings, variant_a)
        run_b = _load_run_file(settings, variant_b)
        grade_a = _load_grade_file(settings, variant_a)
        grade_b = _load_grade_file(settings, variant_b)
        if not all([run_a, run_b, grade_a, grade_b]):
            continue

        shared_case_ids = sorted(set(run_a.cases) & set(run_b.cases) & set(grade_a.cases) & set(grade_b.cases))
        if not shared_case_ids:
            continue

        a_scores = [grade_a.cases[case_id].diagnosis_correctness_score for case_id in shared_case_ids]
        b_scores = [grade_b.cases[case_id].diagnosis_correctness_score for case_id in shared_case_ids]
        accuracy_a = average([float(score) for score in a_scores])
        accuracy_b = average([float(score) for score in b_scores])
        accuracy_delta = accuracy_b - accuracy_a

        a_only = 0
        b_only = 0
        for case_id in shared_case_ids:
            a_correct = grade_a.cases[case_id].diagnosis_correctness_score == 1
            b_correct = grade_b.cases[case_id].diagnosis_correctness_score == 1
            if a_correct and not b_correct:
                a_only += 1
            elif not a_correct and b_correct:
                b_only += 1
        counts = {
            "shared": len(shared_case_ids),
            "a_only": a_only,
            "b_only": b_only,
        }
        exact_p = _mcnemar_exact_p_value(counts["a_only"], counts["b_only"])
        raw_p_values.append(exact_p)

        avg_tokens_a = average([float(run_a.cases[case_id].usage.get("total_tokens", 0)) for case_id in shared_case_ids])
        avg_tokens_b = average([float(run_b.cases[case_id].usage.get("total_tokens", 0)) for case_id in shared_case_ids])
        avg_latency_a = average([run_a.cases[case_id].latency_seconds for case_id in shared_case_ids])
        avg_latency_b = average([run_b.cases[case_id].latency_seconds for case_id in shared_case_ids])
        extra_tokens = avg_tokens_b - avg_tokens_a
        extra_latency = avg_latency_b - avg_latency_a

        additional_correct_per_1000 = accuracy_delta * 1000.0
        tokens_per_additional_correct = None
        seconds_per_additional_correct = None
        if accuracy_delta > 0:
            tokens_per_additional_correct = extra_tokens / accuracy_delta
            seconds_per_additional_correct = extra_latency / accuracy_delta

        examples = []
        for case_id in shared_case_ids:
            a_correct = grade_a.cases[case_id].diagnosis_correctness_score == 1
            b_correct = grade_b.cases[case_id].diagnosis_correctness_score == 1
            if a_correct and not b_correct:
                examples.append(
                    {
                        "case_id": case_id,
                        "variant_a_prediction": run_a.cases[case_id].diagnosis,
                        "variant_b_prediction": run_b.cases[case_id].diagnosis,
                        "variant_b_reasoning_explanation": grade_b.cases[case_id].reasoning_explanation,
                    }
                )
            if len(examples) >= 10:
                break

        comparisons.append(
            {
                "variant_a": variant_a,
                "variant_b": variant_b,
                "comparison": f"{variant_a}_vs_{variant_b}",
                "cases_shared": counts["shared"],
                "variant_a_accuracy": round(accuracy_a, 6),
                "variant_b_accuracy": round(accuracy_b, 6),
                "accuracy_delta_b_minus_a": round(accuracy_delta, 6),
                "absolute_accuracy_delta": round(abs(accuracy_delta), 6),
                "variant_a_only_correct": counts["a_only"],
                "variant_b_only_correct": counts["b_only"],
                "discordant_total": counts["a_only"] + counts["b_only"],
                "mcnemar_exact_p_value": exact_p,
                "mcnemar_chi_square_cc": round(_mcnemar_statistic(counts["a_only"], counts["b_only"]), 4),
                "additional_correct_cases_per_1000": round(additional_correct_per_1000, 3),
                "extra_total_tokens": round(extra_tokens, 2),
                "extra_latency_seconds": round(extra_latency, 3),
                "tokens_per_additional_correct_case": (
                    round(tokens_per_additional_correct, 2) if tokens_per_additional_correct is not None else None
                ),
                "seconds_per_additional_correct_case": (
                    round(seconds_per_additional_correct, 4) if seconds_per_additional_correct is not None else None
                ),
                "variant_a_beats_variant_b_examples": examples,
            }
        )

    adjusted = _holm_bonferroni_adjust(raw_p_values)
    for item, adjusted_p in zip(comparisons, adjusted):
        item["mcnemar_exact_p_value"] = round(float(item["mcnemar_exact_p_value"]), 8)
        item["mcnemar_holm_adjusted_p_value"] = round(adjusted_p, 8)

    if write_path:
        write_json(Path(write_path), comparisons)

    if not comparisons:
        print("No pairwise comparisons available yet.")
        return comparisons

    print("\n=== Pairwise analysis ===")
    for item in comparisons:
        print(
            f"{item['variant_a']} vs {item['variant_b']} | shared={item['cases_shared']} | "
            f"a_only={item['variant_a_only_correct']} | b_only={item['variant_b_only_correct']} | "
            f"p_exact={item['mcnemar_exact_p_value']:.8f} | "
            f"p_holm={item['mcnemar_holm_adjusted_p_value']:.8f} | "
            f"delta={item['accuracy_delta_b_minus_a']:.4f}"
        )
    return comparisons


def analyze_overthinking(settings: StudySettings, write_path: str | None = None) -> list[dict]:
    """Deprecated alias for analyze_pairs."""
    return analyze_pairs(settings, write_path=write_path)
