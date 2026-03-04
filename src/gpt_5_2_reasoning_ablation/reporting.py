from __future__ import annotations

import csv
import itertools
import math
import shutil
from pathlib import Path

from .io_utils import average, read_json, write_json
from .paths import reports_dir
from .schemas import GradeFile, RunFile
from .settings import DEFAULT_REASONING_LEVELS, SUPPORTED_REASONING_LEVELS, StudySettings

CANONICAL_REASONING_ORDER = tuple(DEFAULT_REASONING_LEVELS)
DEPLOYMENT_PRIORITY_PAIRS = (
    ("none", "low"),
    ("low", "high"),
    ("none", "high"),
    ("none", "medium"),
)
WILSON_Z_95 = 1.959963984540054


def _canonical_sort_key(level: str) -> tuple[int, str]:
    if level in CANONICAL_REASONING_ORDER:
        return (CANONICAL_REASONING_ORDER.index(level), level)
    return (len(CANONICAL_REASONING_ORDER), level)


def _load_observed_variant_files(settings: StudySettings) -> dict[str, tuple[RunFile, GradeFile]]:
    result_paths = sorted(Path(settings.results_dir).glob("*.json"))
    score_paths = sorted(Path(settings.scores_dir).glob("*.json"))
    if not result_paths:
        raise ValueError(f"No result JSON files found in {settings.results_dir}.")
    if not score_paths:
        raise ValueError(f"No score JSON files found in {settings.scores_dir}.")

    runs_by_level: dict[str, RunFile] = {}
    for path in result_paths:
        payload = read_json(path)
        if payload is None:
            raise ValueError(f"Result file is unreadable or empty: {path}")
        try:
            run = RunFile.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"Inconsistent result schema in {path}: {exc}") from exc
        level = str(run.variant.get("reasoning_effort", "")).strip()
        if level not in SUPPORTED_REASONING_LEVELS:
            raise ValueError(f"Unexpected variant name {level!r} in result file {path}.")
        runs_by_level[level] = run

    grades_by_level: dict[str, GradeFile] = {}
    for path in score_paths:
        payload = read_json(path)
        if payload is None:
            raise ValueError(f"Score file is unreadable or empty: {path}")
        try:
            grade = GradeFile.model_validate(payload)
        except Exception as exc:
            raise ValueError(f"Inconsistent score schema in {path}: {exc}") from exc
        level = str(grade.variant.get("reasoning_effort", "")).strip()
        if level not in SUPPORTED_REASONING_LEVELS:
            raise ValueError(f"Unexpected variant name {level!r} in score file {path}.")
        grades_by_level[level] = grade

    run_levels = set(runs_by_level)
    score_levels = set(grades_by_level)
    missing_scores = sorted(run_levels - score_levels, key=_canonical_sort_key)
    missing_results = sorted(score_levels - run_levels, key=_canonical_sort_key)
    if missing_scores:
        raise ValueError(f"Missing score files for variants: {missing_scores}")
    if missing_results:
        raise ValueError(f"Missing result files for variants: {missing_results}")

    observed_levels = sorted(run_levels & score_levels, key=_canonical_sort_key)
    if not observed_levels:
        raise ValueError("No complete result/score variant pairs are available.")

    observed: dict[str, tuple[RunFile, GradeFile]] = {
        level: (runs_by_level[level], grades_by_level[level]) for level in observed_levels
    }

    # Validate paired-case comparability across all observed variants.
    baseline_level = observed_levels[0]
    baseline_case_ids = set(observed[baseline_level][0].cases.keys())
    for level, (run, grade) in observed.items():
        run_case_ids = set(run.cases.keys())
        grade_case_ids = set(grade.cases.keys())
        if run_case_ids != grade_case_ids:
            raise ValueError(
                f"Mismatched case IDs for variant {level!r}: results has {len(run_case_ids)}, "
                f"scores has {len(grade_case_ids)}."
            )
        if run_case_ids != baseline_case_ids:
            raise ValueError(
                f"Mismatched case IDs across variants: {baseline_level!r} vs {level!r} "
                f"({len(baseline_case_ids)} vs {len(run_case_ids)})."
            )
    return observed


def validate_committed_inputs(settings: StudySettings) -> dict[str, object]:
    observed = _load_observed_variant_files(settings)
    observed_levels = list(observed.keys())
    pair_count = len(observed_levels) * (len(observed_levels) - 1) // 2
    case_count = len(next(iter(observed.values()))[0].cases)
    return {
        "observed_variants": observed_levels,
        "variants_count": len(observed_levels),
        "case_count_per_variant": case_count,
        "pairwise_comparisons": pair_count,
    }


def _variant_rows(settings: StudySettings) -> list[dict]:
    observed = _load_observed_variant_files(settings)
    rows: list[dict] = []
    for level in observed:
        run, grade = observed[level]

        shared_case_ids = sorted(set(run.cases) & set(grade.cases))
        diagnosis_scores = [grade.cases[case_id].diagnosis_correctness_score for case_id in shared_case_ids]
        latencies = [run.cases[case_id].latency_seconds for case_id in shared_case_ids]
        total_tokens = [float(run.cases[case_id].usage.get("total_tokens", 0)) for case_id in shared_case_ids]
        reasoning_tokens = [float(run.cases[case_id].usage.get("reasoning_tokens", 0)) for case_id in shared_case_ids]

        n = len(shared_case_ids)
        correct = int(sum(diagnosis_scores))
        accuracy = average([float(score) for score in diagnosis_scores])
        ci_low, ci_high = _wilson_interval(correct, n)
        rows.append(
            {
                "reasoning_effort": level,
                "variant_id": run.variant["id"],
                "n": n,
                "correct": correct,
                "accuracy": round(accuracy, 6),
                "accuracy_ci95_low": round(ci_low, 6),
                "accuracy_ci95_high": round(ci_high, 6),
                "avg_total_tokens": round(average(total_tokens), 2),
                "avg_reasoning_tokens": round(average(reasoning_tokens), 2),
                "avg_latency_seconds": round(average(latencies), 3),
            }
        )
    return rows


def _wilson_interval(successes: int, total: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p_hat = successes / total
    z2 = z * z
    denom = 1 + z2 / total
    center = (p_hat + z2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1 - p_hat) / total) + (z2 / (4 * total * total)))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


def _mcnemar_counts(grades_a: GradeFile, grades_b: GradeFile) -> dict[str, int]:
    shared = sorted(set(grades_a.cases) & set(grades_b.cases))
    a_correct_b_incorrect = 0
    a_incorrect_b_correct = 0

    for case_id in shared:
        a_correct = grades_a.cases[case_id].diagnosis_correctness_score == 1
        b_correct = grades_b.cases[case_id].diagnosis_correctness_score == 1
        if a_correct and not b_correct:
            a_correct_b_incorrect += 1
        elif not a_correct and b_correct:
            a_incorrect_b_correct += 1

    return {
        "n": len(shared),
        "a_correct_b_incorrect": a_correct_b_incorrect,
        "a_incorrect_b_correct": a_incorrect_b_correct,
        "discordant_total": a_correct_b_incorrect + a_incorrect_b_correct,
    }


def _mcnemar_exact_p_value(a_correct_b_incorrect: int, a_incorrect_b_correct: int) -> float:
    discordant = a_correct_b_incorrect + a_incorrect_b_correct
    if discordant == 0:
        return 1.0
    min_side = min(a_correct_b_incorrect, a_incorrect_b_correct)
    tail_prob = sum(math.comb(discordant, i) for i in range(min_side + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail_prob)


def _pairwise_rows(settings: StudySettings) -> list[dict]:
    observed = _load_observed_variant_files(settings)
    levels = list(observed.keys())
    variant_stats = {row["reasoning_effort"]: row for row in _variant_rows(settings)}
    rows: list[dict] = []
    raw_p_values: list[float] = []
    for a_level, b_level in itertools.combinations(levels, 2):
        run_a, grade_a = observed[a_level]
        run_b, grade_b = observed[b_level]
        shared_case_ids = sorted(set(run_a.cases) & set(run_b.cases) & set(grade_a.cases) & set(grade_b.cases))
        if not shared_case_ids:
            continue
        counts = _mcnemar_counts(grade_a, grade_b)
        a_correct_b_incorrect = counts["a_correct_b_incorrect"]
        a_incorrect_b_correct = counts["a_incorrect_b_correct"]
        discordant_total = counts["discordant_total"]
        exact_p = _mcnemar_exact_p_value(a_correct_b_incorrect, a_incorrect_b_correct)
        raw_p_values.append(exact_p)

        accuracy_a = float(variant_stats[a_level]["accuracy"])
        accuracy_b = float(variant_stats[b_level]["accuracy"])
        accuracy_delta = accuracy_b - accuracy_a
        extra_tokens = float(variant_stats[b_level]["avg_total_tokens"]) - float(variant_stats[a_level]["avg_total_tokens"])
        extra_latency = float(variant_stats[b_level]["avg_latency_seconds"]) - float(
            variant_stats[a_level]["avg_latency_seconds"]
        )

        tokens_per_additional = None
        seconds_per_additional = None
        if accuracy_delta > 0:
            tokens_per_additional = extra_tokens / accuracy_delta
            seconds_per_additional = extra_latency / accuracy_delta

        rows.append(
            {
                "comparison": f"{a_level}_vs_{b_level}",
                "a_level": a_level,
                "b_level": b_level,
                "n": len(shared_case_ids),
                "a_accuracy": round(accuracy_a, 6),
                "b_accuracy": round(accuracy_b, 6),
                "accuracy_delta_b_minus_a": round(accuracy_delta, 6),
                "absolute_accuracy_delta": round(abs(accuracy_delta), 6),
                "a_correct_b_incorrect": a_correct_b_incorrect,
                "a_incorrect_b_correct": a_incorrect_b_correct,
                "discordant_total": discordant_total,
                "mcnemar_exact_p_value": exact_p,
                "additional_correct_cases_per_1000": round(accuracy_delta * 1000.0, 3),
                "extra_total_tokens": round(extra_tokens, 2),
                "extra_latency_seconds": round(extra_latency, 3),
                "tokens_per_additional_correct_case": (
                    round(tokens_per_additional, 2) if tokens_per_additional is not None else None
                ),
                "seconds_per_additional_correct_case": (
                    round(seconds_per_additional, 4) if seconds_per_additional is not None else None
                ),
            }
        )
    adjusted = _holm_bonferroni_adjust(raw_p_values)
    for row, adjusted_p in zip(rows, adjusted):
        row["mcnemar_exact_p_value"] = round(float(row["mcnemar_exact_p_value"]), 8)
        row["mcnemar_holm_adjusted_p_value"] = round(adjusted_p, 8)
    return rows


def _holm_bonferroni_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    m = len(p_values)
    adjusted_sorted: list[float] = [0.0] * m
    prev = 0.0
    for rank, (_, p_value) in enumerate(indexed):
        adjusted = min(1.0, p_value * (m - rank))
        monotone = max(prev, adjusted)
        adjusted_sorted[rank] = monotone
        prev = monotone
    adjusted_original = [0.0] * m
    for rank, (original_index, _) in enumerate(indexed):
        adjusted_original[original_index] = adjusted_sorted[rank]
    return adjusted_original


def _efficiency_frontier_rows(variant_rows: list[dict]) -> list[dict]:
    # Keep non-dominated variants: maximize accuracy, minimize latency/tokens.
    rows: list[dict] = []
    for candidate in variant_rows:
        dominated = False
        for other in variant_rows:
            if other is candidate:
                continue
            no_worse = (
                other["accuracy"] >= candidate["accuracy"]
                and other["avg_total_tokens"] <= candidate["avg_total_tokens"]
                and other["avg_latency_seconds"] <= candidate["avg_latency_seconds"]
            )
            strictly_better = (
                other["accuracy"] > candidate["accuracy"]
                or other["avg_total_tokens"] < candidate["avg_total_tokens"]
                or other["avg_latency_seconds"] < candidate["avg_latency_seconds"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            rows.append(
                {
                    "reasoning_effort": candidate["reasoning_effort"],
                    "accuracy": candidate["accuracy"],
                    "avg_total_tokens": candidate["avg_total_tokens"],
                    "avg_reasoning_tokens": candidate["avg_reasoning_tokens"],
                    "avg_latency_seconds": candidate["avg_latency_seconds"],
                }
            )
    return rows


def _deployment_view_rows(pairwise_rows: list[dict]) -> list[dict]:
    by_comparison = {row["comparison"]: row for row in pairwise_rows}
    rows: list[dict] = []
    for a_level, b_level in DEPLOYMENT_PRIORITY_PAIRS:
        comparison = f"{a_level}_vs_{b_level}"
        row = by_comparison.get(comparison)
        if row is not None:
            rows.append(row)
    return rows


def _pairwise_p_value_chart_svg(pairwise_rows: list[dict]) -> str:
    width = 860
    height = 360
    margin_left = 80
    margin_right = 40
    margin_top = 40
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    if not pairwise_rows:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            '<rect width="100%" height="100%" fill="#ffffff"/>'
            '<text x="50%" y="50%" text-anchor="middle" fill="#4b5563" font-size="16" font-family="Arial, sans-serif">'
            "No pairwise rows available."
            "</text>"
            "</svg>"
        )

    bars: list[tuple[str, float, float]] = []
    for row in pairwise_rows:
        label = f"{row['a_level']}->{row['b_level']}"
        p_value = max(float(row["mcnemar_exact_p_value"]), 1e-16)
        neg_log10 = -math.log10(p_value)
        bars.append((label, p_value, neg_log10))

    y_max = max(1.0, max(neg_log10 for _, _, neg_log10 in bars) * 1.15)
    zero_y = margin_top + plot_height
    threshold_value = -math.log10(0.05)
    threshold_y = margin_top + plot_height - (threshold_value / y_max) * plot_height
    bar_width = plot_width / max(len(bars), 1) * 0.55

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="50%" y="24" text-anchor="middle" fill="#111827" font-size="18" font-family="Arial, sans-serif">All-pairs McNemar exact p-values</text>',
        f'<line x1="{margin_left}" y1="{zero_y}" x2="{width - margin_right}" y2="{zero_y}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{zero_y}" stroke="#111827" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{threshold_y}" x2="{width - margin_right}" y2="{threshold_y}" stroke="#ef4444" stroke-width="1.2" stroke-dasharray="6 5"/>',
        f'<text x="{width - margin_right - 8}" y="{threshold_y - 6}" text-anchor="end" fill="#ef4444" font-size="11" font-family="Arial, sans-serif">p=0.05</text>',
        f'<text x="{margin_left - 54}" y="{margin_top - 8}" fill="#374151" font-size="11" font-family="Arial, sans-serif">-log10(p)</text>',
    ]

    tick_steps = 4
    for tick_index in range(tick_steps + 1):
        tick_value = (tick_index / tick_steps) * y_max
        y = margin_top + plot_height - (tick_value / y_max) * plot_height
        tick_label = f"{tick_value:.2f}".rstrip("0").rstrip(".")
        parts.append(
            f'<line x1="{margin_left - 4}" y1="{y}" x2="{margin_left}" y2="{y}" stroke="#111827" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 10}" y="{y + 4}" text-anchor="end" fill="#374151" font-size="10" font-family="Arial, sans-serif">{tick_label}</text>'
        )

    slot_width = plot_width / max(len(bars), 1)
    for index, (label, p_value, neg_log10) in enumerate(bars):
        x_center = margin_left + slot_width * (index + 0.5)
        bar_height = (neg_log10 / y_max) * plot_height
        bar_x = x_center - (bar_width / 2)
        bar_y = zero_y - bar_height
        parts.append(
            f'<rect x="{bar_x:.2f}" y="{bar_y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#3b82f6"/>'
        )
        parts.append(
            f'<text x="{x_center:.2f}" y="{zero_y + 20}" text-anchor="middle" fill="#111827" font-size="11" font-family="Arial, sans-serif">{label}</text>'
        )
        parts.append(
            f'<text x="{x_center:.2f}" y="{max(bar_y - 6, margin_top + 12):.2f}" text-anchor="middle" fill="#1f2937" font-size="10" font-family="Arial, sans-serif">p={p_value:.8f}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _write_pairwise_p_value_chart(path: Path, pairwise_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_pairwise_p_value_chart_svg(pairwise_rows))


def _visible_rationale(run: RunFile, case_id: str) -> str:
    case = run.cases[case_id]
    bullets = [item.strip() for item in case.rationale_summary if item and item.strip()]
    if case.api_reasoning_summary:
        bullets.append(f"API summary: {case.api_reasoning_summary.strip()}")
    return " | ".join(bullets)


def export_discordant_cases(
    settings: StudySettings,
    a_level: str = "none",
    b_level: str = "high",
    limit: int = 30,
    write_path: str | None = None,
) -> list[dict]:
    observed = _load_observed_variant_files(settings)
    if a_level not in observed or b_level not in observed:
        return []
    run_a, grade_a = observed[a_level]
    run_b, grade_b = observed[b_level]

    shared_case_ids = sorted(set(run_a.cases) & set(run_b.cases) & set(grade_a.cases) & set(grade_b.cases))
    discordant: list[dict] = []

    for case_id in shared_case_ids:
        a_outcome = grade_a.cases[case_id]
        b_outcome = grade_b.cases[case_id]
        a_correct = a_outcome.diagnosis_correctness_score == 1
        b_correct = b_outcome.diagnosis_correctness_score == 1
        if a_correct == b_correct:
            continue
        discordant.append(
            {
                "case_id": case_id,
                "gold_diagnosis": a_outcome.ground_truth_diagnosis,
                "comparison": f"{a_level}_vs_{b_level}",
                "a_level": a_level,
                "a_prediction": run_a.cases[case_id].diagnosis,
                "a_correctness_label": a_outcome.diagnosis_correctness_label,
                "a_visible_rationale_summary": _visible_rationale(run_a, case_id),
                "a_grader_diagnosis_explanation": a_outcome.diagnosis_explanation,
                "a_grader_reasoning_explanation": a_outcome.reasoning_explanation,
                "b_level": b_level,
                "b_prediction": run_b.cases[case_id].diagnosis,
                "b_correctness_label": b_outcome.diagnosis_correctness_label,
                "b_visible_rationale_summary": _visible_rationale(run_b, case_id),
                "b_grader_diagnosis_explanation": b_outcome.diagnosis_explanation,
                "b_grader_reasoning_explanation": b_outcome.reasoning_explanation,
            }
        )
        if len(discordant) >= limit:
            break

    destination = Path(write_path) if write_path else reports_dir(settings) / f"discordant_{a_level}_vs_{b_level}.json"
    write_json(destination, discordant)
    return discordant


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(x: float) -> str:
    return f"{x:.3f}"


def _format_ci(low: float, high: float) -> str:
    return f"[{low:.3f}, {high:.3f}]"


def _markdown_report(variant_rows: list[dict], pairwise_rows: list[dict], cost_rows: list[dict]) -> str:
    lines = [
        "# Final Statistical Report",
        "",
        "All outputs are derived deterministically from files under `results/` and `scores/`.",
        "",
        "## Per-variant diagnosis accuracy",
        "",
        "| Variant | N | Accuracy | 95% CI | Avg total tokens | Avg reasoning tokens | Avg latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in variant_rows:
        lines.append(
            "| "
            f"{row['reasoning_effort']} | "
            f"{row['n']} | "
            f"{_format_metric(row['accuracy'])} | "
            f"{_format_ci(row['accuracy_ci95_low'], row['accuracy_ci95_high'])} | "
            f"{row['avg_total_tokens']:.2f} | "
            f"{row['avg_reasoning_tokens']:.2f} | "
            f"{row['avg_latency_seconds']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## All-pairs exact McNemar tests",
            "",
            "| Comparison | N | Accuracy A | Accuracy B | |Delta| | A-only correct | B-only correct | Discordant total | Exact p-value | Holm-adjusted p-value |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pairwise_rows:
        lines.append(
            "| "
            f"{row['comparison']} | "
            f"{row['n']} | "
            f"{row['a_accuracy']:.3f} | "
            f"{row['b_accuracy']:.3f} | "
            f"{row['absolute_accuracy_delta']:.3f} | "
            f"{row['a_correct_b_incorrect']} | "
            f"{row['a_incorrect_b_correct']} | "
            f"{row['discordant_total']} | "
            f"{row['mcnemar_exact_p_value']:.8f} | "
            f"{row['mcnemar_holm_adjusted_p_value']:.8f} |"
        )

    lines.extend(
        [
            "",
            "## Efficiency frontier",
            "",
            "| Variant | Accuracy | Avg total tokens | Avg reasoning tokens | Avg latency (s) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in cost_rows:
        lines.append(
            "| "
            f"{row['reasoning_effort']} | "
            f"{_format_metric(row['accuracy'])} | "
            f"{row['avg_total_tokens']:.2f} | "
            f"{row['avg_reasoning_tokens']:.2f} | "
            f"{row['avg_latency_seconds']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def generate_final_artifacts(settings: StudySettings, discordant_limit: int = 30) -> dict[str, Path]:
    validation = validate_committed_inputs(settings)
    variant_rows = _variant_rows(settings)
    pairwise_rows = _pairwise_rows(settings)
    deployment_rows = _deployment_view_rows(pairwise_rows)
    frontier_rows = _efficiency_frontier_rows(variant_rows)

    base_dir = Path(settings.reports_dir)
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir = reports_dir(settings)
    variant_summary_json = base_dir / "variant_summary.json"
    variant_summary_csv = base_dir / "variant_summary.csv"
    pairwise_matrix_json = base_dir / "pairwise_matrix.json"
    pairwise_matrix_csv = base_dir / "pairwise_matrix.csv"
    deployment_views_json = base_dir / "deployment_views.json"
    deployment_views_csv = base_dir / "deployment_views.csv"
    efficiency_frontier_json = base_dir / "efficiency_frontier.json"
    efficiency_frontier_csv = base_dir / "efficiency_frontier.csv"
    pairwise_p_chart_svg = base_dir / "pairwise_mcnemar_p_values.svg"
    report_md = base_dir / "final_report.md"
    validation_json = base_dir / "validation_summary.json"

    write_json(variant_summary_json, variant_rows)
    write_json(pairwise_matrix_json, pairwise_rows)
    write_json(deployment_views_json, deployment_rows)
    write_json(efficiency_frontier_json, frontier_rows)
    write_json(validation_json, validation)
    _write_csv(variant_summary_csv, variant_rows)
    _write_csv(pairwise_matrix_csv, pairwise_rows)
    _write_csv(deployment_views_csv, deployment_rows)
    _write_csv(efficiency_frontier_csv, frontier_rows)
    _write_pairwise_p_value_chart(pairwise_p_chart_svg, pairwise_rows)
    report_md.write_text(_markdown_report(variant_rows, pairwise_rows, frontier_rows))

    discordant_dir = base_dir / "discordant_case_exports"
    discordant_dir.mkdir(parents=True, exist_ok=True)
    discordant_counts: dict[str, int] = {}
    for a_level, b_level in DEPLOYMENT_PRIORITY_PAIRS:
        rows = export_discordant_cases(
            settings,
            a_level=a_level,
            b_level=b_level,
            limit=discordant_limit,
            write_path=str(discordant_dir / f"discordant_{a_level}_vs_{b_level}.json"),
        )
        if rows:
            discordant_counts[f"{a_level}_vs_{b_level}"] = len(rows)

    print("\n=== Final report artifacts ===")
    print(f"variant summary: {variant_summary_json}")
    print(f"pairwise matrix: {pairwise_matrix_json}")
    print(f"deployment views: {deployment_views_json}")
    print(f"efficiency frontier: {efficiency_frontier_json}")
    print(f"pairwise p-value chart: {pairwise_p_chart_svg}")
    print(f"markdown report: {report_md}")
    print(f"validation summary: {validation_json}")
    print(f"discordant exports written: {len(discordant_counts)}")

    if variant_rows:
        print("\n=== Main result (copyable) ===")
        for row in variant_rows:
            print(
                f"{row['reasoning_effort']:>6} | N={row['n']:>4} | acc={row['accuracy']:.3f} "
                f"(95% CI {_format_ci(row['accuracy_ci95_low'], row['accuracy_ci95_high'])}) | "
                f"tokens={row['avg_total_tokens']:.1f} | latency={row['avg_latency_seconds']:.3f}s"
            )

    return {
        "variant_summary_json": variant_summary_json,
        "variant_summary_csv": variant_summary_csv,
        "pairwise_matrix_json": pairwise_matrix_json,
        "pairwise_matrix_csv": pairwise_matrix_csv,
        "deployment_views_json": deployment_views_json,
        "deployment_views_csv": deployment_views_csv,
        "efficiency_frontier_json": efficiency_frontier_json,
        "efficiency_frontier_csv": efficiency_frontier_csv,
        "validation_summary_json": validation_json,
        "pairwise_p_values_chart_svg": pairwise_p_chart_svg,
        "report_md": report_md,
        "discordant_case_exports_dir": discordant_dir,
    }
