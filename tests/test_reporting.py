from __future__ import annotations

from pathlib import Path

import gpt_5_2_reasoning_ablation.reporting as reporting_module
from gpt_5_2_reasoning_ablation.io_utils import write_json
from gpt_5_2_reasoning_ablation.reporting import (
    _mcnemar_exact_p_value,
    _pairwise_rows,
    _variant_rows,
    _wilson_interval,
    export_discordant_cases,
    generate_final_artifacts,
)
from gpt_5_2_reasoning_ablation.settings import StudySettings


def _seed_files(settings: StudySettings, levels: tuple[str, ...] = ("none", "low", "medium", "high")) -> None:
    case_truth = {"C1": "A", "C2": "B", "C3": "C", "C4": "D"}
    correctness_by_level = {
        "none": {"C1": 1, "C2": 0, "C3": 1, "C4": 0},
        "low": {"C1": 1, "C2": 1, "C3": 0, "C4": 0},
        "medium": {"C1": 1, "C2": 1, "C3": 1, "C4": 0},
        "high": {"C1": 0, "C2": 1, "C3": 1, "C4": 1},
    }
    run_meta = {
        "none": {"latency": 1.0, "total_tokens": 100, "reasoning_tokens": 10},
        "low": {"latency": 2.0, "total_tokens": 130, "reasoning_tokens": 20},
        "medium": {"latency": 3.0, "total_tokens": 170, "reasoning_tokens": 40},
        "high": {"latency": 4.0, "total_tokens": 210, "reasoning_tokens": 90},
    }

    for level in levels:
        run_cases = {}
        grade_cases = {}
        for idx, case_id in enumerate(sorted(case_truth.keys()), start=1):
            is_correct = correctness_by_level[level][case_id] == 1
            prediction = case_truth[case_id] if is_correct else f"{case_truth[case_id]}_wrong_{level}"
            run_cases[case_id] = {
                "case_id": case_id,
                "diagnosis": prediction,
                "rationale_summary": [f"{level}_rationale_{case_id}"],
                "raw_output_text": "{}",
                "parsed_output": {"diagnosis": prediction, "rationale_summary": [f"{level}_rationale_{case_id}"]},
                "api_reasoning_summary": f"{level}_reasoning" if level == "high" else None,
                "latency_seconds": run_meta[level]["latency"] + idx / 10.0,
                "usage": {
                    "total_tokens": run_meta[level]["total_tokens"] + idx,
                    "reasoning_tokens": run_meta[level]["reasoning_tokens"] + idx,
                },
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
            grade_cases[case_id] = {
                "case_id": case_id,
                "ground_truth_diagnosis": case_truth[case_id],
                "gold_reasoning_checklist": [f"gold_{case_id}"],
                "predicted_diagnosis": prediction,
                "diagnosis_correctness_score": correctness_by_level[level][case_id],
                "diagnosis_correctness_label": "correct" if is_correct else "incorrect",
                "diagnosis_explanation": "good" if is_correct else "wrong",
                "reasoning_alignment_score": 4 if is_correct else 2,
                "reasoning_alignment_label": "strongly aligned" if is_correct else "mixed",
                "reasoning_explanation": "aligned" if is_correct else "partial",
                "grader_model": "gpt-4.1",
                "grader_timestamp": "2026-01-01T00:00:00+00:00",
            }

        run_payload = {
            "study_name": "gpt-5.2-reasoning-ablation",
            "dataset": {"name": "x", "split": "y"},
            "variant": {"id": f"gpt-5.2__reasoning-{level}", "model": "gpt-5.2", "reasoning_effort": level},
            "run_settings": {},
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "cases": run_cases,
        }
        grade_payload = {
            "study_name": "gpt-5.2-reasoning-ablation",
            "grader_model": "gpt-4.1",
            "variant": {"id": f"gpt-5.2__reasoning-{level}", "model": "gpt-5.2", "reasoning_effort": level},
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "cases": grade_cases,
        }
        write_json(Path(settings.results_dir) / f"gpt-5.2__reasoning-{level}.json", run_payload)
        write_json(Path(settings.scores_dir) / f"gpt-5.2__reasoning-{level}.json", grade_payload)


def test_exact_mcnemar_p_value_is_two_sided():
    assert _mcnemar_exact_p_value(0, 0) == 1.0
    assert _mcnemar_exact_p_value(1, 0) == 1.0
    assert _mcnemar_exact_p_value(4, 0) == 0.125


def test_wilson_interval_bounds():
    low, high = _wilson_interval(successes=8, total=10)
    assert 0 <= low <= high <= 1
    assert low < 0.8 < high


def test_pairwise_and_discordant_exports(tmp_path):
    settings = StudySettings(
        results_dir=str(tmp_path / "results"),
        scores_dir=str(tmp_path / "scores"),
        reports_dir=str(tmp_path / "reports"),
    )
    _seed_files(settings)

    variants = _variant_rows(settings)
    assert [row["reasoning_effort"] for row in variants] == ["none", "low", "medium", "high"]

    pairs = _pairwise_rows(settings)
    assert len(pairs) == 6
    assert [row["comparison"] for row in pairs] == [
        "none_vs_low",
        "none_vs_medium",
        "none_vs_high",
        "low_vs_medium",
        "low_vs_high",
        "medium_vs_high",
    ]
    low_medium = next(row for row in pairs if row["comparison"] == "low_vs_medium")
    assert low_medium["discordant_total"] > 0
    assert isinstance(low_medium["mcnemar_exact_p_value"], float)
    assert 0.0 <= low_medium["mcnemar_exact_p_value"] <= 1.0
    assert "mcnemar_holm_adjusted_p_value" in low_medium
    assert "additional_correct_cases_per_1000" in low_medium
    assert "tokens_per_additional_correct_case" in low_medium

    discordant = export_discordant_cases(settings, a_level="none", b_level="high", limit=10)
    assert len(discordant) == 3
    assert discordant[0]["case_id"] == "C1"
    assert discordant[0]["a_correctness_label"] == "correct"
    assert discordant[0]["b_correctness_label"] == "incorrect"


def test_pairwise_counts_for_partial_variant_sets(tmp_path):
    settings_three = StudySettings(
        results_dir=str(tmp_path / "results3"),
        scores_dir=str(tmp_path / "scores3"),
        reports_dir=str(tmp_path / "reports3"),
    )
    _seed_files(settings_three, levels=("none", "medium", "high"))
    pairs_three = _pairwise_rows(settings_three)
    assert len(pairs_three) == 3
    assert [row["comparison"] for row in pairs_three] == [
        "none_vs_medium",
        "none_vs_high",
        "medium_vs_high",
    ]

    settings_two = StudySettings(
        results_dir=str(tmp_path / "results2"),
        scores_dir=str(tmp_path / "scores2"),
        reports_dir=str(tmp_path / "reports2"),
    )
    _seed_files(settings_two, levels=("none", "high"))
    pairs_two = _pairwise_rows(settings_two)
    assert len(pairs_two) == 1
    assert pairs_two[0]["comparison"] == "none_vs_high"


def test_pairwise_rows_use_full_precision_p_values_for_holm(tmp_path, monkeypatch):
    settings = StudySettings(
        results_dir=str(tmp_path / "results"),
        scores_dir=str(tmp_path / "scores"),
        reports_dir=str(tmp_path / "reports"),
    )
    _seed_files(settings, levels=("none", "low", "medium"))

    raw_p_values = [0.111111115, 0.222222225, 0.333333335]
    p_iter = iter(raw_p_values)
    captured_inputs: dict[str, list[float]] = {}

    monkeypatch.setattr(reporting_module, "_mcnemar_exact_p_value", lambda *_args: next(p_iter))

    def _capture_holm(values: list[float]) -> list[float]:
        captured_inputs["p_values"] = list(values)
        return list(values)

    monkeypatch.setattr(reporting_module, "_holm_bonferroni_adjust", _capture_holm)

    rows = reporting_module._pairwise_rows(settings)

    assert captured_inputs["p_values"] == raw_p_values
    assert captured_inputs["p_values"] != [round(value, 8) for value in raw_p_values]
    assert [row["mcnemar_holm_adjusted_p_value"] for row in rows] == [round(value, 8) for value in raw_p_values]


def test_generate_report_artifacts_from_scratch(tmp_path):
    settings = StudySettings(
        results_dir=str(tmp_path / "results"),
        scores_dir=str(tmp_path / "scores"),
        reports_dir=str(tmp_path / "reports"),
    )
    _seed_files(settings)
    artifacts = generate_final_artifacts(settings, discordant_limit=5)
    assert artifacts["variant_summary_json"].exists()
    assert artifacts["pairwise_matrix_json"].exists()
    assert artifacts["deployment_views_json"].exists()
    assert artifacts["efficiency_frontier_json"].exists()
    assert artifacts["validation_summary_json"].exists()
    assert artifacts["discordant_case_exports_dir"].exists()
