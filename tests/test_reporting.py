from __future__ import annotations

from pathlib import Path

from gpt_5_2_reasoning_ablation.io_utils import write_json
from gpt_5_2_reasoning_ablation.reporting import (
    _mcnemar_exact_p_value,
    _pairwise_rows,
    _variant_rows,
    _wilson_interval,
    export_discordant_cases,
)
from gpt_5_2_reasoning_ablation.settings import StudySettings


def _seed_files(settings: StudySettings) -> None:
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

    for level in ("none", "low", "medium", "high"):
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
    assert len(pairs) == 3
    assert [row["comparison"] for row in pairs] == ["none_vs_low", "low_vs_medium", "medium_vs_high"]
    low_medium = next(row for row in pairs if row["comparison"] == "low_vs_medium")
    assert low_medium["discordant_total"] > 0
    assert isinstance(low_medium["mcnemar_exact_p_value"], float)
    assert 0.0 <= low_medium["mcnemar_exact_p_value"] <= 1.0
    disallowed = {"none_vs_medium", "none_vs_high", "low_vs_high"}
    assert all(row["comparison"] not in disallowed for row in pairs)

    discordant = export_discordant_cases(settings, a_level="none", b_level="high", limit=10)
    assert len(discordant) == 3
    assert discordant[0]["case_id"] == "C1"
    assert discordant[0]["a_correctness_label"] == "correct"
    assert discordant[0]["b_correctness_label"] == "incorrect"
