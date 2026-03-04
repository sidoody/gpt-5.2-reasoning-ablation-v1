from __future__ import annotations

from pathlib import Path

import gpt_5_2_reasoning_ablation.analysis as analysis_module
from gpt_5_2_reasoning_ablation.analysis import analyze_pairs
from gpt_5_2_reasoning_ablation.io_utils import write_json
from gpt_5_2_reasoning_ablation.settings import StudySettings


def _seed_files(settings: StudySettings, levels: tuple[str, ...]) -> None:
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
                "api_reasoning_summary": None,
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
                "diagnosis_explanation": "ok",
                "reasoning_alignment_score": 4 if is_correct else 2,
                "reasoning_alignment_label": "aligned" if is_correct else "mixed",
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


def test_analyze_pairs_emits_all_unique_pairs(tmp_path):
    settings = StudySettings(
        results_dir=str(tmp_path / "results"),
        scores_dir=str(tmp_path / "scores"),
        reports_dir=str(tmp_path / "reports"),
    )
    _seed_files(settings, levels=("none", "low", "medium", "high"))
    rows = analyze_pairs(settings)
    assert len(rows) == 6
    assert [row["comparison"] for row in rows] == [
        "none_vs_low",
        "none_vs_medium",
        "none_vs_high",
        "low_vs_medium",
        "low_vs_high",
        "medium_vs_high",
    ]
    first = rows[0]
    assert "mcnemar_holm_adjusted_p_value" in first
    assert "additional_correct_cases_per_1000" in first
    assert "tokens_per_additional_correct_case" in first


def test_analyze_pairs_handles_missing_intermediate_variant(tmp_path):
    settings = StudySettings(
        results_dir=str(tmp_path / "results"),
        scores_dir=str(tmp_path / "scores"),
        reports_dir=str(tmp_path / "reports"),
    )
    _seed_files(settings, levels=("none", "medium", "high"))
    rows = analyze_pairs(settings)
    assert len(rows) == 3
    assert [row["comparison"] for row in rows] == [
        "none_vs_medium",
        "none_vs_high",
        "medium_vs_high",
    ]


def test_analyze_pairs_uses_full_precision_p_values_for_holm(tmp_path, monkeypatch):
    settings = StudySettings(
        results_dir=str(tmp_path / "results"),
        scores_dir=str(tmp_path / "scores"),
        reports_dir=str(tmp_path / "reports"),
    )
    _seed_files(settings, levels=("none", "low", "medium"))

    raw_p_values = [0.111111115, 0.222222225, 0.333333335]
    p_iter = iter(raw_p_values)
    captured_inputs: dict[str, list[float]] = {}

    monkeypatch.setattr(analysis_module, "_mcnemar_exact_p_value", lambda *_args: next(p_iter))

    def _capture_holm(values: list[float]) -> list[float]:
        captured_inputs["p_values"] = list(values)
        return list(values)

    monkeypatch.setattr(analysis_module, "_holm_bonferroni_adjust", _capture_holm)

    rows = analysis_module.analyze_pairs(settings)

    assert captured_inputs["p_values"] == raw_p_values
    assert captured_inputs["p_values"] != [round(value, 8) for value in raw_p_values]
    assert [row["mcnemar_holm_adjusted_p_value"] for row in rows] == [round(value, 8) for value in raw_p_values]
