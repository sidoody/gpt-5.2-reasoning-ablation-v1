from __future__ import annotations

from gpt_5_2_reasoning_ablation.reporting import generate_final_artifacts
from gpt_5_2_reasoning_ablation.settings import StudySettings


def test_report_regeneration_from_committed_outputs(tmp_path):
    settings = StudySettings(
        results_dir="results",
        scores_dir="scores",
        reports_dir=str(tmp_path / "reports"),
    )
    artifacts = generate_final_artifacts(settings, discordant_limit=5)
    assert artifacts["variant_summary_json"].exists()
    assert artifacts["pairwise_matrix_json"].exists()
    assert artifacts["deployment_views_json"].exists()
    assert artifacts["efficiency_frontier_json"].exists()
    assert artifacts["validation_summary_json"].exists()
