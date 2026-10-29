import hashlib
import json
from types import SimpleNamespace

from adaptive_gpu_orchestrator.modeling import research_report


def result_fixture():
    fold = {
        "held_out_platform": "<GPU-A> & | strange",
        "train_samples": 8, "test_samples": 4,
        "config_rmse": 12.0, "conditioned_rmse": 10.0,
        "config_mae": 8.0, "conditioned_mae": 7.0,
        "rmse_reduction_pct": 16.6666666667,
    }
    evaluation = SimpleNamespace(
        metric="throughput_per_s", config_pooled_rmse=12.0,
        conditioned_pooled_rmse=10.0, rmse_reduction_pct=16.6666666667,
        test_samples=4, folds=[SimpleNamespace(to_dict=lambda: fold)],
    )
    comparison = SimpleNamespace(
        evaluation=evaluation, trial_count=4, excluded_failed=1,
        excluded_missing=2, warnings=("software difference",),
    )
    audit = SimpleNamespace(
        campaign_count=6, platform_count=3,
        warnings=("experimental confounding risk",),
    )
    refit = SimpleNamespace(
        lower_pct=-5.0, upper_pct=23.0, confidence=0.95,
        valid_iterations=19, iterations=20,
        findings=("conditional fixed-platform result",),
    )
    raw = {
        "comparison": {
            "evaluation": {"folds": [fold]},
            "source_sha256": "a" * 64,
            "shared_tags": {"model": "fixture", "workload": "synthetic"},
        },
        "design_audit": {"independent_campaign_replication": True},
        "refitting": {"seed": 42},
    }
    return SimpleNamespace(
        comparison=comparison, design_audit=audit, refitting=refit,
        to_dict=lambda: raw,
    )


def test_all_exports_are_stable_and_valid(tmp_path):
    r = result_fixture()
    paths = research_report.write_research_report(r, tmp_path)
    assert set(paths) == {"json", "markdown", "html", "folds_csv", "sha256"}
    payload = json.loads(paths["json"].read_text())
    assert payload["summary"]["refit_lower_pct"] == -5.0
    assert len(payload["warnings"]) >= 3
    assert "GPU Hardware Transfer" in paths["markdown"].read_text()
    assert "GPU Hardware Transfer" in paths["html"].read_text()
    assert "held_out_platform" in paths["folds_csv"].read_text()
    contents = {name: path.read_bytes() for name, path in paths.items()}
    research_report.write_research_report(r, tmp_path)
    assert contents == {name: path.read_bytes() for name, path in paths.items()}
    digest = hashlib.sha256(contents["json"]).hexdigest()
    assert paths["sha256"].read_text() == f"{digest}  research-report.json\n"


def test_html_escapes_untrusted_platform_and_keeps_caveats():
    payload = research_report.research_report_payload(result_fixture())
    rendered = research_report.render_research_html(payload)
    assert "&lt;GPU-A&gt; &amp; | strange" in rendered
    assert "<GPU-A>" not in rendered
    assert "not calibrated" in rendered


def test_markdown_handles_pipe_and_no_performance_claims():
    payload = research_report.research_report_payload(result_fixture())
    rendered = research_report.render_research_markdown(payload)
    assert "&#124;" in rendered
    assert "16.6667%" in rendered
    assert "synthetic" not in rendered or "research" in rendered.lower()
