import json
from types import SimpleNamespace

from adaptive_gpu_orchestrator import research_cli


def test_parser_requires_hardware_and_output():
    parser = research_cli.build_parser()
    args = parser.parse_args([
        "campaign-a", "--hardware-json", "hardware.json", "--output-dir", "out",
    ])
    assert args.campaign_ids == ["campaign-a"]
    assert args.iterations == 200
    assert not args.allow_software_mismatch


def test_cli_deterministic_read_only_wiring(monkeypatch, tmp_path, capsys):
    hardware = {
        name: {
            "gpu_count": 8, "memory_gib_per_gpu": 80.0,
            "memory_bandwidth_gbps_per_gpu": 3000.0,
            "compute_tflops_per_gpu": 60.0, "interconnect_bandwidth_gbps": 300.0,
        }
        for name in ("A", "B", "C")
    }
    hardware_file = tmp_path / "hardware.json"
    hardware_file.write_text(json.dumps(hardware), encoding="utf-8")
    loaded = []
    called = []

    def load(self, identifier):
        loaded.append(identifier)
        return SimpleNamespace(id=identifier)

    def evaluate(campaigns, **kwargs):
        called.append((tuple(c.id for c in campaigns), kwargs))
        return "validated"

    def export(result, output_dir, *, evidence_manifest):
        assert result == "validated"
        assert evidence_manifest["manifest_sha256"] == "fixture"
        assert output_dir == tmp_path / "out"
        return {"html": output_dir / "research-report.html"}

    monkeypatch.setattr(research_cli.CampaignStore, "load", load)
    monkeypatch.setattr(research_cli, "evaluate_saved_refitting", evaluate)
    monkeypatch.setattr(research_cli, "write_research_report", export)
    monkeypatch.setattr(
        research_cli, "fingerprint_saved_evidence",
        lambda campaigns, *, optimization_store: {"manifest_sha256": "fixture"},
    )
    args = [
        "A-0", "A-1", "B-0", "B-1", "C-0", "C-1",
        "--hardware-json", str(hardware_file),
        "--output-dir", str(tmp_path / "out"),
        "--iterations", "20", "--seed", "21",
    ]
    assert research_cli.main(args) == 0
    assert loaded == args[:6]
    assert called[0][1]["seed"] == 21
    assert len(called[0][1]["measured_hardware"]) == 3
    assert "research-report.html" in capsys.readouterr().out


def test_cli_missing_file_fails_closed(tmp_path, capsys):
    status = research_cli.main([
        "A", "--hardware-json", str(tmp_path / "missing.json"),
        "--output-dir", str(tmp_path / "out"),
    ])
    assert status == 2
    assert "research report failed:" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_duplicate_campaign_rejected_before_load(tmp_path, capsys):
    hardware_file = tmp_path / "hardware.json"
    hardware_file.write_text("{}", encoding="utf-8")
    assert research_cli.main([
        "same", "same", "--hardware-json", str(hardware_file),
        "--output-dir", str(tmp_path / "out"),
    ]) == 2
    assert "unique" in capsys.readouterr().err
