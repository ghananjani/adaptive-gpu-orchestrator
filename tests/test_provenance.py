from adaptive_gpu_orchestrator.evidence import capture_provenance


def test_provenance_captures_only_whitelisted_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("NCCL_DEBUG", "INFO")
    monkeypatch.setenv("SECRET_TOKEN", "do-not-capture")

    result = capture_provenance(tags={"hardware": "h100"})

    assert result.environment["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert result.environment["NCCL_DEBUG"] == "INFO"
    assert "SECRET_TOKEN" not in result.environment
    assert result.tags == {"hardware": "h100"}
    assert result.python_version
    assert result.implementation
