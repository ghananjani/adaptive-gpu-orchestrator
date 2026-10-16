import pytest

from adaptive_gpu_orchestrator.benchmarks import (
    NCCLBaselineSpec,
    NCCLCommunicationBaseline,
    capture_nccl_baseline,
)
from adaptive_gpu_orchestrator.models import NCCLResult


class FakeBenchmark:
    def run(self, **kwargs):
        assert kwargs["begin"] == "8M"
        assert kwargs["end"] == "1G"
        assert kwargs["factor"] == 2
        assert kwargs["gpus"] == 2
        return [
            NCCLResult(
                operation="all_reduce",
                size_bytes=8_388_608,
                time_us=100.0,
                algbw_gbps=80.0,
                busbw_gbps=70.0,
            ),
            NCCLResult(
                operation="all_reduce",
                size_bytes=16_777_216,
                time_us=110.0,
                algbw_gbps=90.0,
                busbw_gbps=80.0,
            ),
        ]


def test_nccl_communication_baseline_capture_and_round_trip():
    spec = NCCLBaselineSpec(gpus=2)
    baseline = capture_nccl_baseline(spec, benchmark=FakeBenchmark())

    assert baseline.summary["busbw_mean_gbps"] == 75.0
    assert baseline.summary["algbw_max_gbps"] == 90.0
    assert baseline.verify() is True
    assert len(baseline.sha256) == 64

    loaded = NCCLCommunicationBaseline.from_dict(baseline.to_dict())
    assert loaded == baseline
    assert loaded.verify() is True


def test_nccl_communication_baseline_detects_tampering():
    spec = NCCLBaselineSpec(gpus=2)
    baseline = capture_nccl_baseline(spec, benchmark=FakeBenchmark())
    payload = baseline.to_dict()
    payload["results"][0]["busbw_gbps"] = 1.0

    with pytest.raises(ValueError, match="SHA-256"):
        NCCLCommunicationBaseline.from_dict(payload)


def test_nccl_baseline_spec_validates_gpu_count():
    with pytest.raises(ValueError, match="GPU count"):
        NCCLBaselineSpec(gpus=0)
