from adaptive_gpu_orchestrator.campaigns import CampaignNCCLBaseline
from adaptive_gpu_orchestrator.models import NCCLResult


def test_campaign_nccl_baseline_summary_and_round_trip():
    baseline = CampaignNCCLBaseline(
        operation="all_reduce",
        source="saved-output",
        binary="all_reduce_perf",
        begin="8M",
        end="1G",
        factor=2,
        gpus=8,
        results=(
            NCCLResult(
                operation="all_reduce",
                size_bytes=8 * 1024 * 1024,
                time_us=10.0,
                algbw_gbps=100.0,
                busbw_gbps=90.0,
            ),
            NCCLResult(
                operation="all_reduce",
                size_bytes=1024 * 1024 * 1024,
                time_us=20.0,
                algbw_gbps=140.0,
                busbw_gbps=126.0,
            ),
        ),
    )

    assert baseline.summary["sample_count"] == 2
    assert baseline.summary["mean_algbw_gbps"] == 120.0
    assert baseline.summary["peak_algbw_gbps"] == 140.0
    assert baseline.summary["mean_busbw_gbps"] == 108.0
    assert baseline.summary["peak_busbw_gbps"] == 126.0
    assert baseline.summary["largest_size_bytes"] == 1024 * 1024 * 1024

    loaded = CampaignNCCLBaseline.from_dict(baseline.to_dict())
    assert loaded == baseline
    assert loaded.summary == baseline.summary
