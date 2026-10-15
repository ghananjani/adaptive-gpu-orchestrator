from adaptive_gpu_orchestrator.telemetry import (
    parse_dcgm_dmon,
    parse_dcgm_exporter,
    summarize_dcgm,
)


def test_parse_concatenated_dcgm_exporter_scrapes():
    text = """
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-0"} 90
DCGM_FI_DEV_GPU_UTIL{gpu="1",UUID="GPU-1"} 60
DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-0"} 500
DCGM_FI_DEV_POWER_USAGE{gpu="1",UUID="GPU-1"} 420
DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-0"} 92
DCGM_FI_DEV_GPU_UTIL{gpu="1",UUID="GPU-1"} 62
DCGM_FI_DEV_POWER_USAGE{gpu="0",UUID="GPU-0"} 510
DCGM_FI_DEV_POWER_USAGE{gpu="1",UUID="GPU-1"} 430
"""

    frames = parse_dcgm_exporter(text, interval_s=2.0)

    assert len(frames) == 2
    assert frames[0].offset_s == 0.0
    assert frames[1].offset_s == 2.0
    assert frames[0].devices[0].metrics["DCGM_FI_DEV_GPU_UTIL"] == 90.0
    summary = summarize_dcgm(frames)
    assert summary[0]["DCGM_FI_DEV_GPU_UTIL"] == 91.0
    assert summary[1]["DCGM_FI_DEV_POWER_USAGE"] == 425.0


def test_parse_dcgm_dmon_frames():
    text = """
# Entity  GRACT  TENSO  DRAMA
Id
GPU 0  0.970  0.930  0.400
GPU 1  0.820  0.700  0.350
GPU 0  0.980  0.940  0.420
GPU 1  0.830  0.710  0.360
"""

    frames = parse_dcgm_dmon(text)

    assert len(frames) == 2
    assert frames[0].devices[0].metrics["GRACT"] == 0.97
    assert frames[1].devices[1].metrics["TENSO"] == 0.71
