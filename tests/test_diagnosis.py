from adaptive_gpu_orchestrator.diagnosis import diagnose
from adaptive_gpu_orchestrator.models import Bottleneck, ClusterSnapshot, GPUDevice, NCCLResult


def gpu(index: int, util: float, used: int = 40000, total: int = 80000, temp: float = 60):
    return GPUDevice(
        index=index,
        name="NVIDIA H100",
        memory_total_mib=total,
        memory_used_mib=used,
        utilization_gpu_pct=util,
        temperature_c=temp,
        power_draw_w=400,
        power_limit_w=700,
    )


def test_no_gpu():
    report = diagnose(ClusterSnapshot(hostname="test"))
    assert report.bottleneck is Bottleneck.NO_GPU


def test_load_imbalance():
    snap = ClusterSnapshot(hostname="test", gpus=[gpu(0, 95), gpu(1, 90), gpu(2, 40), gpu(3, 35)])
    report = diagnose(snap)
    assert report.bottleneck is Bottleneck.LOAD_IMBALANCE


def test_memory_pressure():
    snap = ClusterSnapshot(hostname="test", gpus=[gpu(0, 70, 76000), gpu(1, 72, 75000)])
    report = diagnose(snap)
    assert report.bottleneck is Bottleneck.MEMORY_PRESSURE


def test_underutilized():
    snap = ClusterSnapshot(hostname="test", gpus=[gpu(0, 25), gpu(1, 30)])
    report = diagnose(snap)
    assert report.bottleneck is Bottleneck.UNDERUTILIZED


def test_communication_bound_with_nccl():
    snap = ClusterSnapshot(hostname="test", gpus=[gpu(0, 55), gpu(1, 58)])
    nccl = [NCCLResult("all_reduce", 1024, 20.0, 50.0, 70.0)]
    report = diagnose(snap, nccl=nccl)
    assert report.bottleneck is Bottleneck.COMMUNICATION_BOUND
