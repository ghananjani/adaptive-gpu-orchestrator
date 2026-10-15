from adaptive_gpu_orchestrator.discovery.nvidia_smi import NvidiaSMI


QUERY = """0, NVIDIA H100 80GB HBM3, GPU-abc, 81559, 40000, 73, 55, 61, 420.50, 700.00, 00000000:18:00.0, 9.0
1, NVIDIA H100 80GB HBM3, GPU-def, 81559, 42000, 69, 58, 63, 435.25, 700.00, 00000000:2A:00.0, 9.0
"""

TOPO = """
        GPU0    GPU1    CPU Affinity
GPU0     X      NV4     0-31
GPU1    NV4      X      0-31
"""


def runner(args):
    joined = " ".join(args)
    if "--query-gpu=index" in joined:
        return QUERY
    if "topo -m" in joined:
        return TOPO
    if "--query-gpu=driver_version" in joined:
        return "570.86.15\n570.86.15\n"
    return "NVIDIA-SMI 570.86.15    Driver Version: 570.86.15    CUDA Version: 12.8\n"


def test_query_gpus():
    smi = NvidiaSMI(runner=runner)
    gpus = smi.query_gpus()
    assert len(gpus) == 2
    assert gpus[0].name == "NVIDIA H100 80GB HBM3"
    assert gpus[0].memory_pressure_pct > 49


def test_snapshot_injected_runner():
    smi = NvidiaSMI(runner=runner)
    snapshot = smi.snapshot()
    assert len(snapshot.gpus) == 2
    assert snapshot.driver_version == "570.86.15"
    assert snapshot.cuda_version == "12.8"
    assert len(snapshot.topology) == 1
