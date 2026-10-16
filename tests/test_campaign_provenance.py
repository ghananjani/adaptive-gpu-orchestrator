from adaptive_gpu_orchestrator.campaigns import capture_campaign_provenance
from adaptive_gpu_orchestrator.models import ClusterSnapshot, GPUDevice


def test_campaign_provenance_captures_gpu_driver_cuda_and_tags():
    snapshot = ClusterSnapshot(
        hostname="gpu-node",
        driver_version="580.12",
        cuda_version="13.0",
        gpus=[
            GPUDevice(
                index=0,
                name="NVIDIA H100 80GB HBM3",
                uuid="GPU-123",
                memory_total_mib=81559,
                pci_bus_id="00000000:01:00.0",
                compute_capability="9.0",
            )
        ],
    )

    result = capture_campaign_provenance(
        tags={"workload": "llama70b"},
        snapshot=snapshot,
        package_names=(),
    )

    assert result.cluster_snapshot["hostname"] == "gpu-node"
    assert result.cluster_snapshot["driver_version"] == "580.12"
    assert result.cluster_snapshot["cuda_version"] == "13.0"
    assert result.cluster_snapshot["gpus"][0]["name"].startswith("NVIDIA H100")
    assert result.cluster_snapshot["gpus"][0]["compute_capability"] == "9.0"
    assert result.runtime.tags == {"workload": "llama70b"}
    assert result.package_versions == {}
