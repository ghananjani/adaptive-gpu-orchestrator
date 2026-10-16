from adaptive_gpu_orchestrator.campaigns import capture_campaign_provenance
from adaptive_gpu_orchestrator.models import ClusterSnapshot


def test_campaign_provenance_accepts_explicit_software_and_container_metadata():
    result = capture_campaign_provenance(
        snapshot=ClusterSnapshot(hostname="node"),
        package_names=(),
        cuda_toolkit_version="13.0",
        nccl_version="2.28.3",
        container_image="registry/train:1",
        container_image_digest="sha256:def",
    )

    assert result.cuda_toolkit_version == "13.0"
    assert result.nccl_version == "2.28.3"
    assert result.container_image == "registry/train:1"
    assert result.container_image_digest == "sha256:def"


def test_campaign_provenance_uses_container_environment(monkeypatch):
    monkeypatch.setenv("AGO_CONTAINER_IMAGE", "registry/train:env")
    monkeypatch.setenv("AGO_CONTAINER_IMAGE_DIGEST", "sha256:env")

    result = capture_campaign_provenance(
        snapshot=ClusterSnapshot(hostname="node"),
        package_names=(),
        cuda_toolkit_version="12.8",
        nccl_version="2.27.0",
    )

    assert result.container_image == "registry/train:env"
    assert result.container_image_digest == "sha256:env"
