import subprocess

from adaptive_gpu_orchestrator.campaigns.provenance import probe_runtime_versions


def test_runtime_versions_probe_cuda_toolkit_and_nccl_package():
    def runner(command):
        assert command == ["nvcc", "--version"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "Cuda compilation tools, release 13.0, "
                "V13.0.88"
            ),
            stderr="",
        )

    versions = probe_runtime_versions(
        package_versions={
            "nvidia-nccl-cu13": "2.27.3",
            "nvidia-cuda-runtime-cu13": "13.0.48",
        },
        command_runner=runner,
    )

    assert versions == {
        "cuda_toolkit": "13.0",
        "nccl_package": "2.27.3",
        "nccl_package_name": "nvidia-nccl-cu13",
        "cuda_runtime_package": "13.0.48",
        "cuda_runtime_package_name": "nvidia-cuda-runtime-cu13",
    }


def test_runtime_versions_gracefully_handles_missing_nvcc():
    def runner(command):
        raise FileNotFoundError(command[0])

    versions = probe_runtime_versions(
        package_versions={},
        command_runner=runner,
    )

    assert versions == {}
