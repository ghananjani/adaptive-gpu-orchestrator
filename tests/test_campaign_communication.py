from adaptive_gpu_orchestrator.benchmarks import (
    NCCLBaselineSpec,
    NCCLCommunicationBaseline,
)
from adaptive_gpu_orchestrator.campaigns import (
    CampaignProvenance,
    CampaignSpec,
    compare_campaigns,
    run_campaign,
)
from adaptive_gpu_orchestrator.evidence import RuntimeProvenance
from adaptive_gpu_orchestrator.models import NCCLResult
from adaptive_gpu_orchestrator.optimizer import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationStore,
    ParallelismConfig,
    TrialMetrics,
)
from adaptive_gpu_orchestrator.studies import StudyStore


class Runner:
    def __init__(self, throughput):
        self.throughput = throughput

    def run(self, config):
        return TrialMetrics(
            throughput_per_s=self.throughput,
            p99_latency_ms=100.0,
            energy_joules=1000.0,
        )


def provenance(label):
    return CampaignProvenance(
        runtime=RuntimeProvenance(
            python_version="3.12.0",
            implementation="cpython",
            platform="Linux-test",
            machine="x86_64",
        ),
        cluster_snapshot={
            "hostname": "node",
            "driver_version": "580.12",
            "cuda_version": "13.0",
            "gpus": [{"name": label}],
        },
        package_versions={"torch": "2.9.0"},
        runtime_versions={
            "cuda_toolkit": "13.0",
            "nccl_package": "2.27.3",
        },
    )


def baseline(bus_bw):
    spec = NCCLBaselineSpec(gpus=1)
    return NCCLCommunicationBaseline.create(
        spec,
        [
            NCCLResult(
                operation="all_reduce",
                size_bytes=8_388_608,
                time_us=100.0,
                algbw_gbps=bus_bw + 10,
                busbw_gbps=bus_bw,
            )
        ],
    )


def spec(group):
    return CampaignSpec(
        name=group,
        group=group,
        repeats=2,
        available_gpus=1,
        candidates=(ParallelismConfig(),),
        objective=OptimizationObjective.THROUGHPUT,
        constraints=OptimizationConstraints(),
        adapter="generic",
        benchmark_command=("python", "bench.py"),
        nccl_baseline=NCCLBaselineSpec(gpus=1),
    )


def test_campaign_round_trip_carries_communication_baseline(tmp_path):
    optimizations = OptimizationStore(tmp_path / "optimizations")
    studies = StudyStore(tmp_path / "studies")
    communication = baseline(70.0)

    report = run_campaign(
        spec=spec("h100"),
        provenance=provenance("H100"),
        runner_factory=lambda replicate, env: Runner(100.0),
        optimization_store=optimizations,
        study_store=studies,
        communication_baseline=communication,
    )

    payload = report.to_dict()
    loaded = type(report).from_dict(payload)

    assert loaded.communication_baseline == communication
    assert loaded.spec.nccl_baseline == NCCLBaselineSpec(gpus=1)
    assert loaded.provenance.runtime_versions["nccl_package"] == "2.27.3"


def test_campaign_comparison_adds_nccl_bandwidth_delta(tmp_path):
    optimizations = OptimizationStore(tmp_path / "optimizations")
    studies = StudyStore(tmp_path / "studies")

    h100 = run_campaign(
        spec=spec("h100"),
        provenance=provenance("H100"),
        runner_factory=lambda replicate, env: Runner(100.0),
        optimization_store=optimizations,
        study_store=studies,
        communication_baseline=baseline(70.0),
    )
    h200 = run_campaign(
        spec=spec("h200"),
        provenance=provenance("H200"),
        runner_factory=lambda replicate, env: Runner(120.0),
        optimization_store=optimizations,
        study_store=studies,
        communication_baseline=baseline(84.0),
    )

    study = compare_campaigns(
        [h100, h200],
        optimization_store=optimizations,
        study_store=studies,
    )

    assert any(
        "NCCL mean bus bandwidth h200 vs h100: +20.00%" in finding
        for finding in study.findings
    )
