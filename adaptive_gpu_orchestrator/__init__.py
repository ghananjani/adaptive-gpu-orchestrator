"""Adaptive GPU Orchestrator for AI Workloads."""

from .adapters import (
    AdaptedCommandCanaryRunner,
    PyTorchDistributedMetricsAdapter,
    VLLMMetricsAdapter,
)
from .controller import DeploymentPlan, DeploymentPlanner, DeploymentPolicy, SchedulerBackend
from .correlation import (
    CorrelationReport,
    CorrelationStore,
    correlate_optimization,
    summarize_nccl,
)
from .doctor import Doctor
from .models import ClusterSnapshot, DiagnosisReport, GPUDevice, TopologyLink
from .optimizer import (
    ConfigurationOptimizer,
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationStore,
    ParallelismConfig,
)
from .profiling import ExperimentStore, WorkloadProfile, WorkloadProfiler
from .studies import (
    MetricComparison,
    MetricSummary,
    ParetoPoint,
    StudyGroupSummary,
    StudyReport,
    StudyStore,
    analyze_study,
)
from .telemetry import (
    DCGMDeviceSample,
    DCGMFrame,
    DeviceTelemetry,
    TelemetrySample,
    TelemetrySampler,
    parse_dcgm_dmon,
    parse_dcgm_exporter,
    summarize_dcgm,
)
from .timeseries import (
    GPUTimeSeriesSummary,
    PhaseSummary,
    PhaseWindow,
    TimeSeriesReport,
    TimeSeriesStore,
    analyze_timeseries,
)
from .validation import (
    PlanDiff,
    ValidationDecision,
    ValidationPolicy,
    ValidationReport,
    ValidationStore,
    diff_plans,
    validate_trials,
)

__all__ = [
    "AdaptedCommandCanaryRunner",
    "ClusterSnapshot",
    "ConfigurationOptimizer",
    "CorrelationReport",
    "CorrelationStore",
    "DCGMDeviceSample",
    "DCGMFrame",
    "DeploymentPlan",
    "DeploymentPlanner",
    "DeploymentPolicy",
    "DeviceTelemetry",
    "DiagnosisReport",
    "Doctor",
    "ExperimentStore",
    "GPUDevice",
    "GPUTimeSeriesSummary",
    "MetricComparison",
    "MetricSummary",
    "OptimizationConstraints",
    "OptimizationObjective",
    "OptimizationStore",
    "ParallelismConfig",
    "ParetoPoint",
    "PhaseSummary",
    "PhaseWindow",
    "PlanDiff",
    "PyTorchDistributedMetricsAdapter",
    "SchedulerBackend",
    "StudyGroupSummary",
    "StudyReport",
    "StudyStore",
    "TelemetrySample",
    "TelemetrySampler",
    "TimeSeriesReport",
    "TimeSeriesStore",
    "TopologyLink",
    "VLLMMetricsAdapter",
    "ValidationDecision",
    "ValidationPolicy",
    "ValidationReport",
    "ValidationStore",
    "WorkloadProfile",
    "WorkloadProfiler",
    "analyze_study",
    "analyze_timeseries",
    "correlate_optimization",
    "diff_plans",
    "parse_dcgm_dmon",
    "parse_dcgm_exporter",
    "summarize_dcgm",
    "summarize_nccl",
    "validate_trials",
]
__version__ = "0.9.0a0"
