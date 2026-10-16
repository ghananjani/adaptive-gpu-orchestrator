import json

import pytest

from adaptive_gpu_orchestrator.campaigns import (
    CampaignManifest,
    CampaignSpec,
    load_campaign_manifest,
    save_campaign_manifest,
)
from adaptive_gpu_orchestrator.optimizer import (
    OptimizationConstraints,
    OptimizationObjective,
    ParallelismConfig,
)


def spec():
    return CampaignSpec(
        name="manifest-test",
        group="h100",
        repeats=3,
        available_gpus=2,
        candidates=(
            ParallelismConfig(tensor_parallel=1),
            ParallelismConfig(tensor_parallel=2),
        ),
        objective=OptimizationObjective.THROUGHPUT,
        constraints=OptimizationConstraints(),
        adapter="generic",
        benchmark_command=("python", "bench.py"),
        randomize_candidates=True,
        random_seed=42,
        tags={"hardware": "h100"},
    )


def test_campaign_manifest_round_trip_and_fingerprint(tmp_path):
    target = tmp_path / "campaign.json"
    manifest = save_campaign_manifest(spec(), target)
    loaded = load_campaign_manifest(target)

    assert manifest.sha256 == loaded.sha256
    assert loaded.verify() is True
    assert loaded.spec == spec()
    assert len(loaded.sha256) == 64


def test_campaign_manifest_detects_tampering(tmp_path):
    target = tmp_path / "campaign.json"
    save_campaign_manifest(spec(), target)

    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["spec"]["repeats"] = 99
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        load_campaign_manifest(target)


def test_campaign_manifest_hash_is_deterministic():
    left = CampaignManifest.create(spec())
    right = CampaignManifest.create(spec())

    assert left.sha256 == right.sha256


def test_randomized_campaign_requires_recorded_seed():
    with pytest.raises(ValueError, match="random_seed"):
        CampaignSpec(
            name="missing-seed",
            group="h100",
            repeats=1,
            available_gpus=1,
            candidates=(ParallelismConfig(),),
            objective=OptimizationObjective.THROUGHPUT,
            constraints=OptimizationConstraints(),
            adapter="generic",
            benchmark_command=("python", "bench.py"),
            randomize_candidates=True,
        )
