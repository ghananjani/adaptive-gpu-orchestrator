import json
import sys

from adaptive_gpu_orchestrator.cli import main


def benchmark_command(throughput):
    return [
        sys.executable,
        "-c",
        (
            "import json; "
            f"print(json.dumps({{'throughput_per_s': {throughput}, "
            "'p99_latency_ms': 100.0, 'energy_joules': 1000.0}}))"
        ),
    ]


def run_campaign_cli(tmp_path, capsys, *, name, group, throughput):
    campaigns = tmp_path / "campaigns"
    optimizations = tmp_path / "optimizations"
    studies = tmp_path / "studies"

    args = [
        "campaign",
        "run",
        "--name",
        name,
        "--group",
        group,
        "--repeats",
        "2",
        "--gpus",
        "1",
        "--tp",
        "1",
        "--pp",
        "1",
        "--dp",
        "1",
        "--batch",
        "1",
        "--concurrency",
        "1",
        "--optimization-store",
        str(optimizations),
        "--study-store",
        str(studies),
        "--store-dir",
        str(campaigns),
        "--tag",
        f"hardware={group}",
        "--",
        *benchmark_command(throughput),
    ]
    code = main(args)
    output = capsys.readouterr()

    assert code == 0
    assert f"group={group}" in output.out
    assert "Replicates: 2/2" in output.out

    files = sorted(campaigns.glob("*.json"))
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in files
    ]
    identifier = next(
        item["id"]
        for item in payloads
        if item["spec"]["group"] == group
    )
    return identifier, campaigns, optimizations, studies


def test_campaign_cli_run_list_show_and_compare(tmp_path, capsys):
    h100_id, campaigns, optimizations, studies = run_campaign_cli(
        tmp_path,
        capsys,
        name="h100 benchmark",
        group="h100",
        throughput=100.0,
    )
    h200_id, _, _, _ = run_campaign_cli(
        tmp_path,
        capsys,
        name="h200 benchmark",
        group="h200",
        throughput=125.0,
    )

    list_code = main(
        [
            "campaign",
            "list",
            "--store-dir",
            str(campaigns),
        ]
    )
    listed = capsys.readouterr()
    assert list_code == 0
    assert "group=h100" in listed.out
    assert "group=h200" in listed.out

    show_code = main(
        [
            "campaign",
            "show",
            h100_id,
            "--store-dir",
            str(campaigns),
        ]
    )
    shown = capsys.readouterr()
    assert show_code == 0
    assert f"Campaign: {h100_id}" in shown.out
    assert "hardware=h100" not in shown.err

    compare_code = main(
        [
            "--json",
            "campaign",
            "compare",
            h100_id,
            h200_id,
            "--store-dir",
            str(campaigns),
            "--optimization-store",
            str(optimizations),
            "--study-store",
            str(studies),
        ]
    )
    compared = capsys.readouterr()
    assert compare_code == 0
    payload = json.loads(compared.out)
    assert [group["name"] for group in payload["groups"]] == ["h100", "h200"]
    throughput = next(
        item
        for item in payload["comparisons"]
        if item["metric"] == "throughput_per_s"
    )
    assert throughput["candidate_group"] == "h200"
    assert throughput["relative_improvement_pct"] == 25.0
