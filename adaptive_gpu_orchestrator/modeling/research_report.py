"""Offline, auditable research summaries from validated saved refitting results."""
from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any

from .saved_refitting import SavedRefittingEvaluation


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def research_report_payload(result: SavedRefittingEvaluation) -> dict[str, Any]:
    """Return a stable summary with all caveats and original source provenance."""
    original = result.to_dict()
    audit = result.design_audit
    comparison = result.comparison
    refit = result.refitting
    return {
        "schema_version": 1,
        "kind": "gpu-hardware-research-report",
        "source": original,
        "summary": {
            "metric": comparison.evaluation.metric,
            "campaign_count": audit.campaign_count,
            "platform_count": audit.platform_count,
            "trial_count": comparison.trial_count,
            "excluded_failed": comparison.excluded_failed,
            "excluded_missing": comparison.excluded_missing,
            "baseline_pooled_rmse": comparison.evaluation.config_pooled_rmse,
            "conditioned_pooled_rmse": comparison.evaluation.conditioned_pooled_rmse,
            "rmse_reduction_pct": comparison.evaluation.rmse_reduction_pct,
            "refit_lower_pct": refit.lower_pct,
            "refit_upper_pct": refit.upper_pct,
            "confidence": refit.confidence,
            "valid_iterations": refit.valid_iterations,
            "requested_iterations": refit.iterations,
        },
        "warnings": list(dict.fromkeys((
            *audit.warnings,
            *comparison.warnings,
            *refit.findings,
            "Bootstrap bounds are not calibrated for unseen GPU platforms.",
            "Hardware feature measurements and source campaign authenticity require independent verification.",
        ))),
    }


def render_research_markdown(payload: dict[str, Any]) -> str:
    """Render text from validated report payload without calculating new evidence."""
    s = payload["summary"]
    src = payload["source"]
    fold_rows = src["comparison"]["evaluation"]["folds"]
    lines = [
        "# GPU Hardware Transfer — Research Validation Report",
        "",
        "## Executive summary",
        "",
        f"- Metric: `{s['metric']}`",
        f"- Evidence: {s['campaign_count']} campaigns, {s['platform_count']} hardware platforms, {s['trial_count']} included trials",
        f"- Excluded: {s['excluded_failed']} failed, {s['excluded_missing']} missing/nonfinite metric values",
        f"- Baseline pooled RMSE: {_fmt(s['baseline_pooled_rmse'])}",
        f"- Hardware-conditioned pooled RMSE: {_fmt(s['conditioned_pooled_rmse'])}",
        f"- Signed RMSE reduction: {_fmt(s['rmse_reduction_pct'])}%",
        f"- Refitting sensitivity percentile interval ({s['confidence']:.1%}): {_fmt(s['refit_lower_pct'])}% to {_fmt(s['refit_upper_pct'])}%",
        f"- Valid bootstrap draws: {s['valid_iterations']}/{s['requested_iterations']}",
        "",
        "## Held-out platform folds",
        "",
        "| Platform | Train rows | Test rows | Config RMSE | Conditioned RMSE | Reduction % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fold in fold_rows:
        platform_label = str(fold["held_out_platform"]).replace("|", "&#124;").replace("\\n", " ")
        lines.append(
            f"| {platform_label} | {fold['train_samples']} | "
            f"{fold['test_samples']} | {_fmt(fold['config_rmse'])} | "
            f"{_fmt(fold['conditioned_rmse'])} | {_fmt(fold['rmse_reduction_pct'])} |"
        )
    lines.extend([
        "",
        "## Reproducibility",
        "",
        f"- Source evidence SHA-256: `{src['comparison']['source_sha256']}`",
        f"- Bootstrap seed: `{src['refitting']['seed']}`",
        f"- Platforms with independent campaign replication: `{src['design_audit']['independent_campaign_replication']}`",
        "",
        "## Interpretation and limitations",
        "",
    ])
    lines.extend(f"- {warning}" for warning in payload["warnings"])
    lines.extend(["", "## Full structured evidence", "", "Refer to the accompanying JSON export for campaign IDs, hardware descriptors, software versions and complete audit results.", ""])
    return "\n".join(lines)


def render_research_html(payload: dict[str, Any]) -> str:
    """Self-contained, escaped HTML for offline stakeholders and printing."""
    s = payload["summary"]
    rows = []
    for fold in payload["source"]["comparison"]["evaluation"]["folds"]:
        cells = (
            fold["held_out_platform"], fold["train_samples"], fold["test_samples"],
            _fmt(fold["config_rmse"]), _fmt(fold["conditioned_rmse"]),
            _fmt(fold["rmse_reduction_pct"]),
        )
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells) + "</tr>")
    warnings = "".join(f"<li>{html.escape(str(w))}</li>" for w in payload["warnings"])
    tags = payload["source"]["comparison"]["shared_tags"]
    provenance = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in sorted(tags.items())
    )
    def card(label: str, value: str) -> str:
        return f"<section class='metric'><small>{html.escape(label)}</small><strong>{html.escape(value)}</strong></section>"
    cards = "".join((
        card("Evidence", f"{s['campaign_count']} campaigns / {s['platform_count']} platforms"),
        card("Included trials", str(s["trial_count"])),
        card("Baseline RMSE", _fmt(s["baseline_pooled_rmse"])),
        card("Conditioned RMSE", _fmt(s["conditioned_pooled_rmse"])),
        card("Signed RMSE reduction", _fmt(s["rmse_reduction_pct"]) + "%"),
        card("Refitting percentile bounds", _fmt(s["refit_lower_pct"]) + "% to " + _fmt(s["refit_upper_pct"]) + "%"),
    ))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPU Hardware Transfer — Research Validation</title>
<style>
:root{{color-scheme:dark;font-family:system-ui,-apple-system,sans-serif;background:#10151d;color:#edf1f6}}
body{{margin:0 auto;max-width:1100px;padding:36px 22px;line-height:1.55}}
h1{{font-size:2rem;margin-bottom:0}}h2{{margin-top:40px}}
p,small{{color:#aab6c6}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}}
.metric{{padding:18px;border:1px solid #394556;border-radius:12px;background:#1a2430}}
.metric small,.metric strong{{display:block}}.metric strong{{font-size:1.35rem;margin-top:7px}}
table{{border-collapse:collapse;width:100%;display:block;overflow-x:auto}}
th,td{{padding:10px 14px;border-bottom:1px solid #394556;text-align:left;white-space:nowrap}}
th{{color:#c3d7f7}}li{{margin:8px 0}}code{{overflow-wrap:anywhere}}footer{{margin-top:38px;font-size:.9rem}}
@media print{{:root{{color-scheme:light;background:white;color:#17212d}}body{{max-width:none}}
.metric{{background:#f5f7fa;border-color:#bbb}}p,small{{color:#444}}th,td{{border-color:#bbb}}}}
</style></head><body>
<header><h1>GPU Hardware Transfer — Research Validation</h1>
<p>Offline evaluation of configuration-only vs hardware-conditioned predictions. No measurements are generated by this report.</p></header>
<main><h2>Summary</h2><div class="grid">{cards}</div>
<p>Metric: <strong>{html.escape(str(s['metric']))}</strong> ·
Confidence setting: {s['confidence']:.1%} ·
Valid bootstrap draws: {s['valid_iterations']}/{s['requested_iterations']}</p>
<h2>Held-out platform comparisons</h2>
<table><thead><tr><th>Platform</th><th>Train rows</th><th>Test rows</th>
<th>Config RMSE</th><th>Conditioned RMSE</th><th>Reduction %</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Protocol tags</h2><table><tbody>{provenance}</tbody></table>
<h2>Evidence and limitations</h2><ul>{warnings}</ul>
<p>Excluded observations: {s['excluded_failed']} failed; {s['excluded_missing']} missing/nonfinite metric.</p>
<p>Source included-evidence SHA-256: <code>{html.escape(str(payload['source']['comparison']['source_sha256']))}</code></p>
</main><footer>Research-only. Conditional bootstrap sensitivity is not calibrated new-platform uncertainty. For complete provenance, consult the accompanying JSON.</footer>
</body></html>
"""


def write_research_report(result: SavedRefittingEvaluation, output_dir: str | Path) -> dict[str, Path]:
    """Write deterministic JSON, Markdown, HTML and fold CSV offline."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = research_report_payload(result)
    canonical = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    paths = {
        "json": output / "research-report.json",
        "markdown": output / "research-report.md",
        "html": output / "research-report.html",
        "folds_csv": output / "heldout-folds.csv",
        "sha256": output / "research-report.sha256",
    }
    paths["json"].write_text(canonical, encoding="utf-8")
    paths["markdown"].write_text(render_research_markdown(payload), encoding="utf-8")
    paths["html"].write_text(render_research_html(payload), encoding="utf-8")
    columns = ("held_out_platform", "train_samples", "test_samples", "config_rmse",
               "conditioned_rmse", "config_mae", "conditioned_mae", "rmse_reduction_pct")
    with paths["folds_csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result.comparison.evaluation.folds[i].to_dict()
                         for i in range(len(result.comparison.evaluation.folds)))
    paths["sha256"].write_text(f"{digest}  research-report.json\n", encoding="utf-8")
    return paths
