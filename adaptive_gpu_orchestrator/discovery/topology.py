from __future__ import annotations

import re

from adaptive_gpu_orchestrator.models import TopologyLink

_LINK_WEIGHTS = {"PIX": 5, "PXB": 4, "PHB": 3, "NODE": 2, "SYS": 1}


def _link_weight(raw: str) -> int:
    match = re.fullmatch(r"NV(\d+)", raw)
    if match:
        return 10 * int(match.group(1))
    return _LINK_WEIGHTS.get(raw, 0)


def _gpu_index(token: str) -> int | None:
    match = re.fullmatch(r"GPU(\d+)", token.strip())
    return int(match.group(1)) if match else None


def parse_topology_matrix(text: str) -> list[TopologyLink]:
    """Parse ``nvidia-smi topo -m`` output into unique GPU-to-GPU links."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    header_tokens = re.split(r"\s+", lines[0].strip())
    gpu_columns: list[tuple[int, int]] = []
    for col, token in enumerate(header_tokens):
        idx = _gpu_index(token)
        if idx is not None:
            gpu_columns.append((col, idx))

    links: list[TopologyLink] = []
    seen: set[tuple[int, int]] = set()
    for line in lines[1:]:
        tokens = re.split(r"\s+", line.strip())
        if not tokens:
            continue
        source = _gpu_index(tokens[0])
        if source is None:
            continue

        for header_col, target in gpu_columns:
            value_col = header_col + 1
            if value_col >= len(tokens) or source == target:
                continue
            raw = tokens[value_col]
            if raw in {"X", "N/A", "-"}:
                continue
            edge = tuple(sorted((source, target)))
            if edge in seen:
                continue
            seen.add(edge)
            links.append(
                TopologyLink(
                    source=edge[0],
                    target=edge[1],
                    kind=raw,
                    weight=_link_weight(raw),
                    raw=raw,
                )
            )
    return links
