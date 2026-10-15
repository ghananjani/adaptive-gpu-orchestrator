from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import inf

from adaptive_gpu_orchestrator.models import TopologyLink


@dataclass(frozen=True, slots=True)
class CommunicationEdge:
    """Expected communication volume between two logical ranks."""

    source_rank: int
    target_rank: int
    volume: float


@dataclass(frozen=True, slots=True)
class PlacementResult:
    rank_to_gpu: dict[int, int]
    cost: float
    method: str


def _link_costs(links: list[TopologyLink]) -> dict[tuple[int, int], float]:
    costs: dict[tuple[int, int], float] = {}
    for link in links:
        edge = tuple(sorted((link.source, link.target)))
        costs[edge] = 1.0 / link.weight if link.weight > 0 else 1000.0
    return costs


def placement_cost(
    rank_to_gpu: dict[int, int],
    communication: list[CommunicationEdge],
    topology: list[TopologyLink],
) -> float:
    """Calculate communication-weighted placement cost (lower is better)."""
    link_costs = _link_costs(topology)
    total = 0.0
    for edge in communication:
        if edge.source_rank not in rank_to_gpu or edge.target_rank not in rank_to_gpu:
            raise ValueError("communication graph references an unmapped rank")
        a = rank_to_gpu[edge.source_rank]
        b = rank_to_gpu[edge.target_rank]
        if a == b:
            continue
        physical = tuple(sorted((a, b)))
        total += edge.volume * link_costs.get(physical, 1000.0)
    return total


def _greedy(
    ranks: list[int],
    gpus: list[int],
    communication: list[CommunicationEdge],
    topology: list[TopologyLink],
) -> PlacementResult:
    if not ranks:
        return PlacementResult({}, 0.0, "greedy")

    volumes = {rank: 0.0 for rank in ranks}
    for edge in communication:
        volumes[edge.source_rank] = volumes.get(edge.source_rank, 0.0) + edge.volume
        volumes[edge.target_rank] = volumes.get(edge.target_rank, 0.0) + edge.volume
    ordered = sorted(ranks, key=lambda r: volumes.get(r, 0.0), reverse=True)

    strengths = {gpu: 0 for gpu in gpus}
    for link in topology:
        if link.source in strengths:
            strengths[link.source] += link.weight
        if link.target in strengths:
            strengths[link.target] += link.weight

    mapping: dict[int, int] = {ordered[0]: max(gpus, key=lambda g: strengths.get(g, 0))}
    free = set(gpus) - set(mapping.values())

    for rank in ordered[1:]:
        best_gpu = None
        best_cost = inf
        for gpu in sorted(free):
            trial = dict(mapping)
            trial[rank] = gpu
            relevant = [
                edge
                for edge in communication
                if edge.source_rank in trial and edge.target_rank in trial
            ]
            cost = placement_cost(trial, relevant, topology)
            if cost < best_cost:
                best_gpu, best_cost = gpu, cost
        if best_gpu is None:
            raise ValueError("not enough GPUs for ranks")
        mapping[rank] = best_gpu
        free.remove(best_gpu)

    return PlacementResult(mapping, placement_cost(mapping, communication, topology), "greedy")


def optimize_placement(
    ranks: list[int],
    gpus: list[int],
    communication: list[CommunicationEdge],
    topology: list[TopologyLink],
    *,
    exhaustive_limit: int = 8,
) -> PlacementResult:
    """Find a topology-aware rank placement."""
    if len(ranks) > len(gpus):
        raise ValueError("there must be at least as many GPUs as ranks")
    if len(set(ranks)) != len(ranks) or len(set(gpus)) != len(gpus):
        raise ValueError("ranks and GPUs must be unique")
    if len(ranks) <= exhaustive_limit:
        best_mapping: dict[int, int] | None = None
        best_cost = inf
        for gpu_perm in permutations(gpus, len(ranks)):
            mapping = dict(zip(ranks, gpu_perm, strict=True))
            cost = placement_cost(mapping, communication, topology)
            if cost < best_cost:
                best_mapping, best_cost = mapping, cost
        return PlacementResult(best_mapping or {}, best_cost if best_mapping else 0.0, "exhaustive")
    return _greedy(ranks, gpus, communication, topology)
