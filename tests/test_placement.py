from adaptive_gpu_orchestrator.models import TopologyLink
from adaptive_gpu_orchestrator.optimizer import (
    CommunicationEdge,
    optimize_placement,
    placement_cost,
)


def test_places_chatty_ranks_on_fast_link():
    topology = [
        TopologyLink(0, 1, "NV4", 40, "NV4"),
        TopologyLink(0, 2, "SYS", 1, "SYS"),
        TopologyLink(0, 3, "SYS", 1, "SYS"),
        TopologyLink(1, 2, "SYS", 1, "SYS"),
        TopologyLink(1, 3, "SYS", 1, "SYS"),
        TopologyLink(2, 3, "NV4", 40, "NV4"),
    ]
    comm = [
        CommunicationEdge(0, 1, 100.0),
        CommunicationEdge(2, 3, 10.0),
    ]
    result = optimize_placement([0, 1, 2, 3], [0, 1, 2, 3], comm, topology)
    pair = {result.rank_to_gpu[0], result.rank_to_gpu[1]}
    assert pair in ({0, 1}, {2, 3})
    assert result.method == "exhaustive"


def test_placement_cost_penalizes_slow_links():
    topology = [
        TopologyLink(0, 1, "NV4", 40, "NV4"),
        TopologyLink(0, 2, "SYS", 1, "SYS"),
        TopologyLink(1, 2, "SYS", 1, "SYS"),
    ]
    comm = [CommunicationEdge(0, 1, 100.0)]
    fast = placement_cost({0: 0, 1: 1}, comm, topology)
    slow = placement_cost({0: 0, 1: 2}, comm, topology)
    assert fast < slow
