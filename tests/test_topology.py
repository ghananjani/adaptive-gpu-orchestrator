from adaptive_gpu_orchestrator.discovery.topology import parse_topology_matrix


def test_parse_topology_matrix_unique_edges():
    text = """
            GPU0    GPU1    GPU2    CPU Affinity
    GPU0     X      NV4     SYS     0-31
    GPU1    NV4      X      PHB     0-31
    GPU2    SYS     PHB      X      32-63
    """
    links = parse_topology_matrix(text)
    assert [(x.source, x.target, x.kind) for x in links] == [
        (0, 1, "NV4"),
        (0, 2, "SYS"),
        (1, 2, "PHB"),
    ]
    assert links[0].weight > links[1].weight


def test_empty_topology():
    assert parse_topology_matrix("") == []
