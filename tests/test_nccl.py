from adaptive_gpu_orchestrator.benchmarks.nccl import parse_nccl_output


def test_parse_nccl_output():
    text = """
# nThread 1 nGpus 8 minBytes 8388608 maxBytes 1073741824 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#       size         count      type   redop    root     time   algbw   busbw #wrong
     8388608       2097152     float     sum      -1    40.12   209.1   365.9      0
    16777216       4194304     float     sum      -1    45.50   368.7   645.2      0
    """
    rows = parse_nccl_output(text)
    assert len(rows) == 2
    assert rows[0].size_bytes == 8388608
    assert rows[1].busbw_gbps == 645.2
