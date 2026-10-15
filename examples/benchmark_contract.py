"""Minimal benchmark contract for agoctl optimize.

A real benchmark would launch vLLM, a training step loop, or another workload,
then print measured metrics as one JSON object on stdout.
"""

import json
import os


tp = int(os.environ.get("AGO_TP", "1"))
pp = int(os.environ.get("AGO_PP", "1"))
dp = int(os.environ.get("AGO_DP", "1"))
batch = int(os.environ.get("AGO_BATCH_SIZE", "1"))
concurrency = int(os.environ.get("AGO_CONCURRENCY", "1"))

world_size = tp * pp * dp

print(
    json.dumps(
        {
            "throughput_per_s": batch * concurrency * world_size * 10.0,
            "p99_latency_ms": 250.0 / max(1, tp),
            "energy_joules": 1000.0 + 20.0 * world_size,
            "gpu_util_pct": min(99.0, 45.0 + 4.0 * batch),
            "memory_pressure_pct": min(99.0, 35.0 + 3.0 * batch),
        }
    )
)
