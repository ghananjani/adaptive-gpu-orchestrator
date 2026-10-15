# Workload Adapters

v0.6 lets the optimizer consume native workload summaries instead of requiring every benchmark to emit the generic `TrialMetrics` schema.

## Adapter selection

```bash
agoctl optimize --adapter generic ...
agoctl optimize --adapter vllm ...
agoctl optimize --adapter pytorch-distributed ...
```

The benchmark command still runs once per candidate and receives the existing `AGO_*` environment variables plus:

```text
AGO_METRICS_ADAPTER
```

The final JSON object written to stdout is parsed by the selected adapter.

## vLLM adapter

The adapter recognizes common serving metrics including:

- request throughput
- output-token throughput
- total-token throughput
- TTFT
- TPOT
- ITL
- end-to-end latency
- token counts
- optional GPU utilization, memory pressure and energy

### Normalized fields

`throughput_per_s` uses this preference order:

1. `total_token_throughput`
2. `output_throughput`
3. `request_throughput`
4. already-normalized `throughput_per_s`

`p99_latency_ms` uses:

1. `p99_e2el_ms`
2. already-normalized `p99_latency_ms`
3. `p99_ttft_ms`

This means a latency objective becomes end-to-end p99 when the benchmark provides it and otherwise behaves as a TTFT objective.

TTFT, TPOT, ITL and token-throughput details are also preserved under `TrialMetrics.extras`.

## PyTorch distributed adapter

The adapter is intended for a lightweight JSON summary produced by a distributed training harness.

Supported throughput aliases include:

- `throughput_per_s`
- `tokens_per_s`
- `samples_per_s`
- `sequences_per_s`

Supported p99 step-latency aliases include:

- `p99_latency_ms`
- `p99_step_time_ms`
- `step_time_p99_ms`

Useful phase fields include:

- compute time
- communication time
- all-reduce time
- forward time
- backward time
- optimizer time
- data-loader time
- gradient synchronization time

Nested numeric values under a `phases` object are preserved as `phases.<name>`.

When mean step time is available, the adapter also derives communication and compute fractions.

## Persistence

Adapter-specific numeric metrics are stored in:

```python
TrialMetrics.extras
```

They round-trip through saved optimization results, so later analysis can compare TTFT/ITL or communication fractions even though the optimizer currently scores only the standard normalized fields.

## Design boundary

Adapters normalize metrics only. They do not:

- start vLLM servers
- launch distributed training frameworks
- modify scheduler state
- change optimizer search policy
- reinterpret promotion guardrails

This keeps workload integration separate from scheduling and decision logic.
