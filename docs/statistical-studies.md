# Statistical Experiment Studies

v0.9 adds repeated-run analysis so optimization conclusions do not depend on a single canary execution.

## Inputs

A study is built from named groups of saved optimization IDs.

Example:

```bash
agoctl study analyze \
  --group h100=h100-1,h100-2,h100-3 \
  --group h200=h200-1,h200-2,h200-3
```

The first group is the baseline for pairwise comparisons.

## Group summaries

For every metric exposed by each group's selected trials, the study records:

- sample count
- mean
- minimum and maximum
- sample standard deviation
- two-sided 95% confidence interval

For two to 31 samples, the interval uses the Student-t critical value for the available degrees of freedom. Larger samples use the normal approximation.

A single replicate is still represented, but standard deviation and confidence bounds are unavailable.

## Adapter-specific metrics

Numeric values in `TrialMetrics.extras` are included using the prefix:

```text
extras.<metric-name>
```

This allows repeated-run analysis of vLLM TTFT/TPOT/ITL or distributed-training communication metrics without changing the study schema.

## Configuration stability

Each group records how often the optimizer selected each parallelism configuration.

The dominant-config stability fraction is:

```text
dominant config selections / total runs
```

Low stability is useful evidence that the search result may be sensitive to noise, workload variance or a shallow performance surface.

## Baseline comparisons

Every candidate group is compared to the first group using metrics available in both groups.

The report records:

- baseline and candidate means
- absolute change
- percent change
- direction-aware relative improvement
- Cohen's d when both groups have at least two replicates with non-zero pooled variance

Latency, energy, memory pressure and duration are treated as lower-is-better. Throughput and other metrics are treated as higher-is-better unless future metric metadata says otherwise.

## Pareto frontier

When each selected run exposes:

- throughput
- p99 latency
- energy

v0.9 calculates the non-dominated frontier.

A run is dominated when another run has:

- equal or higher throughput
- equal or lower p99 latency
- equal or lower energy
- and is strictly better in at least one dimension

This keeps multi-objective tradeoffs visible instead of collapsing every run into a single scalar score.

## Interpretation

Confidence intervals and Cohen's d are descriptive evidence, not automatic significance claims.

With fewer than three replicates, the report explicitly warns that intervals and effect sizes are exploratory.

For rigorous publication-quality conclusions, repeat experiments under controlled conditions, record hardware/software versions, randomize run order where practical, and validate assumptions before applying formal hypothesis tests.

## Persistence

Reports are stored under:

```text
.ago/studies/
```

Inspect them with:

```bash
agoctl studies list
agoctl studies show <study-id>
```

## Research extensions

- bootstrap confidence intervals
- robust statistics for non-normal latency tails
- paired-run analysis
- effect-size confidence intervals
- automatic hardware/software provenance
- study manifests for H100/H200 reproduction
- repeated Pareto stability analysis
