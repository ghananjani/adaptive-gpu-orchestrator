# Performance Modeling + Experiment Recommendation

v1.6 turns saved campaign trial history into an interpretable model that can rank the next configurations worth measuring.

## Scope

The first model is intentionally simple and auditable.

It uses a ridge-regularized linear model over:

- log2 tensor parallel size
- log2 pipeline parallel size
- log2 data parallel size
- log2 batch size
- log2 concurrency

This corresponds to a power-law-style scaling surface in the original configuration variables.

No external machine-learning dependency is required.

## Training data

Training consumes every successful optimizer trial from the selected saved campaigns, not only the selected winner.

This matters because a useful performance surface needs negative and intermediate measurements as well as the best configuration.

Default modeled metrics are:

```text
throughput_per_s
p99_latency_ms
energy_joules
memory_pressure_pct
```

Users may request exported adapter metrics explicitly, for example:

```bash
agoctl model train <campaign-id> --metric extra_ttft_ms
```

A custom metric can be modeled when enough numeric trial rows exist.

## Fit diagnostics

Every metric model stores:

- sample count
- distinct configuration count
- coefficients
- feature means and scales
- intercept
- in-sample RMSE
- in-sample MAE
- in-sample R² when target variance is non-zero
- leave-one-out CV RMSE when at least four samples exist
- leave-one-out CV MAE when at least four samples exist

The fit diagnostics are descriptive model-quality signals. They do not prove that the model generalizes outside the measured search space.

## Hardware/software provenance guard

The v1.6 feature vector does not encode GPU type, driver, CUDA, NCCL or container identity.

For that reason, training derives a provenance signature from:

- GPU models
- GPU count
- driver version
- CUDA runtime version
- CUDA toolkit version
- NCCL version
- container image digest

If selected campaign rows contain multiple signatures, training fails by default.

For H100-vs-H200 research, the recommended workflow is to train one H100 model and one H200 model rather than combine them.

An explicit override exists:

```bash
agoctl model train h100-campaign h200-campaign \
  --allow-mixed-hardware
```

The saved model records that multiple hardware signatures were present and warns that platform effects may be conflated.

## Train

```bash
agoctl model train \
  <campaign-id> \
  --campaign-store .ago/campaigns \
  --optimization-store .ago/optimizations \
  --metric throughput_per_s \
  --metric p99_latency_ms \
  --ridge-lambda 1.0
```

Models are saved under:

```text
.ago/models/<model-id>.json
```

Inspect them with:

```bash
agoctl model list
agoctl model show <model-id>
```

## Next-experiment recommendation

Generate a candidate search space and rank configurations without executing them:

```bash
agoctl model recommend <model-id> \
  --metric throughput_per_s \
  --gpus 8 \
  --tp 1,2,4,8 \
  --pp 1,2 \
  --dp 1 \
  --batch 1,2,4,8 \
  --concurrency 1,8 \
  --top-k 5
```

The command reuses the same GPU-budget-aware candidate generator as the optimizer.

Measured configurations are excluded by default.

Use `--include-observed` only when remeasurement is intentional.

## Acquisition score

Each candidate gets two normalized components.

### Exploitation

The model predicts the requested metric.

Known metric directions are:

```text
throughput_per_s      maximize
gpu_util_pct          maximize
p99_latency_ms        minimize
energy_joules         minimize
memory_pressure_pct   minimize
duration_s            minimize
```

For custom metrics, specify:

```bash
--direction maximize
```

or:

```bash
--direction minimize
```

### Exploration

Exploration is the distance from the closest measured configuration after applying the same standardized log2 feature transformation used by the model.

This is a **novelty heuristic**.

It is not:

- a confidence interval
- Bayesian uncertainty
- prediction variance
- a probability of improvement

### Combined ranking

The default score is:

```text
0.75 × exploitation + 0.25 × exploration
```

Change it with:

```bash
--exploration-weight 0.5
```

Set it to zero for pure model exploitation.

## Research workflow

A practical loop is:

```text
campaign trials
     ↓
train model
     ↓
rank unmeasured configs
     ↓
run top recommendation as a real benchmark
     ↓
append campaign evidence
     ↓
retrain
```

This gives the project a path toward active experiment selection without pretending that the initial model is an autonomous optimizer.

## Safety boundary

The modeling subsystem is offline and recommend-only.

It does not:

- run GPU benchmarks
- submit Kubernetes jobs
- submit Slurm jobs
- change GPU clocks or power
- change MIG configuration
- mutate campaign artifacts
- automatically execute recommended experiments

The operator chooses whether to run any recommended configuration.
