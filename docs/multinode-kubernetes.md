# Multi-node Kubernetes Planning

v1.2 adds a recommend-only multi-node Kubernetes renderer based on JobSet.

The implementation follows the current JobSet `jobset.x-k8s.io/v1alpha2` API and Kueue's `kueue.x-k8s.io/queue-name` workload label.

## Deployment shape

A JobSet plan is defined by:

- optimizer/configuration world size
- node count
- GPUs per node

The renderer requires:

```text
nodes × GPUs per node = configuration world size
```

This makes GPU accounting explicit and prevents silently rendering a distributed workload with a different resource shape than the optimizer evaluated.

## Worker model

The JobSet contains one ReplicatedJob named `workers`.

Its `replicas` field equals the requested node count.

Each replica creates one Kubernetes Job containing one worker Pod. Each worker Pod requests and limits exactly `gpus_per_node` NVIDIA GPUs.

This design treats one JobSet worker replica as one physical scheduling unit/node for GPU resource accounting.

## Distributed environment

The renderer retains the standard AGO configuration environment from the deployment plan and adds:

```text
AGO_NODE_COUNT
AGO_GPUS_PER_NODE
AGO_NODE_RANK
AGO_MASTER_ADDR
MASTER_ADDR
MASTER_PORT
```

`AGO_NODE_RANK` is populated through the Kubernetes downward API from:

```text
metadata.labels['jobset.sigs.k8s.io/job-index']
```

The coordinator address follows the JobSet DNS convention used by upstream distributed-training examples:

```text
<jobset>-workers-0-0.<jobset>
```

The default master port is 29500 and can be changed with `--master-port`.

## Kueue

When `--queue-name` is provided, the renderer adds:

```yaml
metadata:
  labels:
    kueue.x-k8s.io/queue-name: <local-queue>
```

The same queue option is supported by the existing single-pod Kubernetes Job renderer.

The renderer does not create LocalQueue or ClusterQueue resources; those remain cluster-administrator concerns.

## Node selection

Any deployment-plan node selector is copied into every worker Pod.

Example:

```bash
--node-selector accelerator=h100
```

This remains a recommendation artifact; it does not verify that matching nodes exist.

## Service account

Use:

```bash
--service-account-name trainer
```

to set `serviceAccountName` on each worker Pod.

No service account is invented by default.

## CLI

Two-node, four-GPU-per-node example:

```bash
agoctl plan \
  --backend kubernetes \
  --available-gpus 8 \
  --tp 8 \
  --image registry.example/train:latest \
  --kubernetes-mode jobset \
  --nodes 2 \
  --gpus-per-node 4 \
  --queue-name gpu-queue \
  --master-port 29500 \
  --node-selector accelerator=h100 \
  -- python train.py
```

If `--gpus-per-node` is omitted, the CLI derives it only when world size is evenly divisible by node count.

## Single-pod compatibility

The existing behavior remains the default:

```bash
agoctl plan --backend kubernetes ...
```

This renders a standard `batch/v1 Job`.

Passing `--nodes` greater than one in single-Job mode is rejected with guidance to use `--kubernetes-mode jobset`.

## Safety boundary

v1.2 only renders manifests.

It does not:

- call `kubectl`
- create JobSet CRDs
- create Kueue resources
- submit workloads
- suspend or resume workloads
- mutate scheduler state

The recommend-only controller and v1.0 autopilot safety model remain unchanged.

## Future extensions

- Kueue topology-aware scheduling annotations
- rack/block topology constraints
- MultiKueue targeting
- JobSet failure policy generation
- elastic worker counts
- launcher-aware torchrun command synthesis
- network-fabric selectors
- PVC/model-cache templates
