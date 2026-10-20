# Research experiment design audit (v1.7)

Run `audit_campaign_design(campaigns, optimization_store=store)` before
interpreting cross-hardware results from `compare_saved_campaigns`.

The audit reads the actual persisted campaign trial rows, and reports:
- Campaign counts per hardware platform (rather than misleading trial counts)
- Replicate counts within each campaign
- Whether each hardware platform is represented by only one campaign
- Whether there are at least two campaigns for each platform
- Reused optimization identifiers across different campaign IDs
- Missing recorded CUDA/NCCL/container software-provenance fields

A platform represented by a single campaign is **confounded** with campaign
identity. In that case, a measured change could arise from hardware, time,
cluster contention, environment or protocol, not the GPU model itself.
Repeated benchmark trials inside the same campaign do not constitute
independent campaign replications.

This is a descriptive design audit and does not prove causal identification.
A green replication flag merely confirms campaign counts, not independence
of machines, prompts, random seeds or temporal effects. Report all warnings
alongside any hardware transfer / model comparison errors.

No additional GPU jobs, container changes or deployments occur.
