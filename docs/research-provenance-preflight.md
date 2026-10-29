# Research provenance preflight (v1.7)

The saved-campaign refitting workflow performs strict metadata checks **before**
fitting any model or writing a research report. It rejects absent or invalid
GPU counts, missing measured hardware signatures, blank CUDA runtime/toolkit,
NCCL or container-image digests, optimization IDs reused by different
campaigns, duplicate trial IDs, and campaigns mixing hardware signatures.

The preflight uses only persisted campaign and optimization evidence.
It cannot independently confirm that a reported GPU descriptor, software
version or benchmark protocol is authentic. Workload/protocol equivalence
remains enforced by the original required campaign-tag comparison. An
explicit software-mismatch override still permits differing **complete**
software versions across platforms for research; it cannot bypass missing
metadata. Existing scripts that previously analyzed incomplete provenance
must first supply the real missing values rather than invent defaults.

This is a fail-closed control for reporting; it does not execute GPU
benchmarks, restart containers or modify source campaign records.
