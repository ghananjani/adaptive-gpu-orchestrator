# Serialized campaign-to-research-report regression coverage

`tests/test_research_disk_e2e_v17.py` exercises the actual, unmocked
`OptimizationStore.save/load`, `CampaignStore.save/load`,
`campaign_trial_rows`, and `ago-research-report` main entrypoint in
a temporary directory. The fixture writes six independent campaigns
from three artificial GPU platforms with fully specified software and
workload provenance, then reads back the persisted JSON before running
the research pipeline.

The test verifies per-fold results, count reconciliation, all five
report files, report checksum and repeat-run byte determinism. A second
test modifies a serialized campaign workload tag and confirms that
report generation fails before output is created.

All hardware labels and performance values here are **synthetic**.
This checks serialization and API interoperability; it does not
establish simulator accuracy, GPU scaling or hardware specifications.
Neither test launches any GPU computation or modifies production
containers or cluster state.
