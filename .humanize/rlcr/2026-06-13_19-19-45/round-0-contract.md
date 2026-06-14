# Round 0 Contract

**Objective:** Implement and validate the holographic length-generalization plan
(docs/superpowers/plans/2026-06-13-holographic-length-gen.md), then launch the
Phase A–D GPU campaign on the emptiest available partition.

**In-scope this round:**
- All code (Tasks 1–10): RoPE model, nested_monoid generator, masked answer-eval,
  configurable floor, plan factories, knob-verification gate, configs.
- Local verification: full unit-test suite + §3.4 knob gate + GPU learnability probe.
- Launch the experimental campaign (Phases A–D) as cluster availability allows.

**Definition of done (Round 0):**
- 21+ unit tests pass; knob gate PASS; learnability confirmed on GPU (train acc ≫ chance).
- Campaign jobs submitted/running; results analyzed into results/*.md + reports/ as
  they land; H1/H2/H3 verdicts recorded.

**Constraints / known blockers:**
- kempner partitions QOSMaxGRESPerUser-capped; user has ~220 queued cron jobs →
  QOSMaxSubmitJobPerUserLimit. Route to gpu_test (idle, separate QoS) and drain
  submissions as slots free.
- GPU runs take hours; results/analysis (Tasks 11–14) complete across later rounds.
