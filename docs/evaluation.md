# Evaluation Plan

## Deterministic analytics

The automated test suite checks:

- policy-driven SLA and cost calculations;
- cross-midnight duration handling;
- temporal-anomaly flagging without source repair;
- weekly aggregation;
- duplicate reconciliation;
- Tier-1 leaderboard filtering;
- repeat-contact proxy logic;
- opportunity construction;
- evidence-backed business-goal construction.

Run:

```powershell
python -m pytest -q
```

## LLM evaluation

Before submission, manually evaluate a fixed set of business questions and generated memo/goal outputs. Record sample size, whether the answer stayed grounded in the analytical snapshot, and the number of material errors.

Recommended check set:

1. Which channel has the highest SLA breach rate?
2. What is the largest modeled opportunity?
3. What is the deterministic business goal and how was its target chosen?
4. What data-quality issue should a CX leader know about?
5. Which complaint themes deserve investigation?

A material error is a fabricated number, a changed target, a claim contradicted by the supplied tables, or a failure to state a key caveat where the answer requires it.
