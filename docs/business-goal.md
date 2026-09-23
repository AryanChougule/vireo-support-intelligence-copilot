# Business Goal Logic

The dashboard derives a concrete business goal without asking the LLM to invent a target.

## Metric

**SLA breach rate**: first response time above the channel-specific first-response SLA in `policy.yaml`.

## Target construction

1. Calculate the SLA breach rate for every historical week.
2. Ignore weeks with fewer than 20 contacts so tiny weeks do not create an unstable benchmark.
3. Find the 25th percentile of weekly SLA breach rates.
4. Use the median breach rate among weeks at or below that percentile as the target benchmark.
5. Compare the full-dataset current breach rate to that historical benchmark.
6. Convert the rate gap into a modeled quarterly exposure reduction using average weekly volume × 13 weeks × the policy SLA-miss credit.

This creates a target from Vireo's own observed data rather than an arbitrary 15%/20% target.

The resulting rupee amount is explicitly **modeled exposure**, not guaranteed realized savings.

Gemini receives these deterministic values and writes the business narrative. It does not calculate or change the target.
