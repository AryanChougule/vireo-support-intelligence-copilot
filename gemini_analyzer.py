from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types
import pandas as pd

# Stable default model. The UI also lets the reviewer switch models if their
# Gemini account exposes a different model.
DEFAULT_MODEL = "gemini-2.5-flash"


SYSTEM_INSTRUCTION = """
You are the Vireo Support Intelligence analyst.

Your job is to interpret deterministic support analytics for a CX leader.
Never invent numbers. Numerical values in the supplied analytical context
are authoritative. If the context does not contain enough evidence, say so.

Important rules:
1. Distinguish observed facts, modeled exposure, hypotheses and recommendations.
2. Do not repair or reinterpret source-data timestamps.
3. Treat the repeat-contact metric as a proxy, not proof of avoidable contacts.
4. Do not rank agents by any metric other than the requested tickets-closed
   leaderboard unless explicitly asked. CSAT/SLA are context.
5. When making a claim, cite the relevant metric/table name and value in prose.
6. Prefer concise business language suitable for Priya Raman, Head of CX.
7. The Python pipeline is the source of truth for calculations.
8. For targets, use the deterministic benchmark supplied by Python. Do not
   invent a target percentage.
9. Modeled exposure is not guaranteed savings; preserve that distinction.
10. If business_goal.available is false, explicitly say that no
    evidence-backed target was identified. Never manufacture a target.
11. Never infer a data-quality count from another metric. Only report a
    data-quality number when the exact field exists in the supplied
    data_quality object.

12. "invalid_created_at" means created_at values that could not be parsed.
    Do not describe missing resolved_at, missing timestamps, or other
    missing fields as invalid_created_at.

13. If a data-quality field is not explicitly present in the context,
    do not estimate or infer it.
"""


def make_client(api_key: str):
    return genai.Client(api_key=api_key.strip())


def generate(client, prompt: str, model: str = DEFAULT_MODEL) -> str:
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=3500,
        ),
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text


def _records(df, n=20):
    if df is None or df.empty:
        return []
    return df.head(n).to_dict(orient="records")

def _latest_records(df, n=80, date_col=None):
    if df is None or df.empty:
        return []

    x = df.copy()

    if date_col and date_col in x.columns:
        x[date_col] = pd.to_datetime(
            x[date_col],
            errors="coerce"
        )
        x = x.sort_values(date_col, ascending=False)

    return x.head(n).to_dict(orient="records")


def build_context(result: Any, max_ticket_rows: int = 12) -> str:
    """Build a compact evidence pack. Python owns all numerical truth."""
    q = result.quality
    ticket_cols = [
        c for c in [
            "ticket_id", "created_at", "channel", "category", "priority",
            "assigned_team", "agent_id", "transfers", "csat_score",
            "refund_amount_inr", "refund_reason_code", "replacement_issued",
            "customer_message", "agent_notes", "source_system",
        ] if c in result.tickets.columns
    ]
    sample = result.tickets[ticket_cols].head(max_ticket_rows).copy()
    for c in ("created_at",):
        if c in sample.columns:
            sample[c] = sample[c].astype(str)
            

    payload = {
        "data_quality": {
            **q,
            "interpretation_rules": {
                "invalid_created_at":
                    "Count of created_at values that failed datetime parsing.",
                "missing_resolved_at":
                    "Count of canonical tickets with resolved_at missing.",
                "canonical_temporal_anomalies":
                    "Count of canonical tickets with invalid timestamp ordering.",
            }
        },
        "business_goal": getattr(result, "business_goal", {}),
        "weekly_metrics": _records(result.weekly_metrics, 100),
        "opportunities": _records(result.opportunities, 20),
        "channel_metrics": _records(result.channel_metrics, 20),
        "category_metrics": _records(result.category_metrics, 50),
        "complaint_themes": _records(result.themes, 20),
        "tier1_leaderboard_latest_weeks":  _latest_records(result.leaderboard, 80, "resolution_week_start",),
        "ticket_sample": sample.to_dict(orient="records"),
    }
    return json.dumps(payload, ensure_ascii=False, default=str)


def business_goal(result: Any, client, model: str = DEFAULT_MODEL) -> str:
    context = build_context(result, max_ticket_rows=0)

    goal = getattr(result, "business_goal", {}) or {}

    if not goal.get("available", False):
        reason = goal.get(
            "reason",
            "The data does not support a defensible target.",
        )

        return (
            "## No evidence-backed business goal identified\n\n"
            f"{reason}\n\n"
            "The system deliberately does not invent a target or "
            "financial impact when the historical data does not "
            "support one."
        )

    prompt = f"""
Create the business goal for Vireo Audio's support analytics assessment.

IMPORTANT:
The Python analytics pipeline has already calculated the business goal.

Python is the source of truth.

You MUST preserve these values exactly:

Metric:
{goal.get("metric")}

Current rate:
{goal.get("current_rate")}

Target rate:
{goal.get("target_rate")}

Improvement gap:
{goal.get("improvement_gap")}

Current volume:
{goal.get("current_volume")}

Avoidable volume per week:
{goal.get("avoidable_volume_per_week")}

Quarterly volume reduction:
{goal.get("quarterly_volume_reduction")}

Unit cost:
{goal.get("unit_cost_inr")}

Modeled quarterly exposure:
{goal.get("quarterly_value_inr")}

Benchmark method:
{goal.get("benchmark_method")}

Baseline period:
{goal.get("baseline_period")}

Historical period:
{goal.get("historical_period")}

Do NOT:
- invent a different target
- change any numerical value
- create additional savings estimates
- round the target into a different percentage
- call modeled exposure guaranteed savings
- introduce an external industry benchmark

Return exactly these sections:

## Business goal

Write one clear sentence:

Reduce [metric] from [current rate] toward [target rate],
representing approximately ₹[quarterly value] in modeled
exposure reduction per quarter.

Use percentage formatting for rate values.

## Why this target

Explain in 2-3 sentences that the target comes from the
historical benchmark calculated by Python.

## Measurement

State:
- metric
- population
- baseline period
- weekly measurement cadence

## Caveat

Explain that the rupee figure is modeled exposure based on
the supplied policy cost and is not guaranteed realized savings.

Analytical context:

{context}
"""

    return generate(client, prompt, model)

def executive_memo(result: Any, client, model: str = DEFAULT_MODEL) -> str:
    context = build_context(result, max_ticket_rows=8)
    prompt = f"""
Write a one-page, nontechnical memo to Priya Raman, Head of CX, based only on
the supplied Vireo analytical context.

Use this structure:
# Vireo Audio Support Intelligence — Executive Memo
**To:** Priya Raman
**Subject:** Weekly support findings and quantified opportunity

## Executive summary
3 concise bullets.

## Business opportunity
If the deterministic business_goal object has available=true,
state the business goal and modeled quarterly exposure exactly as
provided.

If available=false, explicitly state that no evidence-backed
business target was identified and explain why.

Do not invent a target or financial impact.

## Customer / complaint patterns
2-4 evidence-backed observations from categories/themes.

## Operational findings
Discuss SLA, transfers, repeat-contact proxy, and the Tier-1 tickets-closed
leaderboard without creating an overall agent ranking.

## Data quality
Mention the important temporal anomaly finding and that source data was not modified.

## Recommended next steps
3 practical actions or investigations.

Keep it concise enough for one page. Distinguish observed facts from
recommendations and do not invent numbers.

Analytical context:
{context}
"""
    return generate(client, prompt, model)


def deep_analysis(result: Any, client, model: str = DEFAULT_MODEL) -> str:
    context = build_context(result)
    prompt = f"""
Produce an in-depth but concise CX analysis of the Vireo support operation.

Required sections:
## Executive summary
Give 3-5 evidence-backed findings.

## Complaint and customer experience patterns
Identify meaningful complaint patterns from the supplied theme/category evidence.

## Business opportunities
Explain the deterministic business goal and compare the quantified opportunities.
Do not fabricate a target reduction.

## Operational observations
Discuss channels, SLA, transfers and the requested Tier-1 leaderboard.

## Data quality
Explicitly mention important source-data limitations.

## What I would investigate next
Give 3 practical follow-up analyses or interventions.

Analytical context:
{context}
"""
    return generate(client, prompt, model)


def answer_query(result: Any, question: str, client, model: str = DEFAULT_MODEL) -> str:
    question = question.strip()
    if not question:
        raise ValueError("Please enter a question.")

    context = build_context(result)
    prompt = f"""
Answer this user question about the Vireo support data:

{question}

Use only the supplied analytical context. If a direct calculation is not
present, explain that the current analytical snapshot does not contain enough
evidence rather than inventing a number.

Return:
1. A direct answer.
2. The supporting evidence/metrics.
3. Any important caveat.

Analytical context:
{context}
"""
    return generate(client, prompt, model)


def validate_key(api_key: str, model: str = DEFAULT_MODEL) -> tuple[bool, str]:
    try:
        client = make_client(api_key)
        response = client.models.generate_content(
            model=model,
            contents="Reply with exactly: Vireo connection OK",
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=20,
            ),
        )
        text = getattr(response, "text", "") or ""
        return True, text.strip()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
