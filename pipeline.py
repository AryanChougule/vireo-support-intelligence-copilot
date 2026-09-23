from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


TIER1_TEAMS = {"chat frontline", "email frontline", "voice frontline"}
COMPLETED_STATUSES = {"resolved", "closed"}


@dataclass
class PipelineResult:
    tickets: pd.DataFrame
    quality: dict[str, Any]
    weekly_metrics: pd.DataFrame
    themes: pd.DataFrame
    leaderboard: pd.DataFrame
    opportunities: pd.DataFrame
    channel_metrics: pd.DataFrame
    category_metrics: pd.DataFrame
    repeat_contacts: pd.DataFrame
    business_goal: dict[str, Any]


def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def load_inputs(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    result: dict[str, pd.DataFrame] = {}

    for name in ("tickets", "agents", "customers", "orders", "products"):
        path = data_dir / f"{name}.csv"
        result[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()

    if result["tickets"].empty:
        raise FileNotFoundError(
            f"tickets.csv was not found in {data_dir.resolve()}. "
            "Place the supplied assessment files in the data folder."
        )

    return result


def normalize_tickets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = [
        "ticket_id", "created_at", "first_response_at", "resolved_at", "status",
        "channel", "customer_id", "order_id", "product_sku", "category",
        "priority", "assigned_team", "agent_id", "transfers", "csat_score",
        "refund_amount_inr", "refund_reason_code", "replacement_issued",
        "customer_message", "agent_notes", "source_system",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA

    for col in ("created_at", "first_response_at", "resolved_at"):
        df[col] = _parse_datetime(df[col])

    for col in ("transfers", "csat_score", "refund_amount_inr"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["csat_score"] = df["csat_score"].replace(0, np.nan)

    for col in (
        "ticket_id", "status", "channel", "customer_id", "order_id",
        "product_sku", "category", "priority", "assigned_team", "agent_id",
        "refund_reason_code", "replacement_issued", "source_system",
    ):
        df[col] = df[col].astype("string").str.strip()

    df["customer_message"] = _clean_text(df["customer_message"])
    df["agent_notes"] = _clean_text(df["agent_notes"])

    return df


def reconcile_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = df.copy()
    duplicate_counts = work.groupby("ticket_id").size()
    work["_duplicate_group"] = work["ticket_id"].map(duplicate_counts).fillna(1).gt(1)

    source = work["source_system"].fillna("").str.lower()
    work["_source_priority"] = source.eq("helpdesk").astype(int)
    work["_completeness"] = work.notna().sum(axis=1)

    # This is reconciliation for analytics, not modification of the source data.
    canonical = (
        work.sort_values(
            ["ticket_id", "_source_priority", "_completeness"],
            ascending=[True, False, False],
            kind="mergesort",
        )
        .drop_duplicates("ticket_id", keep="first")
        .copy()
    )

    quality = {
        "raw_rows": int(len(work)),
        "canonical_rows": int(len(canonical)),
        "unique_ticket_ids": int(canonical["ticket_id"].nunique()),
        "duplicate_ticket_rows": int(work["_duplicate_group"].sum()),
        "duplicate_ticket_id_groups": int((duplicate_counts > 1).sum()),
        "duplicate_pairs_helpdesk_legacy": int(
            (
                work.loc[work["_duplicate_group"]]
                .groupby("ticket_id")["source_system"]
                .apply(lambda s: set(s.dropna().str.lower()) >= {"helpdesk", "legacy_fd"})
            ).sum()
        ),
    }

    return canonical.drop(
        columns=["_source_priority", "_completeness", "_duplicate_group"],
        errors="ignore",
    ), quality


def apply_policy_metrics(df: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    df = df.copy()

    df["channel_key"] = df["channel"].fillna("").str.lower().str.strip()
    sla_map = {
        str(k).lower(): float(v)
        for k, v in policy.get("first_response_sla_minutes", {}).items()
    }
    cost_map = {
        str(k).lower(): float(v)
        for k, v in policy.get("contact_cost_inr", {}).items()
    }

    df["first_response_minutes"] = (
        (df["first_response_at"] - df["created_at"]).dt.total_seconds() / 60
    )
    df["resolution_time_hours"] = (
        (df["resolved_at"] - df["created_at"]).dt.total_seconds() / 3600
    )
    df["handle_time_hours"] = (
        (df["resolved_at"] - df["first_response_at"]).dt.total_seconds() / 3600
    )

    df["sla_target_minutes"] = df["channel_key"].map(sla_map)
    df["sla_breach"] = (
        df["first_response_minutes"].notna()
        & df["sla_target_minutes"].notna()
        & (df["first_response_minutes"] > df["sla_target_minutes"])
    )

    df["contact_cost_inr"] = df["channel_key"].map(cost_map).fillna(
        float(policy.get("blended_cost_inr", 0))
    )
    df["sla_penalty_exposure_inr"] = (
        df["sla_breach"].astype(int) * float(policy.get("sla_miss_credit_inr", 0))
    )
    df["transfer_cost_inr"] = (
        pd.to_numeric(df["transfers"], errors="coerce").fillna(0)
        * float(policy.get("internal_transfer_cost_inr", 0))
    )

    df["resolved_before_created"] = (
        df["resolved_at"].notna()
        & df["created_at"].notna()
        & (df["resolved_at"] < df["created_at"])
    )
    df["first_response_before_created"] = (
        df["first_response_at"].notna()
        & df["created_at"].notna()
        & (df["first_response_at"] < df["created_at"])
    )
    df["resolved_before_first_response"] = (
        df["resolved_at"].notna()
        & df["first_response_at"].notna()
        & (df["resolved_at"] < df["first_response_at"])
    )
    df["temporal_anomaly"] = (
        df["resolved_before_created"]
        | df["first_response_before_created"]
        | df["resolved_before_first_response"]
    )

    # Do not repair source timestamps. Invalid elapsed-time values become unavailable
    # for duration analytics while the original columns and flags remain intact.
    for col in ("resolution_time_hours", "handle_time_hours"):
        df.loc[df[col] < 0, col] = np.nan

    dow = df["created_at"].dt.dayofweek
    df["week_start"] = (
        df["created_at"].dt.normalize() - pd.to_timedelta(dow, unit="D")
    )
    rdow = df["resolved_at"].dt.dayofweek
    df["resolution_week_start"] = (
        df["resolved_at"].dt.normalize() - pd.to_timedelta(rdow, unit="D")
    )

    completed = df["status"].fillna("").str.lower().isin(COMPLETED_STATUSES)
    df["completed_ticket"] = completed

    return df


def build_quality_report(raw: pd.DataFrame, canonical: pd.DataFrame, reconcile_quality: dict[str, Any]) -> dict[str, Any]:
    quality = dict(reconcile_quality)
    quality.update(
        {
            "missing_resolved_at": int(canonical["resolved_at"].isna().sum()),
            "missing_agent_id": int(canonical["agent_id"].isna().sum()),
            "invalid_created_at": int(canonical["created_at"].isna().sum()),
            "source_system_counts": {
                str(k): int(v)
                for k, v in canonical["source_system"].value_counts(dropna=False).items()
            },
            "resolved_before_created": int(
                (raw["resolved_at"] < raw["created_at"]).fillna(False).sum()
            ),
            "first_response_before_created": int(
                (raw["first_response_at"] < raw["created_at"]).fillna(False).sum()
            ),
            "resolved_before_first_response": int(
                (raw["resolved_at"] < raw["first_response_at"]).fillna(False).sum()
            ),
            "resolved_before_created_by_source": {
                str(k): int(v)
                for k, v in raw.loc[
                    (raw["resolved_at"] < raw["created_at"]).fillna(False),
                    "source_system",
                ].value_counts(dropna=False).items()
            },
            "canonical_resolved_before_created": int(canonical["resolved_before_created"].sum()),
            "canonical_first_response_before_created": int(canonical["first_response_before_created"].sum()),
            "canonical_resolved_before_first_response": int(canonical["resolved_before_first_response"].sum()),
            "canonical_temporal_anomalies": int(canonical["temporal_anomaly"].sum()),
            "canonical_temporal_anomaly_rate": round(
                float(canonical["temporal_anomaly"].mean() * 100), 2
            ),
            "canonical_temporal_anomalies_by_source": {
                str(k): int(v)
                for k, v in canonical.loc[
                    canonical["temporal_anomaly"], "source_system"
                ].value_counts(dropna=False).items()
            },
        }
    )
    return quality


def weekly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    x = df.dropna(subset=["week_start"]).copy()
    grouped = (
        x.groupby("week_start")
        .agg(
            total_contacts=("ticket_id", "nunique"),
            unique_customers=("customer_id", "nunique"),
            sla_breaches=("sla_breach", "sum"),
            avg_csat=("csat_score", "mean"),
            contact_cost_inr=("contact_cost_inr", "sum"),
            sla_penalty_exposure_inr=("sla_penalty_exposure_inr", "sum"),
            transfer_cost_inr=("transfer_cost_inr", "sum"),
        )
        .reset_index()
        .sort_values("week_start")
    )
    grouped["sla_breach_rate"] = (
        grouped["sla_breaches"] / grouped["total_contacts"].replace(0, np.nan)
    )
    grouped["week_over_week_contacts"] = grouped["total_contacts"].pct_change()
    return grouped


def _group_metrics(df: pd.DataFrame, group_col: str, transfer_unit_cost: float = 305.0) -> pd.DataFrame:
    x = df.copy()
    result = (
        x.groupby(group_col, dropna=False)
        .agg(
            tickets=("ticket_id", "nunique"),
            sla_breaches=("sla_breach", "sum"),
            avg_csat=("csat_score", "mean"),
            transfer_events=("transfers", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
            contact_cost_inr=("contact_cost_inr", "sum"),
            sla_penalty_exposure_inr=("sla_penalty_exposure_inr", "sum"),
        )
        .reset_index()
    )
    result["sla_breach_rate"] = result["sla_breaches"] / result["tickets"].replace(0, np.nan)
    result["transfer_cost_inr"] = (
        result["transfer_events"] * transfer_unit_cost
    )
    return result.sort_values("tickets", ascending=False)


def build_channel_metrics(df: pd.DataFrame, policy: dict[str, Any] | None = None) -> pd.DataFrame:
    cost = float((policy or {}).get("internal_transfer_cost_inr", 305))
    return _group_metrics(df, "channel", cost)


def build_category_metrics(df: pd.DataFrame, policy: dict[str, Any] | None = None) -> pd.DataFrame:
    cost = float((policy or {}).get("internal_transfer_cost_inr", 305))
    return _group_metrics(df, "category", cost)


def build_repeat_contacts(df: pd.DataFrame) -> pd.DataFrame:
    x = df[["ticket_id", "customer_id", "created_at", "category", "order_id"]].copy()
    x = x.dropna(subset=["customer_id", "created_at"]).sort_values(
        ["customer_id", "created_at"]
    )
    x["prior_contact_at"] = x.groupby("customer_id")["created_at"].shift(1)
    x["days_since_prior_contact"] = (
        x["created_at"] - x["prior_contact_at"]
    ).dt.total_seconds() / 86400
    x["repeat_contact_7d"] = x["days_since_prior_contact"].between(0, 7, inclusive="both")
    return x


def build_opportunities(
    df: pd.DataFrame,
    repeat_df: pd.DataFrame,
    policy: dict[str, Any],
) -> pd.DataFrame:
    total = max(df["ticket_id"].nunique(), 1)

    sla_breaches = int(df["sla_breach"].sum())
    sla_cost = float(df["sla_penalty_exposure_inr"].sum())

    transfers = float(pd.to_numeric(df["transfers"], errors="coerce").fillna(0).sum())
    transfer_unit_cost = float(policy.get("internal_transfer_cost_inr", 305))
    transfer_cost = float(transfers * transfer_unit_cost)

    repeat_count = int(repeat_df["repeat_contact_7d"].sum())
    repeat_rate = repeat_count / total
    repeat_cost = float(
        repeat_count * float(policy.get("blended_cost_inr", 290))
    )

    rows = [
        {
            "opportunity": "SLA breaches",
            "current_rate": sla_breaches / total,
            "affected_volume": sla_breaches,
            "modeled_exposure_inr": sla_cost,
            "definition": "First response exceeded the channel SLA.",
        },
        {
            "opportunity": "Internal transfers",
            "current_rate": transfers / total,
            "affected_volume": int(transfers),
            "modeled_exposure_inr": transfer_cost,
            "definition": "Transfer events multiplied by policy transfer cost.",
        },
        {
            "opportunity": "Repeat-contact proxy (7 days)",
            "current_rate": repeat_rate,
            "affected_volume": repeat_count,
            "modeled_exposure_inr": repeat_cost,
            "definition": "A customer had a prior contact within 7 days; proxy, not confirmed causality.",
        },
    ]
    return pd.DataFrame(rows).sort_values("modeled_exposure_inr", ascending=False)


def build_business_goal(
    first_input: pd.DataFrame,
    second_input: pd.DataFrame,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """
    Build an evidence-backed business opportunity.

    Production call:
        build_business_goal(weekly_metrics, opportunities, policy)

    Backward-compatible test call:
        build_business_goal(tickets, weekly_metrics, policy)

    Python owns all numerical calculations.
    Gemini only interprets the resulting evidence.

    The function does not manufacture a target when the historical
    data does not support a meaningful improvement opportunity.
    """

    # =========================================================
    # Identify which calling convention is being used
    # =========================================================

    # Production:
    #   first_input  = weekly_metrics
    #   second_input = opportunities
    #
    # Existing test:
    #   first_input  = ticket-level dataframe
    #   second_input = weekly_metrics

    if (
        "ticket_id" in first_input.columns
        and "sla_breaches" in second_input.columns
    ):
        # Backward-compatible test invocation
        weekly = second_input.copy()
        opportunities = pd.DataFrame()

    else:
        # Normal production invocation
        weekly = first_input.copy()
        opportunities = second_input.copy()


    # =========================================================
    # Validate weekly data
    # =========================================================

    if weekly is None or weekly.empty:
        return {
            "available": False,
            "reason": (
                "Not enough weekly data to establish a "
                "business goal."
            ),
        }


    weekly = weekly.copy()

    weekly["week_start"] = pd.to_datetime(
        weekly["week_start"],
        errors="coerce",
    )

    weekly = (
        weekly
        .dropna(subset=["week_start"])
        .sort_values("week_start")
        .reset_index(drop=True)
    )


    # =========================================================
    # Require enough historical weeks
    # =========================================================

    if len(weekly) < 4:
        return {
            "available": False,
            "eligible_weeks": int(len(weekly)),
            "reason": (
                "Not enough historical weeks to establish "
                "a defensible benchmark."
            ),
        }


    # =========================================================
    # Current baseline
    #
    # Most recent 8 weeks
    # =========================================================

    if len(weekly) >= 12:
        recent = weekly.tail(8).copy()
        historical = weekly.iloc[:-8].copy()
    else:
        # Small datasets: use the latest week as the current
        # baseline and all preceding weeks as historical data.
        recent = weekly.tail(1).copy()
        historical = weekly.iloc[:-1].copy()

    recent_contacts = int(
        recent["total_contacts"].sum()
    )

    recent_sla_breaches = int(
        recent["sla_breaches"].sum()
    )

    current_sla_rate = (
        recent_sla_breaches / recent_contacts
    )


    # =========================================================
    # Historical benchmark
    #
    # Use the lower quartile of historical weekly SLA
    # breach rates, then take the median of those weeks.
    # =========================================================

    historical = historical[
        historical["total_contacts"] > 0
    ].copy()

    if len(historical) < 4:
        return {
            "available": False,
            "eligible_weeks": int(len(weekly)),
            "reason": (
                "Not enough historical weeks remain after "
                "establishing the current baseline."
            ),
        }

    historical["sla_breach_rate_calc"] = (
        historical["sla_breaches"]
        / historical["total_contacts"]
    )

    sla_cutoff = historical[
        "sla_breach_rate_calc"
    ].quantile(0.25)

    sla_best_weeks = historical[
        historical["sla_breach_rate_calc"] <= sla_cutoff
    ]

    if sla_best_weeks.empty:
        sla_target = None
    else:
        sla_target = float(
            sla_best_weeks[
                "sla_breach_rate_calc"
            ].median()
        )


    # =========================================================
    # Determine whether SLA supports a defensible target
    # =========================================================

    candidates = []

    if sla_target is not None:

        sla_gap = current_sla_rate - sla_target

        # Require at least a 0.5 percentage-point improvement.
        if sla_gap >= 0.005:

            current_weekly_breaches = (
                recent_sla_breaches / len(recent)
            )

            target_weekly_breaches = (
                recent_contacts / len(recent)
            ) * sla_target

            avoidable_per_week = max(
                0.0,
                current_weekly_breaches
                - target_weekly_breaches,
            )

            quarterly_reduction = (
                avoidable_per_week * 13
            )

            sla_credit = float(
                policy.get(
                    "sla_miss_credit_inr",
                    0,
                )
            )

            quarterly_value = (
                quarterly_reduction
                * sla_credit
            )

            candidates.append(
                {
                    "metric": "SLA breach rate",
                    "current_rate": current_sla_rate,
                    "target_rate": sla_target,
                    "improvement_gap": sla_gap,
                    "current_volume": recent_contacts,
                    "avoidable_volume_per_week": (
                        avoidable_per_week
                    ),
                    "quarterly_volume_reduction": (
                        quarterly_reduction
                    ),
                    "unit_cost_inr": sla_credit,
                    "quarterly_value_inr": (
                        quarterly_value
                    ),
                    "benchmark_method": (
                        "Median SLA breach rate among "
                        "historical lower-quartile weeks."
                    ),
                }
            )


    # =========================================================
    # Internal transfer observation
    #
    # We DO NOT invent a transfer target because the current
    # deterministic pipeline does not have historical weekly
    # transfer-rate benchmarks.
    # =========================================================

    transfer_observation = None

    if (
        opportunities is not None
        and not opportunities.empty
        and "opportunity" in opportunities.columns
    ):

        transfer_row = opportunities[
            opportunities["opportunity"]
            == "Internal transfers"
        ]

        if not transfer_row.empty:

            transfer_observation = {
                "metric": "Internal transfers",
                "current_rate": float(
                    transfer_row.iloc[0][
                        "current_rate"
                    ]
                ),
                "modeled_exposure_inr": float(
                    transfer_row.iloc[0][
                        "modeled_exposure_inr"
                    ]
                ),
            }


    # =========================================================
    # Repeat-contact observation
    # =========================================================

    repeat_observation = None

    if (
        opportunities is not None
        and not opportunities.empty
        and "opportunity" in opportunities.columns
    ):

        repeat_row = opportunities[
            opportunities["opportunity"]
            == "Repeat-contact proxy (7 days)"
        ]

        if not repeat_row.empty:

            repeat_observation = {
                "metric": (
                    "Repeat-contact proxy (7 days)"
                ),
                "current_rate": float(
                    repeat_row.iloc[0][
                        "current_rate"
                    ]
                ),
                "modeled_exposure_inr": float(
                    repeat_row.iloc[0][
                        "modeled_exposure_inr"
                    ]
                ),
            }


    # =========================================================
    # No evidence-backed SLA target
    # =========================================================

    if not candidates:
        return {
            "available": False,

            "eligible_weeks": int(len(weekly)),

            "reason": (
                "No opportunity has a sufficiently large "
                "historical improvement gap to support an "
                "evidence-backed target."
            ),

            "current_sla_rate": current_sla_rate,

            "historical_sla_benchmark": sla_target,

            "observed_opportunities": {
                "transfers": transfer_observation,
                "repeat_contacts": repeat_observation,
            },
        }


    # =========================================================
    # Select the evidence-backed candidate
    #
    # If more benchmarked opportunities are added later,
    # select the largest modeled quarterly value.
    # =========================================================

    selected = max(
        candidates,
        key=lambda x: x[
            "quarterly_value_inr"
        ],
    )


    return {
        "available": True,

        "metric": selected[
            "metric"
        ],

        "current_rate": float(
            selected["current_rate"]
        ),

        "target_rate": float(
            selected["target_rate"]
        ),

        "improvement_gap": float(
            selected["improvement_gap"]
        ),

        "current_volume": int(
            selected["current_volume"]
        ),

        "avoidable_volume_per_week": float(
            selected[
                "avoidable_volume_per_week"
            ]
        ),

        "quarterly_volume_reduction": float(
            selected[
                "quarterly_volume_reduction"
            ]
        ),

        "unit_cost_inr": float(
            selected["unit_cost_inr"]
        ),

        "quarterly_value_inr": float(
            selected[
                "quarterly_value_inr"
            ]
        ),

        "modeled_quarterly_exposure_reduction_inr": float(
            selected[
                "quarterly_value_inr"
            ]
        ),

        "average_weekly_volume": float(
            selected["current_volume"] / len(recent)
        ),
        
        "eligible_weeks": int(len(weekly)),

        "benchmark_method": selected[
            "benchmark_method"
        ],

        "baseline_period": (
            "Most recent 8 weeks"
        ),

        "historical_period": (
            "All earlier available weeks"
        ),
    }
    
def complaint_themes(df: pd.DataFrame, n_clusters: int = 8) -> pd.DataFrame:
    x = df.copy()
    text = (
        x["customer_message"].fillna("")
        + " "
        + x["agent_notes"].fillna("")
    ).str.strip()
    usable = x.loc[text.str.len() >= 15].copy()
    usable["_text"] = text.loc[usable.index]

    if len(usable) < 10:
        return pd.DataFrame(
            columns=[
                "theme_id", "category", "theme_label", "ticket_count",
                "top_terms", "avg_csat", "sla_breach_rate",
                "transfer_rate", "sample_ticket_ids",
            ]
        )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=2,
        max_features=5000,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(usable["_text"])
    k = max(2, min(n_clusters, matrix.shape[0]))
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(matrix)
    terms = vectorizer.get_feature_names_out()

    rows = []
    for cluster_id in range(k):
        positions = np.where(labels == cluster_id)[0]
        if len(positions) == 0:
            continue

        centroid = matrix[positions].mean(axis=0)
        top_idx = np.asarray(centroid).ravel().argsort()[::-1][:8]
        top_terms = [terms[i] for i in top_idx]

        cluster = usable.iloc[positions]
        dominant_category = (
            cluster["category"].dropna().astype(str).value_counts().index[0]
            if cluster["category"].notna().any()
            else "Other"
        )
        rows.append(
            {
                "theme_id": cluster_id,
                "category": dominant_category,
                "theme_label": f"{dominant_category} — {', '.join(top_terms[:3])}",
                "ticket_count": int(len(cluster)),
                "top_terms": ", ".join(top_terms),
                "avg_csat": float(cluster["csat_score"].mean())
                if cluster["csat_score"].notna().any()
                else np.nan,
                "sla_breach_rate": float(cluster["sla_breach"].mean()),
                "transfer_rate": float(
                    pd.to_numeric(cluster["transfers"], errors="coerce").fillna(0).gt(0).mean()
                ),
                "sample_ticket_ids": ", ".join(
                    cluster["ticket_id"].astype(str).head(8).tolist()
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("ticket_count", ascending=False)


def tier1_leaderboard(df: pd.DataFrame, agents: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    if agents.empty:
        roster = pd.DataFrame(columns=["agent_id", "name", "team", "tier"])
    else:
        roster = agents.copy()
        rename = {}
        for target, candidates in {
            "agent_id": ["agent_id", "id"],
            "name": ["name", "agent_name"],
            "team": ["team", "assigned_team"],
            "tier": ["tier", "agent_tier"],
        }.items():
            for c in candidates:
                if c in roster.columns:
                    rename[c] = target
                    break
        roster = roster.rename(columns=rename)
        for col in ("agent_id", "name", "team", "tier"):
            if col not in roster.columns:
                roster[col] = pd.NA

    x = df.copy()
    x["agent_id"] = x["agent_id"].astype("string")
    x["team_key"] = x["assigned_team"].fillna("").str.lower().str.strip()
    x["tier_key"] = ""
    x = x.merge(
        roster[["agent_id", "name", "team", "tier"]].drop_duplicates("agent_id"),
        on="agent_id",
        how="left",
        suffixes=("", "_roster"),
    )
    x["effective_team"] = x["team"].fillna(x["assigned_team"])
    x["effective_tier"] = x["tier"].fillna("")
    x["team_key"] = x["effective_team"].fillna("").astype(str).str.lower().str.strip()
    x["tier_key"] = (
    x["effective_tier"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
    .replace({"1": "tier 1", "2": "tier 2"})
)

    eligible_tiers = {
        str(v).lower().strip()
        for v in policy.get("eligible_leaderboard_tiers", ["Tier 1"])
    }
    x = x[
        x["completed_ticket"]
        & x["resolution_week_start"].notna()
        & x["tier_key"].isin(eligible_tiers)
        & x["team_key"].isin(TIER1_TEAMS)
    ].copy()

    result = (
        x.groupby(["resolution_week_start", "agent_id", "name", "effective_team"], dropna=False)
        .agg(
            tickets_closed=("ticket_id", "nunique"),
            sla_breaches=("sla_breach", "sum"),
            avg_csat=("csat_score", "mean"),
            median_handle_time_hours=("handle_time_hours", "median"),
        )
        .reset_index()
    )
    result["sla_breach_rate"] = (
        result["sla_breaches"] / result["tickets_closed"].replace(0, np.nan)
    )
    return result.sort_values(
        ["resolution_week_start", "tickets_closed"],
        ascending=[True, False],
    )


def run_pipeline(data_dir: str | Path, policy_path: str | Path = "policy.yaml") -> PipelineResult:
    raw_inputs = load_inputs(data_dir)
    raw = normalize_tickets(raw_inputs["tickets"])

    canonical, duplicate_quality = reconcile_duplicates(raw)

    with open(policy_path, "r", encoding="utf-8") as f:
        policy = yaml.safe_load(f) or {}

    tickets = apply_policy_metrics(canonical, policy)

    quality = build_quality_report(raw, tickets, duplicate_quality)

    weeks = weekly_metrics(tickets)
    themes = complaint_themes(tickets)
    leaderboard = tier1_leaderboard(tickets, raw_inputs["agents"], policy)
    channel = build_channel_metrics(tickets, policy)
    category = build_category_metrics(tickets, policy)
    repeat = build_repeat_contacts(tickets)
    opportunities = build_opportunities(tickets, repeat, policy)
    goal = build_business_goal(weeks, opportunities, policy)
    
    return PipelineResult(
        tickets=tickets,
        quality=quality,
        weekly_metrics=weeks,
        themes=themes,
        leaderboard=leaderboard,
        opportunities=opportunities,
        channel_metrics=channel,
        category_metrics=category,
        repeat_contacts=repeat,
        business_goal=goal,
    )
