import numpy as np
import pandas as pd

from pipeline import (
    apply_policy_metrics,
    build_opportunities,
    build_repeat_contacts,
    reconcile_duplicates,
    tier1_leaderboard,
    weekly_metrics,
)


POLICY = {
    "first_response_sla_minutes": {"chat": 15, "email": 480, "voice": 120, "social": 240},
    "contact_cost_inr": {"chat": 210, "email": 260, "voice": 520, "social": 240},
    "blended_cost_inr": 290,
    "internal_transfer_cost_inr": 305,
    "sla_miss_credit_inr": 350,
    "eligible_leaderboard_tiers": ["Tier 1"],
}


def base_ticket(**overrides):
    row = {
        "ticket_id": "T1",
        "created_at": pd.Timestamp("2025-01-01 23:00"),
        "first_response_at": pd.Timestamp("2025-01-02 00:00"),
        "resolved_at": pd.Timestamp("2025-01-02 01:00"),
        "status": "resolved",
        "channel": "email",
        "customer_id": "C1",
        "order_id": "O1",
        "product_sku": "P1",
        "category": "Returns",
        "priority": "normal",
        "assigned_team": "Email Frontline",
        "agent_id": "A1",
        "transfers": 1,
        "csat_score": 4,
        "refund_amount_inr": 0,
        "refund_reason_code": "",
        "replacement_issued": "no",
        "customer_message": "I need help with my order.",
        "agent_notes": "Resolved.",
        "source_system": "helpdesk",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_policy_metrics_and_costs():
    df = apply_policy_metrics(base_ticket(), POLICY)
    assert df.loc[0, "first_response_minutes"] == 60
    assert bool(df.loc[0, "sla_breach"]) is False
    assert df.loc[0, "contact_cost_inr"] == 260
    assert df.loc[0, "transfer_cost_inr"] == 305


def test_cross_midnight_duration_preserves_date():
    df = apply_policy_metrics(
        base_ticket(
            created_at=pd.Timestamp("2025-01-01 23:00"),
            first_response_at=pd.Timestamp("2025-01-02 00:30"),
            resolved_at=pd.Timestamp("2025-01-02 02:00"),
        ),
        POLICY,
    )
    assert round(df.loc[0, "resolution_time_hours"], 4) == 3
    assert round(df.loc[0, "handle_time_hours"], 4) == 1.5


def test_temporal_anomaly_is_flagged_without_repairing_source():
    df = apply_policy_metrics(
        base_ticket(
            resolved_at=pd.Timestamp("2024-12-31 20:00"),
        ),
        POLICY,
    )
    assert bool(df.loc[0, "resolved_before_created"]) is True
    assert pd.isna(df.loc[0, "resolution_time_hours"])
    assert df.loc[0, "resolved_at"] == pd.Timestamp("2024-12-31 20:00")


def test_weekly_metrics():
    df = apply_policy_metrics(
        pd.concat(
            [
                base_ticket(ticket_id="T1"),
                base_ticket(
                    ticket_id="T2",
                    created_at=pd.Timestamp("2025-01-03 10:00"),
                    first_response_at=pd.Timestamp("2025-01-03 11:00"),
                ),
            ],
            ignore_index=True,
        ),
        POLICY,
    )
    weekly = weekly_metrics(df)
    assert int(weekly.loc[0, "total_contacts"]) == 2
    assert int(weekly.loc[0, "sla_breaches"]) == 0


def test_duplicate_reconciliation_prefers_helpdesk_and_keeps_legacy_only():
    df = pd.concat(
        [
            base_ticket(ticket_id="D1", source_system="legacy_fd"),
            base_ticket(ticket_id="D1", source_system="helpdesk"),
            base_ticket(ticket_id="L1", source_system="legacy_fd"),
        ],
        ignore_index=True,
    )
    canonical, quality = reconcile_duplicates(df)
    assert len(canonical) == 2
    assert canonical.loc[canonical.ticket_id == "D1", "source_system"].iloc[0] == "helpdesk"
    assert quality["duplicate_ticket_id_groups"] == 1


def test_tier1_leaderboard_excludes_tier2():
    df = apply_policy_metrics(
        pd.concat(
            [
                base_ticket(ticket_id="T1", agent_id="A1"),
                base_ticket(ticket_id="T2", agent_id="A2"),
            ],
            ignore_index=True,
        ),
        POLICY,
    )
    agents = pd.DataFrame(
        [
            {"agent_id": "A1", "name": "Tier One", "team": "Email Frontline", "tier": "Tier 1"},
            {"agent_id": "A2", "name": "Tier Two", "team": "Email Frontline", "tier": "Tier 2"},
        ]
    )
    lb = tier1_leaderboard(df, agents, POLICY)
    assert set(lb["agent_id"]) == {"A1"}


def test_repeat_contact_proxy():
    df = apply_policy_metrics(
        pd.concat(
            [
                base_ticket(ticket_id="T1", customer_id="C1", created_at=pd.Timestamp("2025-01-01 10:00")),
                base_ticket(ticket_id="T2", customer_id="C1", created_at=pd.Timestamp("2025-01-03 10:00")),
            ],
            ignore_index=True,
        ),
        POLICY,
    )
    repeat = build_repeat_contacts(df)
    assert int(repeat["repeat_contact_7d"].sum()) == 1


def test_opportunity_rows():
    df = apply_policy_metrics(base_ticket(), POLICY)
    repeat = build_repeat_contacts(df)
    opp = build_opportunities(df, repeat, POLICY)
    assert set(opp["opportunity"]) == {
        "SLA breaches",
        "Internal transfers",
        "Repeat-contact proxy (7 days)",
    }


def test_business_goal_uses_historical_weekly_benchmark():
    from pipeline import build_business_goal

    rows = []
    for week, rate in enumerate([0.10, 0.12, 0.14, 0.20, 0.22]):
        created = pd.Timestamp("2025-01-06") + pd.Timedelta(days=7 * week)
        for i in range(50):
            first = created + pd.Timedelta(minutes=5 if i < int(50 * (1 - rate)) else 30)
            rows.append(base_ticket(
                ticket_id=f"T{week}_{i}",
                created_at=created + pd.Timedelta(minutes=i),
                first_response_at=first + pd.Timedelta(minutes=i),
                channel="chat",
            ).iloc[0].to_dict())
    df = apply_policy_metrics(pd.DataFrame(rows), POLICY)
    weekly = weekly_metrics(df)
    goal = build_business_goal(df, weekly, POLICY)
    assert goal["eligible_weeks"] == 5
    assert goal["target_rate"] <= goal["current_rate"]
    assert goal["modeled_quarterly_exposure_reduction_inr"] >= 0
