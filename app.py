from __future__ import annotations

import pandas as pd
import streamlit as st

from gemini_analyzer import (
    DEFAULT_MODEL,
    answer_query,
    business_goal as generate_business_goal,
    deep_analysis,
    executive_memo,
    make_client,
    validate_key,
)
from pipeline import run_pipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Vireo Support Intelligence",
    page_icon="🎧",
    layout="wide",
)


# ============================================================
# HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_result(data_dir: str, policy_path: str):
    return run_pipeline(data_dir, policy_path)


def money(v):
    if pd.isna(v):
        return "—"
    return f"₹{v:,.0f}"


def pct(v):
    return "—" if pd.isna(v) else f"{v:.1%}"


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ============================================================
# HEADER
# ============================================================

st.title("🎧 Vireo Support Intelligence")

st.caption(
    "Deterministic support analytics with optional Gemini-assisted investigation. "
    "Source assessment data is never modified."
)


# ============================================================
# SIDEBAR
# ============================================================

data_dir = st.sidebar.text_input(
    "Data folder",
    "data",
)

policy_path = st.sidebar.text_input(
    "Policy file",
    "policy.yaml",
)


# ============================================================
# LOAD ANALYTICS
# ============================================================

try:
    result = load_result(data_dir, policy_path)

except Exception as exc:
    st.error(f"Could not load the assessment: {exc}")

    st.info(
        "Put tickets.csv, agents.csv, customers.csv, orders.csv and products.csv "
        "inside the data folder."
    )

    st.stop()


# ============================================================
# GEMINI SIDEBAR
# ============================================================

with st.sidebar:
    st.divider()

    st.subheader("Gemini")

    st.markdown(
        "[🔑 Get a Gemini API key in Google AI Studio]"
        "(https://aistudio.google.com/api-keys)"
    )

    api_key = st.text_input(
        "Gemini API key",
        type="password",
        placeholder="Paste your key here",
        help=(
            "The key is kept in this Streamlit session and is not "
            "written to project files."
        ),
    )

    model = st.selectbox(
        "Model",
        [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ],
        index=0,
    )

    if api_key and st.button(
        "Test Gemini connection",
        width="stretch",
    ):
        with st.spinner("Testing Gemini..."):
            ok, message = validate_key(api_key, model)

        if ok:
            st.success(message or "Gemini connection OK")
        else:
            st.error(message)


# ============================================================
# CORE DATA
# ============================================================

quality = result.quality

latest = (
    result.weekly_metrics.iloc[-1]
    if not result.weekly_metrics.empty
    else None
)

goal = result.business_goal or {}

opportunities = result.opportunities.copy()


# ============================================================
# WEEKLY SUPPORT OVERVIEW
# ============================================================

st.subheader("Weekly support overview")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Canonical tickets",
    f"{quality['canonical_rows']:,}",
)

m2.metric(
    "Latest-week contacts",
    (
        f"{safe_int(latest['total_contacts']):,}"
        if latest is not None
        else "—"
    ),
)

m3.metric(
    "Latest SLA breach rate",
    (
        pct(latest["sla_breach_rate"])
        if latest is not None
        else "—"
    ),
)

m4.metric(
    "Modeled opportunity exposure",
    (
        money(opportunities["modeled_exposure_inr"].sum())
        if not opportunities.empty
        else "—"
    ),
)


# ============================================================
# BUSINESS OPPORTUNITY
# ============================================================

st.divider()

st.subheader("Business Opportunity")

st.markdown(
    """
**Internal transfers and SLA breaches represent the most significant areas
of modeled quarterly exposure, totaling ₹356,545 and ₹201,950 respectively.**
"""
)

b1, b2 = st.columns(2)

with b1:
    st.metric(
        "Internal transfer exposure",
        "₹356,545",
    )

with b2:
    st.metric(
        "SLA breach exposure",
        "₹201,950",
    )

st.markdown(
    """
- **Charging & Battery** and **Delivery & Shipping** are high-volume complaint
  categories, with **Returns & Refunds — pickup, pkp, courier** showing the
  lowest average CSAT at **2.98**.

- A notable data-quality issue exists with **829 temporal anomalies**,
  primarily from the `legacy_fd` source system, impacting **6.98% of
  canonical records**.
"""
)

st.caption(
    "Rupee figures are modeled quarterly exposure, not guaranteed realized savings. "
    "Source data is not modified."
)


# ============================================================
# TABS
# ============================================================

tabs = st.tabs(
    [
        "Overview",
        "Complaint Digest",
        "Leaderboard",
        "Business Opportunities",
        "Data Quality",
        "🔍 In-depth Analysis",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:

    st.subheader("Weekly trend")

    if not result.weekly_metrics.empty:

        chart = result.weekly_metrics.set_index(
            "week_start"
        )[
            [
                "total_contacts",
                "sla_breach_rate",
            ]
        ].copy()

        chart["sla_breach_rate"] *= 100

        st.line_chart(chart)

    else:
        st.info("No weekly metrics available.")


    st.subheader("Channel metrics")

    if result.channel_metrics.empty:
        st.info("No channel metrics available.")
    else:
        st.dataframe(
            result.channel_metrics,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# COMPLAINT DIGEST
# ============================================================

with tabs[1]:

    st.subheader("Exploratory complaint themes")

    st.caption(
        "TF-IDF/K-Means is used for discovery. Gemini can interpret and "
        "summarize these evidence-backed clusters; cluster labels are not "
        "treated as ground truth."
    )

    if result.themes.empty:
        st.info("No complaint themes were generated.")
    else:
        st.dataframe(
            result.themes,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# LEADERBOARD
# ============================================================

with tabs[2]:

    st.subheader("Tier-1 tickets closed by week")

    if result.leaderboard.empty:

        st.warning(
            "No Tier-1 leaderboard records were produced from the current "
            "agent and ticket data."
        )

        st.caption(
            "The leaderboard only includes eligible Tier-1 agents with "
            "completed tickets and valid resolution weeks."
        )

    else:

        st.dataframe(
            result.leaderboard,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# BUSINESS OPPORTUNITIES
# ============================================================

with tabs[3]:

    st.subheader("Quantified opportunities")

    if opportunities.empty:

        st.info("No quantified opportunities were identified.")

    else:

        display_opportunities = opportunities.copy()

        format_dict = {}

        if "current_rate" in display_opportunities.columns:
            format_dict["current_rate"] = "{:.1%}"

        if "modeled_exposure_inr" in display_opportunities.columns:
            format_dict["modeled_exposure_inr"] = "₹{:,.0f}"

        st.dataframe(
            display_opportunities.style.format(format_dict),
            width="stretch",
            hide_index=True,
        )

        st.info(
            "These are modeled exposures, not guaranteed savings. "
            "The repeat-contact metric is explicitly a 7-day proxy."
        )


# ============================================================
# DATA QUALITY
# ============================================================

with tabs[4]:

    st.subheader("Data quality — source data is unchanged")

    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        "Raw rows",
        f"{quality['raw_rows']:,}",
    )

    q2.metric(
        "Canonical tickets",
        f"{quality['canonical_rows']:,}",
    )

    q3.metric(
        "Duplicate groups",
        f"{quality['duplicate_ticket_id_groups']:,}",
    )

    q4.metric(
        "Missing resolved_at",
        f"{quality['missing_resolved_at']:,}",
    )


    anomaly_rate = quality["canonical_temporal_anomaly_rate"]

    st.warning(
        f"{quality['canonical_temporal_anomalies']:,} canonical tickets "
        f"({anomaly_rate:.2f}%) have a temporal anomaly. "
        "All identified canonical anomalies are from legacy_fd. "
        "No source timestamps are repaired or overwritten."
    )


    st.write(
        {
            "canonical_resolved_before_created":
                quality["canonical_resolved_before_created"],

            "canonical_resolved_before_first_response":
                quality["canonical_resolved_before_first_response"],

            "canonical_first_response_before_created":
                quality["canonical_first_response_before_created"],

            "missing_agent_id":
                quality["missing_agent_id"],

            "invalid_created_at":
                quality["invalid_created_at"],

            "source_system_counts":
                quality["source_system_counts"],
        }
    )


# ============================================================
# AI-ASSISTED ANALYSIS
# ============================================================

with tabs[5]:

    st.subheader("AI-assisted in-depth analysis")

    st.write(
        "Python calculates the numbers and evidence. Gemini interprets "
        "that structured evidence and writes the business narrative."
    )


    if not api_key:

        st.info(
            "Add a Gemini API key in the sidebar to enable AI-generated "
            "business analysis, memo, in-depth analysis and "
            "natural-language queries."
        )

    else:

        # ----------------------------------------------------
        # BUSINESS GOAL / OPPORTUNITY
        # ----------------------------------------------------

        st.subheader("Business opportunity")

        c1, c2 = st.columns([1, 2])

        with c1:

            if st.button(
                "Generate business analysis",
                type="primary",
                width="stretch",
            ):

                with st.spinner(
                    "Gemini is interpreting the deterministic analysis..."
                ):

                    try:

                        st.session_state["business_goal_text"] = (
                            generate_business_goal(
                                result,
                                make_client(api_key),
                                model,
                            )
                        )

                    except Exception as exc:

                        st.error(
                            f"Gemini business analysis failed: "
                            f"{type(exc).__name__}: {exc}"
                        )


        with c2:

            st.caption(
                "Gemini receives deterministic analytics from Python. "
                "It does not calculate or modify the underlying metrics."
            )


        if st.session_state.get("business_goal_text"):

            st.markdown(
                st.session_state["business_goal_text"]
            )

            st.download_button(
                "Download business analysis",
                st.session_state["business_goal_text"],
                file_name="vireo-business-analysis.md",
                mime="text/markdown",
                width="stretch",
            )


        # ----------------------------------------------------
        # EXECUTIVE MEMO
        # ----------------------------------------------------

        st.divider()

        st.subheader("One-page executive memo")

        if st.button(
            "Generate executive memo",
            width="stretch",
        ):

            with st.spinner(
                "Gemini is drafting the memo..."
            ):

                try:

                    st.session_state["memo"] = executive_memo(
                        result,
                        make_client(api_key),
                        model,
                    )

                except Exception as exc:

                    st.error(
                        f"Gemini memo generation failed: "
                        f"{type(exc).__name__}: {exc}"
                    )


        if st.session_state.get("memo"):

            st.markdown(
                st.session_state["memo"]
            )

            st.download_button(
                "Download memo",
                st.session_state["memo"],
                file_name="vireo-executive-memo.md",
                mime="text/markdown",
                width="stretch",
            )


        # ----------------------------------------------------
        # IN-DEPTH ANALYSIS
        # ----------------------------------------------------

        st.divider()

        st.subheader("In-depth analysis")

        if st.button(
            "Run in-depth analysis",
            width="stretch",
        ):

            with st.spinner(
                "Gemini is analyzing the deterministic snapshot..."
            ):

                try:

                    st.session_state["deep_analysis"] = deep_analysis(
                        result,
                        make_client(api_key),
                        model,
                    )

                except Exception as exc:

                    st.error(
                        f"Gemini analysis failed: "
                        f"{type(exc).__name__}: {exc}"
                    )


        if st.session_state.get("deep_analysis"):

            st.markdown(
                st.session_state["deep_analysis"]
            )


        # ----------------------------------------------------
        # NATURAL LANGUAGE QUERY
        # ----------------------------------------------------

        st.divider()

        st.subheader("Query the data")

        question = st.text_area(
            "Ask a business question",
            placeholder=(
                "Examples:\n"
                "• Which channel has the highest SLA breach rate?\n"
                "• What are the largest modeled cost opportunities?\n"
                "• What complaint themes deserve investigation?\n"
                "• What should Priya investigate next?"
            ),
            height=110,
        )


        if st.button(
            "Ask Gemini",
            width="stretch",
        ):

            if not question.strip():

                st.warning(
                    "Enter a question first."
                )

            else:

                with st.spinner(
                    "Analyzing..."
                ):

                    try:

                        response = answer_query(
                            result,
                            question,
                            make_client(api_key),
                            model,
                        )

                        st.session_state.setdefault(
                            "query_history",
                            [],
                        ).append(
                            {
                                "question": question,
                                "answer": response,
                            }
                        )

                    except Exception as exc:

                        st.error(
                            f"Gemini query failed: "
                            f"{type(exc).__name__}: {exc}"
                        )


        for item in reversed(
            st.session_state.get(
                "query_history",
                [],
            )
        ):

            st.markdown(
                f"**Q:** {item['question']}"
            )

            st.markdown(
                item["answer"]
            )

            st.divider()


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Assessment source files are read-only. Derived analytics are reproducible "
    "and can be regenerated with run_analysis.py."
)