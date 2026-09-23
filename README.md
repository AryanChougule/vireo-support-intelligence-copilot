# Vireo Support Intelligence Copilot

A small, reproducible support-analytics tool for the Vireo Audio Task 1 assessment.

## What it does

- Weekly support-volume and SLA analysis
- Exploratory complaint-theme discovery
- Tier-1 Chat/Email/Voice Frontline tickets-closed leaderboard
- Policy-driven contact/transfer/SLA exposure
- 7-day repeat-contact proxy
- Explicit data-quality reporting
- Evidence-backed business-goal calculation
- Gemini-generated business-goal narrative
- Gemini-generated one-page executive memo
- Gemini in-depth analysis
- Natural-language questions over the analytical snapshot

## Core design principle

**Python owns numerical truth. Gemini owns language and synthesis.**

The deterministic pipeline calculates counts, rates, costs, duplicate reconciliation, timestamp anomalies, leaderboard results, and the business-goal benchmark. Gemini is only given that structured evidence plus a small contextual ticket sample.

## Source-data rule

**The supplied assessment data is never modified.**

Duplicate reconciliation and temporal validation are derived analytics only. Source timestamps remain untouched. Records with invalid elapsed-time relationships are flagged; their invalid duration values are not repaired.

## Quick start — Windows

Put the supplied assessment files in `data\\`.

Then double-click:

```text
setup_and_run.bat
```

The script:

1. Finds a Python installation.
2. Creates `.venv`.
3. Installs `requirements.txt`.
4. Runs the deterministic analysis.
5. Runs the test suite.
6. Starts Streamlit.

Python itself must already be installed.

## Manual start

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python run_analysis.py --data-dir data
python -m pytest -q
streamlit run app.py
```

## Gemini

Gemini is optional. Open the app and enter a Gemini API key in the sidebar.

The AI section provides four actions:

1. **Generate business goal** — turns the deterministic benchmark into executive wording.
2. **Generate executive memo** — creates the one-page memo to Priya Raman.
3. **Run in-depth analysis** — interprets the complete analytical snapshot.
4. **Ask Gemini** — answers natural-language business questions using the supplied snapshot.

The key is not written to the project.

## Business goal logic

The tool does not invent a target such as 15%. It derives the target from historical Vireo performance: the median SLA breach rate among sufficiently sized weeks at or below the historical 25th percentile. The quarterly rupee figure uses average weekly volume × 13 weeks × the policy SLA-miss credit.

See `docs/business-goal.md`.

## Outputs

Derived files are written to `outputs/`:

- `weekly_metrics.csv`
- `complaint_themes.csv`
- `tier1_leaderboard.csv`
- `business_opportunities.csv`
- `channel_metrics.csv`
- `category_metrics.csv`
- `repeat_contacts_proxy.csv`
- `data_quality.json`
- `business_goal.json`

Raw assessment files are excluded from Git by `.gitignore`.

## Known limitations

- Complaint clustering is exploratory, not a manually verified taxonomy.
- Repeat contact is a 7-day same-customer proxy, not proof of avoidable repeat issues.
- Modeled exposure is not guaranteed savings.
- Gemini answers only from the structured analytical snapshot; it is not an arbitrary SQL interface.
- Temporal anomalies in `legacy_fd` are surfaced rather than silently repaired.
