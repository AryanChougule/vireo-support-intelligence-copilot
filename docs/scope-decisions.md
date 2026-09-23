# Scope Decisions

## Changed / narrowed

- **Agent leaderboard:** kept the requested tickets-closed/week leaderboard instead of creating an opaque overall agent score. This preserves the client's explicit ask and avoids pretending CSAT/SLA are directly comparable to throughput.
- **Data repair:** deliberately did not repair timestamps or overwrite source rows. The dataset's temporal anomalies are shown as data-quality findings.
- **Business target:** did not hard-code an arbitrary target such as 15%. The target is derived from Vireo's own historical weekly distribution.
- **AI role:** Gemini does not own arithmetic or target selection. It writes and interprets deterministic evidence.

## Deliberately omitted

- real-time ingestion;
- production authentication;
- predictive staffing/forecasting;
- automated intervention workflows;
- a fully autonomous SQL agent;
- production observability and deployment infrastructure.

These were omitted to keep the assessment artifact small, reproducible, and focused on the stated weekly support-intelligence use case.
