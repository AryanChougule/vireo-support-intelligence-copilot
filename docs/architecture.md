# Architecture

```text
Raw assessment CSVs
        |
        v
Schema + timestamp normalization
        |
        v
Duplicate audit/reconciliation
        |
        v
Policy-driven deterministic analytics
   |        |        |        |
   v        v        v        v
Weekly   Themes   Leaderboard Opportunities
metrics  (NLP)              |
   \_________________________/
             |
             v
       Streamlit dashboard
             |
             +---- optional Gemini layer
                    |
                    +-- in-depth analysis
                    +-- natural-language queries
```

## Design rule

Python owns numerical truth:

- counts
- rates
- SLA
- costs
- leaderboard
- duplicate reconciliation
- temporal validation

Gemini owns language tasks:

- interpretation
- synthesis
- business narrative
- natural-language answers

The raw source files are never modified.
