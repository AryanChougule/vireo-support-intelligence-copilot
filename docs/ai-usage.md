# AI Usage

## Gemini

Gemini is optional and is used for:

- business-goal wording;
- one-page executive memo drafting;
- in-depth CX analysis;
- natural-language questions over the structured analytical snapshot;
- interpreting exploratory complaint themes.

The Python pipeline owns numerical truth: ticket counts, duplicate reconciliation, SLA calculations, transfer cost, repeat-contact proxy, leaderboard, data-quality flags, and the business-goal benchmark.

The Gemini API key is entered in the Streamlit session and is not hard-coded or written to project files.

The tool sends a compact analytical snapshot and a small ticket sample for contextual interpretation. It does **not** send the entire raw dataset to Gemini.

## Cost disclosure

No API call is made unless the reviewer enters a Gemini API key and presses a Gemini action. Actual API cost depends on the selected model, token usage, account pricing, and applicable free tier. The app does not claim a fixed cost without a real billing measurement.

## Discarded / narrowed AI work

The project deliberately avoids using Gemini for raw-data arithmetic, timestamp repair, duplicate resolution, agent scoring, or arbitrary target selection. Those approaches were rejected because deterministic code provides a more reproducible source of truth for those tasks.
