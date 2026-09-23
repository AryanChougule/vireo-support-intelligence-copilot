from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--policy", default="policy.yaml")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = run_pipeline(args.data_dir, args.policy)

    result.weekly_metrics.to_csv(out / "weekly_metrics.csv", index=False)
    result.themes.to_csv(out / "complaint_themes.csv", index=False)
    result.leaderboard.to_csv(out / "tier1_leaderboard.csv", index=False)
    result.opportunities.to_csv(out / "business_opportunities.csv", index=False)
    result.channel_metrics.to_csv(out / "channel_metrics.csv", index=False)
    result.category_metrics.to_csv(out / "category_metrics.csv", index=False)
    result.repeat_contacts.to_csv(out / "repeat_contacts_proxy.csv", index=False)

    (out / "data_quality.json").write_text(
        json.dumps(result.quality, indent=2, default=str), encoding="utf-8"
    )
    (out / "business_goal.json").write_text(
        json.dumps(result.business_goal, indent=2, default=str), encoding="utf-8"
    )

    print(json.dumps({"data_quality": result.quality, "business_goal": result.business_goal}, indent=2, default=str))
    print("\nAnalysis complete. Derived files written to outputs/.")
    print("Raw assessment files were not modified.")


if __name__ == "__main__":
    main()
