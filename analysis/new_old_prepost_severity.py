# -*- coding: utf-8 -*-
"""Pre/post descriptive severity metrics for new and old vehicles."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wealth_newness_hypothesis as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "wealth_newness_20260919"
DAYS = {"pre": 54, "post": 92}


def pct_change(post, pre):
    return np.nan if pre == 0 else (post / pre - 1) * 100


def main():
    people, fines, panel, _ = base.prepare()
    people["vehicle_group"] = np.select(
        [people["vehicle_age"].le(5), people["vehicle_age"].ge(11)],
        ["новые_0_5", "старые_11плюс"],
        default="средние_6_10",
    )
    keep = ["новые_0_5", "старые_11плюс"]
    clients = people[people["vehicle_group"].isin(keep)].groupby("vehicle_group").size()

    y = fines.join(people[["vehicle_group"]], on="client_id")
    y = y[y["vehicle_group"].isin(keep)].copy()
    y = y[y["date"].lt(base.base.PRE_END) | y["date"].ge(base.base.POST_START)].copy()
    y["period"] = np.where(y["date"] < base.base.PRE_END, "pre", "post")
    y["light_risk"] = 1 - y["high_risk"]
    y["light_amount"] = 1 - y["severe_amount"]

    fuel = panel.join(people[["vehicle_group"]], on="client_id")
    fuel = fuel[fuel["vehicle_group"].isin(keep)]
    liters = fuel.groupby(["vehicle_group", "period"])["liters"].sum()

    rows = []
    for group in keep:
        n_clients = int(clients[group])
        for period in ["pre", "post"]:
            d = y[(y["vehicle_group"] == group) & (y["period"] == period)]
            total_liters = float(liters.loc[(group, period)])
            exposure_months = n_clients * DAYS[period] / 30
            for definition, severe_col, light_col in [
                ("опасное_по_типу", "high_risk", "light_risk"),
                ("крупное_по_сумме", "severe_amount", "light_amount"),
            ]:
                severe = int(d[severe_col].sum())
                light = int(d[light_col].sum())
                total = severe + light
                rows.append({
                    "vehicle_group": group,
                    "period": period,
                    "severity_definition": definition,
                    "clients": n_clients,
                    "liters": total_liters,
                    "all_fines": total,
                    "severe_fines": severe,
                    "light_fines": light,
                    "severe_share_pct": severe / total * 100,
                    "light_share_pct": light / total * 100,
                    "severe_per_1000_clients_30d": severe / exposure_months * 1000,
                    "light_per_1000_clients_30d": light / exposure_months * 1000,
                    "severe_per_1000_liters": severe / total_liters * 1000,
                    "light_per_1000_liters": light / total_liters * 1000,
                })
    levels = pd.DataFrame(rows)
    levels.to_csv(OUT / "new_old_severity_prepost_levels.csv", index=False)

    change_rows = []
    metrics = [
        "severe_share_pct", "light_share_pct",
        "severe_per_1000_clients_30d", "light_per_1000_clients_30d",
        "severe_per_1000_liters", "light_per_1000_liters",
    ]
    for (group, definition), d in levels.groupby(["vehicle_group", "severity_definition"]):
        d = d.set_index("period")
        row = {"vehicle_group": group, "severity_definition": definition}
        for metric in metrics:
            pre, post = d.loc["pre", metric], d.loc["post", metric]
            row[f"{metric}_pre"] = pre
            row[f"{metric}_post"] = post
            row[f"{metric}_change"] = post - pre
            row[f"{metric}_pct_change"] = pct_change(post, pre)
        change_rows.append(row)
    changes = pd.DataFrame(change_rows)
    changes.to_csv(OUT / "new_old_severity_prepost_changes.csv", index=False)
    print(changes.to_string(index=False))


if __name__ == "__main__":
    main()
