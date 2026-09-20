# -*- coding: utf-8 -*-
"""Sensitivity checks: new cars versus old cars under physical fuel shortage."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wealth_newness_hypothesis as base


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "wealth_newness_20260919"


def main():
    p, fines, panel, _ = base.prepare()
    attrs = [
        "region", "gender", "age", "log_price", "cars", "hp", "engine_liters",
        "fines25", "wealth_high", "wealth_low", "vehicle_age",
    ]
    changes = panel.pivot(index="client_id", columns="period", values=["liters_30", "fuel_tx_30"])
    changes.columns = [f"{metric}_{period}" for metric, period in changes.columns]
    changes = changes.join(p[attrs]).join(
        panel.drop_duplicates("client_id").set_index("client_id")[["physical_shortage"]]
    )
    changes["delta_liters_30"] = changes["liters_30_post"] - changes["liters_30_pre"]
    changes["delta_fuel_tx_30"] = changes["fuel_tx_30_post"] - changes["fuel_tx_30_pre"]

    fine_cols = attrs + ["wealth_high", "wealth_low"]
    fine_cols = list(dict.fromkeys(fine_cols))
    x = fines.join(p[fine_cols], on="client_id").merge(
        panel[["region", "physical_shortage"]].drop_duplicates(), on="region"
    )
    x = x[x["date"].lt(base.base.PRE_END) | x["date"].ge(base.base.POST_START)].copy()
    x["post"] = x["date"].ge(base.base.POST_START).astype(int)

    rows = []
    for cutoff in [3, 5, 7]:
        # Compare clearly new cars with clearly old cars; omit the middle group.
        for d in [changes, x]:
            d[f"new_{cutoff}"] = d["vehicle_age"].le(cutoff).astype(int)
        change_use = changes[(changes["vehicle_age"] <= cutoff) | (changes["vehicle_age"] >= 11)].copy()
        fine_use = x[(x["vehicle_age"] <= cutoff) | (x["vehicle_age"] >= 11)].copy()
        group = f"new_{cutoff}"

        rhs_change = (
            f"C(region)+physical_shortage*{group}+C(gender)+C(age)+log_price+cars+"
            "hp+engine_liters+fines25+wealth_high+wealth_low"
        )
        for outcome in ["delta_liters_30", "delta_fuel_tx_30"]:
            d = change_use.dropna().copy()
            fit = smf.ols(f"{outcome} ~ {rhs_change}", d).fit(
                cov_type="cluster", cov_kwds={"groups": d["region"]}
            )
            term = f"physical_shortage:{group}"
            lo, hi = fit.conf_int().loc[term]
            rows.append([cutoff, outcome, term, fit.params[term], lo, hi, fit.pvalues[term], "difference"])

        rhs_fine = (
            f"C(region)*post + {group}+post:{group}+physical_shortage:{group}+"
            f"post:physical_shortage:{group}+C(gender)+C(age)+log_price+cars+hp+"
            "engine_liters+fines25+wealth_high+wealth_low"
        )
        for outcome in ["high_risk", "severe_amount"]:
            d = fine_use.dropna().copy()
            fit = smf.glm(f"{outcome} ~ {rhs_fine}", d, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": d["region"]}
            )
            term = f"post:physical_shortage:{group}"
            lo, hi = fit.conf_int().loc[term]
            rows.append([
                cutoff, outcome, term, np.exp(fit.params[term]), np.exp(lo), np.exp(hi),
                fit.pvalues[term], "OR",
            ])

    result = pd.DataFrame(
        rows, columns=["new_age_cutoff", "outcome", "term", "estimate", "ci_low", "ci_high", "p_value", "scale"]
    )
    result["q_value_bh"] = multipletests(result["p_value"], method="fdr_bh")[1]
    result.to_csv(OUT / "new_vs_old_sensitivity.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
