# -*- coding: utf-8 -*-
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import region_shortage_hypotheses as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "physical_shortage_20260919"
OUT.mkdir(parents=True, exist_ok=True)


def z(s):
    return (s - s.mean()) / s.std(ddof=0)


def fit_ols_change(w, outcome, group):
    needed = [outcome, "region", "physical_shortage", group, "gender", "age", "vehicle_age", "log_price", "cars", "hp", "engine_liters", "fines25"]
    d = w.dropna(subset=needed).copy()
    formula = (
        f"{outcome} ~ C(region) + {group} + physical_shortage:{group} + "
        "C(gender)+C(age)+vehicle_age+log_price+cars+hp+engine_liters+fines25"
    )
    fit = smf.ols(formula, d).fit(cov_type="cluster", cov_kwds={"groups": d["region"]})
    term = f"physical_shortage:{group}"
    lo, hi = fit.conf_int().loc[term]
    return [outcome, group, term, fit.params[term], lo, hi, fit.pvalues[term], "OLS change; region FE; region-clustered"]


def fit_count_long(d, group, exposure):
    required = ["fines", "post", "region", "physical_shortage", group, "gender", "age", "vehicle_age", "log_price", "cars", "hp", "engine_liters", "fines25", exposure]
    use = d.dropna(subset=required).copy()
    use = use[use[exposure] > 0]
    if exposure == "days":
        use["offset"] = np.log(use["days"] / 30)
        outcome = "fines_per_30d"
    else:
        use["offset"] = np.log(use["liters"] / 1000)
        outcome = "fines_per_1000_liters"
    # Region×period absorbs arbitrary changes in camera/enforcement intensity.
    # The triple interaction asks whether the client group changes differently
    # as physical shortage becomes more severe, within each region-period cell.
    formula = (
        f"fines ~ C(region)*post + {group} + post:{group} + physical_shortage:{group} + "
        f"post:physical_shortage:{group} + C(gender)+C(age)+vehicle_age+log_price+cars+hp+engine_liters+fines25"
    )
    fit = smf.glm(formula, use, family=sm.families.Poisson(), offset=use["offset"]).fit(
        cov_type="cluster", cov_kwds={"groups": use["region"]}
    )
    term = f"post:physical_shortage:{group}"
    lo, hi = fit.conf_int().loc[term]
    return [outcome, group, term, np.exp(fit.params[term]), np.exp(lo), np.exp(hi), fit.pvalues[term], "Poisson GLM; region×period FE; region-clustered IRR"]


def fit_severity(fines, profile, regions, group, outcome):
    cols = ["region", "gender", "age", "vehicle_age", "log_price", "cars", "hp", "engine_liters", "fines25", group]
    x = fines.join(profile[cols], on="client_id").merge(regions[["region", "physical_shortage"]], on="region")
    x["post"] = x["date"].ge(base.POST_START).astype(int)
    required = [outcome, "post", "region", "physical_shortage", group, "gender", "age", "vehicle_age", "log_price", "cars", "hp", "engine_liters", "fines25"]
    x = x.dropna(subset=required)
    formula = (
        f"{outcome} ~ C(region)*post + {group} + post:{group} + physical_shortage:{group} + "
        f"post:physical_shortage:{group} + C(gender)+C(age)+vehicle_age+log_price+cars+hp+engine_liters+fines25"
    )
    fit = smf.glm(formula, x, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": x["region"]}
    )
    term = f"post:physical_shortage:{group}"
    lo, hi = fit.conf_int().loc[term]
    return [outcome, group, term, np.exp(fit.params[term]), np.exp(lo), np.exp(hi), fit.pvalues[term], "Binomial GLM; region×period FE; region-clustered OR"]


def raw_group_changes(panel, regions):
    d = panel.merge(regions[["region", "physical_shortage", "physical_group"]], on="region")
    groups = ["wealth_high", "risky25", "multi_car", "powerful_car", "new_car"]
    parts = []
    for group in groups:
        g = d.groupby(["physical_group", group, "period"], observed=True).agg(
            clients=("client_id", "nunique"), liters_30=("liters_30", "mean"), fuel_tx_30=("fuel_tx_30", "mean"), fines_30=("fines_30", "mean")
        ).reset_index()
        q = g.pivot(index=["physical_group", group], columns="period", values=["clients", "liters_30", "fuel_tx_30", "fines_30"])
        q.columns = [f"{a}_{b}" for a, b in q.columns]
        q = q.reset_index()
        for metric in ["liters_30", "fuel_tx_30", "fines_30"]:
            q[f"delta_{metric}"] = q[f"{metric}_post"] - q[f"{metric}_pre"]
        q["group_variable"] = group
        parts.append(q)
    return pd.concat(parts, ignore_index=True, sort=False)


def main():
    profile, fuel, fines = base.load_data()
    profile["wealth_high"] = profile["wealth_q"].eq("Q4").astype(int)
    profile["powerful_car"] = profile["hp"].ge(profile["hp"].quantile(.75)).astype(int)
    profile["new_car"] = profile["vehicle_age"].le(5).astype(int)
    panel = base.build_client_period(profile, fuel, fines)
    regions = base.region_table(panel)
    regions["physical_shortage"] = (z(-regions["liters_pct_change"]) + z(-regions["tx_pct_change"])) / 2
    regions["physical_group"] = pd.qcut(regions["physical_shortage"], 4, labels=["низкий", "умеренный", "высокий", "очень высокий"])
    d = panel.merge(regions[["region", "physical_shortage", "physical_group"]], on="region")

    profile_cols = ["region", "gender", "age", "vehicle_age", "log_price", "cars", "hp", "engine_liters", "fines25", "wealth_high", "risky25", "multi_car", "powerful_car", "new_car"]
    w = d.pivot(index="client_id", columns="period", values=["liters_30", "fuel_tx_30", "fines_30"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.join(profile[profile_cols]).join(regions.set_index("region")[["physical_shortage"]], on="region")
    for metric in ["liters_30", "fuel_tx_30", "fines_30"]:
        w[f"delta_{metric}"] = w[f"{metric}_post"] - w[f"{metric}_pre"]

    groups = ["wealth_high", "risky25", "multi_car", "powerful_car", "new_car"]
    rows = []
    for group in groups:
        for outcome in ["delta_liters_30", "delta_fuel_tx_30", "delta_fines_30"]:
            rows.append(fit_ols_change(w, outcome, group))
        rows.append(fit_count_long(d, group, "days"))
        rows.append(fit_count_long(d, group, "liters"))
        for outcome in ["severe_amount", "high_risk"]:
            rows.append(fit_severity(fines, profile, regions, group, outcome))
    results = pd.DataFrame(rows, columns=["outcome", "group", "term", "estimate", "ci_low", "ci_high", "p_value", "model"])
    results["q_value_bh"] = multipletests(results["p_value"], method="fdr_bh")[1]

    regions.to_csv(OUT / "physical_shortage_by_region.csv", index=False)
    results.to_csv(OUT / "heterogeneity_models.csv", index=False)
    raw_group_changes(panel, regions).to_csv(OUT / "group_changes.csv", index=False)
    w.reset_index().to_csv(OUT / "client_changes.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
