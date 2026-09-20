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
OUT = ROOT / "outputs" / "wealth_newness_20260919"
OUT.mkdir(parents=True, exist_ok=True)


def z(s):
    return (s - s.mean()) / s.std(ddof=0)


def prepare():
    p, fuel, fines = base.load_data()
    # Approximate household vehicle assets. For multi-car clients the source has
    # vehicle-level prices; rebuild a deduplicated sum rather than using median price.
    raw = pd.read_csv(ROOT / "clients_demographics.csv", sep=";", decimal=",")
    raw["signup"] = pd.to_datetime(raw["subscription_creation_date"])
    v = raw.sort_values(["auto_document_id", "signup"]).drop_duplicates("auto_document_id")
    fleet = v.groupby("client_id").agg(fleet_value=("price", "sum"), priced_cars=("price", "count"))
    p = p.join(fleet)
    p["log_fleet_value"] = np.log1p(p["fleet_value"])
    p["vehicle_age_sq"] = p["vehicle_age"] ** 2

    # Wealth proxy = vehicle assets unusually expensive for the client's region
    # and vehicle age. This prevents a new but mass-market car from mechanically
    # being classified as wealthier solely because it is new.
    use = p.dropna(subset=["log_fleet_value", "vehicle_age", "region"])
    wealth_fit = smf.ols("log_fleet_value ~ vehicle_age + vehicle_age_sq + C(region)", use).fit()
    p.loc[use.index, "wealth_residual"] = wealth_fit.resid
    p["wealth_group"] = pd.qcut(p["wealth_residual"], [0, .25, .75, 1], labels=["низкий", "средний", "высокий"])
    p["wealth_high"] = p["wealth_group"].eq("высокий").astype(int)
    p["wealth_low"] = p["wealth_group"].eq("низкий").astype(int)
    p["age_group"] = pd.cut(p["vehicle_age"], [-1, 5, 10, 100], labels=["новая_0_5", "средняя_6_10", "старая_11плюс"])
    p["new_car"] = p["age_group"].eq("новая_0_5").astype(int)
    p["old_car"] = p["age_group"].eq("старая_11плюс").astype(int)

    panel = base.build_client_period(p, fuel, fines)
    regions = base.region_table(panel)
    regions["physical_shortage"] = (z(-regions["liters_pct_change"]) + z(-regions["tx_pct_change"])) / 2
    panel = panel.merge(regions[["region", "physical_shortage"]], on="region")
    return p, fines, panel, regions


def fit_change_models(p, panel):
    attrs = ["region", "gender", "age", "vehicle_age", "log_price", "cars", "hp", "engine_liters", "fines25", "wealth_high", "wealth_low", "new_car", "old_car"]
    w = panel.pivot(index="client_id", columns="period", values=["liters_30", "fuel_tx_30", "fines_30"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.join(p[attrs]).join(panel.drop_duplicates("client_id").set_index("client_id")[["physical_shortage"]])
    for metric in ["liters_30", "fuel_tx_30", "fines_30"]:
        w[f"delta_{metric}"] = w[f"{metric}_post"] - w[f"{metric}_pre"]
    rows = []
    formula_rhs = (
        "C(region)+physical_shortage*(wealth_high+wealth_low+new_car+old_car)+"
        "C(gender)+C(age)+log_price+cars+hp+engine_liters+fines25"
    )
    for outcome in ["delta_liters_30", "delta_fuel_tx_30", "delta_fines_30"]:
        needed = [outcome, "physical_shortage", "wealth_high", "wealth_low", "new_car", "old_car", "region", "gender", "age", "log_price", "cars", "hp", "engine_liters", "fines25"]
        d = w.dropna(subset=needed)
        fit = smf.ols(f"{outcome} ~ {formula_rhs}", d).fit(cov_type="cluster", cov_kwds={"groups": d["region"]})
        for term in ["physical_shortage:wealth_high", "physical_shortage:wealth_low", "physical_shortage:new_car", "physical_shortage:old_car"]:
            lo, hi = fit.conf_int().loc[term]
            rows.append([outcome, term, fit.params[term], lo, hi, fit.pvalues[term], "difference", "OLS change; region FE"])
    return w, rows


def fit_fine_models(p, fines, panel, rows):
    cols = ["region", "gender", "age", "log_price", "cars", "hp", "engine_liters", "fines25", "wealth_high", "wealth_low", "new_car", "old_car"]
    x = fines.join(p[cols], on="client_id").merge(panel[["region", "physical_shortage"]].drop_duplicates(), on="region")
    # Exclude the transition week so the fine model uses the same clean
    # pre/post windows as the fuel panel: Apr 1–May 24 and Jun 1–Aug 31.
    x = x[x["date"].lt(base.PRE_END) | x["date"].ge(base.POST_START)].copy()
    x["post"] = x["date"].ge(base.POST_START).astype(int)
    rhs = (
        "C(region)*post + wealth_high+wealth_low+new_car+old_car + "
        "post:(wealth_high+wealth_low+new_car+old_car) + "
        "physical_shortage:(wealth_high+wealth_low+new_car+old_car) + "
        "post:physical_shortage:(wealth_high+wealth_low+new_car+old_car) + "
        "C(gender)+C(age)+log_price+cars+hp+engine_liters+fines25"
    )
    for outcome in ["severe_amount", "high_risk"]:
        d = x.dropna()
        fit = smf.glm(f"{outcome} ~ {rhs}", d, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d["region"]})
        for group in ["wealth_high", "wealth_low", "new_car", "old_car"]:
            term = f"post:physical_shortage:{group}"
            lo, hi = fit.conf_int().loc[term]
            rows.append([outcome, term, np.exp(fit.params[term]), np.exp(lo), np.exp(hi), fit.pvalues[term], "OR", "Binomial GLM; region×period FE"])
    return rows


def summaries(p, w):
    profile = p.groupby(["wealth_group", "age_group"], observed=True).agg(
        clients=("region", "size"), fleet_value_median=("fleet_value", "median"), cars_mean=("cars", "mean"), fines25_mean=("fines25", "mean")
    ).reset_index()
    change = w.join(p[["wealth_group", "age_group"]]).groupby(["wealth_group", "age_group"], observed=True).agg(
        clients=("region", "size"), delta_liters=("delta_liters_30", "mean"), delta_tx=("delta_fuel_tx_30", "mean"), delta_fines=("delta_fines_30", "mean")
    ).reset_index()
    return profile, change


def main():
    p, fines, panel, regions = prepare()
    w, rows = fit_change_models(p, panel)
    rows = fit_fine_models(p, fines, panel, rows)
    results = pd.DataFrame(rows, columns=["outcome", "term", "estimate", "ci_low", "ci_high", "p_value", "scale", "model"])
    results["q_value_bh"] = multipletests(results["p_value"], method="fdr_bh")[1]
    profile, change = summaries(p, w)
    results.to_csv(OUT / "wealth_newness_models.csv", index=False)
    profile.to_csv(OUT / "segment_profiles.csv", index=False)
    change.to_csv(OUT / "segment_changes.csv", index=False)
    regions.to_csv(OUT / "physical_shortage_regions.csv", index=False)
    print(results.sort_values("p_value").to_string(index=False))


if __name__ == "__main__":
    main()
