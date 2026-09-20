# -*- coding: utf-8 -*-
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.generalized_estimating_equations import GEE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "region_shortage_20260919"
OUT.mkdir(parents=True, exist_ok=True)

PRE_START, PRE_END, PRE_DAYS = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-25"), 54
POST_START, POST_END, POST_DAYS = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-09-01"), 92
REGION_NAMES = {
    77: "Москва", 50: "Московская область", 78: "Санкт-Петербург", 16: "Татарстан",
    66: "Свердловская область", 54: "Новосибирская область", 47: "Ленинградская область",
    52: "Нижегородская область", 23: "Краснодарский край", 63: "Самарская область",
    74: "Челябинская область", 42: "Кемеровская область", 72: "Тюменская область",
    2: "Башкортостан", 24: "Красноярский край", 55: "Омская область",
}


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def load_data():
    c = pd.read_csv(ROOT / "clients_demographics.csv", sep=";", decimal=",")
    f = pd.read_csv(ROOT / "fuel_transaction.csv", sep=";", decimal=",")
    y = pd.read_csv(ROOT / "fines_2026.csv", sep=";")
    c["signup"] = pd.to_datetime(c["subscription_creation_date"])
    f["date"] = pd.to_datetime(f["order_datetime"])
    y["date"] = pd.to_datetime(y["bill_offence_date"])

    c["hp"] = pd.to_numeric(c["engine_type"].str.extract(r"\(([0-9.]+)")[0], errors="coerce")
    c["engine"] = pd.to_numeric(c["engine_type"].str.extract(r"^([0-9.]+)")[0], errors="coerce")
    v = c.sort_values(["auto_document_id", "signup"]).drop_duplicates("auto_document_id")
    fine_cols = [f"{m}_2025_fines" for m in ["april", "may", "jun", "jul", "aug"]]
    v["fines25"] = v[fine_cols].sum(axis=1)
    p = v.groupby("client_id").agg(
        region=("kladr_code", "first"), gender=("gender", "first"), age=("age_type_code", "first"),
        cars=("auto_document_id", "nunique"), price=("price", "median"), auto_year=("auto_year", "median"),
        hp=("hp", "median"), engine_liters=("engine", "median"), fines25=("fines25", "sum"),
    )
    p["region_name"] = p["region"].map(REGION_NAMES)
    p["vehicle_age"] = 2026 - p["auto_year"]
    p["log_price"] = np.log1p(p["price"])
    p["risky25"] = p["fines25"].gt(0).astype(int)
    p["multi_car"] = p["cars"].gt(1).astype(int)
    p["wealth_q"] = pd.qcut(p["price"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")

    f = f[f["date"].between(PRE_START, POST_END, inclusive="left")].copy()
    f = f[f["order_fuel_volume"].between(.01, 120) & f["order_fuel_price_1liter"].between(30, 150)]
    f["spend"] = f["order_fuel_volume"] * f["order_fuel_price_1liter"]
    y = y[y["date"].between(PRE_START, POST_END, inclusive="left")].copy()
    y["severe_amount"] = y["total_fine_amount"].ge(150000).astype(int)
    y["high_risk"] = y["offence_short_statement"].str.contains(
        "40-60|60-80|более чем на 80|красный|телефона|обочине|встречного|пешехода", regex=True
    ).astype(int)
    return p, f, y


def build_client_period(p, fuel, fines):
    rows = []
    for label, start, end, days, post in [
        ("pre", PRE_START, PRE_END, PRE_DAYS, 0), ("post", POST_START, POST_END, POST_DAYS, 1)
    ]:
        base = p.copy()
        fx = fuel[fuel["date"].between(start, end, inclusive="left")].groupby("client_id").agg(
            fuel_tx=("order_id", "size"), liters=("order_fuel_volume", "sum"), spend=("spend", "sum")
        )
        fy = fines[fines["date"].between(start, end, inclusive="left")].groupby("client_id").agg(
            fines=("bill_id", "size"), severe_fines=("severe_amount", "sum"), high_risk_fines=("high_risk", "sum"),
            fine_amount=("total_fine_amount", "sum")
        )
        base = base.join(fx).join(fy)
        for col in ["fuel_tx", "liters", "spend", "fines", "severe_fines", "high_risk_fines", "fine_amount"]:
            base[col] = base[col].fillna(0)
        base["fuel_tx_30"] = base["fuel_tx"] * 30 / days
        base["liters_30"] = base["liters"] * 30 / days
        base["fines_30"] = base["fines"] * 30 / days
        base["price_vw"] = np.where(base["liters"] > 0, base["spend"] / base["liters"], np.nan)
        base["period"] = label
        base["post"] = post
        base["days"] = days
        rows.append(base.reset_index())
    return pd.concat(rows, ignore_index=True)


def region_table(panel):
    g = panel.groupby(["region", "region_name", "period"]).agg(
        clients=("client_id", "nunique"), fuel_tx_30=("fuel_tx_30", "mean"), liters_30=("liters_30", "mean"),
        spend_sum=("spend", "sum"), liters_sum=("liters", "sum"), fines_30=("fines_30", "mean"), fines=("fines", "sum"),
        severe_fines=("severe_fines", "sum"), high_risk_fines=("high_risk_fines", "sum"),
    ).reset_index()
    g["price_vw"] = g["spend_sum"] / g["liters_sum"]
    g["fines_per_1000_liters"] = g["fines"] / g["liters_sum"] * 1000
    g["severe_share"] = g["severe_fines"] / g["fines"]
    g["high_risk_share"] = g["high_risk_fines"] / g["fines"]
    wide = g.pivot(index=["region", "region_name"], columns="period")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for col in ["fuel_tx_30", "liters_30", "price_vw", "fines_30", "fines_per_1000_liters", "severe_share", "high_risk_share"]:
        wide[f"{col}_change"] = wide[f"{col}_post"] - wide[f"{col}_pre"]
    wide["liters_pct_change"] = wide["liters_30_change"] / wide["liters_30_pre"] * 100
    wide["tx_pct_change"] = wide["fuel_tx_30_change"] / wide["fuel_tx_30_pre"] * 100
    # High values mean a larger supply/demand squeeze: falling liters/frequency plus rising price.
    wide["shortage_index"] = (zscore(-wide["liters_pct_change"]) + zscore(-wide["tx_pct_change"]) + zscore(wide["price_vw_change"])) / 3
    wide["shortage_group"] = pd.qcut(wide["shortage_index"], 4, labels=["низкий", "умеренный", "высокий", "очень высокий"])
    return wide.sort_values("shortage_index", ascending=False)


def permutation_spearman(x, y, reps=20000, seed=42):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    observed = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    perm = np.array([stats.spearmanr(x, rng.permutation(y)).statistic for _ in range(reps)])
    p = (1 + np.sum(np.abs(perm) >= abs(observed))) / (reps + 1)
    return observed, p


def models(panel, regions):
    d = panel.merge(regions[["region", "shortage_index", "shortage_group"]], on="region", how="left")
    d["offset_time"] = np.log(d["days"] / 30)
    controls = "C(gender)+C(age)+vehicle_age+log_price+cars+hp+engine_liters+fines25+C(region)"
    results = []

    needed = ["fines", "post", "shortage_index", "gender", "age", "vehicle_age", "log_price", "cars", "hp", "engine_liters", "fines25", "region"]
    use_count = d.dropna(subset=needed).copy()
    fit = smf.glm(
        f"fines ~ post + post:shortage_index + {controls}", data=use_count,
        family=sm.families.Poisson(), offset=use_count["offset_time"]
    ).fit(cov_type="cluster", cov_kwds={"groups": use_count["region"]})
    for term in ["post", "post:shortage_index"]:
        lo, hi = fit.conf_int().loc[term]
        results.append(["fine_count_per_30d", term, np.exp(fit.params[term]), np.exp(lo), np.exp(hi), fit.pvalues[term], "Poisson GLM, region-clustered IRR"])

    # Exposure-adjusted rate. Restrict to client-periods with fuel purchases.
    e = d[d["liters"] > 0].copy()
    e["offset_liters"] = np.log(e["liters"] / 1000)
    e = e.dropna(subset=needed).copy()
    fit2 = smf.glm(
        f"fines ~ post + post:shortage_index + {controls}", data=e,
        family=sm.families.Poisson(), offset=e["offset_liters"]
    ).fit(cov_type="cluster", cov_kwds={"groups": e["region"]})
    for term in ["post", "post:shortage_index"]:
        lo, hi = fit2.conf_int().loc[term]
        results.append(["fines_per_1000_liters", term, np.exp(fit2.params[term]), np.exp(lo), np.exp(hi), fit2.pvalues[term], "Poisson GLM, region-clustered IRR"])

    # Client-level changes and heterogeneity.
    w = d.pivot(index="client_id", columns="period", values=["liters_30", "fuel_tx_30", "fines_30"]).copy()
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.join(d.drop_duplicates("client_id").set_index("client_id")[["shortage_index", "wealth_q", "multi_car", "risky25", "gender", "age", "vehicle_age", "log_price", "cars", "hp", "engine_liters", "region"]])
    for stem in ["liters_30", "fuel_tx_30", "fines_30"]:
        w[f"delta_{stem}"] = w[f"{stem}_post"] - w[f"{stem}_pre"]
    w["wealth_high"] = w["wealth_q"].eq("Q4").astype(int)
    for outcome in ["delta_liters_30", "delta_fuel_tx_30", "delta_fines_30"]:
        needed = [outcome, "shortage_index", "wealth_high", "multi_car", "risky25", "gender", "age", "vehicle_age", "log_price", "cars", "hp", "engine_liters", "region"]
        use = w.dropna(subset=needed).copy()
        ols = smf.ols(
            f"{outcome} ~ shortage_index*(wealth_high + multi_car + risky25) + C(gender)+C(age)+vehicle_age+log_price+cars+hp+engine_liters",
            data=use
        ).fit(cov_type="cluster", cov_kwds={"groups": use["region"]})
        for term in ["shortage_index", "shortage_index:wealth_high", "shortage_index:multi_car", "shortage_index:risky25"]:
            lo, hi = ols.conf_int().loc[term]
            results.append([outcome, term, ols.params[term], lo, hi, ols.pvalues[term], "OLS, region-clustered"])
    return pd.DataFrame(results, columns=["outcome", "term", "estimate", "ci_low", "ci_high", "p_value", "model"]), d, w
def fine_severity_models(fines, p, regions):
    x = fines.join(p[["region", "gender", "age", "vehicle_age", "log_price", "cars", "hp", "engine_liters", "fines25"]], on="client_id")
    x = x.merge(regions[["region", "shortage_index"]], on="region", how="left")
    x["post"] = x["date"].ge(POST_START).astype(int)
    controls = "C(gender)+C(age)+vehicle_age+log_price+cars+hp+engine_liters+fines25+C(region)"
    rows = []
    for outcome in ["severe_amount", "high_risk"]:
        fit = GEE.from_formula(
            f"{outcome} ~ post + post:shortage_index + {controls}", groups="client_id", data=x,
            family=sm.families.Binomial(), cov_struct=Exchangeable()
        ).fit()
        for term in ["post", "post:shortage_index"]:
            lo, hi = fit.conf_int().loc[term]
            rows.append([outcome, term, np.exp(fit.params[term]), np.exp(lo), np.exp(hi), fit.pvalues[term], "Binomial GEE OR"])
    return pd.DataFrame(rows, columns=["outcome", "term", "estimate", "ci_low", "ci_high", "p_value", "model"])


def subgroup_table(panel, regions):
    d = panel.merge(regions[["region", "shortage_group"]], on="region")
    d["wealth"] = np.where(d["wealth_q"].eq("Q4"), "дорогая машина (Q4)", "остальные")
    d["baseline_style"] = np.where(d["risky25"].eq(1), "были штрафы в 2025", "без штрафов в 2025")
    out = []
    for dims in [["shortage_group"], ["shortage_group", "wealth"], ["shortage_group", "multi_car"], ["shortage_group", "baseline_style"]]:
        g = d.groupby(dims + ["period"], observed=True).agg(
            clients=("client_id", "nunique"), liters_30=("liters_30", "mean"), fuel_tx_30=("fuel_tx_30", "mean"), fines_30=("fines_30", "mean")
        ).reset_index()
        keys = dims
        wide = g.pivot(index=keys, columns="period", values=["clients", "liters_30", "fuel_tx_30", "fines_30"])
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        wide = wide.reset_index()
        for stem in ["liters_30", "fuel_tx_30", "fines_30"]:
            wide[f"delta_{stem}"] = wide[f"{stem}_post"] - wide[f"{stem}_pre"]
        wide["breakdown"] = "+".join(dims)
        out.append(wide)
    return pd.concat(out, ignore_index=True, sort=False)


def make_figure(regions):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    specs = [
        ("price_vw_change", "liters_pct_change", "Цена и доступность топлива", "Изменение цены, руб./л", "Изменение литров, %"),
        ("shortage_index", "fines_30_change", "Абсолютное число штрафов", "Индекс дефицита", "Изменение штрафов на 30 дней"),
        ("shortage_index", "fines_per_1000_liters_change", "Штрафы относительно мобильности", "Индекс дефицита", "Изменение штрафов на 1 000 л"),
    ]
    for ax, (x, y, title, xlabel, ylabel) in zip(axes, specs):
        ax.scatter(regions[x], regions[y], s=45, color="#FFCC00", edgecolor="#333333", zorder=3)
        for _, row in regions.iterrows():
            ax.annotate(str(int(row["region"])), (row[x], row[y]), xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.axhline(0, color="#888888", linewidth=.8)
        ax.axvline(0, color="#888888", linewidth=.8)
        ax.grid(alpha=.2)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.suptitle("Региональный топливный шок и нарушения ПДД, апрель–август 2026", fontsize=14, fontweight="bold")
    fig.savefig(OUT / "regional_shortage_and_fines.png", dpi=180)
    plt.close(fig)


def main():
    p, fuel, fines = load_data()
    panel = build_client_period(p, fuel, fines)
    regions = region_table(panel)
    model_results, full_panel, client_changes = models(panel, regions)
    severity = fine_severity_models(fines, p, regions)
    subgroups = subgroup_table(panel, regions)
    make_figure(regions)

    corr_rows = []
    for outcome in ["fines_30_change", "severe_share_change", "high_risk_share_change"]:
        rho, pv = permutation_spearman(regions["shortage_index"], regions[outcome])
        corr_rows.append([outcome, rho, pv, len(regions)])
    correlations = pd.DataFrame(corr_rows, columns=["outcome", "spearman_rho", "permutation_p", "regions"])

    regions.to_csv(OUT / "region_exposure_and_outcomes.csv", index=False)
    model_results.to_csv(OUT / "client_models.csv", index=False)
    severity.to_csv(OUT / "severity_models.csv", index=False)
    correlations.to_csv(OUT / "region_permutation_tests.csv", index=False)
    subgroups.to_csv(OUT / "subgroup_changes.csv", index=False)
    client_changes.reset_index().to_csv(OUT / "client_changes.csv", index=False)

    print(json.dumps({
        "regions": regions[["region", "region_name", "shortage_index", "liters_pct_change", "price_vw_change", "fines_30_change", "severe_share_change"]].to_dict("records"),
        "models": model_results.to_dict("records"), "severity": severity.to_dict("records"),
        "correlations": correlations.to_dict("records"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
