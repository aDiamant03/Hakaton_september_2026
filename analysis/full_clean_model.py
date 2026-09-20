# -*- coding: utf-8 -*-
"""Full mathematical model on the clean datasets.

Produces transparent OLS/LPM estimates with cluster-robust standard errors.
The estimand is the change after the shortage in new versus old vehicles,
with the change allowed to vary by a standardized regional physical-shortage
index.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, t

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "fuel_cleaning_20260919"
OUT = ROOT / "outputs" / "wealth_newness_clean_20260920"
OUT.mkdir(exist_ok=True)

PRE_START = pd.Timestamp("2026-04-01")
PRE_END = pd.Timestamp("2026-05-25")
POST_START = pd.Timestamp("2026-06-01")
POST_END = pd.Timestamp("2026-09-01")
DAYS = {"pre": (PRE_END - PRE_START).days, "post": (POST_END - POST_START).days}


def load_clean():
    c = pd.read_csv(SRC / "clients_demographics_clean.csv", sep=";", decimal=",")
    fuel = pd.read_csv(SRC / "fuel_transaction_clean.csv", sep=";", decimal=",")
    fines = pd.read_csv(SRC / "fines_2026_clean.csv", sep=";")
    c["signup"] = pd.to_datetime(c["subscription_creation_date"], errors="coerce")
    fuel["date"] = pd.to_datetime(fuel["order_datetime"], errors="coerce")
    fines["date"] = pd.to_datetime(fines["bill_offence_date"], errors="coerce")
    return c, fuel, fines


def make_people(c):
    v = c.sort_values(["auto_document_id", "signup"]).drop_duplicates("auto_document_id")
    fine_cols = [f"{m}_2025_fines" for m in ["april", "may", "jun", "jul", "aug"]]
    v["fines25_row"] = v[fine_cols].sum(axis=1, min_count=1)
    people = v.groupby("client_id").agg(
        region=("kladr_code", "first"),
        gender=("gender", "first"),
        age_type_code=("age_type_code", "first"),
        auto_year=("auto_year", "median"),
        price=("price", "median"),
        cars=("auto_document_id", "nunique"),
        fines25=("fines25_row", "sum"),
    )
    people["vehicle_age"] = 2026 - people["auto_year"]
    people["log_price"] = np.log1p(pd.to_numeric(people["price"], errors="coerce"))
    people["vehicle_group"] = np.select(
        [people["vehicle_age"].le(5), people["vehicle_age"].ge(11)],
        ["new", "old"], default="middle",
    )
    return people


def dangerous_flags(fines):
    txt = fines["offence_short_statement"].fillna("")
    fines["dangerous"] = txt.str.contains(
        r"40-60|60-80|более чем на 80|красн|телефон|обочин|встречн|пешеход",
        case=False, regex=True,
    ).astype(int)
    fines["large_amount"] = pd.to_numeric(fines["total_fine_amount"], errors="coerce").ge(150000).astype(int)
    fines["period"] = np.select(
        [fines["date"].between(PRE_START, PRE_END, inclusive="left"), fines["date"].between(POST_START, POST_END, inclusive="left")],
        ["pre", "post"], default=None,
    )
    return fines[fines["period"].notna()].copy()


def region_shortage(people, fuel):
    keep = people[people["vehicle_group"].isin(["new", "old"])].copy()
    rows = []
    for period, start, end in [("pre", PRE_START, PRE_END), ("post", POST_START, POST_END)]:
        d = fuel[fuel["date"].between(start, end, inclusive="left")].merge(
            keep[["region"]].reset_index(), on="client_id", how="inner"
        )
        agg = d.groupby("region").agg(liters=("order_fuel_volume", "sum"), transactions=("order_id", "size"))
        n = keep.groupby("region").size().rename("clients")
        z = pd.concat([n, agg], axis=1).fillna(0)
        z["liters_30"] = z["liters"] / z["clients"] * 30 / DAYS[period]
        z["transactions_30"] = z["transactions"] / z["clients"] * 30 / DAYS[period]
        z["period"] = period
        rows.append(z.reset_index())
    reg = pd.concat(rows, ignore_index=True)
    wide = reg.pivot(index="region", columns="period", values=["liters_30", "transactions_30"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["liters_pct_change"] = (wide["liters_30_post"] / wide["liters_30_pre"] - 1) * 100
    wide["transactions_pct_change"] = (wide["transactions_30_post"] / wide["transactions_30_pre"] - 1) * 100
    def zscore(x):
        return (x - x.mean()) / x.std(ddof=0)
    wide["shortage_index"] = (zscore(-wide["liters_pct_change"]) + zscore(-wide["transactions_pct_change"])) / 2
    wide["shortage_std"] = wide["shortage_index"]
    wide.to_csv(OUT / "region_shortage_clean_full.csv", index=False)
    return wide[["region", "shortage_std"]]


def design_matrix(df, outcome, with_period=True):
    d = df.copy()
    d["post"] = (d["period"] == "post").astype(float) if with_period else 0.0
    d["new"] = (d["vehicle_group"] == "new").astype(float)
    d["post_new"] = d["post"] * d["new"]
    d["post_shortage"] = d["post"] * d["shortage_std"]
    d["new_shortage"] = d["new"] * d["shortage_std"]
    d["post_new_shortage"] = d["post"] * d["new"] * d["shortage_std"]
    base = ["post", "new", "post_new", "post_shortage", "new_shortage", "post_new_shortage", "vehicle_age", "log_price", "cars", "fines25"]
    x = d[base].astype(float).copy()
    for col in ["region", "gender", "age_type_code"]:
        dum = pd.get_dummies(d[col].astype(str), prefix=col, drop_first=True, dtype=float)
        x = pd.concat([x, dum], axis=1)
    x.insert(0, "const", 1.0)
    mask = x.notna().all(axis=1) & d[outcome].notna()
    return d.loc[mask].copy(), x.loc[mask].to_numpy(dtype=float), list(x.columns)


def fit_ols(d, X, y, clusters, cluster_name):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        xtx_inv = np.linalg.pinv(X.T @ X)
        meat = np.zeros((X.shape[1], X.shape[1]))
        groups = pd.Series(clusters).astype(str).to_numpy()
        uniq = np.unique(groups)
        for g in uniq:
            ix = groups == g
            xg = X[ix]
            eg = resid[ix]
            v = xg.T @ eg
            meat += np.outer(v, v)
    # CR1 small-sample correction; inference uses t_{G-1} for few region clusters.
    G = len(uniq); n, k = X.shape
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        cov = xtx_inv @ meat @ xtx_inv * (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    stat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    df = max(G - 1, 1) if cluster_name == "region" else 1e9
    p = 2 * t.sf(np.abs(stat), df=df)
    ci_lo = beta - t.ppf(.975, df) * se
    ci_hi = beta + t.ppf(.975, df) * se
    return beta, se, stat, p, ci_lo, ci_hi, G


def run_model(df, outcome, model_family):
    d, X, names = design_matrix(df, outcome)
    y = d[outcome].to_numpy(dtype=float)
    rows = []
    for cluster_name, clusters in [("region", d["region"]), ("client", d["client_id"])]:
        beta, se, stat, p, lo, hi, G = fit_ols(d, X, y, clusters, cluster_name)
        for term in ["post", "new", "post_new", "post_shortage", "new_shortage", "post_new_shortage"]:
            j = names.index(term)
            rows.append({"model": model_family, "outcome": outcome, "cluster": cluster_name, "term": term, "estimate": beta[j], "std_error": se[j], "stat": stat[j], "p_value": p[j], "ci_low": lo[j], "ci_high": hi[j], "n_obs": len(d), "n_clusters": G, "unit": "percentage points" if outcome in ["dangerous", "large_amount"] else "outcome units"})
    return pd.DataFrame(rows)


def main():
    c, fuel, fines = load_clean()
    people = make_people(c)
    reg = region_shortage(people, fuel)
    people = people.reset_index().merge(reg, on="region", how="left").set_index("client_id")
    fines = dangerous_flags(fines).merge(people.reset_index(), on="client_id", how="inner")
    fines = fines[fines["vehicle_group"].isin(["new", "old"])].copy()
    models = [run_model(fines, "dangerous", "fine_LPM_dangerous"), run_model(fines, "large_amount", "fine_LPM_large_amount")]

    # Client-period fuel outcomes: zeros are real zero purchases in a period.
    client = people[people["vehicle_group"].isin(["new", "old"])].reset_index()
    panels = []
    for period, start, end in [("pre", PRE_START, PRE_END), ("post", POST_START, POST_END)]:
        fx = fuel[fuel["date"].between(start, end, inclusive="left")].groupby("client_id").agg(
            liters=("order_fuel_volume", "sum"), transactions=("order_id", "size")
        )
        x = client.join(fx, on="client_id")
        x[["liters", "transactions"]] = x[["liters", "transactions"]].fillna(0)
        x["liters_30"] = x["liters"] * 30 / DAYS[period]
        x["transactions_30"] = x["transactions"] * 30 / DAYS[period]
        x["period"] = period
        panels.append(x)
    panel = pd.concat(panels, ignore_index=True)
    for outcome in ["liters_30", "transactions_30"]:
        models.append(run_model(panel, outcome, f"client_period_OLS_{outcome}"))
    result = pd.concat(models, ignore_index=True)
    result.to_csv(OUT / "model_coefficients_clean.csv", index=False)

    dictionary = pd.DataFrame([
        ["Y_it", "dangerous", "1 если штраф относится к опасному типу; иначе 0", "Основной исход"],
        ["Y_it", "large_amount", "1 если total_fine_amount >= 150000", "Робастностный исход"],
        ["L_it", "liters_30", "литры клиента за период × 30 / длина периода", "Механизм: объём использования"],
        ["T_it", "transactions_30", "число заправок × 30 / длина периода", "Механизм: частота использования"],
        ["Post_t", "post", "1 для 1 июня—31 августа; 0 для 1 апреля—24 мая", "Индикатор дефицита"],
        ["New_i", "new", "1 если 2026 − медианный год выпуска <= 5", "Новые авто"],
        ["S_r", "shortage_std", "среднее z(-изменение литров) и z(-изменение транзакций) по региону", "Физический дефицит; +1 = сильнее"],
        ["Z_i", "controls", "возраст авто, ln(1+цена), число авто, штрафы 2025, пол, возрастная категория", "Контрольные переменные"],
        ["alpha_r", "region FE", "набор индикаторов региона, один регион исключён как база", "Контроль постоянных различий регистрации"],
        ["beta_3", "post_new", "ключевой коэффициент DiD", "Разница изменения опасной доли: новые минус старые после дефицита"],
        ["beta_7", "post_new_shortage", "дополнительный коэффициент тройного взаимодействия", "Проверка неоднородности по силе регионального дефицита"],
    ], columns=["symbol", "variable", "definition", "role"])
    dictionary.to_csv(OUT / "model_dictionary_clean.csv", index=False)

    key = result[(result.term == "post_new_shortage") & (result.cluster == "region")]
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
