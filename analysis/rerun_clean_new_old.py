# -*- coding: utf-8 -*-
"""Recompute new/old vehicle severity analysis on the clean datasets only."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "fuel_cleaning_20260919"
OUT = ROOT / "outputs" / "wealth_newness_clean_20260920"
OUT.mkdir(parents=True, exist_ok=True)

PRE_START = pd.Timestamp("2026-04-01")
PRE_END = pd.Timestamp("2026-05-25")
POST_START = pd.Timestamp("2026-06-01")
POST_END = pd.Timestamp("2026-09-01")
DAYS = {"pre": (PRE_END - PRE_START).days, "post": (POST_END - POST_START).days}


def read_clean():
    c = pd.read_csv(SRC / "clients_demographics_clean.csv", sep=";", decimal=",")
    f = pd.read_csv(SRC / "fuel_transaction_clean.csv", sep=";", decimal=",")
    y = pd.read_csv(SRC / "fines_2026_clean.csv", sep=";")
    c["signup"] = pd.to_datetime(c["subscription_creation_date"], errors="coerce")
    f["date"] = pd.to_datetime(f["order_datetime"], errors="coerce")
    y["date"] = pd.to_datetime(y["bill_offence_date"], errors="coerce")
    return c, f, y


def build_people(c):
    v = c.sort_values(["auto_document_id", "signup"]).drop_duplicates("auto_document_id")
    people = v.groupby("client_id").agg(
        region=("kladr_code", "first"),
        auto_year=("auto_year", "median"),
        cars=("auto_document_id", "nunique"),
        price=("price", "median"),
    )
    people["vehicle_age"] = 2026 - people["auto_year"]
    people["vehicle_group"] = np.select(
        [people["vehicle_age"].le(5), people["vehicle_age"].ge(11)],
        ["новые_0_5", "старые_11плюс"],
        default="средние_6_10",
    )
    return people


def flag_severity(y):
    text = y["offence_short_statement"].fillna("")
    # Type-based primary definition: high-risk manoeuvres and large speed excess.
    y["dangerous_flag"] = text.str.contains(
        r"40-60|60-80|более чем на 80|красн|телефон|обочин|встречн|пешеход",
        case=False, regex=True,
    )
    # Independent robustness definition based on the monetary amount.
    y["large_amount_flag"] = pd.to_numeric(y["total_fine_amount"], errors="coerce").ge(150000)
    return y


def main():
    c, fuel, fines = read_clean()
    people = build_people(c)
    fines = flag_severity(fines)
    rows = []
    for period, start, end in [
        ("pre", PRE_START, PRE_END), ("post", POST_START, POST_END)
    ]:
        fuel_p = fuel[fuel["date"].between(start, end, inclusive="left")]
        fines_p = fines[fines["date"].between(start, end, inclusive="left")]
        fuel_agg = fuel_p.groupby("client_id").agg(
            transactions=("order_id", "size"), liters=("order_fuel_volume", "sum")
        )
        fine_agg = fines_p.groupby("client_id").agg(
            all_fines=("bill_id", "size"),
            dangerous_fines=("dangerous_flag", "sum"),
            large_amount_fines=("large_amount_flag", "sum"),
        )
        for group in ["новые_0_5", "старые_11плюс"]:
            n_clients = int(people[people["vehicle_group"].eq(group)].index.nunique())
            client_ids = people.index[people["vehicle_group"].eq(group)]
            d = fines_p[fines_p["client_id"].isin(client_ids)]
            fsub = fuel_p[fuel_p["client_id"].isin(client_ids)]
            n_fines = len(d)
            dangerous = int(d["dangerous_flag"].sum())
            light = n_fines - dangerous
            large = int(d["large_amount_flag"].sum())
            small = n_fines - large
            liters = float(fsub["order_fuel_volume"].sum())
            exposure_months = n_clients * DAYS[period] / 30
            rows.extend([
                {"vehicle_group": group, "period": period, "severity_definition": "опасное_по_типу", "clients": n_clients, "liters": liters, "all_fines": n_fines, "severe_fines": dangerous, "light_fines": light, "severe_share_pct": 100 * dangerous / n_fines if n_fines else np.nan, "light_share_pct": 100 * light / n_fines if n_fines else np.nan, "severe_per_1000_clients_30d": dangerous / exposure_months * 1000 if exposure_months else np.nan, "light_per_1000_clients_30d": light / exposure_months * 1000 if exposure_months else np.nan, "severe_per_1000_liters": dangerous / liters * 1000 if liters else np.nan, "light_per_1000_liters": light / liters * 1000 if liters else np.nan},
                {"vehicle_group": group, "period": period, "severity_definition": "крупное_по_сумме", "clients": n_clients, "liters": liters, "all_fines": n_fines, "severe_fines": large, "light_fines": small, "severe_share_pct": 100 * large / n_fines if n_fines else np.nan, "light_share_pct": 100 * small / n_fines if n_fines else np.nan, "severe_per_1000_clients_30d": large / exposure_months * 1000 if exposure_months else np.nan, "light_per_1000_clients_30d": small / exposure_months * 1000 if exposure_months else np.nan, "severe_per_1000_liters": large / liters * 1000 if liters else np.nan, "light_per_1000_liters": small / liters * 1000 if liters else np.nan},
            ])
    levels = pd.DataFrame(rows)
    changes = []
    metrics = ["severe_share_pct", "light_share_pct", "severe_per_1000_clients_30d", "light_per_1000_clients_30d", "severe_per_1000_liters", "light_per_1000_liters"]
    for (group, definition), d in levels.groupby(["vehicle_group", "severity_definition"]):
        d = d.set_index("period")
        out = {"vehicle_group": group, "severity_definition": definition}
        for m in metrics:
            pre, post = d.loc["pre", m], d.loc["post", m]
            out[f"{m}_pre"] = pre
            out[f"{m}_post"] = post
            out[f"{m}_change"] = post - pre
            out[f"{m}_pct_change"] = (post / pre - 1) * 100 if pre else np.nan
        changes.append(out)
    changes = pd.DataFrame(changes)
    counts = people["vehicle_group"].value_counts().rename_axis("vehicle_group").reset_index(name="clients")
    # Region-level clean diagnostics: same clean clients, fuel and fines.
    keep = people["vehicle_group"].isin(["новые_0_5", "старые_11плюс"])
    pp = people.loc[keep, ["region", "vehicle_group"]].reset_index()
    fp = fuel.merge(pp[["client_id", "region", "vehicle_group"]], on="client_id", how="inner")
    yp = fines.merge(pp[["client_id", "region", "vehicle_group"]], on="client_id", how="inner")
    region_rows = []
    for period, start, end in [("pre", PRE_START, PRE_END), ("post", POST_START, POST_END)]:
        ff = fp[fp["date"].between(start, end, inclusive="left")]
        yy = yp[yp["date"].between(start, end, inclusive="left")]
        for (region, group), d in yy.groupby(["region", "vehicle_group"]):
            tx = ff[(ff["region"] == region) & (ff["vehicle_group"] == group)]
            n_clients = int(pp[(pp["region"] == region) & (pp["vehicle_group"] == group)]["client_id"].nunique())
            region_rows.append({
                "region": region, "vehicle_group": group, "period": period,
                "clients": n_clients, "all_fines": len(d),
                "dangerous_fines": int(d["dangerous_flag"].sum()),
                "light_fines": int((~d["dangerous_flag"]).sum()),
                "dangerous_share_pct": 100 * d["dangerous_flag"].mean(),
                "liters": float(tx["order_fuel_volume"].sum()),
                "transactions": len(tx),
            })
    region_levels = pd.DataFrame(region_rows)
    region_wide = region_levels.pivot_table(index=["region", "vehicle_group"], columns="period", values=["all_fines", "dangerous_share_pct", "liters", "transactions"], aggfunc="first")
    region_wide.columns = [f"{a}_{b}" for a, b in region_wide.columns]
    region_wide = region_wide.reset_index()
    for metric in ["dangerous_share_pct", "liters", "transactions"]:
        region_wide[f"{metric}_change"] = region_wide[f"{metric}_post"] - region_wide[f"{metric}_pre"]
        region_wide[f"{metric}_pct_change"] = (region_wide[f"{metric}_post"] / region_wide[f"{metric}_pre"] - 1) * 100
    shortage = region_wide.groupby("region").agg(
        liters_pct_change=("liters_pct_change", "mean"), transactions_pct_change=("transactions_pct_change", "mean")
    ).reset_index()
    shortage["physical_shortage"] = (-shortage["liters_pct_change"] - shortage["transactions_pct_change"])
    region_wide = region_wide.merge(shortage[["region", "physical_shortage"]], on="region", how="left")
    region_wide.to_csv(OUT / "region_vehicle_clean.csv", index=False)
    chart_dir = OUT / "theory_charts"
    chart_dir.mkdir(exist_ok=True)
    chart_prepost = levels[levels["severity_definition"].eq("опасное_по_типу")].copy()
    chart_prepost["days"] = chart_prepost["period"].map(DAYS)
    chart_prepost.to_csv(chart_dir / "chart_data_prepost_clean.csv", index=False)
    ch = changes[changes["severity_definition"].eq("опасное_по_типу")].copy()
    ch = ch.rename(columns={
        "severe_share_pct_change": "dangerous_share_change_pp",
        "severe_share_pct_pct_change": "dangerous_share_pct_change",
        "severe_per_1000_clients_30d_pct_change": "dangerous_rate_pct_change",
        "light_per_1000_clients_30d_pct_change": "light_rate_pct_change",
    })
    ch[["vehicle_group", "dangerous_share_change_pp", "dangerous_share_pct_change", "dangerous_rate_pct_change", "light_rate_pct_change"]].to_csv(chart_dir / "chart_data_changes_clean.csv", index=False)
    region_wide.to_csv(chart_dir / "chart_data_regions_clean.csv", index=False)
    method = pd.DataFrame([
        ["Источник клиентов", "clients_demographics_clean.csv", "Clean-демография задаёт состав клиентов и возраст авто"],
        ["Источник топлива", "fuel_transaction_clean.csv", "Clean-транзакции, периоды Apr 1–May 24 и Jun 1–Aug 31"],
        ["Источник штрафов", "fines_2026_clean.csv", "Clean-штрафы, те же периоды; 25–31 мая исключено"],
        ["Новые авто", "vehicle_age <= 5", "2026 − медианный год выпуска клиента"],
        ["Старые авто", "vehicle_age >= 11", "6–10 лет исключены из контрастного сравнения"],
        ["Опасные по типу", "speed 40+; red; phone; shoulder; oncoming; pedestrian", "Остальные нарушения — лёгкие"],
        ["Крупные по сумме", "total_fine_amount >= 150000", "Робастностная альтернативная классификация"],
    ], columns=["item", "rule", "comment"])
    levels.to_csv(OUT / "new_old_severity_prepost_levels_clean.csv", index=False)
    changes.to_csv(OUT / "new_old_severity_prepost_changes_clean.csv", index=False)
    counts.to_csv(OUT / "vehicle_group_counts_clean.csv", index=False)
    method.to_csv(OUT / "method_clean.csv", index=False)
    sig_rows = []
    lor_rows = []
    for group in ["новые_0_5", "старые_11плюс"]:
        d = levels[(levels["vehicle_group"].eq(group)) & (levels["severity_definition"].eq("опасное_по_типу"))].set_index("period")
        a, b = int(d.loc["pre", "severe_fines"]), int(d.loc["pre", "light_fines"])
        c2, e = int(d.loc["post", "severe_fines"]), int(d.loc["post", "light_fines"])
        chi2, p, _, _ = chi2_contingency([[a, b], [c2, e]], correction=False)
        lor = np.log(c2 * b / (e * a))
        var = 1 / a + 1 / b + 1 / c2 + 1 / e
        lor_rows.append((group, lor, var))
        sig_rows.append({"group": group, "chi2": chi2, "p_value_pre_post": p, "odds_ratio_post_vs_pre": np.exp(lor)})
    diff = lor_rows[0][1] - lor_rows[1][1]
    se = np.sqrt(lor_rows[0][2] + lor_rows[1][2])
    sig_rows.append({"group": "разница эффектов: новые минус старые", "chi2": np.nan, "p_value_pre_post": 2 * norm.sf(abs(diff / se)), "odds_ratio_post_vs_pre": np.exp(diff)})
    pd.DataFrame(sig_rows).to_csv(OUT / "significance_clean.csv", index=False)
    print(counts.to_string(index=False))
    print(changes[["vehicle_group", "severity_definition", "severe_share_pct_pre", "severe_share_pct_post", "severe_share_pct_change", "severe_per_1000_clients_30d_pct_change", "light_per_1000_clients_30d_pct_change"]].to_string(index=False))


if __name__ == "__main__":
    main()
