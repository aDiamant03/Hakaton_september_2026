from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "working_data" / "night_hypothesis"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2026-04-01")
POST = pd.Timestamp("2026-06-01")
END = pd.Timestamp("2026-09-01")


def two_prop_test(x1: int, n1: int, x0: int, n0: int) -> dict:
    p1, p0 = x1 / n1, x0 / n0
    pooled = (x1 + x0) / (n1 + n0)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
    z = (p1 - p0) / se
    return {"pre_rate": p0, "post_rate": p1, "difference_pp": (p1 - p0) * 100,
            "z": z, "p_value": 2 * stats.norm.sf(abs(z))}


def paired_share_test(data: pd.DataFrame, flag: pd.Series) -> dict:
    d = data[["client_id", "post"]].copy()
    d["flag"] = flag.astype(int).to_numpy()
    cp = d.groupby(["client_id", "post"]).agg(events=("flag", "size"), flagged=("flag", "sum")).reset_index()
    cp["share"] = cp["flagged"] / cp["events"]
    wide = cp.pivot(index="client_id", columns="post", values="share").dropna()
    delta = wide[True] - wide[False]
    se = delta.std(ddof=1) / np.sqrt(len(delta))
    return {"paired_clients": len(delta), "pre_mean": float(wide[False].mean()), "post_mean": float(wide[True].mean()),
            "difference_pp": float(delta.mean() * 100), "ci95_low_pp": float((delta.mean() - 1.96 * se) * 100),
            "ci95_high_pp": float((delta.mean() + 1.96 * se) * 100),
            "p_value": float(stats.ttest_1samp(delta, 0).pvalue)}


fuel = pd.read_csv(
    ROOT / "fuel_transaction.csv", sep=";", decimal=",",
    usecols=["order_id", "order_datetime", "order_fuel_volume", "order_fuel_price_1liter", "client_id"],
)
fuel["dt"] = pd.to_datetime(fuel.pop("order_datetime"))
fuel = fuel.loc[fuel["dt"].between(START, END, inclusive="left")].copy()
fuel["month"] = fuel["dt"].dt.month
fuel["post"] = fuel["dt"].ge(POST)
fuel["night"] = (fuel["dt"].dt.hour >= 22) | (fuel["dt"].dt.hour < 6)
fuel["valid_fuel"] = fuel["order_fuel_volume"].gt(0) & fuel["order_fuel_price_1liter"].between(50, 130)

fines = pd.read_csv(
    ROOT / "fines_2026.csv", sep=";",
    usecols=["client_id", "bill_id", "offence_short_statement", "region_name", "bill_offence_date"],
)
fines["dt"] = pd.to_datetime(fines.pop("bill_offence_date"))
fines = fines.loc[fines["dt"].between(START, END, inclusive="left")].copy()
fines["month"] = fines["dt"].dt.month
fines["post"] = fines["dt"].ge(POST)
fines["night"] = (fines["dt"].dt.hour >= 22) | (fines["dt"].dt.hour < 6)

# 1. Night refueling: transaction-weighted and paired client-level estimates.
valid = fuel.loc[fuel["valid_fuel"]].copy()
monthly_fuel = valid.groupby("month", as_index=False).agg(
    transactions=("order_id", "size"), night_transactions=("night", "sum"), clients=("client_id", "nunique")
)
monthly_fuel["night_share"] = monthly_fuel["night_transactions"] / monthly_fuel["transactions"]
monthly_fuel.to_csv(OUT / "monthly_night_refueling.csv", index=False)

pre_fuel = valid.loc[~valid["post"]]
post_fuel = valid.loc[valid["post"]]
night_refuel_txn = two_prop_test(int(post_fuel["night"].sum()), len(post_fuel), int(pre_fuel["night"].sum()), len(pre_fuel))

client_period = valid.groupby(["client_id", "post"]).agg(night=("night", "sum"), total=("night", "size")).reset_index()
client_period["share"] = client_period["night"] / client_period["total"]
paired = client_period.pivot(index="client_id", columns="post", values="share").dropna()
delta = paired[True] - paired[False]
paired_test = stats.ttest_1samp(delta, 0)
night_refuel_client = {
    "paired_clients": len(delta), "pre_mean": float(paired[False].mean()), "post_mean": float(paired[True].mean()),
    "difference_pp": float(delta.mean() * 100), "ci95_low_pp": float((delta.mean() - 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))) * 100),
    "ci95_high_pp": float((delta.mean() + 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))) * 100),
    "p_value": float(paired_test.pvalue),
}

# 2. Night violations and attention-related offence composition.
monthly_fines = fines.groupby("month", as_index=False).agg(
    fines=("bill_id", "size"), night_fines=("night", "sum"), clients=("client_id", "nunique")
)
monthly_fines["night_share"] = monthly_fines["night_fines"] / monthly_fines["fines"]
monthly_fines.to_csv(OUT / "monthly_night_fines.csv", index=False)
pre_fines, post_fines = fines.loc[~fines["post"]], fines.loc[fines["post"]]
night_fines_test = two_prop_test(int(post_fines["night"].sum()), len(post_fines), int(pre_fines["night"].sum()), len(pre_fines))
night_fines_paired = paired_share_test(fines, fines["night"])

phone = fines["offence_short_statement"].eq("Использование телефона за рулем")
broad_attention = fines["offence_short_statement"].isin([
    "Использование телефона за рулем", "Проезд на красный сигнал светофора", "Пересечение стоп-линии",
    "Нарушение разметки", "Не пристегнут ремень безопасности",
])
attention_tests = {}
for name, flag in {"phone": phone, "broad_attention_proxy": broad_attention}.items():
    attention_tests[name] = two_prop_test(
        int(flag[fines["post"]].sum()), int(fines["post"].sum()),
        int(flag[~fines["post"]].sum()), int((~fines["post"]).sum()),
    )

# 3. Unfamiliar-route proxy: violation outside registration region.
profile = pd.read_csv(ROOT / "working_data" / "client_profile.csv", usecols=["client_id", "registration_region"])
geo = fines.merge(profile, on="client_id", how="left")
geo["region_known"] = geo["registration_region"].notna() & geo["region_name"].notna()
geo["outside_home"] = geo["region_known"] & geo["region_name"].ne(geo["registration_region"])
known = geo.loc[geo["region_known"]]
outside_test = two_prop_test(
    int(known.loc[known["post"], "outside_home"].sum()), int(known["post"].sum()),
    int(known.loc[~known["post"], "outside_home"].sum()), int((~known["post"]).sum()),
)
outside_paired = paired_share_test(known, known["outside_home"])
region_pairs = known.groupby(["registration_region", "region_name"]).size().sort_values(ascending=False).head(30)
region_pairs.to_csv(OUT / "top_region_pairs.csv")

# 3b. Do clients who shifted more strongly to night refueling also show a larger
# change in monthly fine counts? This is an exploratory within-client association.
fuel_client = valid.groupby(["client_id", "post"]).agg(night=("night", "sum"), total=("night", "size")).reset_index()
fuel_client["night_share"] = fuel_client["night"] / fuel_client["total"]
night_wide = fuel_client.pivot(index="client_id", columns="post", values="night_share").dropna()
fine_counts = fines.groupby(["client_id", "post"]).size().unstack(fill_value=0).reindex(night_wide.index, fill_value=0)
for period in (False, True):
    if period not in fine_counts:
        fine_counts[period] = 0
client_link = pd.DataFrame({
    "night_share_change": night_wide[True] - night_wide[False],
    "fines_per_month_change": fine_counts[True] / 3 - fine_counts[False] / 2,
})
rho, rho_p = stats.spearmanr(client_link["night_share_change"], client_link["fines_per_month_change"])
client_link["night_shift_quartile"] = pd.qcut(client_link["night_share_change"].rank(method="first"), 4, labels=False) + 1
night_shift_quartiles = client_link.groupby("night_shift_quartile", as_index=False).agg(
    clients=("fines_per_month_change", "size"), mean_night_share_change=("night_share_change", "mean"),
    mean_fines_per_month_change=("fines_per_month_change", "mean"),
)
night_shift_quartiles.to_csv(OUT / "client_night_shift_quartiles.csv", index=False)
client_shift_link = {"clients": len(client_link), "spearman_rho": float(rho), "p_value": float(rho_p)}

# 4. Short-window event study around each valid refueling. Count fines in 6h before and after.
# The before-after difference removes a large part of stable client risk; comparing that
# difference for night versus day refueling is still associational because trips are unobserved.
fine_times = {cid: np.sort(g["dt"].astype("int64").to_numpy()) for cid, g in fines.groupby("client_id")}
six_h = np.int64(pd.Timedelta(hours=6).value)
rows = []
for cid, g in valid.groupby("client_id", sort=False):
    ft = fine_times.get(cid)
    if ft is None:
        before = np.zeros(len(g), dtype=int)
        after = np.zeros(len(g), dtype=int)
    else:
        t = g["dt"].astype("int64").to_numpy()
        before = np.searchsorted(ft, t, side="left") - np.searchsorted(ft, t - six_h, side="left")
        after = np.searchsorted(ft, t + six_h, side="right") - np.searchsorted(ft, t, side="right")
    tmp = g[["client_id", "month", "post", "night"]].copy()
    tmp["before_6h"] = before
    tmp["after_6h"] = after
    tmp["delta_6h"] = after - before
    rows.append(tmp)
event = pd.concat(rows, ignore_index=True)
event_summary = event.groupby(["post", "night"], as_index=False).agg(
    refuelings=("client_id", "size"), clients=("client_id", "nunique"),
    before_6h=("before_6h", "mean"), after_6h=("after_6h", "mean"), delta_6h=("delta_6h", "mean"),
)
event_summary.to_csv(OUT / "refueling_event_window.csv", index=False)

# Cluster bootstrap of the difference-in-differences in mean after-minus-before counts.
client_cells = event.groupby(["client_id", "night"])["delta_6h"].mean().unstack()
client_cells = client_cells.dropna()
client_cells["contrast"] = client_cells[True] - client_cells[False]
rng = np.random.default_rng(20260919)
vals = client_cells["contrast"].to_numpy()
boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(2000)])
window_test = {
    "clients_with_day_and_night_refueling": len(vals), "night_minus_day_before_after_contrast": float(vals.mean()),
    "ci95_low": float(np.quantile(boot, 0.025)), "ci95_high": float(np.quantile(boot, 0.975)),
    "p_value_ttest": float(stats.ttest_1samp(vals, 0).pvalue),
}

# Sensitivity to the clock-hour definition of night.
sensitivity_rows = []
for label, start_hour, end_hour in [("21:00–06:59", 21, 7), ("22:00–05:59", 22, 6),
                                    ("23:00–04:59", 23, 5), ("00:00–05:59", 0, 6)]:
    if start_hour == 0:
        fuel_flag = valid["dt"].dt.hour.lt(end_hour)
        fine_flag = fines["dt"].dt.hour.lt(end_hour)
    else:
        fuel_flag = valid["dt"].dt.hour.ge(start_hour) | valid["dt"].dt.hour.lt(end_hour)
        fine_flag = fines["dt"].dt.hour.ge(start_hour) | fines["dt"].dt.hour.lt(end_hour)
    fr = two_prop_test(int(fuel_flag[valid["post"]].sum()), int(valid["post"].sum()),
                       int(fuel_flag[~valid["post"]].sum()), int((~valid["post"]).sum()))
    fi = two_prop_test(int(fine_flag[fines["post"]].sum()), int(fines["post"].sum()),
                       int(fine_flag[~fines["post"]].sum()), int((~fines["post"]).sum()))
    sensitivity_rows.append({"night_definition": label, "refueling_pre_share": fr["pre_rate"],
                             "refueling_post_share": fr["post_rate"], "refueling_change_pp": fr["difference_pp"],
                             "refueling_p_value": fr["p_value"], "fines_pre_share": fi["pre_rate"],
                             "fines_post_share": fi["post_rate"], "fines_change_pp": fi["difference_pp"],
                             "fines_p_value": fi["p_value"]})
pd.DataFrame(sensitivity_rows).to_csv(OUT / "night_definition_sensitivity.csv", index=False)

result = {
    "definitions": {"pre": "2026-04-01..2026-05-31", "post": "2026-06-01..2026-08-31", "night": "22:00..05:59",
                    "valid_refueling": "volume>0 and 50<=price_per_liter<=130"},
    "night_refueling_transaction_weighted": night_refuel_txn,
    "night_refueling_paired_clients": night_refuel_client,
    "night_fines_transaction_weighted": night_fines_test,
    "night_fines_paired_clients": night_fines_paired,
    "attention_offence_composition": attention_tests,
    "outside_home_region_proxy": outside_test,
    "outside_home_region_paired_clients": outside_paired,
    "client_night_shift_vs_fines_change": client_shift_link,
    "refueling_6h_event_window": window_test,
    "coverage": {"valid_refuelings": len(valid), "fines": len(fines), "clients_fuel": valid["client_id"].nunique(),
                 "clients_fines": fines["client_id"].nunique(), "known_home_region_fines": len(known)},
}
(OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
