from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp" / "research_pydeps"))
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from statsmodels.discrete.conditional_models import ConditionalPoisson  # noqa: E402


WORK = ROOT / "working_data" / "final_research"
OUT = ROOT / "outputs" / "final"
WORK.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

PANEL_PATH = ROOT / "working_data" / "client_month_analysis" / "client_month_panel_quality.csv"
CLIENT_PATH = ROOT / "working_data" / "preanalysis" / "client_did.csv"
REGION_PATH = ROOT / "working_data" / "preanalysis" / "region_hypothesis.csv"
MAP_CHECK_PATH = ROOT / "working_data" / "preanalysis" / "map_validation.csv"

MONTH_NAMES = {4: "Апрель", 5: "Май", 6: "Июнь", 7: "Июль", 8: "Август"}


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "<0,001"
    return f"{p:.3f}".replace(".", ",")


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    x = values.dropna().astype(float)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    crit = float(stats.t.ppf(0.975, len(x) - 1))
    return mean, mean - crit * se, mean + crit * se


def make_long(panel: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "client_id", "month_num", "registration_region", "gender", "age_type_code",
        "vehicle_count", "vehicle_price_median_rub", "vehicle_year_median",
        "engine_power_hp_median", "engine_displacement_l_median", "subscription_creation_date",
    ]
    rows = []
    for year in (2025, 2026):
        d = panel[id_cols].copy()
        d["year"] = year
        d["fines_count"] = panel[f"fines_{year}"].astype(float)
        d["year_2026"] = int(year == 2026)
        d["post_crisis"] = panel["month_num"].ge(6).astype(int)
        d["post_x_year_2026"] = d["post_crisis"] * d["year_2026"]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def fit_conditional_poisson(data: pd.DataFrame, terms: list[str]):
    x = data[terms].astype(float)
    model = ConditionalPoisson(data["fines_count"].astype(float), x, groups=data["client_id"])
    return model.fit(method="BFGS", disp=False, maxiter=250)


def base_terms(data: pd.DataFrame) -> list[str]:
    for month in (5, 6, 7, 8):
        data[f"month_{month}"] = data["month_num"].eq(month).astype(int)
    return ["year_2026", "month_5", "month_6", "month_7", "month_8"]


def model_row(name: str, sample: pd.DataFrame, result, term: str, note: str) -> dict:
    coef = float(result.params[term])
    se = float(result.bse[term])
    lo, hi = result.conf_int().loc[term]
    return {
        "model": name,
        "sample_clients": int(sample["client_id"].nunique()),
        "model_observations": int(result.nobs),
        "term": term,
        "coef_log_rate": coef,
        "std_error": se,
        "irr": float(np.exp(coef)),
        "ci95_low": float(np.exp(lo)),
        "ci95_high": float(np.exp(hi)),
        "p_value": float(result.pvalues[term]),
        "interpretation": note,
    }


def prepare_clients(panel: pd.DataFrame, clients: pd.DataFrame) -> pd.DataFrame:
    clients = clients.copy()
    first = panel.drop_duplicates("client_id").set_index("client_id")
    clients["full_2025_coverage_flag"] = pd.to_datetime(clients["subscription_creation_date"]).le("2025-04-01")
    clients["subscription_tenure_months"] = clients["client_id"].map(first["subscription_tenure_months"])
    clients["vehicle_age_2026"] = 2026 - clients["vehicle_year_median"]
    clients["log_vehicle_price"] = np.log1p(clients["vehicle_price_median_rub"])
    clients["single_vehicle_flag"] = clients["vehicle_count"].eq(1)
    clients["old_cheap"] = clients["old_car_flag"] & clients["cheap_car_flag"]
    clients["new_expensive"] = clients["new_car_flag"] & clients["expensive_car_flag"]
    clients["fuel_intensity_q"] = clients["fuel_intensity_segment"].map({
        "Q1: низкая": 1, "Q2": 2, "Q3": 3, "Q4: высокая": 4,
    })
    clients["fuel_exposure_q"] = clients["fuel_exposure_quartile"].map({
        "Q1: низкая": 1, "Q2": 2, "Q3": 3, "Q4: высокая": 4,
    })

    pre = panel.loc[panel["month_num"].le(5)].groupby("client_id", as_index=False).agg(
        baseline_fuel_quality_share=("fuel_txn_standard_band", "sum"),
        baseline_fuel_positive_txn=("fuel_txn_positive", "sum"),
        baseline_negative_txn=("fuel_txn_negative", "sum"),
        baseline_low_price_txn=("fuel_txn_low_price", "sum"),
        baseline_high_price_txn=("fuel_txn_high_price", "sum"),
    )
    pre["fuel_quality_share"] = pre["baseline_fuel_quality_share"].div(pre["baseline_fuel_positive_txn"].replace(0, np.nan))
    pre["negative_txn_share"] = pre["baseline_negative_txn"].div((pre["baseline_fuel_positive_txn"] + pre["baseline_negative_txn"]).replace(0, np.nan))
    pre["low_price_share"] = pre["baseline_low_price_txn"].div(pre["baseline_fuel_positive_txn"].replace(0, np.nan))
    pre["high_price_share"] = pre["baseline_high_price_txn"].div(pre["baseline_fuel_positive_txn"].replace(0, np.nan))
    clients = clients.merge(pre[["client_id", "fuel_quality_share", "negative_txn_share", "low_price_share", "high_price_share"]], on="client_id", how="left")

    april = panel.loc[panel["month_num"].eq(4), ["client_id", "fuel_volume_positive_l", "fuel_txn_positive", "fuel_spend_standard_band_rub"]].copy()
    april = april.rename(columns={
        "fuel_volume_positive_l": "april_fuel_volume_l",
        "fuel_txn_positive": "april_fuel_orders",
        "fuel_spend_standard_band_rub": "april_fuel_spend_standard_rub",
    })
    clients = clients.merge(april, on="client_id", how="left")
    clients["april_fuel_exposure_index"] = clients["april_fuel_spend_standard_rub"].div(clients["vehicle_price_median_rub"])
    mask = clients["april_fuel_exposure_index"].notna()
    clients.loc[mask, "april_fuel_exposure_q"] = pd.qcut(clients.loc[mask, "april_fuel_exposure_index"], 4, labels=False, duplicates="drop") + 1

    region = pd.read_csv(REGION_PATH)[["registration_region", "price_change_pct"]].rename(columns={"price_change_pct": "region_price_shock_pct"})
    clients = clients.merge(region, on="registration_region", how="left")
    return clients


def build_models(panel: pd.DataFrame, clients: pd.DataFrame):
    long = make_long(panel)
    long = long.merge(clients, on="client_id", how="left", suffixes=("", "_client"))
    terms = base_terms(long)
    model_rows = []

    main_res = fit_conditional_poisson(long, terms + ["post_x_year_2026"])
    model_rows.append(model_row("Основная conditional Poisson", long, main_res, "post_x_year_2026", "IRR<1 означает снижение частоты штрафов после мая"))

    # Additive client and month fixed effects through within-client demeaning.
    ols = long.copy()
    for col in ["fines_count", *terms, "post_x_year_2026"]:
        ols[f"w_{col}"] = ols[col] - ols.groupby("client_id")[col].transform("mean")
    xcols = [f"w_{c}" for c in [*terms, "post_x_year_2026"]]
    ols_res = sm.OLS(ols["w_fines_count"], ols[xcols]).fit(cov_type="cluster", cov_kwds={"groups": ols["client_id"]})
    coef = float(ols_res.params["w_post_x_year_2026"])
    lo, hi = ols_res.conf_int().loc["w_post_x_year_2026"]
    additive = pd.DataFrame([{
        "model": "Линейная TWFE, кластер SE по клиенту", "clients": ols["client_id"].nunique(), "observations": len(ols),
        "effect_fines_per_client_month": coef, "std_error": float(ols_res.bse["w_post_x_year_2026"]),
        "ci95_low": float(lo), "ci95_high": float(hi), "p_value": float(ols_res.pvalues["w_post_x_year_2026"]),
    }])

    # Event study, April is reference month.
    event_terms = terms.copy()
    for month in (5, 6, 7, 8):
        long[f"event_{month}"] = long["year_2026"] * long[f"month_{month}"]
        event_terms.append(f"event_{month}")
    event_res = fit_conditional_poisson(long, event_terms)
    event_rows = [{"month_num": 4, "month_name": MONTH_NAMES[4], "irr": 1.0, "ci95_low": 1.0, "ci95_high": 1.0, "p_value": np.nan, "note": "Референс"}]
    for month in (5, 6, 7, 8):
        term = f"event_{month}"
        lo, hi = event_res.conf_int().loc[term]
        event_rows.append({
            "month_num": month, "month_name": MONTH_NAMES[month], "irr": float(np.exp(event_res.params[term])),
            "ci95_low": float(np.exp(lo)), "ci95_high": float(np.exp(hi)), "p_value": float(event_res.pvalues[term]),
            "note": "Плацебо до кризиса" if month == 5 else "После начала кризиса",
        })
    event = pd.DataFrame(event_rows)

    # Preferred heterogeneity: quartiles among clients with positive pre-crisis volume.
    intensity = long.loc[long["fuel_intensity_q"].notna()].copy()
    intensity["q_centered"] = intensity["fuel_intensity_q"].astype(float) - 2.5
    intensity["year_q"] = intensity["year_2026"] * intensity["q_centered"]
    intensity["treat_q"] = intensity["post_x_year_2026"] * intensity["q_centered"]
    int_terms = base_terms(intensity) + ["post_x_year_2026", "year_q", "treat_q"]
    intensity_res = fit_conditional_poisson(intensity, int_terms)
    model_rows.append(model_row("Гетерогенность по топливной активности", intensity, intensity_res, "treat_q", "IRR на переход в следующий квартиль активности"))
    cov = intensity_res.cov_params()
    segment_rows = []
    for q in (1, 2, 3, 4):
        z = q - 2.5
        effect = float(intensity_res.params["post_x_year_2026"] + z * intensity_res.params["treat_q"])
        var = float(cov.loc["post_x_year_2026", "post_x_year_2026"] + z*z*cov.loc["treat_q", "treat_q"] + 2*z*cov.loc["post_x_year_2026", "treat_q"])
        se = np.sqrt(var)
        segment_rows.append({
            "fuel_intensity_q": q, "segment": f"Q{q}" + (": низкая" if q == 1 else ": высокая" if q == 4 else ""),
            "clients": int(clients["fuel_intensity_q"].eq(q).sum()), "irr_post": float(np.exp(effect)),
            "effect_pct": float((np.exp(effect) - 1) * 100), "ci95_low": float(np.exp(effect - 1.96*se)),
            "ci95_high": float(np.exp(effect + 1.96*se)),
        })
    segments = pd.DataFrame(segment_rows)

    # Main-effect robustness.
    sample_masks = {
        "Все клиенты": pd.Series(True, index=long.index),
        "Полная история 2025": long["full_2025_coverage_flag"].fillna(False),
        "Один автомобиль": long["single_vehicle_flag"].fillna(False),
        "Полная история + один автомобиль": long["full_2025_coverage_flag"].fillna(False) & long["single_vehicle_flag"].fillna(False),
    }
    robust_main = []
    for name, mask in sample_masks.items():
        d = long.loc[mask].copy()
        res = fit_conditional_poisson(d, base_terms(d) + ["post_x_year_2026"])
        robust_main.append(model_row(name, d, res, "post_x_year_2026", "Основной эффект"))
    robust_main = pd.DataFrame(robust_main)

    # Interaction robustness on fuel-volume quartiles.
    robust_int = []
    int_masks = {
        "Все покупатели топлива": intensity.index,
        "Полная история 2025": intensity.index[intensity["full_2025_coverage_flag"].fillna(False)],
        "Один автомобиль": intensity.index[intensity["single_vehicle_flag"].fillna(False)],
        "Полная история + один автомобиль": intensity.index[intensity["full_2025_coverage_flag"].fillna(False) & intensity["single_vehicle_flag"].fillna(False)],
    }
    for name, idx in int_masks.items():
        d = intensity.loc[idx].copy()
        res = fit_conditional_poisson(d, base_terms(d) + ["post_x_year_2026", "year_q", "treat_q"])
        robust_int.append(model_row(name, d, res, "treat_q", "IRR на следующий квартиль активности"))
    robust_int = pd.DataFrame(robust_int)

    # April-only pre-period exposure.
    april_d = long.loc[long["april_fuel_exposure_q"].notna()].copy()
    april_d["april_q_centered"] = april_d["april_fuel_exposure_q"].astype(float) - 2.5
    april_d["year_april_q"] = april_d["year_2026"] * april_d["april_q_centered"]
    april_d["treat_april_q"] = april_d["post_x_year_2026"] * april_d["april_q_centered"]
    april_res = fit_conditional_poisson(april_d, base_terms(april_d) + ["post_x_year_2026", "year_april_q", "treat_april_q"])
    model_rows.append(model_row("Плацебо сегментации: только апрель", april_d, april_res, "treat_april_q", "Устойчивость к определению baseline"))

    # Continuous effect modifiers: activity vs vehicle characteristics.
    mod = long.copy()
    modifier_defs = {
        "spend": np.log1p(mod["baseline_fuel_spend_standard_rub"].fillna(0)),
        "price": mod["log_vehicle_price"],
        "age": mod["vehicle_age_2026"],
        "power": mod["engine_power_hp_median_client"].fillna(mod["engine_power_hp_median"]),
    }
    complete = pd.Series(True, index=mod.index)
    for key, values in modifier_defs.items():
        mod[f"z_{key}"] = (values - values.mean()) / values.std(ddof=0)
        complete &= mod[f"z_{key}"].notna()
    mod = mod.loc[complete].copy()
    mod_terms = base_terms(mod) + ["post_x_year_2026"]
    for key in modifier_defs:
        mod[f"year_{key}"] = mod["year_2026"] * mod[f"z_{key}"]
        mod[f"treat_{key}"] = mod["post_x_year_2026"] * mod[f"z_{key}"]
        mod_terms += [f"year_{key}", f"treat_{key}"]
    mod_res = fit_conditional_poisson(mod, mod_terms)
    modifier_rows = []
    labels = {"spend": "Докризисные расходы на топливо", "price": "Стоимость автомобиля", "age": "Возраст автомобиля", "power": "Мощность двигателя"}
    for key, label in labels.items():
        term = f"treat_{key}"
        lo, hi = mod_res.conf_int().loc[term]
        modifier_rows.append({
            "modifier": label, "coef_log_rate": float(mod_res.params[term]), "irr_per_1sd": float(np.exp(mod_res.params[term])),
            "ci95_low": float(np.exp(lo)), "ci95_high": float(np.exp(hi)), "p_value": float(mod_res.pvalues[term]),
        })
    modifiers = pd.DataFrame(modifier_rows)

    # Mechanism-consistent fuel dynamics by baseline volume quartile.
    p2 = panel.merge(clients[["client_id", "fuel_intensity_q"]], on="client_id", how="left")
    p2["crisis"] = p2["month_num"].ge(6)
    dyn = p2.loc[p2["fuel_intensity_q"].notna()].groupby(["fuel_intensity_q", "crisis"], as_index=False).agg(
        clients=("client_id", "nunique"), orders=("fuel_txn_positive", "sum"), volume_l=("fuel_volume_positive_l", "sum"),
    )
    dyn["months"] = dyn["crisis"].map({False: 2, True: 3})
    dyn["orders_per_client_month"] = dyn["orders"] / dyn["clients"] / dyn["months"]
    dyn["volume_per_client_month_l"] = dyn["volume_l"] / dyn["clients"] / dyn["months"]
    dyn["liters_per_order"] = dyn["volume_l"] / dyn["orders"]
    w = dyn.pivot(index="fuel_intensity_q", columns="crisis")
    fuel_dynamics = pd.DataFrame(index=w.index)
    fuel_dynamics["segment"] = [f"Q{int(q)}" for q in fuel_dynamics.index]
    for metric in ["orders_per_client_month", "volume_per_client_month_l", "liters_per_order"]:
        fuel_dynamics[f"{metric}_pre"] = w[metric][False]
        fuel_dynamics[f"{metric}_crisis"] = w[metric][True]
        fuel_dynamics[f"{metric}_change_pct"] = (w[metric][True] / w[metric][False] - 1) * 100
    fuel_dynamics = fuel_dynamics.reset_index(drop=True)

    return long, pd.DataFrame(model_rows), additive, event, segments, robust_main, robust_int, modifiers, fuel_dynamics


def client_level_tests(clients: pd.DataFrame):
    # Original vehicle hypothesis.
    vehicle = []
    for label, mask in {
        "Старая и дешёвая": clients["old_cheap"],
        "Новая и дорогая": clients["new_expensive"],
        "Остальные": ~(clients["old_cheap"] | clients["new_expensive"]),
    }.items():
        vals = clients.loc[mask.fillna(False), "fine_did"]
        m, lo, hi = mean_ci(vals)
        vehicle.append({"segment": label, "clients": vals.notna().sum(), "fine_did": m, "ci95_low": lo, "ci95_high": hi})
    vehicle = pd.DataFrame(vehicle)
    a = clients.loc[clients["old_cheap"].fillna(False), "fine_did"]
    b = clients.loc[clients["new_expensive"].fillna(False), "fine_did"]
    vehicle_p = float(stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)

    # Adjusted DiD at client level.
    d = clients.copy()
    d["fuel_exposure_q_num"] = d["fuel_exposure_q"].astype(float)
    formula = "fine_did ~ fuel_exposure_q_num + log_vehicle_price + vehicle_age_2026 + engine_power_hp_median + vehicle_count + full_2025_coverage_flag + C(registration_region) + C(age_type_code) + C(gender)"
    adj = smf.ols(formula, data=d).fit(cov_type="HC3")
    lo, hi = adj.conf_int().loc["fuel_exposure_q_num"]
    adjusted = pd.DataFrame([{
        "model": "Adjusted client-level DiD", "clients": int(adj.nobs), "effect_per_exposure_quartile": float(adj.params["fuel_exposure_q_num"]),
        "std_error": float(adj.bse["fuel_exposure_q_num"]), "ci95_low": float(lo), "ci95_high": float(hi), "p_value": float(adj.pvalues["fuel_exposure_q_num"]),
    }])

    # Deterministic discovery/validation split.
    d2 = clients.loc[clients["fuel_exposure_index"].notna()].copy()
    d2["split"] = d2["client_id"].map(lambda x: "Validation" if int(hashlib.md5(x.encode()).hexdigest()[-1], 16) >= 8 else "Discovery")
    cuts = d2.loc[d2["split"].eq("Discovery"), "fuel_exposure_index"].quantile([.25, .5, .75]).to_numpy()
    d2["split_q"] = np.digitize(d2["fuel_exposure_index"], cuts, right=True) + 1
    validation = []
    for split, g in d2.groupby("split"):
        trend = stats.linregress(g["split_q"], g["fine_did"])
        q1 = g.loc[g["split_q"].eq(1), "fine_did"]
        q4 = g.loc[g["split_q"].eq(4), "fine_did"]
        extreme = stats.ttest_ind(q4, q1, equal_var=False, nan_policy="omit")
        validation.append({"sample": split, "clients": len(g), "trend_per_quartile": trend.slope, "trend_p_value": trend.pvalue, "q4_minus_q1": q4.mean()-q1.mean(), "q4_vs_q1_p_value": extreme.pvalue})
    return vehicle, vehicle_p, adjusted, pd.DataFrame(validation), cuts


def build_workbook(path: Path, tables: dict[str, pd.DataFrame], headline: dict):
    import xlsxwriter
    from xlsxwriter.utility import xl_col_to_name

    wb = xlsxwriter.Workbook(path, {"constant_memory": False, "nan_inf_to_errors": True})
    wb.set_properties({"title": "DANO — доказательство гипотезы о топливном кризисе", "author": "Исследовательская команда DANO"})
    navy, teal, amber, red, ink, pale = "#0B1F33", "#0F8B8D", "#F2A900", "#C94040", "#1F2937", "#F4F7FA"
    title = wb.add_format({"bold": True, "font_size": 20, "font_color": "#FFFFFF", "bg_color": navy, "align": "left", "valign": "vcenter"})
    subtitle = wb.add_format({"bold": True, "font_size": 12, "font_color": navy, "bg_color": "#DDECF0", "text_wrap": True, "valign": "vcenter"})
    section = wb.add_format({"bold": True, "font_size": 13, "font_color": "#FFFFFF", "bg_color": teal, "align": "left"})
    header = wb.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": navy, "border": 0, "text_wrap": True, "valign": "vcenter"})
    body = wb.add_format({"font_color": ink, "border": 1, "border_color": "#D8E0E8", "valign": "top"})
    body_wrap = wb.add_format({"font_color": ink, "border": 1, "border_color": "#D8E0E8", "text_wrap": True, "valign": "top"})
    num = wb.add_format({"font_color": ink, "border": 1, "border_color": "#D8E0E8", "num_format": "0.000"})
    pct = wb.add_format({"font_color": ink, "border": 1, "border_color": "#D8E0E8", "num_format": "0.0%"})
    pval = wb.add_format({"font_color": ink, "border": 1, "border_color": "#D8E0E8", "num_format": "0.000"})
    note = wb.add_format({"font_color": "#52606D", "font_size": 10, "text_wrap": True, "valign": "top"})
    kpi = wb.add_format({"bold": True, "font_size": 24, "font_color": navy, "align": "center", "valign": "vcenter", "bg_color": "#FFFFFF", "border": 1, "border_color": "#C7D4E0"})
    kpi_label = wb.add_format({"bold": True, "font_size": 10, "font_color": "#52606D", "align": "center", "valign": "vcenter", "bg_color": "#FFFFFF", "border": 1, "border_color": "#C7D4E0", "text_wrap": True})
    good = wb.add_format({"bg_color": "#DDF3ED", "font_color": "#0B6B57"})
    bad = wb.add_format({"bg_color": "#FCE8E8", "font_color": red})

    def write_frame(ws, df, start_row=0, start_col=0, widths=None, autofilter=True):
        for j, col in enumerate(df.columns):
            ws.write(start_row, start_col+j, col, header)
        for i, row in enumerate(df.itertuples(index=False), start_row+1):
            for j, value in enumerate(row):
                if pd.isna(value):
                    ws.write_blank(i, start_col+j, None, body)
                elif isinstance(value, (float, np.floating)):
                    ws.write_number(i, start_col+j, float(value), pval if "p_value" in df.columns[j] or "pvalue" in df.columns[j] else num)
                elif isinstance(value, (int, np.integer, bool, np.bool_)):
                    ws.write(i, start_col+j, value, body)
                else:
                    ws.write(i, start_col+j, str(value), body_wrap)
        if autofilter and len(df):
            ws.autofilter(start_row, start_col, start_row+len(df), start_col+len(df.columns)-1)
        ws.freeze_panes(start_row+1, start_col)
        for j, col in enumerate(df.columns):
            width = (widths or {}).get(col, min(max(len(col)+2, 12), 28))
            ws.set_column(start_col+j, start_col+j, width)

    ws = wb.add_worksheet("Главный вывод")
    ws.hide_gridlines(2)
    ws.set_column("A:A", 2); ws.set_column("B:H", 18)
    ws.set_row(0, 34)
    ws.merge_range("B1:H1", "DANO — готовое доказательство гипотезы", title)
    ws.merge_range("B3:H4", "После начала топливного кризиса частота штрафов снизилась примерно на 10%. Снижение сосредоточено среди клиентов с высокой докризисной топливной активностью; цена и возраст автомобиля сами по себе разницу не объясняют.", subtitle)
    labels = [("B6:C6", "B7:C8", "Общий эффект", headline["main_effect_pct"]), ("D6:E6", "D7:E8", "Q1: низкая активность", headline["q1_effect_pct"]), ("F6:G6", "F7:G8", "Q4: высокая активность", headline["q4_effect_pct"])]
    for lrange, vrange, lab, val in labels:
        ws.merge_range(lrange, lab, kpi_label)
        ws.merge_range(vrange, f"{val:.1f}%", kpi)
    ws.merge_range("B10:H10", "Почему это сильная гипотеза", section)
    bullets = [
        "Предиктор задан до кризиса — меньше риск перепутать причину и следствие.",
        "Модель сравнивает тех же клиентов и те же календарные месяцы 2025 и 2026 годов.",
        "Майский плацебо-коэффициент ≈1,00; после июня появляется лаговый спад.",
        "Результат сохраняется при полной истории 2025, одном автомобиле и апрельском baseline.",
        "Исходная идея про старые дешёвые и новые дорогие машины проверена, но не подтверждена (p=0,732).",
    ]
    for i, text in enumerate(bullets, 11):
        ws.merge_range(i, 1, i, 7, "• " + text, note)
        ws.set_row(i, 25)
    ws.merge_range("B18:H18", "Как защищать вывод", section)
    ws.merge_range("B19:H22", "Корректная формулировка: данные согласуются с механизмом снижения мобильности, но не доказывают, что именно цена бензина была единственной причиной. Мы наблюдаем штрафы, а не километраж; регион — регион регистрации клиента, а не АЗС. Поэтому рекомендуем продуктовый эксперимент, а не безусловное причинное обещание.", note)
    ws.merge_range("B24:H24", "Навигация", section)
    nav = [("Метод и дизайн", "Схема сравнения и определения"), ("Результаты моделей", "Коэффициенты и доверительные интервалы"), ("Event study", "Проверка динамики по месяцам"), ("Сегменты", "Q1–Q4 топливной активности"), ("Robustness", "Проверки устойчивости"), ("Клиенты", "Модельная таблица с созданными переменными")]
    for i, (sheet, desc) in enumerate(nav, 25):
        ws.write_url(i, 1, f"internal:'{sheet}'!A1", string=sheet, cell_format=body)
        ws.merge_range(i, 2, i, 7, desc, body_wrap)

    method = wb.add_worksheet("Метод и дизайн")
    method.hide_gridlines(2); method.set_column("A:A", 2); method.set_column("B:H", 19)
    method.merge_range("B1:H1", "Исследовательский дизайн", title)
    method.merge_range("B3:H3", "Вопрос: как кризис после конца мая 2026 года связан с частотой штрафов и для кого эффект сильнее?", subtitle)
    rows = [
        ("Единица наблюдения", "клиент × календарный месяц × год"),
        ("Период", "апрель–август 2025 и 2026; июнь–август — post"),
        ("Outcome", "число штрафов клиента в месяц"),
        ("Основная модель", "Conditional Poisson с фиксированными эффектами клиента; month FE; year 2026; year2026×post"),
        ("Главная гетерогенность", "year2026×post×квартиль докризисного положительного объёма топлива"),
        ("Почему не цена авто", "цена/возраст — слабые прокси достатка; проверяем их как альтернативу и контроли"),
        ("Стандартное топливо", "для ценовых метрик 50–130 ₽/л; raw-операции не удаляются"),
    ]
    method.write_row("B5", ["Элемент", "Решение"], header)
    for r, (a, b) in enumerate(rows, 5):
        method.write(r, 1, a, body); method.merge_range(r, 2, r, 7, b, body_wrap); method.set_row(r, 32)

    for name, df in tables.items():
        if name in {"Главный вывод", "Метод и дизайн"}: continue
        ws2 = wb.add_worksheet(name[:31])
        ws2.hide_gridlines(2)
        write_frame(ws2, df)
        ws2.set_default_row(18)
        if "p_value" in df.columns:
            c = df.columns.get_loc("p_value")
            ws2.conditional_format(1, c, len(df), c, {"type": "cell", "criteria": "<", "value": 0.05, "format": good})
            ws2.conditional_format(1, c, len(df), c, {"type": "cell", "criteria": ">=", "value": 0.05, "format": bad})

    # Native Excel charts on compact evidence sheets.
    ev = wb.get_worksheet_by_name("Event study")
    ev_chart = wb.add_chart({"type": "line"})
    ev_chart.add_series({"name": "IRR", "categories": "='Event study'!$B$2:$B$6", "values": "='Event study'!$C$2:$C$6", "line": {"color": teal, "width": 2.5}, "marker": {"type": "circle", "size": 7, "border": {"color": teal}, "fill": {"color": "#FFFFFF"}}})
    ev_chart.set_title({"name": "Динамика эффекта относительно апреля"}); ev_chart.set_y_axis({"name": "IRR", "min": 0.6, "max": 1.15, "major_gridlines": {"visible": True}}); ev_chart.set_legend({"none": True}); ev_chart.set_size({"width": 720, "height": 380})
    ev.insert_chart("I2", ev_chart)
    sg = wb.get_worksheet_by_name("Сегменты")
    sg_chart = wb.add_chart({"type": "column"})
    sg_chart.add_series({"name": "Изменение, %", "categories": "='Сегменты'!$B$2:$B$5", "values": "='Сегменты'!$E$2:$E$5", "fill": {"color": amber}, "border": {"none": True}})
    sg_chart.set_title({"name": "Эффект кризиса по докризисной активности"}); sg_chart.set_y_axis({"name": "%", "major_gridlines": {"visible": True}}); sg_chart.set_legend({"none": True}); sg_chart.set_size({"width": 720, "height": 380})
    sg.insert_chart("I2", sg_chart)
    wb.close()


def main():
    panel = pd.read_csv(PANEL_PATH, parse_dates=["month", "subscription_creation_date"])
    clients = pd.read_csv(CLIENT_PATH, parse_dates=["subscription_creation_date"])
    clients = prepare_clients(panel, clients)
    long, models, additive, event, segments, robust_main, robust_int, modifiers, fuel_dynamics = build_models(panel, clients)
    vehicle, vehicle_p, adjusted, validation, cuts = client_level_tests(clients)

    # Aggregate crisis signal.
    panel["crisis"] = panel["month_num"].ge(6)
    pf = panel.groupby("crisis", as_index=False).agg(clients=("client_id", "nunique"), months=("month_num", "nunique"), orders=("fuel_txn_positive", "sum"), volume=("fuel_volume_positive_l", "sum"), std_volume=("fuel_volume_standard_band_l", "sum"), std_spend=("fuel_spend_standard_band_rub", "sum"))
    pf["orders_pcm"] = pf["orders"] / pf["clients"] / pf["months"]
    pf["volume_pcm"] = pf["volume"] / pf["clients"] / pf["months"]
    pf["liters_per_order"] = pf["volume"] / pf["orders"]
    pf["price"] = pf["std_spend"] / pf["std_volume"]
    pre, post = pf.set_index("crisis").loc[False], pf.set_index("crisis").loc[True]
    crisis_signal = pd.DataFrame([
        ("Средняя цена standard band", pre["price"], post["price"], (post["price"] / pre["price"] - 1) * 100),
        ("Литры на клиента-месяц", pre["volume_pcm"], post["volume_pcm"], (post["volume_pcm"] / pre["volume_pcm"] - 1) * 100),
        ("Заказы на клиента-месяц", pre["orders_pcm"], post["orders_pcm"], (post["orders_pcm"] / pre["orders_pcm"] - 1) * 100),
        ("Литры на заказ", pre["liters_per_order"], post["liters_per_order"], (post["liters_per_order"] / pre["liters_per_order"] - 1) * 100),
    ], columns=["metric", "pre", "crisis", "change_pct"])

    variable_dictionary = pd.DataFrame([
        ("post_crisis", "1 для июня–августа", "Воздействие"), ("year_2026", "1 для 2026 года", "Воздействие"),
        ("post_x_year_2026", "post_crisis × year_2026", "Главный коэффициент"), ("fines_count", "Число штрафов клиента в месяц", "Outcome"),
        ("fine_did", "Средний (2026−2025) в post минус средний (2026−2025) в pre", "Outcome для EDA"),
        ("baseline_fuel_volume_l", "Средний положительный объём в апреле–мае 2026", "Докризисная активность"),
        ("baseline_fuel_orders", "Среднее число положительных заказов в апреле–мае", "Докризисная активность"),
        ("baseline_fuel_spend_standard_rub", "Средние расходы при цене 50–130 ₽/л в апреле–мае", "Докризисная активность"),
        ("fuel_intensity_q", "Квартиль baseline volume среди положительных покупателей", "Основная сегментация"),
        ("fuel_exposure_index", "baseline standard spend / median vehicle price", "Альтернативная сегментация"),
        ("vehicle_age_2026", "2026 − медианный год выпуска", "Автомобиль"), ("log_vehicle_price", "ln(1+медианная стоимость)", "Автомобиль"),
        ("vehicle_price_percentile_within_year", "Перцентиль цены среди машин того же года", "Автомобиль"),
        ("old_cheap", "возраст ≥15 и цена в нижнем квартиле", "Проверка исходной идеи"),
        ("new_expensive", "возраст ≤5 и цена в верхнем квартиле", "Проверка исходной идеи"),
        ("full_2025_coverage_flag", "подписка создана не позднее 01.04.2025", "Качество/robustness"),
        ("single_vehicle_flag", "у клиента один автомобиль", "Качество/robustness"),
        ("fuel_quality_share", "доля positive операций в диапазоне 50–130 ₽/л", "Качество"),
        ("region_price_shock_pct", "изменение standard-band цены post/pre по региону регистрации", "Регион"),
    ], columns=["variable", "definition", "role"])

    sources = pd.DataFrame([
        ("Задание", "Задание.pdf", "Исследовательский вопрос и данные"),
        ("Критерии", "Критерии.pdf", "Модель, значимость, robustness, визуализация, ограничения"),
        ("Литература 1", "https://www.sciencedirect.com/science/article/pii/S0001457519306499", "Рост цен топлива связан со снижением аварий через сокращение поездок/изменение поведения"),
        ("Литература 2", "https://www.sciencedirect.com/science/article/pii/S0001457523002439", "Панель ЕС: 10% роста цены связано примерно с 1,4% снижения столкновений"),
        ("Литература 3", "https://www.sciencedirect.com/science/article/pii/S0925753513001276", "Эффект цен топлива неоднороден по тяжести и территории"),
    ], columns=["source", "url_or_file", "use"])

    map_validation = pd.read_csv(MAP_CHECK_PATH)
    regions = pd.read_csv(REGION_PATH)
    quality = pd.read_csv(ROOT / "working_data" / "preanalysis" / "data_quality_preanalysis.csv")

    tables = {
        "Кризисный сигнал": crisis_signal,
        "Результаты моделей": models,
        "Аддитивная модель": additive,
        "Event study": event,
        "Сегменты": segments,
        "Robustness": robust_main,
        "Robustness сегментов": robust_int,
        "Модификаторы": modifiers,
        "Топливная динамика": fuel_dynamics,
        "Автомобили": vehicle,
        "Adjusted DiD": adjusted,
        "Валидация": validation,
        "Регионы": regions,
        "Проверка карт": map_validation,
        "Качество данных": quality,
        "Переменные": variable_dictionary,
        "Источники": sources,
        "Клиенты": clients,
    }
    for name, frame in tables.items():
        safe = name.lower().replace(" ", "_")
        frame.to_csv(WORK / f"{safe}.csv", index=False)
    long.to_csv(WORK / "model_panel_client_month_year.csv", index=False)

    main = models.loc[models["model"].eq("Основная conditional Poisson")].iloc[0]
    q1, q4 = segments.iloc[0], segments.iloc[-1]
    headline = {
        "main_effect_pct": (main["irr"] - 1) * 100,
        "main_irr": main["irr"], "main_ci": [main["ci95_low"], main["ci95_high"]], "main_p": main["p_value"],
        "q1_effect_pct": q1["effect_pct"], "q1_ci": [q1["ci95_low"], q1["ci95_high"]],
        "q4_effect_pct": q4["effect_pct"], "q4_ci": [q4["ci95_low"], q4["ci95_high"]],
        "interaction_irr": models.loc[models["model"].eq("Гетерогенность по топливной активности"), "irr"].iloc[0],
        "interaction_p": models.loc[models["model"].eq("Гетерогенность по топливной активности"), "p_value"].iloc[0],
        "vehicle_test_p": vehicle_p,
        "clients": int(panel["client_id"].nunique()),
        "vehicle_price_q25": float(clients["vehicle_price_median_rub"].quantile(.25)),
        "vehicle_price_q75": float(clients["vehicle_price_median_rub"].quantile(.75)),
        "discovery_cutoffs": cuts.tolist(),
        "map_passed": int(map_validation["validation_status"].eq("OK").sum()),
        "map_total": int(len(map_validation)),
    }
    (WORK / "headline.json").write_text(json.dumps(headline, ensure_ascii=False, indent=2), encoding="utf-8")
    build_workbook(OUT / "DANO_доказательство_гипотезы.xlsx", tables, headline)
    print(json.dumps(headline, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
