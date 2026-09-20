from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "client_month_analysis"
WORK = ROOT / "working_data" / "client_month_analysis"
FIG = ROOT / "figures" / "client_month_analysis"
for directory in (OUT, WORK, FIG):
    directory.mkdir(parents=True, exist_ok=True)

MONTHS = [4, 5, 6, 7, 8]
MONTH_NAMES = {4: "Апрель", 5: "Май", 6: "Июнь", 7: "Июль", 8: "Август"}
FINE_2025 = {
    4: "april_2025_fines",
    5: "may_2025_fines",
    6: "jun_2025_fines",
    7: "jul_2025_fines",
    8: "aug_2025_fines",
}
REGION_MAP = {
    "77": "Москва",
    "50": "Московская область",
    "78": "Санкт-Петербург",
    "16": "Республика Татарстан",
    "66": "Свердловская область",
    "54": "Новосибирская область",
    "47": "Ленинградская область",
    "52": "Нижегородская область",
    "23": "Краснодарский край",
    "63": "Самарская область",
    "74": "Челябинская область",
    "42": "Кемеровская область",
    "72": "Тюменская область",
    "02": "Республика Башкортостан",
    "24": "Красноярский край",
    "55": "Омская область",
}

# Sensitivity band for ordinary gasoline/diesel. Values outside it are preserved
# and flagged, not silently corrected or deleted.
STANDARD_PRICE_MIN = 50.0
STANDARD_PRICE_MAX = 130.0


def mode_or_first(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return np.nan
    modes = values.mode()
    return modes.iloc[0] if not modes.empty else values.iloc[0]


def parse_engine(engine: pd.Series) -> tuple[pd.Series, pd.Series]:
    displacement = pd.to_numeric(
        engine.astype("string")
        .str.extract(r"^\s*([0-9]+(?:[.,][0-9]+)?)", expand=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )
    power = pd.to_numeric(
        engine.astype("string")
        .str.extract(r"\(([0-9]+(?:[.,][0-9]+)?)\s*л\.с\.", expand=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )
    return displacement, power


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan))


def build_profile(demographics: pd.DataFrame) -> pd.DataFrame:
    demographics = demographics.copy()
    demographics["kladr_code"] = demographics["kladr_code"].astype("string").str.zfill(2)
    demographics["engine_displacement_l"], demographics["engine_power_hp"] = parse_engine(
        demographics["engine_type"]
    )

    vehicle_pairs = (
        demographics.groupby(["client_id", "auto_document_id"], as_index=False, dropna=False)
        .agg(
            gender=("gender", mode_or_first),
            age_type_code=("age_type_code", mode_or_first),
            auto_year=("auto_year", "median"),
            vehicle_price_rub=("price", "median"),
            kladr_code=("kladr_code", mode_or_first),
            engine_displacement_l=("engine_displacement_l", "median"),
            engine_power_hp=("engine_power_hp", "median"),
            subscription_creation_date=("subscription_creation_date", "min"),
            **{column: (column, "max") for column in FINE_2025.values()},
            fines_last_6_month=("fines_last_6_month", "max"),
            fines_last_12_month=("fines_last_12_month", "max"),
            fines_last_24_month=("fines_last_24_month", "max"),
            fines_last_36_month=("fines_last_36_month", "max"),
        )
    )

    profile = (
        vehicle_pairs.groupby("client_id", as_index=False)
        .agg(
            gender=("gender", mode_or_first),
            age_type_code=("age_type_code", mode_or_first),
            kladr_code=("kladr_code", mode_or_first),
            vehicle_count=("auto_document_id", "nunique"),
            vehicle_price_median_rub=("vehicle_price_rub", "median"),
            vehicle_year_median=("auto_year", "median"),
            engine_power_hp_median=("engine_power_hp", "median"),
            engine_displacement_l_median=("engine_displacement_l", "median"),
            subscription_creation_date=("subscription_creation_date", "min"),
            **{f"fines_2025_{month:02d}": (column, "sum") for month, column in FINE_2025.items()},
            fines_last_6_month=("fines_last_6_month", "sum"),
            fines_last_12_month=("fines_last_12_month", "sum"),
            fines_last_24_month=("fines_last_24_month", "sum"),
            fines_last_36_month=("fines_last_36_month", "sum"),
        )
    )
    profile["registration_region"] = profile["kladr_code"].map(REGION_MAP).fillna("Другой/неизвестный")
    return profile


def add_fuel_flags(fuel: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    fuel = fuel.copy()
    fuel["month_num"] = fuel["order_datetime"].dt.month
    fuel["is_positive_volume"] = fuel["order_fuel_volume"] > 0
    fuel["is_negative_volume"] = fuel["order_fuel_volume"] < 0
    fuel["is_low_price"] = fuel["order_fuel_price_1liter"] < STANDARD_PRICE_MIN
    fuel["is_high_price"] = fuel["order_fuel_price_1liter"] > STANDARD_PRICE_MAX
    fuel["is_standard_price"] = fuel["order_fuel_price_1liter"].between(
        STANDARD_PRICE_MIN, STANDARD_PRICE_MAX, inclusive="both"
    )
    fuel["is_standard_purchase"] = fuel["is_positive_volume"] & fuel["is_standard_price"]
    fuel["is_positive_low_price"] = fuel["is_positive_volume"] & fuel["is_low_price"]
    fuel["is_positive_high_price"] = fuel["is_positive_volume"] & fuel["is_high_price"]
    positive = fuel.loc[fuel["is_positive_volume"], "order_fuel_volume"]
    extreme_volume_threshold = float(positive.quantile(0.995))
    fuel["is_extreme_volume"] = fuel["is_positive_volume"] & (
        fuel["order_fuel_volume"] > extreme_volume_threshold
    )
    fuel["spend_net_rub"] = fuel["order_fuel_volume"] * fuel["order_fuel_price_1liter"]
    fuel["volume_positive_l"] = fuel["order_fuel_volume"].where(fuel["is_positive_volume"], 0.0)
    fuel["volume_negative_abs_l"] = (-fuel["order_fuel_volume"]).where(fuel["is_negative_volume"], 0.0)
    fuel["spend_positive_rub"] = fuel["spend_net_rub"].where(fuel["is_positive_volume"], 0.0)
    fuel["volume_standard_l"] = fuel["order_fuel_volume"].where(fuel["is_standard_purchase"], 0.0)
    fuel["spend_standard_rub"] = fuel["spend_net_rub"].where(fuel["is_standard_purchase"], 0.0)
    return fuel, extreme_volume_threshold


def aggregate_fuel_month(fuel: pd.DataFrame) -> pd.DataFrame:
    keys = ["client_id", "month_num"]
    base = fuel.groupby(keys, as_index=False).agg(
        fuel_txn_all=("order_id", "size"),
        fuel_txn_positive=("is_positive_volume", "sum"),
        fuel_txn_negative=("is_negative_volume", "sum"),
        fuel_txn_low_price=("is_positive_low_price", "sum"),
        fuel_txn_high_price=("is_positive_high_price", "sum"),
        fuel_txn_standard_band=("is_standard_purchase", "sum"),
        fuel_txn_extreme_volume=("is_extreme_volume", "sum"),
        fuel_volume_net_l=("order_fuel_volume", "sum"),
        fuel_volume_positive_l=("volume_positive_l", "sum"),
        fuel_volume_negative_abs_l=("volume_negative_abs_l", "sum"),
        fuel_volume_standard_band_l=("volume_standard_l", "sum"),
        fuel_spend_net_rub=("spend_net_rub", "sum"),
        fuel_spend_positive_rub=("spend_positive_rub", "sum"),
        fuel_spend_standard_band_rub=("spend_standard_rub", "sum"),
        fuel_price_mean_raw=("order_fuel_price_1liter", "mean"),
        fuel_price_median_raw=("order_fuel_price_1liter", "median"),
    )
    standard_median = (
        fuel.loc[fuel["is_standard_purchase"]]
        .groupby(keys)["order_fuel_price_1liter"]
        .median()
        .rename("fuel_price_median_standard_band")
        .reset_index()
    )
    base = base.merge(standard_median, on=keys, how="left")
    base["fuel_price_weighted_standard_band"] = safe_divide(
        base["fuel_spend_standard_band_rub"], base["fuel_volume_standard_band_l"]
    )
    base["fuel_low_price_share_positive"] = safe_divide(base["fuel_txn_low_price"], base["fuel_txn_positive"])
    base["fuel_high_price_share_positive"] = safe_divide(base["fuel_txn_high_price"], base["fuel_txn_positive"])
    base["fuel_negative_txn_share_all"] = safe_divide(base["fuel_txn_negative"], base["fuel_txn_all"])
    return base


def build_panel(profile: pd.DataFrame, fines: pd.DataFrame, fuel_month: pd.DataFrame) -> pd.DataFrame:
    fines = fines.copy()
    fines["month_num"] = fines["bill_offence_date"].dt.month
    fines["fine_amount_rub"] = fines["total_fine_amount"] / 100.0
    text = fines["offence_short_statement"].fillna("")
    fines["is_severe_offence"] = text.str.contains(
        r"40-60|60-80|более чем на 80|красн|встречн|телефон", case=False, regex=True
    )
    fines["is_speeding"] = text.str.contains("превышение скорости", case=False, regex=False)
    fine_month = (
        fines.groupby(["client_id", "month_num"], as_index=False)
        .agg(
            fines_2026=("bill_id", "size"),
            fine_amount_2026_rub=("fine_amount_rub", "sum"),
            fine_amount_median_2026_rub=("fine_amount_rub", "median"),
            severe_fines_2026=("is_severe_offence", "sum"),
            speeding_fines_2026=("is_speeding", "sum"),
            offence_types_2026=("offence_short_statement", "nunique"),
            offence_regions_2026=("region_name", "nunique"),
        )
    )

    grid = pd.MultiIndex.from_product(
        [profile["client_id"], MONTHS], names=["client_id", "month_num"]
    ).to_frame(index=False)
    panel = grid.merge(profile, on="client_id", how="left")
    fine_lookup = profile.set_index("client_id")[[f"fines_2025_{m:02d}" for m in MONTHS]]
    for month in MONTHS:
        idx = panel["month_num"].eq(month)
        panel.loc[idx, "fines_2025"] = panel.loc[idx, "client_id"].map(
            fine_lookup[f"fines_2025_{month:02d}"]
        )
    panel = panel.merge(fine_month, on=["client_id", "month_num"], how="left")
    panel = panel.merge(fuel_month, on=["client_id", "month_num"], how="left")

    zero_columns = [
        "fines_2026", "fine_amount_2026_rub", "severe_fines_2026", "speeding_fines_2026",
        "offence_types_2026", "offence_regions_2026", "fuel_txn_all", "fuel_txn_positive",
        "fuel_txn_negative", "fuel_txn_low_price", "fuel_txn_high_price", "fuel_txn_standard_band",
        "fuel_txn_extreme_volume", "fuel_volume_net_l", "fuel_volume_positive_l",
        "fuel_volume_negative_abs_l", "fuel_volume_standard_band_l", "fuel_spend_net_rub",
        "fuel_spend_positive_rub", "fuel_spend_standard_band_rub",
    ]
    panel[zero_columns] = panel[zero_columns].fillna(0)
    panel["month"] = pd.to_datetime(dict(year=2026, month=panel["month_num"], day=1))
    panel["month_name"] = panel["month_num"].map(MONTH_NAMES)
    panel["period"] = np.where(panel["month_num"] <= 5, "До кризиса", "Кризис")
    panel["subscription_tenure_months"] = (
        (panel["month"].dt.year - panel["subscription_creation_date"].dt.year) * 12
        + panel["month"].dt.month
        - panel["subscription_creation_date"].dt.month
    ).clip(lower=0)
    panel["fines_change_abs_2026_vs_2025"] = panel["fines_2026"] - panel["fines_2025"]
    panel["fines_change_pct_2026_vs_2025"] = safe_divide(
        panel["fines_2026"] - panel["fines_2025"], panel["fines_2025"]
    )
    panel["fuel_quality_flag"] = np.select(
        [
            panel["fuel_txn_all"].eq(0),
            panel["fuel_txn_negative"].gt(0) | panel["fuel_txn_low_price"].gt(0) | panel["fuel_txn_high_price"].gt(0),
        ],
        ["Нет операций", "Есть флаги"],
        default="Без флагов",
    )
    ordered = [
        "client_id", "month", "month_num", "month_name", "period",
        "gender", "age_type_code", "kladr_code", "registration_region",
        "vehicle_count", "vehicle_price_median_rub", "vehicle_year_median",
        "engine_power_hp_median", "engine_displacement_l_median",
        "subscription_creation_date", "subscription_tenure_months",
        "fines_2025", "fines_2026", "fines_change_abs_2026_vs_2025",
        "fines_change_pct_2026_vs_2025", "fine_amount_2026_rub",
        "fine_amount_median_2026_rub", "severe_fines_2026", "speeding_fines_2026",
        "offence_types_2026", "offence_regions_2026",
        "fuel_txn_all", "fuel_txn_positive", "fuel_txn_negative", "fuel_txn_standard_band",
        "fuel_txn_low_price", "fuel_txn_high_price", "fuel_txn_extreme_volume",
        "fuel_volume_net_l", "fuel_volume_positive_l", "fuel_volume_negative_abs_l",
        "fuel_volume_standard_band_l", "fuel_spend_net_rub", "fuel_spend_positive_rub",
        "fuel_spend_standard_band_rub", "fuel_price_mean_raw", "fuel_price_median_raw",
        "fuel_price_median_standard_band", "fuel_price_weighted_standard_band",
        "fuel_low_price_share_positive", "fuel_high_price_share_positive",
        "fuel_negative_txn_share_all", "fuel_quality_flag",
        "fines_last_6_month", "fines_last_12_month", "fines_last_24_month", "fines_last_36_month",
    ]
    return panel[ordered].sort_values(["client_id", "month_num"]).reset_index(drop=True)


def build_monthly(panel: pd.DataFrame, fuel: pd.DataFrame) -> pd.DataFrame:
    monthly = panel.groupby("month_num", as_index=False).agg(
        clients=("client_id", "nunique"),
        fines_2025=("fines_2025", "sum"),
        fines_2026=("fines_2026", "sum"),
        active_fine_clients_2025=("fines_2025", lambda s: int((s > 0).sum())),
        active_fine_clients_2026=("fines_2026", lambda s: int((s > 0).sum())),
        fine_amount_2026_rub=("fine_amount_2026_rub", "sum"),
        fuel_clients=("fuel_txn_positive", lambda s: int((s > 0).sum())),
        fuel_txn_all=("fuel_txn_all", "sum"),
        fuel_txn_positive=("fuel_txn_positive", "sum"),
        fuel_txn_negative=("fuel_txn_negative", "sum"),
        fuel_txn_low_price=("fuel_txn_low_price", "sum"),
        fuel_txn_high_price=("fuel_txn_high_price", "sum"),
        fuel_txn_standard_band=("fuel_txn_standard_band", "sum"),
        fuel_volume_net_l=("fuel_volume_net_l", "sum"),
        fuel_volume_positive_l=("fuel_volume_positive_l", "sum"),
        fuel_volume_standard_band_l=("fuel_volume_standard_band_l", "sum"),
        fuel_spend_standard_band_rub=("fuel_spend_standard_band_rub", "sum"),
    )
    price = fuel.groupby("month_num")["order_fuel_price_1liter"].median().rename("fuel_price_median_raw")
    standard_price = (
        fuel.loc[fuel["is_standard_purchase"]]
        .groupby("month_num")["order_fuel_price_1liter"]
        .median()
        .rename("fuel_price_median_standard_band")
    )
    monthly = monthly.merge(price, on="month_num", how="left").merge(standard_price, on="month_num", how="left")
    monthly["fuel_price_weighted_standard_band"] = safe_divide(
        monthly["fuel_spend_standard_band_rub"], monthly["fuel_volume_standard_band_l"]
    )
    monthly["month"] = pd.to_datetime(dict(year=2026, month=monthly["month_num"], day=1))
    monthly["month_name"] = monthly["month_num"].map(MONTH_NAMES)
    monthly["fines_per_client_2025"] = monthly["fines_2025"] / monthly["clients"]
    monthly["fines_per_client_2026"] = monthly["fines_2026"] / monthly["clients"]
    monthly["fines_change_pct"] = safe_divide(
        monthly["fines_2026"] - monthly["fines_2025"], monthly["fines_2025"]
    )
    return monthly


def build_regions(panel: pd.DataFrame) -> pd.DataFrame:
    region = panel.loc[panel["registration_region"].ne("Другой/неизвестный")].groupby(
        "registration_region", as_index=False
    ).agg(
        clients=("client_id", "nunique"),
        fines_2025=("fines_2025", "sum"),
        fines_2026=("fines_2026", "sum"),
        fine_amount_2026_rub=("fine_amount_2026_rub", "sum"),
        fuel_clients=("fuel_txn_positive", lambda s: s.index[panel.loc[s.index, "fuel_txn_positive"].gt(0)].size),
        fuel_txn_positive=("fuel_txn_positive", "sum"),
        fuel_txn_negative=("fuel_txn_negative", "sum"),
        fuel_txn_low_price=("fuel_txn_low_price", "sum"),
        fuel_txn_high_price=("fuel_txn_high_price", "sum"),
        fuel_txn_standard_band=("fuel_txn_standard_band", "sum"),
        fuel_volume_positive_l=("fuel_volume_positive_l", "sum"),
        fuel_volume_standard_band_l=("fuel_volume_standard_band_l", "sum"),
        fuel_spend_standard_band_rub=("fuel_spend_standard_band_rub", "sum"),
    )
    # Count unique clients with at least one positive fuel transaction in the five-month window.
    active_fuel = (
        panel.loc[panel["fuel_txn_positive"].gt(0)]
        .groupby("registration_region")["client_id"].nunique()
    )
    region["fuel_clients"] = region["registration_region"].map(active_fuel).fillna(0).astype(int)
    medians = (
        panel.loc[panel["fuel_price_median_standard_band"].notna()]
        .groupby("registration_region")["fuel_price_median_standard_band"]
        .median()
    )
    region["fuel_price_median_standard_band"] = region["registration_region"].map(medians)
    region["fuel_price_weighted_standard_band"] = safe_divide(
        region["fuel_spend_standard_band_rub"], region["fuel_volume_standard_band_l"]
    )
    region["fines_per_client_2025"] = region["fines_2025"] / region["clients"]
    region["fines_per_client_2026"] = region["fines_2026"] / region["clients"]
    region["fines_change_pct"] = safe_divide(
        region["fines_2026"] - region["fines_2025"], region["fines_2025"]
    )
    region["fuel_volume_per_fuel_client_l"] = safe_divide(
        region["fuel_volume_positive_l"], region["fuel_clients"]
    )
    return region.sort_values("clients", ascending=False).reset_index(drop=True)


def build_client_changes(panel: pd.DataFrame) -> pd.DataFrame:
    working = panel.copy()
    period = working["month_num"].le(5).map({True: "pre", False: "crisis"})
    working["analysis_period"] = period
    metrics = [
        "fines_2025", "fines_2026", "fuel_volume_positive_l",
        "fuel_volume_standard_band_l", "fuel_spend_standard_band_rub",
    ]
    means = working.pivot_table(index="client_id", columns="analysis_period", values=metrics, aggfunc="mean")
    means.columns = [f"{metric}_{period}_monthly" for metric, period in means.columns]
    means = means.reset_index()
    price = working.pivot_table(
        index="client_id", columns="analysis_period", values="fuel_price_median_standard_band", aggfunc="median"
    )
    price.columns = [f"fuel_price_standard_{period}" for period in price.columns]
    price = price.reset_index()
    changes = means.merge(price, on="client_id", how="outer")
    pairs = [
        ("fines_2025", "fines_2025"),
        ("fines_2026", "fines_2026"),
        ("fuel_volume_positive_l", "fuel_volume_positive_l"),
        ("fuel_volume_standard_band_l", "fuel_volume_standard_band_l"),
    ]
    for prefix, label in pairs:
        pre, crisis = f"{prefix}_pre_monthly", f"{prefix}_crisis_monthly"
        changes[f"{label}_change_abs"] = changes[crisis] - changes[pre]
        changes[f"{label}_change_pct"] = safe_divide(changes[crisis] - changes[pre], changes[pre])
    changes["fuel_price_standard_change_abs"] = changes["fuel_price_standard_crisis"] - changes["fuel_price_standard_pre"]
    changes["fuel_price_standard_change_pct"] = safe_divide(
        changes["fuel_price_standard_crisis"] - changes["fuel_price_standard_pre"],
        changes["fuel_price_standard_pre"],
    )
    return changes


def correlation_matrix(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in data.columns]
    return data[available].corr(method="spearman", min_periods=100)


def histogram_table(series: pd.Series, edges: list[float], label: str) -> pd.DataFrame:
    categories = pd.cut(series, bins=edges, right=False, include_lowest=True)
    counts = categories.value_counts(sort=False)
    result = counts.rename("count").reset_index().rename(columns={series.name: "interval"})
    if "interval" not in result.columns:
        result = result.rename(columns={result.columns[0]: "interval"})
    result["interval"] = result["interval"].astype(str)
    result["metric"] = label
    result["share"] = result["count"] / result["count"].sum()
    return result[["metric", "interval", "count", "share"]]


def main() -> None:
    demographics = pd.read_csv(
        ROOT / "clients_demographics.csv", sep=";", decimal=",", dtype={"kladr_code": "string"},
        parse_dates=["subscription_creation_date"]
    )
    fines = pd.read_csv(ROOT / "fines_2026.csv", sep=";", parse_dates=["bill_offence_date"])
    fuel = pd.read_csv(
        ROOT / "fuel_transaction.csv", sep=";", decimal=",", parse_dates=["order_datetime"]
    )
    start, end = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-09-01")
    fines = fines.loc[fines["bill_offence_date"].between(start, end, inclusive="left")].copy()
    fuel = fuel.loc[fuel["order_datetime"].between(start, end, inclusive="left")].copy()

    profile = build_profile(demographics)
    fuel, extreme_volume_threshold = add_fuel_flags(fuel)
    fuel_month = aggregate_fuel_month(fuel)
    panel = build_panel(profile, fines, fuel_month)
    monthly = build_monthly(panel, fuel)
    regions = build_regions(panel)
    changes = build_client_changes(panel)

    level_columns = [
        "fines_2025", "fines_2026", "fine_amount_2026_rub", "severe_fines_2026",
        "speeding_fines_2026", "fuel_txn_positive", "fuel_volume_positive_l",
        "fuel_volume_standard_band_l", "fuel_price_median_standard_band",
        "fuel_spend_standard_band_rub", "vehicle_count", "vehicle_price_median_rub",
        "vehicle_year_median", "engine_power_hp_median", "subscription_tenure_months",
    ]
    change_columns = [
        "fines_2025_change_abs", "fines_2026_change_abs", "fuel_volume_positive_l_change_abs",
        "fuel_volume_standard_band_l_change_abs", "fuel_price_standard_change_abs",
        "fuel_price_standard_change_pct",
    ]
    corr_level = correlation_matrix(panel, level_columns)
    corr_change = correlation_matrix(changes, change_columns)

    price_hist = histogram_table(
        fuel["order_fuel_price_1liter"],
        [0, 25, 40, 50, 55, 60, 65, 70, 75, 80, 90, 110, 130, 150, 200, 250, 300, 350, 400, np.inf],
        "Цена, ₽/л",
    )
    volume_hist = histogram_table(
        fuel["order_fuel_volume"],
        [-np.inf, -100, -50, -20, 0, 10, 20, 30, 40, 50, 60, 80, 100, 150, 250, 500, np.inf],
        "Объём, л",
    )
    histograms = pd.concat([price_hist, volume_hist], ignore_index=True)

    quality_rows = [
        ("Период анализа", "2026-04-01 — 2026-08-31", "Фиксированное окно задания"),
        ("Транзакции топлива", int(len(fuel)), "Все операции, включая возвраты/корректировки"),
        ("Положительный объём", int(fuel["is_positive_volume"].sum()), "Основной сценарий покупок"),
        ("Отрицательный объём", int(fuel["is_negative_volume"].sum()), "Сохранены как вероятные возвраты/корректировки"),
        (f"Цена ниже {STANDARD_PRICE_MIN:.0f} ₽/л", int((fuel["order_fuel_price_1liter"] < STANDARD_PRICE_MIN).sum()), "Флаг; не удалено"),
        (f"Цена {STANDARD_PRICE_MIN:.0f}–{STANDARD_PRICE_MAX:.0f} ₽/л и объём > 0", int(fuel["is_standard_purchase"].sum()), "Основной робастный сценарий"),
        (f"Цена выше {STANDARD_PRICE_MAX:.0f} ₽/л", int((fuel["order_fuel_price_1liter"] > STANDARD_PRICE_MAX).sum()), "Флаг; возможен иной продукт/тип топлива"),
        ("Цена 500 ₽/л и выше", int((fuel["order_fuel_price_1liter"] >= 500).sum()), "В текущих данных отсутствует"),
        ("Минимальная цена, ₽/л", float(fuel["order_fuel_price_1liter"].min()), "Raw"),
        ("Максимальная цена, ₽/л", float(fuel["order_fuel_price_1liter"].max()), "Raw"),
        ("Минимальный объём, л", float(fuel["order_fuel_volume"].min()), "Raw"),
        ("Максимальный объём, л", float(fuel["order_fuel_volume"].max()), "Raw"),
        ("Порог экстремального положительного объёма, л", extreme_volume_threshold, "99,5-й перцентиль; только флаг"),
        ("Операции выше порога объёма", int(fuel["is_extreme_volume"].sum()), "Сохранены; используйте чувствительный анализ"),
    ]
    quality = pd.DataFrame(quality_rows, columns=["indicator", "value", "interpretation"])

    profile.to_csv(WORK / "client_profile.csv", index=False)
    panel.to_csv(WORK / "client_month_panel_quality.csv", index=False)
    monthly.to_csv(WORK / "monthly_summary_quality.csv", index=False)
    regions.to_csv(WORK / "region_summary_quality.csv", index=False)
    changes.to_csv(WORK / "client_changes.csv", index=False)
    corr_level.to_csv(WORK / "correlation_level_spearman.csv")
    corr_change.to_csv(WORK / "correlation_change_spearman.csv")
    quality.to_csv(WORK / "fuel_quality_summary.csv", index=False)
    histograms.to_csv(WORK / "fuel_histograms.csv", index=False)

    metadata = {
        "panel_rows": int(len(panel)),
        "panel_columns": int(panel.shape[1]),
        "unique_clients": int(panel["client_id"].nunique()),
        "months": MONTHS,
        "fines_2025_total": int(panel["fines_2025"].sum()),
        "fines_2026_total": int(panel["fines_2026"].sum()),
        "fuel_transactions": int(panel["fuel_txn_all"].sum()),
        "fuel_negative_transactions": int(panel["fuel_txn_negative"].sum()),
        "fuel_low_price_transactions": int(panel["fuel_txn_low_price"].sum()),
        "fuel_high_price_transactions": int(panel["fuel_txn_high_price"].sum()),
        "standard_price_min": STANDARD_PRICE_MIN,
        "standard_price_max": STANDARD_PRICE_MAX,
        "extreme_volume_threshold": extreme_volume_threshold,
        "price_min": float(fuel["order_fuel_price_1liter"].min()),
        "price_max": float(fuel["order_fuel_price_1liter"].max()),
        "volume_min": float(fuel["order_fuel_volume"].min()),
        "volume_max": float(fuel["order_fuel_volume"].max()),
    }
    (WORK / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
