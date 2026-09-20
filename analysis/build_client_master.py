from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp" / "client_master"
TMP.mkdir(parents=True, exist_ok=True)

MONTHS = [4, 5, 6, 7, 8]


def mode_or_blank(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    if values.empty:
        return ""
    modes = values.mode()
    return modes.iloc[0] if not modes.empty else values.iloc[0]


profile = pd.read_csv(
    ROOT / "working_data" / "client_profile.csv",
    dtype={"client_id": "string", "kladr_code": "string"},
    parse_dates=["subscription_creation_date"],
)
vehicles = pd.read_csv(
    ROOT / "working_data" / "vehicle_pairs_deduplicated.csv",
    dtype={"client_id": "string", "auto_document_id": "string"},
)
fines = pd.read_csv(
    ROOT / "fines_2026.csv",
    sep=";",
    dtype={"client_id": "string"},
    parse_dates=["bill_offence_date"],
)
fuel = pd.read_csv(
    ROOT / "fuel_transaction.csv",
    sep=";",
    decimal=",",
    dtype={"client_id": "string"},
    parse_dates=["order_datetime"],
)

start, end = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-09-01")
fines = fines.loc[fines["bill_offence_date"].between(start, end, inclusive="left")].copy()
fuel = fuel.loc[fuel["order_datetime"].between(start, end, inclusive="left")].copy()
fines["month"] = fines["bill_offence_date"].dt.month
fuel["month"] = fuel["order_datetime"].dt.month
fines["fine_amount_rub"] = fines["total_fine_amount"] / 100
fines["severe_offence"] = fines["offence_short_statement"].str.contains(
    r"40-60|60-80|более чем на 80|красный|встречного|телефона",
    case=False,
    regex=True,
)
fuel["positive_order"] = fuel["order_fuel_volume"] > 0
fuel["positive_volume_l"] = fuel["order_fuel_volume"].clip(lower=0)
fuel["net_spend_rub"] = fuel["order_fuel_volume"] * fuel["order_fuel_price_1liter"]
fuel["positive_spend_rub"] = fuel["positive_volume_l"] * fuel["order_fuel_price_1liter"]

vehicle_lists = (
    vehicles.groupby("client_id", as_index=False)
    .agg(
        auto_document_ids=("auto_document_id", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
        auto_marks=("auto_mark", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
        auto_colors=("color", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
    )
)

fine_month = fines.pivot_table(
    index="client_id", columns="month", values="bill_id", aggfunc="count", fill_value=0
)
fine_month = fine_month.reindex(columns=MONTHS, fill_value=0)
fine_month.columns = [f"fines_2026_{month:02d}" for month in MONTHS]
fine_month = fine_month.reset_index()

fine_total = (
    fines.groupby("client_id", as_index=False)
    .agg(
        fines_2026_apr_aug_total=("bill_id", "count"),
        fine_amount_2026_total_rub=("fine_amount_rub", "sum"),
        severe_fines_2026_total=("severe_offence", "sum"),
        offence_types_2026=("offence_short_statement", "nunique"),
        top_offence_2026=("offence_short_statement", mode_or_blank),
        offence_regions_2026=("region_name", "nunique"),
        top_offence_region_2026=("region_name", mode_or_blank),
    )
)

fuel_transactions = fuel.pivot_table(
    index="client_id", columns="month", values="order_id", aggfunc="count", fill_value=0
).reindex(columns=MONTHS, fill_value=0)
fuel_transactions.columns = [f"fuel_transactions_2026_{month:02d}" for month in MONTHS]
fuel_transactions = fuel_transactions.reset_index()

fuel_volume = fuel.pivot_table(
    index="client_id", columns="month", values="positive_volume_l", aggfunc="sum", fill_value=0
).reindex(columns=MONTHS, fill_value=0)
fuel_volume.columns = [f"fuel_volume_l_2026_{month:02d}" for month in MONTHS]
fuel_volume = fuel_volume.reset_index()

fuel_total = (
    fuel.groupby("client_id", as_index=False)
    .agg(
        fuel_transactions_apr_aug_total=("order_id", "count"),
        fuel_positive_orders_apr_aug_total=("positive_order", "sum"),
        fuel_volume_l_apr_aug_total=("positive_volume_l", "sum"),
        fuel_net_volume_l_apr_aug_total=("order_fuel_volume", "sum"),
        fuel_spend_positive_rub=("positive_spend_rub", "sum"),
        fuel_spend_net_rub=("net_spend_rub", "sum"),
        fuel_price_median_rub_l=("order_fuel_price_1liter", "median"),
        fuel_price_mean_rub_l=("order_fuel_price_1liter", "mean"),
        fuel_active_months=("month", "nunique"),
    )
)

base_columns = [
    "client_id",
    "gender",
    "age_type_code",
    "kladr_code",
    "registration_region",
    "vehicle_count",
    "vehicle_year_median",
    "vehicle_price_median",
    "engine_power_hp_median",
    "engine_displacement_l_median",
    "subscription_creation_date",
    *[f"fines_2025_{month:02d}" for month in MONTHS],
    "fines_last_6_month",
    "fines_last_12_month",
    "fines_last_24_month",
    "fines_last_36_month",
]
master = profile[base_columns].merge(vehicle_lists, on="client_id", how="left")
for frame in (fine_month, fine_total, fuel_transactions, fuel_volume, fuel_total):
    master = master.merge(frame, on="client_id", how="left", validate="one_to_one")

count_prefixes = (
    "fines_2026_",
    "fine_amount_2026_",
    "severe_fines_2026_",
    "offence_types_2026",
    "offence_regions_2026",
    "fuel_transactions_",
    "fuel_positive_orders_",
    "fuel_volume_l_",
    "fuel_net_volume_l_",
    "fuel_spend_",
    "fuel_active_months",
)
numeric_fill = [column for column in master.columns if column.startswith(count_prefixes)]
master[numeric_fill] = master[numeric_fill].fillna(0)
for column in ["top_offence_2026", "top_offence_region_2026"]:
    master[column] = master[column].fillna("")

master["fines_2025_apr_aug_total"] = master[
    [f"fines_2025_{month:02d}" for month in MONTHS]
].sum(axis=1)
master["fines_change_2026_minus_2025"] = (
    master["fines_2026_apr_aug_total"] - master["fines_2025_apr_aug_total"]
)
master["fines_change_pct_2026_vs_2025"] = np.where(
    master["fines_2025_apr_aug_total"] > 0,
    master["fines_2026_apr_aug_total"] / master["fines_2025_apr_aug_total"] - 1,
    np.nan,
)

ordered = [
    "client_id",
    "gender",
    "age_type_code",
    "kladr_code",
    "registration_region",
    "vehicle_count",
    "auto_document_ids",
    "auto_marks",
    "auto_colors",
    "vehicle_year_median",
    "vehicle_price_median",
    "engine_power_hp_median",
    "engine_displacement_l_median",
    "subscription_creation_date",
    *[f"fines_2025_{month:02d}" for month in MONTHS],
    "fines_2025_apr_aug_total",
    "fines_last_6_month",
    "fines_last_12_month",
    "fines_last_24_month",
    "fines_last_36_month",
    *[f"fines_2026_{month:02d}" for month in MONTHS],
    "fines_2026_apr_aug_total",
    "fine_amount_2026_total_rub",
    "severe_fines_2026_total",
    "offence_types_2026",
    "top_offence_2026",
    "offence_regions_2026",
    "top_offence_region_2026",
    "fines_change_2026_minus_2025",
    "fines_change_pct_2026_vs_2025",
    *[f"fuel_transactions_2026_{month:02d}" for month in MONTHS],
    "fuel_transactions_apr_aug_total",
    "fuel_positive_orders_apr_aug_total",
    *[f"fuel_volume_l_2026_{month:02d}" for month in MONTHS],
    "fuel_volume_l_apr_aug_total",
    "fuel_net_volume_l_apr_aug_total",
    "fuel_spend_positive_rub",
    "fuel_spend_net_rub",
    "fuel_price_median_rub_l",
    "fuel_price_mean_rub_l",
    "fuel_active_months",
]
master = master[ordered].sort_values("client_id").reset_index(drop=True)
master["subscription_creation_date"] = master["subscription_creation_date"].dt.strftime("%Y-%m-%d")

checks = {
    "rows": int(len(master)),
    "unique_client_ids": int(master["client_id"].nunique()),
    "duplicate_client_ids": int(master["client_id"].duplicated().sum()),
    "columns": int(master.shape[1]),
    "fines_2025_total": int(master["fines_2025_apr_aug_total"].sum()),
    "fines_2026_total": int(master["fines_2026_apr_aug_total"].sum()),
    "fuel_transactions_total": int(master["fuel_transactions_apr_aug_total"].sum()),
    "fuel_positive_volume_total": float(master["fuel_volume_l_apr_aug_total"].sum()),
    "source_fines_2026_rows": int(len(fines)),
    "source_fuel_rows": int(len(fuel)),
}
assert checks["rows"] == checks["unique_client_ids"] == profile["client_id"].nunique()
assert checks["duplicate_client_ids"] == 0
assert checks["fines_2026_total"] == checks["source_fines_2026_rows"]
assert checks["fuel_transactions_total"] == checks["source_fuel_rows"]

master.to_csv(TMP / "client_master.tsv", sep="\t", index=False, na_rep="")
(TMP / "schema.json").write_text(
    json.dumps(
        {
            "columns": ordered,
            "numeric_columns": [
                column
                for column in ordered
                if column
                not in {
                    "client_id",
                    "gender",
                    "age_type_code",
                    "kladr_code",
                    "registration_region",
                    "auto_document_ids",
                    "auto_marks",
                    "auto_colors",
                    "subscription_creation_date",
                    "top_offence_2026",
                    "top_offence_region_2026",
                }
            ],
            "date_columns": ["subscription_creation_date"],
            "percent_columns": ["fines_change_pct_2026_vs_2025"],
            "checks": checks,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(json.dumps(checks, ensure_ascii=False, indent=2))
