# -*- coding: utf-8 -*-
"""Regional descriptive diagnostics for the old-car shortage hypothesis.

This table is deliberately descriptive. The causal-style inference is performed
in wealth_newness_hypothesis.py with region and region-by-period controls.
"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "wealth_newness_20260919"
PRE_START, PRE_END = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-25")
POST_START, POST_END = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-09-01")
REGION_NAMES = {
    77: "Москва", 50: "Московская область", 78: "Санкт-Петербург", 16: "Татарстан",
    66: "Свердловская область", 54: "Новосибирская область", 47: "Ленинградская область",
    52: "Нижегородская область", 23: "Краснодарский край", 63: "Самарская область",
    74: "Челябинская область", 42: "Кемеровская область", 72: "Тюменская область",
    2: "Башкортостан", 24: "Красноярский край", 55: "Омская область",
}


def main():
    clients = pd.read_csv(ROOT / "clients_demographics.csv", sep=";", decimal=",")
    fines = pd.read_csv(ROOT / "fines_2026.csv", sep=";")
    fuel = pd.read_csv(ROOT / "fuel_transaction.csv", sep=";", decimal=",")

    clients["signup"] = pd.to_datetime(clients["subscription_creation_date"])
    vehicles = clients.sort_values(["auto_document_id", "signup"]).drop_duplicates("auto_document_id")
    people = vehicles.groupby("client_id").agg(
        region=("kladr_code", "first"),
        auto_year=("auto_year", "median"),
    )
    people["region_name"] = people["region"].map(REGION_NAMES)
    people["old_car"] = (2026 - people["auto_year"] >= 11).astype(int)

    fines["date"] = pd.to_datetime(fines["bill_offence_date"])
    fines = fines[fines["date"].between(PRE_START, POST_END, inclusive="left")].copy()
    fines = fines.join(people[["region", "old_car"]], on="client_id", how="inner", rsuffix="_client")
    # Use the client's home region consistently with the shortage exposure.
    # The raw fine's region_name can describe the place of the offence instead.
    fines["region_name"] = fines["region"].map(REGION_NAMES)
    fines["period"] = np.where(fines["date"] < PRE_END, "pre", np.where(fines["date"] >= POST_START, "post", "gap"))
    fines = fines[fines["period"].ne("gap")].copy()
    text = fines["offence_short_statement"].fillna("")
    fines["high_risk"] = text.str.contains(
        "40-60|60-80|более чем на 80|красный|телефона|обочине|встречного|пешехода", regex=True
    )
    fines["severe_amount"] = fines["total_fine_amount"].ge(150000)
    fines["advanced_camera"] = text.str.contains("телефона|ремень", regex=True)

    grouped = fines.groupby(["region", "region_name", "old_car", "period"]).agg(
        fines=("bill_id", "size"),
        high_risk_share=("high_risk", "mean"),
        severe_share=("severe_amount", "mean"),
    ).reset_index()
    wide = grouped.pivot(index=["region", "region_name", "old_car"], columns="period")
    wide.columns = [f"{metric}_{period}" for metric, period in wide.columns]
    wide = wide.reset_index()
    for metric in ["high_risk_share", "severe_share"]:
        wide[f"delta_{metric}"] = wide[f"{metric}_post"] - wide[f"{metric}_pre"]

    old = wide[wide["old_car"].eq(1)].set_index(["region", "region_name"])
    other = wide[wide["old_car"].eq(0)].set_index(["region", "region_name"])
    result = pd.DataFrame(index=old.index.intersection(other.index))
    result["old_fines_pre"] = old["fines_pre"]
    result["old_fines_post"] = old["fines_post"]
    result["other_fines_pre"] = other["fines_pre"]
    result["other_fines_post"] = other["fines_post"]
    result["high_risk_did_pp"] = 100 * (old["delta_high_risk_share"] - other["delta_high_risk_share"])
    result["severe_amount_did_pp"] = 100 * (old["delta_severe_share"] - other["delta_severe_share"])

    pre = fines[fines["period"].eq("pre")]
    surveillance = pre.groupby(["region", "region_name"]).agg(
        pre_fines=("bill_id", "size"),
        phone_seatbelt_share=("advanced_camera", "mean"),
    )
    result = result.join(surveillance)

    fuel["date"] = pd.to_datetime(fuel["order_datetime"])
    fuel = fuel[fuel["date"].between(PRE_START, PRE_END, inclusive="left")].copy()
    fuel = fuel[fuel["order_fuel_volume"].between(0.01, 120)]
    fuel = fuel.join(people[["region"]], on="client_id", how="inner")
    liters = fuel.groupby("region")["order_fuel_volume"].sum()
    result["pre_fines_per_1000_liters"] = result.index.get_level_values("region").map(
        (pre.groupby("region")["bill_id"].size() / liters * 1000).to_dict()
    )

    shortage = pd.read_csv(OUT / "physical_shortage_regions.csv").set_index("region")
    result["physical_shortage"] = result.index.get_level_values("region").map(shortage["physical_shortage"])
    result["liters_pct_change"] = result.index.get_level_values("region").map(shortage["liters_pct_change"])
    result["tx_pct_change"] = result.index.get_level_values("region").map(shortage["tx_pct_change"])
    result = result.reset_index().sort_values("physical_shortage", ascending=False)
    result.to_csv(OUT / "old_car_regional_diagnostics.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
