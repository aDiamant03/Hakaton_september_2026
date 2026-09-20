from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
WORKING = ROOT / "working_data"
ANALYSIS = ROOT / "analysis"
FIGURES = ROOT / "figures"
for directory in (WORKING, ANALYSIS, FIGURES):
    directory.mkdir(parents=True, exist_ok=True)

MONTHS = [4, 5, 6, 7, 8]
MONTH_LABELS = {4: "Апрель", 5: "Май", 6: "Июнь", 7: "Июль", 8: "Август"}
FINE_2025_COLUMNS = {
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


def mode_or_first(series: pd.Series):
    values = series.dropna()
    if values.empty:
        return np.nan
    modes = values.mode()
    return modes.iloc[0] if not modes.empty else values.iloc[0]


def parse_engine(engine: pd.Series) -> tuple[pd.Series, pd.Series]:
    displacement = pd.to_numeric(
        engine.str.extract(r"^\s*([0-9]+(?:[.,][0-9]+)?)", expand=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    power = pd.to_numeric(
        engine.str.extract(r"\(([0-9]+(?:[.,][0-9]+)?)\s*л\.с\.", expand=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    return displacement, power


def bootstrap_mean_ci(values: np.ndarray, seed: int = 20260919, draws: int = 4000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = np.empty(draws)
    chunk = 200
    for start in range(0, draws, chunk):
        n = min(chunk, draws - start)
        sample = rng.choice(values, size=(n, len(values)), replace=True)
        means[start : start + n] = sample.mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def load_sources():
    demographics = pd.read_csv(
        ROOT / "clients_demographics.csv",
        sep=";",
        decimal=",",
        dtype={"kladr_code": "string"},
        parse_dates=["subscription_creation_date"],
    )
    demographics["kladr_code"] = demographics["kladr_code"].str.zfill(2)
    demographics["engine_displacement_l"], demographics["engine_power_hp"] = parse_engine(
        demographics["engine_type"]
    )
    fines = pd.read_csv(
        ROOT / "fines_2026.csv",
        sep=";",
        parse_dates=["bill_offence_date"],
    )
    fuel = pd.read_csv(
        ROOT / "fuel_transaction.csv",
        sep=";",
        decimal=",",
        parse_dates=["order_datetime"],
    )
    return demographics, fines, fuel


def build_analysis():
    demographics, fines_raw, fuel_raw = load_sources()

    # One record per client-vehicle pair. Repeated rows for the same pair carry
    # duplicated history, so max() avoids double counting the same 2025 fines.
    vehicle_pairs = (
        demographics.groupby(["client_id", "auto_document_id"], as_index=False)
        .agg(
            gender=("gender", mode_or_first),
            age_type_code=("age_type_code", mode_or_first),
            auto_mark=("auto_mark", mode_or_first),
            auto_year=("auto_year", "median"),
            price=("price", "median"),
            color=("color", mode_or_first),
            kladr_code=("kladr_code", mode_or_first),
            engine_displacement_l=("engine_displacement_l", "median"),
            engine_power_hp=("engine_power_hp", "median"),
            subscription_creation_date=("subscription_creation_date", "min"),
            **{col: (col, "max") for col in FINE_2025_COLUMNS.values()},
            fines_last_6_month=("fines_last_6_month", "max"),
            fines_last_12_month=("fines_last_12_month", "max"),
            fines_last_24_month=("fines_last_24_month", "max"),
            fines_last_36_month=("fines_last_36_month", "max"),
        )
    )

    client_profile = (
        vehicle_pairs.groupby("client_id", as_index=False)
        .agg(
            gender=("gender", mode_or_first),
            age_type_code=("age_type_code", mode_or_first),
            kladr_code=("kladr_code", mode_or_first),
            vehicle_count=("auto_document_id", "nunique"),
            vehicle_price_median=("price", "median"),
            vehicle_year_median=("auto_year", "median"),
            engine_power_hp_median=("engine_power_hp", "median"),
            engine_displacement_l_median=("engine_displacement_l", "median"),
            subscription_creation_date=("subscription_creation_date", "min"),
            **{f"fines_2025_{month:02d}": (column, "sum") for month, column in FINE_2025_COLUMNS.items()},
            fines_last_6_month=("fines_last_6_month", "sum"),
            fines_last_12_month=("fines_last_12_month", "sum"),
            fines_last_24_month=("fines_last_24_month", "sum"),
            fines_last_36_month=("fines_last_36_month", "sum"),
        )
    )
    client_profile["registration_region"] = client_profile["kladr_code"].map(REGION_MAP).fillna("Другой/неизвестный")

    start, end = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-09-01")
    fines = fines_raw.loc[fines_raw["bill_offence_date"].between(start, end, inclusive="left")].copy()
    fines["month"] = fines["bill_offence_date"].dt.month
    fines["fine_amount_rub"] = fines["total_fine_amount"] / 100
    fines["severe_offence"] = fines["offence_short_statement"].str.contains(
        r"40-60|60-80|более чем на 80|красный|встречного|телефона", case=False, regex=True
    )

    fuel = fuel_raw.loc[fuel_raw["order_datetime"].between(start, end, inclusive="left")].copy()
    fuel["month"] = fuel["order_datetime"].dt.month
    fuel["fuel_spend_rub"] = fuel["order_fuel_volume"] * fuel["order_fuel_price_1liter"]
    fuel["positive_volume_l"] = fuel["order_fuel_volume"].clip(lower=0)
    fuel["positive_order"] = fuel["order_fuel_volume"] > 0

    fine_month = (
        fines.groupby(["client_id", "month"], as_index=False)
        .agg(
            fines_2026=("bill_id", "size"),
            fine_amount_2026_rub=("fine_amount_rub", "sum"),
            severe_fines_2026=("severe_offence", "sum"),
            offence_regions=("region_name", "nunique"),
        )
    )
    fuel_month = (
        fuel.groupby(["client_id", "month"], as_index=False)
        .agg(
            fuel_orders=("positive_order", "sum"),
            fuel_volume_l=("positive_volume_l", "sum"),
            fuel_net_volume_l=("order_fuel_volume", "sum"),
            fuel_price_mean=("order_fuel_price_1liter", "mean"),
            fuel_price_median=("order_fuel_price_1liter", "median"),
            fuel_spend_rub=("fuel_spend_rub", "sum"),
        )
    )

    grid = pd.MultiIndex.from_product(
        [client_profile["client_id"], MONTHS], names=["client_id", "month"]
    ).to_frame(index=False)
    panel = grid.merge(client_profile, on="client_id", how="left")
    panel["fines_2025"] = panel.apply(lambda row: row[f"fines_2025_{int(row['month']):02d}"], axis=1)
    panel = panel.merge(fine_month, on=["client_id", "month"], how="left")
    panel = panel.merge(fuel_month, on=["client_id", "month"], how="left")
    zero_columns = [
        "fines_2026",
        "fine_amount_2026_rub",
        "severe_fines_2026",
        "offence_regions",
        "fuel_orders",
        "fuel_volume_l",
        "fuel_net_volume_l",
        "fuel_spend_rub",
    ]
    panel[zero_columns] = panel[zero_columns].fillna(0)
    panel["month_name"] = panel["month"].map(MONTH_LABELS)
    panel["crisis_period"] = np.where(panel["month"] >= 6, "Кризис (июнь–август)", "До кризиса (апрель–май)")

    monthly = (
        panel.groupby("month", as_index=False)
        .agg(
            clients=("client_id", "nunique"),
            fines_2025=("fines_2025", "sum"),
            fines_2026=("fines_2026", "sum"),
            active_fine_clients_2025=("fines_2025", lambda x: int((x > 0).sum())),
            active_fine_clients_2026=("fines_2026", lambda x: int((x > 0).sum())),
            fuel_orders=("fuel_orders", "sum"),
            fuel_clients=("fuel_orders", lambda x: int((x > 0).sum())),
            fuel_volume_l=("fuel_volume_l", "sum"),
            fuel_net_volume_l=("fuel_net_volume_l", "sum"),
            fuel_spend_rub=("fuel_spend_rub", "sum"),
            fuel_price_mean=("fuel_price_mean", "mean"),
            fuel_price_median=("fuel_price_median", "median"),
        )
    )
    monthly["month_name"] = monthly["month"].map(MONTH_LABELS)
    monthly["fines_per_client_2025"] = monthly["fines_2025"] / monthly["clients"]
    monthly["fines_per_client_2026"] = monthly["fines_2026"] / monthly["clients"]
    monthly["active_share_2025"] = monthly["active_fine_clients_2025"] / monthly["clients"]
    monthly["active_share_2026"] = monthly["active_fine_clients_2026"] / monthly["clients"]
    monthly["fines_change_pct"] = monthly["fines_2026"] / monthly["fines_2025"] - 1

    client_totals = (
        panel.groupby("client_id", as_index=False)
        .agg(
            fines_2025=("fines_2025", "sum"),
            fines_2026=("fines_2026", "sum"),
            fine_amount_2026_rub=("fine_amount_2026_rub", "sum"),
            severe_fines_2026=("severe_fines_2026", "sum"),
            fuel_orders=("fuel_orders", "sum"),
            fuel_volume_l=("fuel_volume_l", "sum"),
            fuel_net_volume_l=("fuel_net_volume_l", "sum"),
            fuel_spend_rub=("fuel_spend_rub", "sum"),
            fuel_price_mean=("fuel_price_mean", "mean"),
            fuel_price_median=("fuel_price_median", "median"),
        )
        .merge(
            client_profile[
                [
                    "client_id",
                    "gender",
                    "age_type_code",
                    "registration_region",
                    "vehicle_count",
                    "vehicle_price_median",
                    "vehicle_year_median",
                    "engine_power_hp_median",
                    "engine_displacement_l_median",
                ]
            ],
            on="client_id",
            how="left",
        )
    )
    client_totals["fine_change_2026_minus_2025"] = client_totals["fines_2026"] - client_totals["fines_2025"]

    corr_columns = {
        "fines_2025": "Штрафы 2025",
        "fines_2026": "Штрафы 2026",
        "fuel_orders": "Заправки",
        "fuel_volume_l": "Литры",
        "fuel_spend_rub": "Расходы на топливо",
        "fuel_price_median": "Медианная цена топлива",
        "vehicle_count": "Число автомобилей",
        "vehicle_price_median": "Стоимость автомобиля",
        "engine_power_hp_median": "Мощность двигателя",
    }
    correlations = client_totals[list(corr_columns)].corr(method="spearman").rename(
        index=corr_columns, columns=corr_columns
    )

    region_month = (
        panel.groupby(["registration_region", "month"], as_index=False)
        .agg(
            clients=("client_id", "nunique"),
            fines_2025=("fines_2025", "sum"),
            fines_2026=("fines_2026", "sum"),
            fuel_orders=("fuel_orders", "sum"),
            fuel_volume_l=("fuel_volume_l", "sum"),
            fuel_price_median=("fuel_price_median", "median"),
        )
    )
    region_month["fines_per_client_2025"] = region_month["fines_2025"] / region_month["clients"]
    region_month["fines_per_client_2026"] = region_month["fines_2026"] / region_month["clients"]

    region_summary = (
        region_month.groupby("registration_region", as_index=False)
        .agg(
            clients=("clients", "max"),
            fines_per_client_2025=("fines_per_client_2025", "sum"),
            fines_per_client_2026=("fines_per_client_2026", "sum"),
            fuel_orders=("fuel_orders", "sum"),
            fuel_volume_l=("fuel_volume_l", "sum"),
        )
    )
    pre_price = region_month[region_month["month"] <= 5].groupby("registration_region")["fuel_price_median"].mean()
    crisis_price = region_month[region_month["month"] >= 6].groupby("registration_region")["fuel_price_median"].mean()
    region_summary["price_pre"] = region_summary["registration_region"].map(pre_price)
    region_summary["price_crisis"] = region_summary["registration_region"].map(crisis_price)
    region_summary["price_change_pct"] = region_summary["price_crisis"] / region_summary["price_pre"] - 1
    region_summary["fines_change_pct"] = (
        region_summary["fines_per_client_2026"] / region_summary["fines_per_client_2025"] - 1
    )
    region_summary = region_summary.sort_values("clients", ascending=False)

    offence_period = np.where(fines["month"] <= 5, "До кризиса", "Кризис")
    offence_mix = (
        fines.assign(period=offence_period)
        .groupby(["offence_short_statement", "period"], as_index=False)
        .agg(fines=("bill_id", "size"))
    )
    totals_by_period = offence_mix.groupby("period")["fines"].transform("sum")
    offence_mix["share"] = offence_mix["fines"] / totals_by_period
    offence_pivot = offence_mix.pivot(index="offence_short_statement", columns="period", values=["fines", "share"]).fillna(0)
    offence_pivot.columns = [f"{metric}_{period}" for metric, period in offence_pivot.columns]
    offence_pivot = offence_pivot.reset_index()
    for col in ["share_До кризиса", "share_Кризис"]:
        if col not in offence_pivot:
            offence_pivot[col] = 0.0
    offence_pivot["share_change_pp"] = (
        offence_pivot["share_Кризис"] - offence_pivot["share_До кризиса"]
    ) * 100
    offence_pivot = offence_pivot.sort_values("fines_Кризис", ascending=False)

    price_panel = panel.loc[
        panel["fuel_orders"] > 0, ["fuel_price_median", "fines_2026", "month"]
    ].dropna()
    price_panel["price_decile"] = pd.qcut(
        price_panel["fuel_price_median"], 10, duplicates="drop"
    )
    price_bins = (
        price_panel.groupby("price_decile", observed=True)
        .agg(
            mean_price_rub=("fuel_price_median", "mean"),
            mean_fines=("fines_2026", "mean"),
            observations=("fines_2026", "size"),
        )
        .reset_index(drop=True)
    )
    price_bins.insert(0, "decile", np.arange(1, len(price_bins) + 1))

    difference = client_totals["fine_change_2026_minus_2025"].to_numpy()
    ci_low, ci_high = bootstrap_mean_ci(difference)
    wilcoxon = stats.wilcoxon(
        client_totals["fines_2026"], client_totals["fines_2025"], zero_method="zsplit"
    )
    paired_summary = pd.DataFrame(
        [
            {
                "metric": "Среднее число штрафов на клиента, апрель–август 2025",
                "value": client_totals["fines_2025"].mean(),
            },
            {
                "metric": "Среднее число штрафов на клиента, апрель–август 2026",
                "value": client_totals["fines_2026"].mean(),
            },
            {"metric": "Средняя парная разница 2026−2025", "value": difference.mean()},
            {"metric": "Нижняя граница 95% bootstrap ДИ", "value": ci_low},
            {"metric": "Верхняя граница 95% bootstrap ДИ", "value": ci_high},
            {"metric": "Wilcoxon p-value", "value": wilcoxon.pvalue},
        ]
    )

    quality_rows = [
        ("Демография: строки", len(demographics)),
        ("Демография: уникальные клиенты", demographics["client_id"].nunique()),
        ("Демография: уникальные документы авто", demographics["auto_document_id"].nunique()),
        ("Повторы client_id", int(demographics["client_id"].duplicated().sum())),
        ("Повторы auto_document_id", int(demographics["auto_document_id"].duplicated().sum())),
        ("Повторы пары client_id + auto_document_id", int(demographics.duplicated(["client_id", "auto_document_id"]).sum())),
        ("Штрафы: строки", len(fines_raw)),
        ("Штрафы вне апреля–августа", int((~fines_raw["bill_offence_date"].between(start, end, inclusive="left")).sum())),
        ("Топливо: строки", len(fuel_raw)),
        ("Топливо вне апреля–августа", int((~fuel_raw["order_datetime"].between(start, end, inclusive="left")).sum())),
        ("Транзакции с отрицательным объемом", int((fuel_raw["order_fuel_volume"] < 0).sum())),
        ("Цена топлива > 150 руб./л", int((fuel_raw["order_fuel_price_1liter"] > 150).sum())),
        ("Пропуски пола", int(demographics["gender"].isna().sum())),
        ("Пропуски возрастной группы", int(demographics["age_type_code"].isna().sum())),
        ("Пропуски стоимости авто", int(demographics["price"].isna().sum())),
        ("Пропуски цвета авто", int(demographics["color"].isna().sum())),
    ]
    data_quality = pd.DataFrame(quality_rows, columns=["check", "value"])

    hypotheses = pd.DataFrame(
        [
            [
                "H1",
                "После начала кризиса рост цены топлива связан со снижением числа штрафов на клиента сверх обычной сезонности 2025 года.",
                "Клиент-месяц",
                "Difference-in-differences по месяцам 2025/2026; count-модель для fines_2026 с контролями и кластеризацией ошибок по клиенту.",
                "Эффект исчезает после учета сезонности, региона и прошлой штрафной активности.",
            ],
            [
                "H2",
                "Связь нелинейна: штрафы уменьшаются только после заметного роста цены относительно личного апрельского уровня.",
                "Клиент-месяц",
                "Бины/сплайн изменения цены; сравнение Poisson/negative binomial; permutation test порога.",
                "Нет порога или результат определяется единичными дорогими покупками.",
            ],
            [
                "H3",
                "Регионы с большим ростом цены и падением объема заправок показывают более сильное снижение штрафов.",
                "Регион-месяц",
                "Региональная панель, взаимодействие кризис × изменение цены/объема, взвешивание по числу клиентов.",
                "Региональная связь пропадает при учете московской агломерации или состава клиентов.",
            ],
            [
                "H4",
                "Во время кризиса меняется состав нарушений: доля более тяжелых нарушений растет даже при снижении общего числа штрафов.",
                "Штраф/клиент-месяц",
                "Тест долей и логистическая модель severe_offence с контролями региона и повторных нарушителей.",
                "Доли стабильны либо изменение объясняется регионами/камерами фиксации.",
            ],
            [
                "H5",
                "Клиенты, сохранившие или увеличившие объем топлива после мая, имеют рост штрафов: это группа с высокой интенсивностью вождения.",
                "Клиент-месяц",
                "Within-client изменение литров и штрафов; лаг топлива; отрицательная биномиальная модель; анализ подгрупп.",
                "Связь исчезает при контроле прошлых штрафов или возникает только одновременно, без временного порядка.",
            ],
        ],
        columns=["id", "hypothesis", "unit", "test", "rejection"],
    )

    # Persist clean and aggregated data.
    vehicle_pairs.to_csv(WORKING / "vehicle_pairs_deduplicated.csv", index=False)
    client_profile.to_csv(WORKING / "client_profile.csv", index=False)
    panel.to_csv(WORKING / "client_month_panel.csv", index=False)
    client_totals.to_csv(ANALYSIS / "client_totals.csv", index=False)
    monthly.to_csv(ANALYSIS / "monthly_summary.csv", index=False)
    correlations.to_csv(ANALYSIS / "spearman_correlations.csv")
    region_summary.to_csv(ANALYSIS / "region_summary.csv", index=False)
    region_month.to_csv(ANALYSIS / "region_month.csv", index=False)
    offence_pivot.to_csv(ANALYSIS / "offence_mix.csv", index=False)
    price_bins.to_csv(ANALYSIS / "price_fines_bins.csv", index=False)
    paired_summary.to_csv(ANALYSIS / "paired_2025_2026.csv", index=False)
    data_quality.to_csv(ANALYSIS / "data_quality.csv", index=False)
    hypotheses.to_csv(ANALYSIS / "hypotheses.csv", index=False)

    metadata = {
        "analysis_window": "2026-04-01 through 2026-08-31",
        "primary_unit": "client_id x month",
        "clients": int(client_profile["client_id"].nunique()),
        "vehicle_pairs": int(len(vehicle_pairs)),
        "panel_rows": int(len(panel)),
        "fines_2026_in_window": int(len(fines)),
        "fuel_transactions_in_window": int(len(fuel)),
        "mean_paired_change": float(difference.mean()),
        "mean_paired_change_ci95": [float(ci_low), float(ci_high)],
        "wilcoxon_p": float(wilcoxon.pvalue),
    }
    (ANALYSIS / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    workbook_payload = {
        "metadata": metadata,
        "monthly": monthly.replace({np.nan: None}).to_dict(orient="records"),
        "paired": paired_summary.replace({np.nan: None}).to_dict(orient="records"),
        "correlation_labels": correlations.columns.tolist(),
        "correlations": correlations.replace({np.nan: None}).values.tolist(),
        "regions": region_summary.replace({np.nan: None}).to_dict(orient="records"),
        "offences": offence_pivot.replace({np.nan: None}).to_dict(orient="records"),
        "price_bins": price_bins.replace({np.nan: None}).to_dict(orient="records"),
        "quality": data_quality.to_dict(orient="records"),
        "hypotheses": hypotheses.to_dict(orient="records"),
    }
    (ANALYSIS / "workbook_data.json").write_text(
        json.dumps(workbook_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    build_analysis()
