from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "working_data" / "preanalysis"
OUT = ROOT / "outputs" / "preanalysis"
FIG = ROOT / "figures" / "preanalysis"
for directory in (WORK, OUT, FIG):
    directory.mkdir(parents=True, exist_ok=True)

PANEL = ROOT / "working_data" / "client_month_analysis" / "client_month_panel_quality.csv"
GEOJSON = ROOT / "resources" / "geodata" / "candidates" / "russia_subjects_github.json"
MONTH_NAMES = {4: "Апрель", 5: "Май", 6: "Июнь", 7: "Июль", 8: "Август"}


def mean_ci(series: pd.Series) -> tuple[float, float, float, float]:
    values = series.dropna().to_numpy(float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    critical = float(stats.t.ppf(0.975, len(values) - 1))
    return mean, se, mean - critical * se, mean + critical * se


def point_in_ring(x: float, y: float, ring: list) -> bool:
    inside = False
    previous = len(ring) - 1
    for current, point in enumerate(ring):
        xi, yi = point[:2]
        xj, yj = ring[previous][:2]
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        previous = current
    return inside


def point_in_geometry(x: float, y: float, geometry: dict) -> bool:
    polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    for rings in polygons:
        if point_in_ring(x, y, rings[0]) and not any(point_in_ring(x, y, hole) for hole in rings[1:]):
            return True
    return False


def build_map_validation() -> pd.DataFrame:
    name_map = {
        "Москва": "город федерального значения Москва",
        "Санкт-Петербург": "город федерального значения Санкт-Петербург",
        "Республика Татарстан": "Республика Татарстан (Татарстан)",
    }
    control_points = {
        "Москва": (37.6173, 55.7558, "Москва"),
        "Московская область": (37.9400, 55.8000, "Балашиха"),
        "Санкт-Петербург": (30.3351, 59.9343, "Санкт-Петербург"),
        "Республика Татарстан": (49.1064, 55.7961, "Казань"),
        "Свердловская область": (60.5975, 56.8389, "Екатеринбург"),
        "Новосибирская область": (82.9204, 55.0302, "Новосибирск"),
        "Ленинградская область": (30.1282, 59.5684, "Гатчина"),
        "Нижегородская область": (44.0020, 56.3269, "Нижний Новгород"),
        "Краснодарский край": (38.9760, 45.0355, "Краснодар"),
        "Самарская область": (50.1018, 53.1959, "Самара"),
        "Челябинская область": (61.4026, 55.1644, "Челябинск"),
        "Кемеровская область": (86.0873, 55.3552, "Кемерово"),
        "Тюменская область": (65.5343, 57.1530, "Тюмень"),
        "Республика Башкортостан": (55.9721, 54.7388, "Уфа"),
        "Красноярский край": (92.8932, 56.0153, "Красноярск"),
        "Омская область": (73.3686, 54.9893, "Омск"),
    }
    geo = json.loads(GEOJSON.read_text(encoding="utf-8"))
    features = {feature["properties"].get("NL_NAME_1"): feature for feature in geo["features"]}
    rows = []
    for kladr, region in {
        "77": "Москва", "50": "Московская область", "78": "Санкт-Петербург",
        "16": "Республика Татарстан", "66": "Свердловская область", "54": "Новосибирская область",
        "47": "Ленинградская область", "52": "Нижегородская область", "23": "Краснодарский край",
        "63": "Самарская область", "74": "Челябинская область", "42": "Кемеровская область",
        "72": "Тюменская область", "02": "Республика Башкортостан", "24": "Красноярский край",
        "55": "Омская область",
    }.items():
        geo_name = name_map.get(region, region)
        feature = features.get(geo_name)
        lon, lat, control_name = control_points[region]
        matched = feature is not None
        inside = matched and point_in_geometry(lon, lat, feature["geometry"])
        rows.append({
            "kladr_code": kladr,
            "dataset_region": region,
            "geojson_region": geo_name if matched else "",
            "control_point": control_name,
            "control_lon": lon,
            "control_lat": lat,
            "name_matched": matched,
            "control_point_inside_polygon": inside,
            "validation_status": "OK" if matched and inside else "ПРОВЕРИТЬ",
        })
    return pd.DataFrame(rows)


def main() -> None:
    panel = pd.read_csv(PANEL, parse_dates=["month", "subscription_creation_date"])
    panel["is_pre"] = panel["month_num"] <= 5
    panel["yoy_fines_delta"] = panel["fines_2026"] - panel["fines_2025"]

    # Event-study: same clients and same calendar months, so 2025 absorbs seasonality.
    event_rows = []
    for (month_num, month_name), group in panel.groupby(["month_num", "month_name"], sort=True):
        mean, se, low, high = mean_ci(group["yoy_fines_delta"])
        event_rows.append({
            "month_num": month_num,
            "month_name": month_name,
            "period": "До кризиса" if month_num <= 5 else "Кризис",
            "clients": group["client_id"].nunique(),
            "fines_per_client_2025": group["fines_2025"].mean(),
            "fines_per_client_2026": group["fines_2026"].mean(),
            "yoy_delta_per_client": mean,
            "ci95_low": low,
            "ci95_high": high,
            "ci95_minus": mean - low,
            "ci95_plus": high - mean,
        })
    event = pd.DataFrame(event_rows)

    # One client-level difference-in-differences observation.
    client_period = panel.groupby(["client_id", "is_pre"], as_index=False).agg(
        yoy_fines_delta=("yoy_fines_delta", "mean"),
        fines_2026=("fines_2026", "mean"),
        fines_2025=("fines_2025", "mean"),
        fuel_volume_positive_l=("fuel_volume_positive_l", "mean"),
        fuel_orders=("fuel_txn_positive", "mean"),
        fuel_price_standard=("fuel_price_median_standard_band", "median"),
    )
    wide = client_period.pivot(index="client_id", columns="is_pre")
    clients = pd.DataFrame(index=wide.index)
    clients["fine_did"] = wide["yoy_fines_delta"][False] - wide["yoy_fines_delta"][True]
    clients["fine_change_2026"] = wide["fines_2026"][False] - wide["fines_2026"][True]
    clients["fuel_volume_change"] = wide["fuel_volume_positive_l"][False] - wide["fuel_volume_positive_l"][True]
    clients["fuel_orders_change"] = wide["fuel_orders"][False] - wide["fuel_orders"][True]
    clients["fuel_price_change_abs"] = wide["fuel_price_standard"][False] - wide["fuel_price_standard"][True]
    clients = clients.reset_index().merge(
        panel[["client_id", "registration_region", "gender", "age_type_code", "vehicle_count", "engine_power_hp_median"]].drop_duplicates("client_id"),
        on="client_id", how="left"
    )
    did_mean, did_se, did_low, did_high = mean_ci(clients["fine_did"])
    did_test = stats.ttest_1samp(clients["fine_did"], 0, nan_policy="omit")

    # Fuel dynamics: indices instead of raw totals make the crisis trajectory readable.
    fuel_month = panel.groupby(["month_num", "month_name"], as_index=False).agg(
        clients=("client_id", "nunique"),
        orders=("fuel_txn_positive", "sum"),
        volume_l=("fuel_volume_positive_l", "sum"),
        standard_volume_l=("fuel_volume_standard_band_l", "sum"),
        standard_spend_rub=("fuel_spend_standard_band_rub", "sum"),
    )
    fuel_month["price_standard_weighted"] = fuel_month["standard_spend_rub"] / fuel_month["standard_volume_l"]
    fuel_month["orders_per_client"] = fuel_month["orders"] / fuel_month["clients"]
    fuel_month["volume_per_client_l"] = fuel_month["volume_l"] / fuel_month["clients"]
    fuel_month["liters_per_order"] = fuel_month["volume_l"] / fuel_month["orders"]
    for column in ["price_standard_weighted", "orders_per_client", "volume_per_client_l", "liters_per_order"]:
        april = fuel_month.loc[fuel_month["month_num"].eq(4), column].iloc[0]
        fuel_month[f"{column}_index_april"] = fuel_month[column] / april * 100

    period_fuel = panel.assign(period=np.where(panel["is_pre"], "До кризиса", "Кризис")).groupby("period", as_index=False).agg(
        clients=("client_id", "nunique"), months=("month_num", "nunique"), orders=("fuel_txn_positive", "sum"),
        volume_l=("fuel_volume_positive_l", "sum"), standard_volume_l=("fuel_volume_standard_band_l", "sum"),
        standard_spend_rub=("fuel_spend_standard_band_rub", "sum"),
    )
    period_fuel["orders_per_client_month"] = period_fuel["orders"] / period_fuel["clients"] / period_fuel["months"]
    period_fuel["volume_per_client_month_l"] = period_fuel["volume_l"] / period_fuel["clients"] / period_fuel["months"]
    period_fuel["liters_per_order"] = period_fuel["volume_l"] / period_fuel["orders"]
    period_fuel["price_standard_weighted"] = period_fuel["standard_spend_rub"] / period_fuel["standard_volume_l"]

    # Region-level design: price first stage vs seasonally adjusted fine outcome.
    region_period = panel.groupby(["registration_region", "is_pre"], as_index=False).agg(
        clients=("client_id", "nunique"),
        yoy_fines_delta=("yoy_fines_delta", "mean"),
        fines_2025=("fines_2025", "mean"), fines_2026=("fines_2026", "mean"),
        fuel_volume_per_client=("fuel_volume_positive_l", "mean"),
        standard_volume_l=("fuel_volume_standard_band_l", "sum"),
        standard_spend_rub=("fuel_spend_standard_band_rub", "sum"),
    )
    region_period["price_standard_weighted"] = region_period["standard_spend_rub"] / region_period["standard_volume_l"]
    region_wide = region_period.pivot(index="registration_region", columns="is_pre")
    regions = pd.DataFrame(index=region_wide.index)
    regions["clients"] = region_wide["clients"][True]
    regions["fine_did_per_client_month"] = region_wide["yoy_fines_delta"][False] - region_wide["yoy_fines_delta"][True]
    regions["price_pre"] = region_wide["price_standard_weighted"][True]
    regions["price_crisis"] = region_wide["price_standard_weighted"][False]
    regions["price_change_pct"] = regions["price_crisis"] / regions["price_pre"] - 1
    regions["fuel_volume_pre_per_client_month"] = region_wide["fuel_volume_per_client"][True]
    regions["fuel_volume_crisis_per_client_month"] = region_wide["fuel_volume_per_client"][False]
    regions["fuel_volume_change_pct"] = regions["fuel_volume_crisis_per_client_month"] / regions["fuel_volume_pre_per_client_month"] - 1
    regions = regions.drop(index="Другой/неизвестный", errors="ignore").reset_index()
    price_fines_corr = stats.spearmanr(regions["price_change_pct"], regions["fine_did_per_client_month"])
    price_volume_corr = stats.spearmanr(regions["price_change_pct"], regions["fuel_volume_change_pct"])

    # Mechanism and heterogeneity by pre-crisis fuel intensity.
    baseline = panel.loc[panel["is_pre"]].groupby("client_id", as_index=False).agg(
        baseline_fuel_volume_l=("fuel_volume_positive_l", "mean"),
        baseline_fuel_orders=("fuel_txn_positive", "mean"),
        baseline_fuel_spend_standard_rub=("fuel_spend_standard_band_rub", "mean"),
    )
    clients = clients.merge(baseline, on="client_id", how="left")
    clients["fuel_intensity_segment"] = "Нет покупок до кризиса"
    positive = clients["baseline_fuel_volume_l"] > 0
    clients.loc[positive, "fuel_intensity_segment"] = pd.qcut(
        clients.loc[positive, "baseline_fuel_volume_l"], 4,
        labels=["Q1: низкая", "Q2", "Q3", "Q4: высокая"]
    ).astype(str)
    segment_order = ["Нет покупок до кризиса", "Q1: низкая", "Q2", "Q3", "Q4: высокая"]
    segment_rows = []
    for segment in segment_order:
        values = clients.loc[clients["fuel_intensity_segment"].eq(segment), "fine_did"]
        mean, se, low, high = mean_ci(values)
        segment_rows.append({"segment": segment, "clients": values.notna().sum(), "fine_did_mean": mean, "ci95_low": low, "ci95_high": high, "ci95_minus": mean - low, "ci95_plus": high - mean})
    segments = pd.DataFrame(segment_rows)
    client_mechanism_corr = stats.spearmanr(clients["fuel_volume_change"], clients["fine_did"], nan_policy="omit")

    # Vehicle and affordability/use heterogeneity. Absolute car price alone is a weak
    # proxy for resources, so the preferred pre-treatment measure combines fuel use
    # and the value of the client's vehicle portfolio.
    vehicle_profile = panel.drop_duplicates("client_id")[[
        "client_id", "vehicle_price_median_rub", "vehicle_year_median", "vehicle_count",
        "engine_power_hp_median", "engine_displacement_l_median", "subscription_creation_date",
    ]]
    clients = clients.merge(vehicle_profile, on="client_id", how="left", suffixes=("", "_profile"))
    clients["vehicle_age_2026"] = 2026 - clients["vehicle_year_median"]
    clients["log_vehicle_price"] = np.log1p(clients["vehicle_price_median_rub"])
    clients["full_2025_coverage_flag"] = clients["subscription_creation_date"] <= pd.Timestamp("2025-04-01")
    clients["single_vehicle_flag"] = clients["vehicle_count"].eq(1)
    clients["fuel_exposure_index"] = (
        clients["baseline_fuel_spend_standard_rub"] / clients["vehicle_price_median_rub"]
    )
    exposure_mask = clients["fuel_exposure_index"].notna()
    clients.loc[exposure_mask, "fuel_exposure_quartile"] = pd.qcut(
        clients.loc[exposure_mask, "fuel_exposure_index"], 4,
        labels=["Q1: низкая", "Q2", "Q3", "Q4: высокая"]
    ).astype(str)
    exposure_order = ["Q1: низкая", "Q2", "Q3", "Q4: высокая"]
    exposure_rows = []
    for quartile in exposure_order:
        group = clients.loc[clients["fuel_exposure_quartile"].eq(quartile)]
        mean, se, low, high = mean_ci(group["fine_did"])
        exposure_rows.append({
            "fuel_exposure_quartile": quartile,
            "clients": len(group),
            "fine_did_mean": mean,
            "ci95_low": low,
            "ci95_high": high,
            "ci95_minus": mean - low,
            "ci95_plus": high - mean,
            "baseline_fuel_spend_median_rub": group["baseline_fuel_spend_standard_rub"].median(),
            "vehicle_price_median_rub": group["vehicle_price_median_rub"].median(),
            "baseline_fuel_volume_mean_l": group["baseline_fuel_volume_l"].mean(),
        })
    exposure_segments = pd.DataFrame(exposure_rows)
    exposure_numeric = clients["fuel_exposure_quartile"].map(dict(zip(exposure_order, [1, 2, 3, 4])))
    exposure_test = clients.loc[exposure_numeric.notna(), ["fine_did"]].copy()
    exposure_test["quartile"] = exposure_numeric.dropna().astype(float)
    exposure_trend = stats.linregress(exposure_test["quartile"], exposure_test["fine_did"])
    q1 = clients.loc[clients["fuel_exposure_quartile"].eq("Q1: низкая"), "fine_did"]
    q4 = clients.loc[clients["fuel_exposure_quartile"].eq("Q4: высокая"), "fine_did"]
    exposure_extreme_test = stats.ttest_ind(q4, q1, equal_var=False, nan_policy="omit")
    robustness_rows = []
    robustness_samples = {
        "Все клиенты": pd.Series(True, index=clients.index),
        "Полная история 2025": clients["full_2025_coverage_flag"],
        "Один автомобиль": clients["single_vehicle_flag"],
        "Полная история + один автомобиль": clients["full_2025_coverage_flag"] & clients["single_vehicle_flag"],
    }
    for sample_name, sample_mask in robustness_samples.items():
        sample = clients.loc[sample_mask & clients["fuel_exposure_quartile"].notna()].copy()
        sample["quartile_numeric"] = sample["fuel_exposure_quartile"].map(dict(zip(exposure_order, [1, 2, 3, 4]))).astype(float)
        trend = stats.linregress(sample["quartile_numeric"], sample["fine_did"])
        sample_q1 = sample.loc[sample["fuel_exposure_quartile"].eq("Q1: низкая"), "fine_did"]
        sample_q4 = sample.loc[sample["fuel_exposure_quartile"].eq("Q4: высокая"), "fine_did"]
        extreme = stats.ttest_ind(sample_q4, sample_q1, equal_var=False, nan_policy="omit")
        robustness_rows.append({
            "sample": sample_name,
            "clients": len(sample),
            "q1_fine_did": sample_q1.mean(),
            "q4_fine_did": sample_q4.mean(),
            "q4_minus_q1": sample_q4.mean() - sample_q1.mean(),
            "q4_vs_q1_pvalue": extreme.pvalue,
            "trend_per_quartile": trend.slope,
            "trend_pvalue": trend.pvalue,
        })
    exposure_robustness = pd.DataFrame(robustness_rows)

    price_q25, price_q75 = clients["vehicle_price_median_rub"].quantile([0.25, 0.75])
    age_q25, age_q75 = clients["vehicle_age_2026"].quantile([0.25, 0.75])
    clients["vehicle_extreme_segment"] = "Остальные"
    clients.loc[
        clients["vehicle_price_median_rub"].le(price_q25) & clients["vehicle_age_2026"].ge(age_q75),
        "vehicle_extreme_segment",
    ] = "Старая и дешёвая"
    clients.loc[
        clients["vehicle_price_median_rub"].ge(price_q75) & clients["vehicle_age_2026"].le(age_q25),
        "vehicle_extreme_segment",
    ] = "Новая и дорогая"
    car_rows = []
    for segment in ["Старая и дешёвая", "Новая и дорогая", "Остальные"]:
        group = clients.loc[clients["vehicle_extreme_segment"].eq(segment)]
        mean, se, low, high = mean_ci(group["fine_did"])
        car_rows.append({
            "vehicle_segment": segment, "clients": len(group), "fine_did_mean": mean,
            "ci95_low": low, "ci95_high": high, "ci95_minus": mean - low, "ci95_plus": high - mean,
            "fuel_volume_change_mean_l": group["fuel_volume_change"].mean(),
            "baseline_fuel_volume_mean_l": group["baseline_fuel_volume_l"].mean(),
        })
    vehicle_segments = pd.DataFrame(car_rows)
    old_cheap = clients.loc[clients["vehicle_extreme_segment"].eq("Старая и дешёвая"), "fine_did"]
    new_expensive = clients.loc[clients["vehicle_extreme_segment"].eq("Новая и дорогая"), "fine_did"]
    vehicle_extreme_test = stats.ttest_ind(old_cheap, new_expensive, equal_var=False, nan_policy="omit")

    # Price percentile within the same vehicle year separates age from relative value.
    clients["vehicle_price_percentile_within_year"] = clients.groupby("vehicle_year_median")[
        "vehicle_price_median_rub"
    ].rank(pct=True)
    clients["new_car_flag"] = clients["vehicle_age_2026"].le(5)
    clients["old_car_flag"] = clients["vehicle_age_2026"].ge(15)
    clients["expensive_car_flag"] = clients["vehicle_price_median_rub"].ge(price_q75)
    clients["cheap_car_flag"] = clients["vehicle_price_median_rub"].le(price_q25)

    # Offence mix: test whether composition changed, even if total fines fell.
    fines = pd.read_csv(ROOT / "fines_2026.csv", sep=";", parse_dates=["bill_offence_date"])
    fines = fines.loc[fines["bill_offence_date"].between("2026-04-01", "2026-09-01", inclusive="left")].copy()
    fines["period"] = np.where(fines["bill_offence_date"].dt.month <= 5, "До кризиса", "Кризис")
    statement = fines["offence_short_statement"].fillna("")
    fines["speeding"] = statement.str.contains("Превышение скорости", case=False, regex=False)
    fines["severe"] = statement.str.contains(r"40-60|60-80|более чем на 80|красн|встречн|телефон", case=False, regex=True)
    offence = fines.groupby("period", as_index=False).agg(
        fines=("bill_id", "size"), speeding=("speeding", "sum"), severe=("severe", "sum"),
        fine_amount_mean_rub=("total_fine_amount", lambda s: s.mean() / 100),
    )
    offence["speeding_share"] = offence["speeding"] / offence["fines"]
    offence["severe_share"] = offence["severe"] / offence["fines"]
    offence["months"] = offence["period"].map({"До кризиса": 2, "Кризис": 3})
    offence["fines_per_month"] = offence["fines"] / offence["months"]

    def two_prop_test(column: str):
        lookup = offence.set_index("period")
        x1, n1 = lookup.loc["До кризиса", [column, "fines"]]
        x2, n2 = lookup.loc["Кризис", [column, "fines"]]
        pooled = (x1 + x2) / (n1 + n2)
        z = (x2 / n2 - x1 / n1) / np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
        return float(z), float(2 * stats.norm.sf(abs(z)))

    speed_z, speed_p = two_prop_test("speeding")
    severe_z, severe_p = two_prop_test("severe")

    # Data quality and coverage checks needed before modeling.
    demographics = pd.read_csv(ROOT / "clients_demographics.csv", sep=";", decimal=",", dtype={"kladr_code": "string"})
    fines_all = pd.read_csv(ROOT / "fines_2026.csv", sep=";", parse_dates=["bill_offence_date"])
    fuel_all = pd.read_csv(ROOT / "fuel_transaction.csv", sep=";", decimal=",", parse_dates=["order_datetime"])
    quality = pd.DataFrame([
        ("Демография: строки", len(demographics), "Исходная таблица"),
        ("Демография: уникальные клиенты", demographics["client_id"].nunique(), "База панели"),
        ("Дубликаты client_id × auto_document_id", demographics.duplicated(["client_id", "auto_document_id"]).sum(), "Перед агрегацией схлопываются"),
        ("Клиенты с несколькими автомобилями", int((demographics.groupby("client_id")["auto_document_id"].nunique() > 1).sum()), "Штрафы 2025 суммируются после дедупликации автомобиля"),
        ("Пропуски gender", demographics["gender"].isna().sum(), "Оставить неизвестными"),
        ("Пропуски price автомобиля", demographics["price"].isna().sum(), "Не заменять нулём"),
        ("Пропуски color автомобиля", demographics["color"].isna().sum(), "Не используется в основной модели"),
        ("Штрафы: дубликаты bill_id", fines_all.duplicated("bill_id").sum(), "Нет"),
        ("Топливо: дубликаты order_id", fuel_all.duplicated("order_id").sum(), "Нет"),
        ("События с client_id вне демографии", int((~fines_all["client_id"].isin(demographics["client_id"])).sum() + (~fuel_all["client_id"].isin(demographics["client_id"])).sum()), "Нет"),
        ("Штрафы вне окна апрель–август", int((~fines_all["bill_offence_date"].between("2026-04-01", "2026-09-01", inclusive="left")).sum()), "Март и сентябрь исключены"),
        ("Топливо вне окна апрель–август", int((~fuel_all["order_datetime"].between("2026-04-01", "2026-09-01", inclusive="left")).sum()), "Март и сентябрь исключены"),
        ("Отрицательный объём, апрель–август", int(panel["fuel_txn_negative"].sum()), "Вероятные возвраты/корректировки; raw сохранён"),
        ("Положительные операции дешевле 50 ₽/л", int(panel["fuel_txn_low_price"].sum()), "Флаг, не доказанная ошибка"),
        ("Положительные операции дороже 130 ₽/л", int(panel["fuel_txn_high_price"].sum()), "Флаг, возможен иной продукт"),
    ], columns=["check", "value", "decision"])

    map_validation = build_map_validation()

    hypotheses = pd.DataFrame([
        ("H1 — фаворит", "После конца мая штрафов стало меньше через сокращение мобильности; эффект сильнее у клиентов с высокой докризисной топливной нагрузкой — расходами на топливо относительно стоимости автомобиля.", "Отрицательный общий DiD и дополнительное снижение fine DiD от Q1 к Q4 fuel_exposure_index.", "DiD/event-study + post×fuel_exposure_quartile; отдельно проверить полную историю и клиентов с одной машиной.", "Поддерживается предварительно"),
        ("H2", "Чем сильнее региональный рост цены топлива, тем сильнее снижение штрафов.", "Отрицательная связь регионального изменения цены и fine DiD.", "Региональная интенсивность × post; взвешивание по числу клиентов; leave-one-region-out.", "Пока не поддерживается"),
        ("H3", "Снижение числа заправок и объёма отражает снижение мобильности и связано со снижением штрафов.", "Положительная связь Δобъёма и fine DiD: сильнее падение топлива — сильнее падение штрафов.", "Медиаторный анализ; не включать топливо в модель общего эффекта.", "Связь есть, но очень слабая"),
        ("H4", "Даже при снижении общего числа штрафов доля тяжёлых нарушений могла вырасти из-за более свободных дорог.", "Рост доли severe/speeding в кризис.", "Двухвыборочный тест долей; логит по отдельным штрафам.", "Не поддерживается"),
        ("H5", "Кризис сильнее повлиял на клиентов с высокой докризисной топливной активностью.", "Более отрицательный fine DiD в верхних квартилях baseline fuel volume.", "Взаимодействие post × baseline fuel intensity; устойчивость по заказам и объёму.", "Поддерживается описательно"),
        ("H6", "Дефицит проявляется не только ценой, но и меньшим размером одной заправки и меньшей частотой покупок.", "Снижение liters/order и orders/client после мая.", "Event-study индексов; сравнение с апрелем и маем.", "Поддерживается описательно"),
        ("H7", "Владельцы старых дешёвых автомобилей сильнее сокращают нарушения, чем владельцы новых дорогих автомобилей.", "Более отрицательный fine DiD в группе old+cheap.", "Крайние квартильные группы; затем непрерывные age, log(price) и их взаимодействие.", "Простое деление пока не поддерживается"),
    ], columns=["hypothesis_id", "hypothesis", "expected_pattern", "test_plan", "preliminary_status"])

    new_variables = pd.DataFrame([
        ("post_crisis", "1 для июня–августа 2026, иначе 0", "Воздействие", "Главная временная переменная"),
        ("year_2026", "1 для наблюдений 2026 года", "Воздействие", "Для двухлетней DiD-панели"),
        ("post_x_year_2026", "post_crisis × year_2026", "Воздействие", "Главный коэффициент общего эффекта"),
        ("yoy_fines_delta", "fines_2026 − fines_2025 для того же месяца", "Outcome", "Убирает месячную сезонность первого порядка"),
        ("fine_did", "Средний yoy gap в кризис − средний yoy gap до кризиса", "Outcome", "Клиентский показатель для EDA и гетерогенности"),
        ("vehicle_age_2026", "2026 − медианный год выпуска автомобилей клиента", "Автомобиль", "Возраст автомобиля"),
        ("log_vehicle_price", "ln(1 + медианная стоимость автомобиля)", "Автомобиль", "Снижает влияние очень дорогих машин"),
        ("vehicle_price_percentile_within_year", "Перцентиль цены среди машин того же года выпуска", "Автомобиль", "Отделяет дороговизну от возраста"),
        ("new_car_flag", "vehicle_age_2026 ≤ 5", "Автомобиль", "Новая машина"),
        ("old_car_flag", "vehicle_age_2026 ≥ 15", "Автомобиль", "Старая машина"),
        ("cheap_car_flag / expensive_car_flag", "Нижний / верхний квартиль стоимости", "Автомобиль", "Для понятных подвыборок"),
        ("old_cheap / new_expensive", "Пересечения крайних квартилей возраста и цены", "Автомобиль", "Ваша исходная идея; использовать как robustness"),
        ("baseline_fuel_volume_l", "Средний положительный объём в апреле–мае", "Докризисная активность", "Не зависит от будущего кризисного поведения"),
        ("baseline_fuel_orders", "Среднее число положительных заказов в апреле–мае", "Докризисная активность", "Частота использования топлива"),
        ("baseline_fuel_spend_standard_rub", "Средние расходы в диапазоне 50–130 ₽/л в апреле–мае", "Докризисная активность", "Числитель основной сегментации"),
        ("fuel_exposure_index", "baseline_fuel_spend_standard_rub / vehicle_price_median_rub", "Фаворит", "Прокси топливной нагрузки, не дохода"),
        ("fuel_exposure_quartile", "Квартиль fuel_exposure_index", "Фаворит", "Интерпретируемое взаимодействие с кризисом"),
        ("region_price_shock_pct", "Цена 50–130 ₽/л: кризис / pre − 1 по региону регистрации", "Регион", "Интенсивность ценового шока"),
        ("full_2025_coverage_flag", "subscription_creation_date ≤ 2025-04-01", "Качество", "Ключевая проверка сопоставимости 2025 и 2026"),
        ("single_vehicle_flag", "vehicle_count = 1", "Качество", "Убирает неоднозначность медианной машины клиента"),
        ("fuel_quality_share", "Доля положительных операций в диапазоне 50–130 ₽/л", "Качество", "Проверка чувствительности к типу топлива"),
    ], columns=["variable", "definition", "block", "research_role"])

    findings = {
        "fine_did_mean": did_mean, "fine_did_ci95_low": did_low, "fine_did_ci95_high": did_high,
        "fine_did_pvalue": float(did_test.pvalue),
        "region_price_fines_spearman": float(price_fines_corr.statistic), "region_price_fines_pvalue": float(price_fines_corr.pvalue),
        "region_price_volume_spearman": float(price_volume_corr.statistic), "region_price_volume_pvalue": float(price_volume_corr.pvalue),
        "client_volume_fine_did_spearman": float(client_mechanism_corr.statistic), "client_volume_fine_did_pvalue": float(client_mechanism_corr.pvalue),
        "speeding_share_z": speed_z, "speeding_share_pvalue": speed_p,
        "severe_share_z": severe_z, "severe_share_pvalue": severe_p,
        "map_regions_checked": int(len(map_validation)), "map_regions_passed": int(map_validation["validation_status"].eq("OK").sum()),
        "fuel_exposure_trend_per_quartile": float(exposure_trend.slope),
        "fuel_exposure_trend_pvalue": float(exposure_trend.pvalue),
        "fuel_exposure_q4_minus_q1": float(q4.mean() - q1.mean()),
        "fuel_exposure_q4_vs_q1_pvalue": float(exposure_extreme_test.pvalue),
        "old_cheap_fine_did": float(old_cheap.mean()),
        "new_expensive_fine_did": float(new_expensive.mean()),
        "old_cheap_vs_new_expensive_pvalue": float(vehicle_extreme_test.pvalue),
        "vehicle_price_q25": float(price_q25), "vehicle_price_q75": float(price_q75),
        "vehicle_age_q25": float(age_q25), "vehicle_age_q75": float(age_q75),
    }

    outputs = {
        "monthly_event_study.csv": event,
        "monthly_fuel_indices.csv": fuel_month,
        "period_fuel_summary.csv": period_fuel,
        "client_did.csv": clients,
        "region_hypothesis.csv": regions,
        "fuel_intensity_segments.csv": segments,
        "offence_composition.csv": offence,
        "data_quality_preanalysis.csv": quality,
        "map_validation.csv": map_validation,
        "hypotheses_preanalysis.csv": hypotheses,
        "fuel_exposure_segments.csv": exposure_segments,
        "vehicle_extreme_segments.csv": vehicle_segments,
        "new_variables_dictionary.csv": new_variables,
        "fuel_exposure_robustness.csv": exposure_robustness,
    }
    for name, frame in outputs.items():
        frame.to_csv(WORK / name, index=False)
    (WORK / "findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(findings, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
