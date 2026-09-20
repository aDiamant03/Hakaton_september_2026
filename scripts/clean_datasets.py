from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SOURCE_DIR = Path("/Library/Everything/hacaton_full")
OUTPUT_DIR = Path("/Library/Everything/hakaton/outputs/fuel_cleaning_20260919")
CUTOFF = pd.Timestamp("2025-04-01")
MIN_FUEL_PRICE = 50.0
HIGH_FUEL_PRICE_REVIEW = 150.0

# Brands whose cars in this dataset are electric or range-extended/hybrid.
ELECTRIFIED_BRANDS = {
    "AITO",
    "AVATR",
    "BYD",
    "DEEPAL",
    "DENZA",
    "EVOLUTE",
    "LIXIANG",
    "ORA",
    "POLESTAR",
    "QIYUAN",
    "SERES",
    "TESLA",
    "VOYAH",
    "ZEEKR",
}


def read_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clients = pd.read_csv(SOURCE_DIR / "clients_demographics.csv", sep=";", decimal=",")
    fines = pd.read_csv(SOURCE_DIR / "fines_2026.csv", sep=";")
    fuel = pd.read_csv(SOURCE_DIR / "fuel_transaction.csv", sep=";", decimal=",")
    return clients, fines, fuel


def electrified_mask(clients: pd.DataFrame) -> pd.Series:
    engine = clients["engine_type"].fillna("").astype(str).str.lower()
    displacement = pd.to_numeric(
        engine.str.extract(r"^\s*([0-9]+(?:[\.,][0-9]+)?)")[0].str.replace(",", ".", regex=False),
        errors="coerce",
    )
    text_flag = engine.str.contains(r"электр|электро|гибрид", regex=True)
    return text_flag | displacement.eq(0) | clients["auto_mark"].isin(ELECTRIFIED_BRANDS)


def qstats(series: pd.Series) -> dict[str, float | int]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    q1, median, q3 = s.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    return {
        "n": int(s.size),
        "min": float(s.min()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(s.max()),
        "iqr_low": float(q1 - 1.5 * iqr),
        "iqr_high": float(q3 + 1.5 * iqr),
    }


def save_boxplots(
    clients: pd.DataFrame,
    clients_clean: pd.DataFrame,
    fines: pd.DataFrame,
    fines_clean: pd.DataFrame,
    fuel: pd.DataFrame,
    fuel_clean: pd.DataFrame,
) -> None:
    width, height = 1680, 1080
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    font = ImageFont.truetype(font_path, 22)
    small = ImageFont.truetype(font_path, 18)
    title_font = ImageFont.truetype(bold_path, 34)
    panel_title_font = ImageFont.truetype(bold_path, 22)
    draw.text((width // 2, 28), "Боксплоты до и после очистки", font=title_font, fill="#111827", anchor="ma")
    specs = [
        ([fuel["order_fuel_price_1liter"], fuel_clean["order_fuel_price_1liter"]], "Цена топлива, руб./л"),
        ([fuel["order_fuel_volume"], fuel_clean["order_fuel_volume"]], "Объем операции, л"),
        ([clients["price"].dropna(), clients_clean["price"].dropna()], "Стоимость автомобиля, руб."),
        ([fines["total_fine_amount"] / 100, fines_clean["total_fine_amount"] / 100], "Сумма штрафа, руб."),
    ]
    panels = [(50, 100, 800, 500), (880, 100, 1630, 500), (50, 570, 800, 970), (880, 570, 1630, 970)]
    rng = np.random.default_rng(20260919)
    for (data, panel_title), (x0, y0, x1, y1) in zip(specs, panels):
        draw.rounded_rectangle((x0, y0, x1, y1), radius=12, outline="#CBD5E1", width=2, fill="#FAFCFE")
        draw.text(((x0 + x1) // 2, y0 + 20), panel_title, font=panel_title_font, fill="#111827", anchor="ma")
        series = [pd.to_numeric(pd.Series(s), errors="coerce").dropna().to_numpy() for s in data]
        ymin = min(float(np.min(s)) for s in series)
        ymax = max(float(np.max(s)) for s in series)
        pad = (ymax - ymin) * 0.04 or 1
        ymin = ymin - pad if ymin < 0 else max(0, ymin - pad)
        ymax += pad
        plot_top, plot_bottom = y0 + 65, y1 - 55
        plot_left, plot_right = x0 + 105, x1 - 30

        def py(value: float) -> float:
            return plot_bottom - (value - ymin) / (ymax - ymin) * (plot_bottom - plot_top)

        for tick in np.linspace(ymin, ymax, 5):
            yy = py(float(tick))
            draw.line((plot_left, yy, plot_right, yy), fill="#E5E7EB", width=1)
            label = f"{tick:,.0f}".replace(",", " ")
            draw.text((plot_left - 8, yy), label, font=small, fill="#64748B", anchor="rm")
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#94A3B8", width=2)
        positions = [plot_left + (plot_right - plot_left) * 0.34, plot_left + (plot_right - plot_left) * 0.72]
        for idx, (values, xpos) in enumerate(zip(series, positions)):
            q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            iqr = q3 - q1
            low_bound, high_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            inside = values[(values >= low_bound) & (values <= high_bound)]
            whisk_low, whisk_high = float(inside.min()), float(inside.max())
            outliers = values[(values < low_bound) | (values > high_bound)]
            if len(outliers) > 1800:
                outliers = rng.choice(outliers, size=1800, replace=False)
            box_half = 58
            draw.line((xpos, py(whisk_low), xpos, py(whisk_high)), fill="#334155", width=3)
            draw.line((xpos - 22, py(whisk_low), xpos + 22, py(whisk_low)), fill="#334155", width=3)
            draw.line((xpos - 22, py(whisk_high), xpos + 22, py(whisk_high)), fill="#334155", width=3)
            fill = "#CBD5E1" if idx == 0 else "#93C5FD"
            draw.rectangle((xpos - box_half, py(q3), xpos + box_half, py(q1)), fill=fill, outline="#334155", width=3)
            draw.line((xpos - box_half, py(med), xpos + box_half, py(med)), fill="#111827", width=4)
            for outlier in outliers:
                jitter = float(rng.uniform(-40, 40))
                draw.ellipse((xpos + jitter - 2, py(float(outlier)) - 2, xpos + jitter + 2, py(float(outlier)) + 2), fill="#B91C1C")
            label = "Исходные" if idx == 0 else "Очищенные"
            draw.text((xpos, y1 - 36), label, font=font, fill="#334155", anchor="ma")
    image.save(OUTPUT_DIR / "boxplots_before_after.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clients, fines, fuel = read_sources()

    subscription_date = pd.to_datetime(clients["subscription_creation_date"], errors="raise")
    registration_row = subscription_date.gt(CUTOFF)
    electrified_row = electrified_mask(clients)

    registration_clients = set(clients.loc[registration_row, "client_id"])
    electrified_clients = set(clients.loc[electrified_row, "client_id"])
    removed_clients = registration_clients | electrified_clients

    # A client is removed consistently from all tables. Fuel transactions cannot be tied to a specific car.
    clients_client_filter = clients["client_id"].isin(removed_clients)
    fines_client_filter = fines["client_id"].isin(removed_clients)
    fuel_client_filter = fuel["client_id"].isin(removed_clients)

    fuel_dt = pd.to_datetime(fuel["order_datetime"], errors="raise")
    first_positive_dt = fuel.loc[fuel["order_fuel_volume"].gt(0)].assign(_dt=fuel_dt).groupby("client_id")["_dt"].min()
    first_positive_for_row = fuel["client_id"].map(first_positive_dt)
    negative_volume = fuel["order_fuel_volume"].lt(0)
    unexplained_negative = negative_volume & ~first_positive_for_row.lt(fuel_dt)
    strange_price = fuel["order_fuel_price_1liter"].lt(MIN_FUEL_PRICE)
    high_price_review = fuel["order_fuel_price_1liter"].gt(HIGH_FUEL_PRICE_REVIEW)

    clients_clean = clients.loc[~clients_client_filter].copy()
    fines_clean = fines.loc[~fines_client_filter].copy()
    fuel_clean = fuel.loc[~(fuel_client_filter | strange_price | unexplained_negative)].copy()

    clients_clean.to_csv(
        OUTPUT_DIR / "clients_demographics_clean.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig"
    )
    fines_clean.to_csv(OUTPUT_DIR / "fines_2026_clean.csv", sep=";", index=False, encoding="utf-8-sig")
    fuel_clean.to_csv(
        OUTPUT_DIR / "fuel_transaction_clean.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig"
    )
    fuel_clean.loc[high_price_review].to_csv(
        OUTPUT_DIR / "fuel_high_price_review.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig"
    )

    # Compact, auditable exclusion registry. Reasons are not inferred from attached PDF instructions.
    exclusion_parts = []
    for dataset, df, id_col, client_mask in [
        ("clients_demographics", clients, "auto_document_id", clients_client_filter),
        ("fines_2026", fines, "bill_id", fines_client_filter),
        ("fuel_transaction", fuel, "order_id", fuel_client_filter),
    ]:
        reg = df["client_id"].isin(registration_clients)
        ev = df["client_id"].isin(electrified_clients)
        reason = np.select(
            [reg & ev, reg, ev],
            ["registration_after_2025-04-01_and_electrified", "registration_after_2025-04-01", "electrified_vehicle"],
            default="",
        )
        part = pd.DataFrame({"dataset": dataset, "row_id": df[id_col], "client_id": df["client_id"], "reason": reason})
        part = part.loc[client_mask]
        exclusion_parts.append(part)
    fuel_extra_reason = np.select(
        [strange_price & unexplained_negative, strange_price, unexplained_negative],
        ["strange_fuel_price_and_unexplained_negative", "strange_fuel_price", "unexplained_negative_volume"],
        default="",
    )
    fuel_extra = pd.DataFrame(
        {"dataset": "fuel_transaction", "row_id": fuel["order_id"], "client_id": fuel["client_id"], "reason": fuel_extra_reason}
    )
    fuel_extra = fuel_extra.loc[~fuel_client_filter & (strange_price | unexplained_negative)]
    exclusion_parts.append(fuel_extra)
    exclusions = pd.concat(exclusion_parts, ignore_index=True)
    exclusions.to_csv(OUTPUT_DIR / "cleaning_exclusions.csv", sep=";", index=False, encoding="utf-8-sig")

    # Counts use a non-overlapping hierarchy so every removed row is counted once.
    counts = []
    for dataset, original, clean, reg_mask, ev_mask in [
        ("clients_demographics.csv", clients, clients_clean, clients["client_id"].isin(registration_clients), clients["client_id"].isin(electrified_clients)),
        ("fines_2026.csv", fines, fines_clean, fines["client_id"].isin(registration_clients), fines["client_id"].isin(electrified_clients)),
        ("fuel_transaction.csv", fuel, fuel_clean, fuel["client_id"].isin(registration_clients), fuel["client_id"].isin(electrified_clients)),
    ]:
        row = {
            "dataset": dataset,
            "before": int(len(original)),
            "after": int(len(clean)),
            "removed": int(len(original) - len(clean)),
            "registration": int(reg_mask.sum()),
            "electrified_after_registration": int((ev_mask & ~reg_mask).sum()),
            "low_price_after_client_filters": 0,
            "unexplained_negative_after_other_filters": 0,
        }
        if dataset == "fuel_transaction.csv":
            base = ~(reg_mask | ev_mask)
            row["low_price_after_client_filters"] = int((strange_price & base).sum())
            row["unexplained_negative_after_other_filters"] = int((unexplained_negative & base & ~strange_price).sum())
        counts.append(row)

    retained_negative = int((fuel_clean["order_fuel_volume"] < 0).sum())
    all_negative = int(negative_volume.sum())
    explained_negative = int((negative_volume & ~unexplained_negative).sum())

    stats = {
        "counts": counts,
        "parameters": {
            "registration_cutoff": CUTOFF.date().isoformat(),
            "fuel_price_min": MIN_FUEL_PRICE,
            "high_fuel_price_review": HIGH_FUEL_PRICE_REVIEW,
            "electrified_brands": sorted(ELECTRIFIED_BRANDS),
        },
        "clients": {
            "unique_registration_clients": len(registration_clients),
            "unique_electrified_clients": len(electrified_clients),
            "unique_removed_clients": len(removed_clients),
        },
        "negative_fuel": {
            "all_negative_rows": all_negative,
            "with_prior_positive_in_source": explained_negative,
            "without_prior_positive_in_source": int(unexplained_negative.sum()),
            "retained_after_all_filters": retained_negative,
        },
        "high_price_review": {
            "threshold": HIGH_FUEL_PRICE_REVIEW,
            "rows_in_source": int(high_price_review.sum()),
            "rows_after_client_filters": int((high_price_review & ~fuel_client_filter).sum()),
            "rows_retained_in_clean_dataset": int((high_price_review & ~(fuel_client_filter | strange_price | unexplained_negative)).sum()),
        },
        "boxplot_stats": {
            "fuel_price_before": qstats(fuel["order_fuel_price_1liter"]),
            "fuel_price_after": qstats(fuel_clean["order_fuel_price_1liter"]),
            "fuel_volume_before": qstats(fuel["order_fuel_volume"]),
            "fuel_volume_after": qstats(fuel_clean["order_fuel_volume"]),
            "car_price_before": qstats(clients["price"]),
            "car_price_after": qstats(clients_clean["price"]),
            "fine_rubles_before": qstats(fines["total_fine_amount"] / 100),
            "fine_rubles_after": qstats(fines_clean["total_fine_amount"] / 100),
        },
    }
    (OUTPUT_DIR / "cleaning_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    save_boxplots(clients, clients_clean, fines, fines_clean, fuel, fuel_clean)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
