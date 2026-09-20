"""Build a slide-ready chart of absolute light/heavy fine counts.

The two comparison windows have equal length (54 days). The transition week
between them is excluded so the chart does not mix the onset of the crisis
with either the pre or post period.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "fuel_cleaning_20260919" / "fines_2026_clean.csv"
OUT = ROOT / "outputs" / "presentation_charts_v2_20260920"
OUT.mkdir(parents=True, exist_ok=True)

PRE_START = pd.Timestamp("2026-04-01")
PRE_END = pd.Timestamp("2026-05-25")  # exclusive: 54 days
POST_START = pd.Timestamp("2026-06-01")
POST_END = pd.Timestamp("2026-07-25")  # exclusive: 54 days

DANGEROUS = {
    "Превышение скорости на 40-60 км/ч",
    "Превышение скорости на 60-80 км/ч",
    "Превышение скорости более чем на 80 км/ч",
    "Проезд на красный сигнал светофора",
    "Использование телефона за рулем",
    "Движение по обочине",
    "Выезд на полосу встречного движения или на трамвайные пути встречного направления",
    "Не пропустил пешехода",
}

BLUE = "#2E54FF"
NAVY = "#0A1D4D"
TEXT = "#0A1D4D"
MUTED = "#68738A"
GRID = "#DFE6F3"
PANEL = "#EEF3FF"


def spaced(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", " ")


def signed_pct(value: float) -> str:
    return f"{value:+.1f}%".replace(".", ",")


def main() -> None:
    fines = pd.read_csv(SOURCE, sep=";")
    fines["date"] = pd.to_datetime(fines["bill_offence_date"], errors="raise")
    fines["severity"] = np.where(
        fines["offence_short_statement"].isin(DANGEROUS), "Тяжёлые", "Лёгкие"
    )
    fines["period"] = np.select(
        [
            fines["date"].between(PRE_START, PRE_END, inclusive="left"),
            fines["date"].between(POST_START, POST_END, inclusive="left"),
        ],
        ["До кризиса", "После кризиса"],
        default="Исключено",
    )

    chart_data = (
        fines.loc[fines["period"].ne("Исключено")]
        .groupby(["period", "severity"], observed=True)["bill_id"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=["До кризиса", "После кризиса"], columns=["Лёгкие", "Тяжёлые"])
    )
    chart_data.to_csv(
        OUT / "штрафы_до_после_равные_периоды.csv",
        encoding="utf-8-sig",
    )

    light = chart_data["Лёгкие"].to_numpy(dtype=float)
    heavy = chart_data["Тяжёлые"].to_numpy(dtype=float)
    light_change = (light[1] / light[0] - 1) * 100
    heavy_change = (heavy[1] / heavy[0] - 1) * 100

    width, height = 1920, 1080
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    regular_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    title_font = ImageFont.truetype(bold_path, 54)
    subtitle_font = ImageFont.truetype(regular_path, 25)
    axis_font = ImageFont.truetype(regular_path, 22)
    axis_bold = ImageFont.truetype(bold_path, 25)
    value_font = ImageFont.truetype(bold_path, 31)
    legend_font = ImageFont.truetype(bold_path, 23)
    note_font = ImageFont.truetype(regular_path, 18)

    draw.text((120, 72), "Количество штрафов до и после кризиса", font=title_font, fill=TEXT)
    draw.text(
        (120, 145),
        "Абсолютные значения • два одинаковых периода по 54 дня • переходная неделя 25–31 мая исключена",
        font=subtitle_font,
        fill=MUTED,
    )

    pill = (1270, 76, 1795, 148)
    draw.rounded_rectangle(pill, radius=28, fill=PANEL)
    draw.text(
        ((pill[0] + pill[2]) // 2, (pill[1] + pill[3]) // 2),
        f"Лёгкие: {signed_pct(light_change)}   •   Тяжёлые: {signed_pct(heavy_change)}",
        font=legend_font,
        fill=TEXT,
        anchor="mm",
    )

    chart_left, chart_top, chart_right, chart_bottom = 190, 265, 1780, 850
    ymax = 20000
    for tick in range(0, ymax + 1, 5000):
        y = chart_bottom - (tick / ymax) * (chart_bottom - chart_top)
        draw.line((chart_left, y, chart_right, y), fill=GRID, width=2)
        draw.text((chart_left - 25, y), spaced(tick), font=axis_font, fill=MUTED, anchor="rm")

    centers = [600, 1380]
    bar_width = 220
    gap = 28
    values = [(light[0], heavy[0]), (light[1], heavy[1])]
    period_labels = [
        ("До кризиса", "1 апреля — 24 мая"),
        ("После кризиса", "1 июня — 24 июля"),
    ]
    for center, (light_value, heavy_value), labels in zip(centers, values, period_labels):
        for offset, value, color in [(-bar_width / 2 - gap / 2, light_value, BLUE), (bar_width / 2 + gap / 2, heavy_value, NAVY)]:
            x0 = center + offset - bar_width / 2
            x1 = x0 + bar_width
            y0 = chart_bottom - (value / ymax) * (chart_bottom - chart_top)
            draw.rounded_rectangle((x0, y0, x1, chart_bottom), radius=14, fill=color)
            draw.text(((x0 + x1) / 2, y0 - 18), spaced(value), font=value_font, fill=TEXT, anchor="ms")
        draw.text((center, 900), labels[0], font=axis_bold, fill=TEXT, anchor="ma")
        draw.text((center, 938), labels[1], font=axis_font, fill=MUTED, anchor="ma")

    legend_y = 213
    draw.rounded_rectangle((720, legend_y, 756, legend_y + 24), radius=7, fill=BLUE)
    draw.text((772, legend_y + 12), "Лёгкие", font=legend_font, fill=TEXT, anchor="lm")
    draw.rounded_rectangle((970, legend_y, 1006, legend_y + 24), radius=7, fill=NAVY)
    draw.text((1022, legend_y + 12), "Тяжёлые", font=legend_font, fill=TEXT, anchor="lm")

    draw.text(
        (120, 1008),
        "Тяжёлые: скорость 40+ км/ч, красный свет, телефон, обочина, встречная полоса и непропуск пешехода. Остальные — лёгкие.",
        font=note_font,
        fill=MUTED,
    )
    draw.text(
        (120, 1040),
        "Источник: очищенный реестр штрафов; единица подсчёта — уникальный bill_id.",
        font=note_font,
        fill=MUTED,
    )

    output = OUT / "10_штрафы_до_после_абсолютные.png"
    image.save(output)

    print(chart_data.to_string())
    print(f"Лёгкие: {signed_pct(light_change)}")
    print(f"Тяжёлые: {signed_pct(heavy_change)}")
    print(output)


if __name__ == "__main__":
    main()
