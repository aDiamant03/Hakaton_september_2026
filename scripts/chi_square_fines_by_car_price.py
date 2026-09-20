from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Library/Everything/hakaton")
INPUT_DIR = ROOT / "outputs" / "fuel_cleaning_20260919"
OUTPUT_DIR = ROOT / "outputs" / "fines_chi_square_20260919"
CRISIS_START = pd.Timestamp("2026-06-01")
CRISIS_END = pd.Timestamp("2026-08-24 23:59:59.999999")


def chi_square(table: np.ndarray) -> dict:
    table = np.asarray(table, dtype=float)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    value = float(((table - expected) ** 2 / expected).sum())
    df = int((table.shape[0] - 1) * (table.shape[1] - 1))
    if df == 1:
        p = math.erfc(math.sqrt(value / 2.0))
    elif df == 2:
        p = math.exp(-value / 2.0)
    else:
        raise ValueError(f"Unsupported degrees of freedom: {df}")
    v = math.sqrt(value / (table.sum() * min(table.shape[0] - 1, table.shape[1] - 1)))
    return {
        "chi2": value,
        "df": df,
        "p": p,
        "cramers_v": v,
        "expected": expected.tolist(),
        "min_expected": float(expected.min()),
    }


def draw_chart(summary: dict) -> None:
    image = Image.new("RGB", (1600, 1050), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    font = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    title = ImageFont.truetype(bold, 38)
    head = ImageFont.truetype(bold, 24)
    body = ImageFont.truetype(font, 20)
    small = ImageFont.truetype(font, 16)
    draw.text((800, 38), "Количество штрафов у дешёвых и дорогих автомобилей", font=title, fill="#172033", anchor="ma")
    draw.text((800, 88), "Кризисный период: 01.06–24.08.2026. Единица наблюдения: автомобиль", font=body, fill="#52627A", anchor="ma")

    shares = summary["shares"]
    categories = ["0 штрафов", "1 штраф", "2 и более"]
    colors = {"Дешевые авто": "#2563EB", "Дорогие авто": "#F97316"}
    rect = (70, 150, 1060, 755)
    draw.rounded_rectangle(rect, radius=14, fill="#FAFCFE", outline="#CBD5E1", width=2)
    draw.text(((rect[0] + rect[2]) / 2, rect[1] + 28), "Распределение автомобилей по числу штрафов", font=head, fill="#172033", anchor="ma")
    left, right, top, bottom = 150, 1015, 260, 680
    for pct in [0, .25, .5, .75, 1.0]:
        y = bottom - pct * (bottom - top)
        draw.line((left, y, right, y), fill="#E5E7EB", width=1)
        draw.text((left - 15, y), f"{pct*100:.0f}%", font=small, fill="#52627A", anchor="rm")
    group_centers = [330, 760]
    bar_w, gap = 92, 12
    for center, group in zip(group_centers, ["Дешевые авто", "Дорогие авто"]):
        start = center - (3 * bar_w + 2 * gap) / 2
        for j, category in enumerate(categories):
            share = shares[group][category]
            x0 = start + j * (bar_w + gap)
            y0 = bottom - share * (bottom - top)
            draw.rectangle((x0, y0, x0 + bar_w, bottom), fill=colors[group])
            draw.text((x0 + bar_w / 2, y0 - 13), f"{share*100:.1f}%", font=small, fill="#172033", anchor="ms")
            draw.text((x0 + bar_w / 2, bottom + 18), category.replace(" штрафов", "\nштрафов").replace(" штраф", "\nштраф"), font=small, fill="#52627A", anchor="ma", align="center")
        draw.text((center, 748), group, font=head, fill=colors[group], anchor="ma")

    rect2 = (1100, 150, 1530, 755)
    draw.rounded_rectangle(rect2, radius=14, fill="#FAFCFE", outline="#CBD5E1", width=2)
    test = summary["chi_square_2x3"]
    draw.text(((rect2[0] + rect2[2]) / 2, rect2[1] + 28), "χ²-тест Пирсона", font=head, fill="#172033", anchor="ma")
    lines = [
        f"χ²({test['df']}) = {test['chi2']:.2f}",
        f"p-value = {test['p']:.4g}",
        f"Cramér’s V = {test['cramers_v']:.3f}",
        f"Мин. ожидаемая = {test['min_expected']:.1f}",
    ]
    for i, line in enumerate(lines):
        draw.text((1150, 260 + i * 70), line, font=head if i < 3 else body, fill="#172033")
    conclusion = "Распределения\nразличаются значимо" if test["p"] < .05 else "Значимых различий\nне обнаружено"
    draw.multiline_text((1150, 585), conclusion, font=head, fill="#14532D", spacing=8)

    draw.rounded_rectangle((70, 805, 1530, 990), radius=14, fill="#FFF4CC", outline="#E8C85A", width=2)
    draw.text((105, 835), "Интерпретация", font=head, fill="#172033")
    draw.multiline_text((105, 880), summary["slide_conclusion"], font=body, fill="#334155", spacing=10)
    image.save(OUTPUT_DIR / "fines_chi_square.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cars = pd.read_csv(INPUT_DIR / "clients_demographics_clean.csv", sep=";", decimal=",")
    fines = pd.read_csv(INPUT_DIR / "fines_2026_clean.csv", sep=";", decimal=",")
    fines["date"] = pd.to_datetime(fines["bill_offence_date"], errors="raise")

    # One independent observation per vehicle. Duplicate vehicle records are collapsed.
    vehicle = cars.groupby("auto_document_id", as_index=False).agg(
        client_id=("client_id", "first"),
        price=("price", "median"),
        auto_mark=("auto_mark", "first"),
    ).dropna(subset=["price"])
    low_cutoff = float(vehicle["price"].quantile(1 / 3))
    high_cutoff = float(vehicle["price"].quantile(2 / 3))
    vehicle["group"] = np.where(vehicle["price"].le(low_cutoff), "Дешевые авто", np.where(vehicle["price"].ge(high_cutoff), "Дорогие авто", "Средняя группа"))
    vehicle = vehicle.loc[vehicle["group"].ne("Средняя группа")].copy()

    crisis_fines = fines.loc[fines["date"].between(CRISIS_START, CRISIS_END)].copy()
    counts = crisis_fines.groupby("auto_document_id")["bill_id"].nunique().rename("fine_count")
    vehicle["fine_count"] = vehicle["auto_document_id"].map(counts).fillna(0).astype(int)
    vehicle["fine_category"] = pd.cut(vehicle["fine_count"], bins=[-1, 0, 1, np.inf], labels=["0 штрафов", "1 штраф", "2 и более"])
    vehicle["any_fine"] = vehicle["fine_count"].gt(0)

    groups = ["Дешевые авто", "Дорогие авто"]
    categories = ["0 штрафов", "1 штраф", "2 и более"]
    table_2x3 = pd.crosstab(vehicle["group"], vehicle["fine_category"]).reindex(index=groups, columns=categories, fill_value=0)
    table_2x2 = pd.crosstab(vehicle["group"], vehicle["any_fine"]).reindex(index=groups, columns=[False, True], fill_value=0)
    test_2x3 = chi_square(table_2x3.to_numpy())
    test_2x2 = chi_square(table_2x2.to_numpy())
    shares = {group: {category: float(table_2x3.loc[group, category] / table_2x3.loc[group].sum()) for category in categories} for group in groups}
    means = vehicle.groupby("group")["fine_count"].agg(["count", "sum", "mean"]).reindex(groups)

    cheap_any = float(table_2x2.loc["Дешевые авто", True] / table_2x2.loc["Дешевые авто"].sum())
    expensive_any = float(table_2x2.loc["Дорогие авто", True] / table_2x2.loc["Дорогие авто"].sum())
    if test_2x3["p"] < .05:
        slide_conclusion = (
            f"Распределение числа штрафов статистически различается. Автомобили без штрафов: "
            f"{shares['Дешевые авто']['0 штрафов']*100:.1f}% среди дешёвых и {shares['Дорогие авто']['0 штрафов']*100:.1f}% среди дорогих.\n"
            f"Cramér’s V = {test_2x3['cramers_v']:.3f}: связь {'слабая' if test_2x3['cramers_v'] < .1 else 'умеренная'}; тест не доказывает причинность."
        )
    else:
        slide_conclusion = "Статистически значимых различий в распределении числа штрафов между группами не обнаружено."

    summary = {
        "period": {"start": str(CRISIS_START.date()), "end": str(CRISIS_END.date())},
        "unit": "vehicle (auto_document_id)",
        "group_definition": {"cheap_max": low_cutoff, "expensive_min": high_cutoff, "middle_third": "excluded"},
        "observed_2x3": {group: {category: int(table_2x3.loc[group, category]) for category in categories} for group in groups},
        "shares": shares,
        "chi_square_2x3": test_2x3,
        "observed_any_fine_2x2": {group: {"no_fine": int(table_2x2.loc[group, False]), "any_fine": int(table_2x2.loc[group, True])} for group in groups},
        "any_fine_shares": {"cheap": cheap_any, "expensive": expensive_any, "difference_pp": (cheap_any - expensive_any) * 100},
        "chi_square_any_fine": test_2x2,
        "descriptive_counts": means.reset_index().to_dict(orient="records"),
        "slide_conclusion": slide_conclusion,
    }
    vehicle.to_csv(OUTPUT_DIR / "vehicle_fine_categories.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "fines_chi_square_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    table_2x3.to_csv(OUTPUT_DIR / "fines_contingency_table.csv", encoding="utf-8-sig")
    draw_chart(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
