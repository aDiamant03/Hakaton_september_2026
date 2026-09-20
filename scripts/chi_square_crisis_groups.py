from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Library/Everything/hakaton")
INPUT_DIR = ROOT / "outputs" / "fuel_cleaning_20260919"
OUTPUT_DIR = ROOT / "outputs" / "fuel_chi_square_20260919"

spec = importlib.util.spec_from_file_location("activity_did", ROOT / "scripts" / "analyze_activity_did.py")
did = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(did)


def chi_square_2x2(table: np.ndarray) -> dict[str, float | list]:
    table = np.asarray(table, dtype=float)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    chi2 = float(((table - expected) ** 2 / expected).sum())
    p = math.erfc(math.sqrt(chi2 / 2.0))
    n = float(table.sum())
    v = math.sqrt(chi2 / n)
    yates = float((((np.abs(table - expected) - 0.5).clip(min=0)) ** 2 / expected).sum())
    p_yates = math.erfc(math.sqrt(yates / 2.0))
    return {
        "chi2": chi2,
        "df": 1,
        "p": p,
        "cramers_v": v,
        "expected": expected.tolist(),
        "min_expected": float(expected.min()),
        "chi2_yates": yates,
        "p_yates": p_yates,
    }


def make_test(frame: pd.DataFrame, column: str, label_true: str, label_false: str) -> dict:
    groups = ["Дешевые авто", "Дорогие авто"]
    table = np.array([
        [int(frame.loc[frame.group.eq(group), column].sum()), int((~frame.loc[frame.group.eq(group), column]).sum())]
        for group in groups
    ])
    stats = chi_square_2x2(table)
    p_cheap = table[0, 0] / table[0].sum()
    p_expensive = table[1, 0] / table[1].sum()
    odds_ratio = (table[0, 0] * table[1, 1]) / (table[0, 1] * table[1, 0]) if np.all(table > 0) else float("nan")
    stats.update({
        "rows": groups,
        "columns": [label_true, label_false],
        "observed": table.tolist(),
        "cheap_share": p_cheap,
        "expensive_share": p_expensive,
        "percentage_point_difference": p_cheap - p_expensive,
        "risk_ratio": p_cheap / p_expensive if p_expensive else float("nan"),
        "odds_ratio": odds_ratio,
    })
    return stats


def draw_result(result: dict) -> None:
    image = Image.new("RGB", (1600, 1050), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    font = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    title = ImageFont.truetype(bold, 36)
    head = ImageFont.truetype(bold, 24)
    body = ImageFont.truetype(font, 20)
    small = ImageFont.truetype(font, 16)

    draw.text((800, 35), "Дешёвые и дорогие автомобили во время кризиса", font=title, fill="#172033", anchor="ma")
    draw.text((800, 82), "Качественный признак активности в кризис: низкая / высокая относительно медианы 16,17 л на клиента в неделю", font=body, fill="#52627A", anchor="ma")

    test = result["low_activity_test"]
    shares = [test["cheap_share"], test["expensive_share"]]
    labels = ["Дешёвые авто", "Дорогие авто"]
    colors = ["#2563EB", "#F97316"]
    x0, y0, x1, y1 = 80, 155, 960, 745
    draw.rounded_rectangle((x0, y0, x1, y1), radius=14, fill="#FAFCFE", outline="#CBD5E1", width=2)
    draw.text(((x0 + x1) / 2, y0 + 25), "Доля клиентов с низкой активностью в кризис", font=head, fill="#172033", anchor="ma")
    base_y = y1 - 90
    max_h = 390
    bar_w = 230
    xs = [240, 600]
    for x, share, label, color in zip(xs, shares, labels, colors):
        h = share * max_h
        draw.rectangle((x, base_y - h, x + bar_w, base_y), fill=color)
        draw.text((x + bar_w / 2, base_y - h - 18), f"{share*100:.1f}%", font=head, fill="#172033", anchor="ms")
        draw.text((x + bar_w / 2, base_y + 28), label, font=body, fill="#334155", anchor="ma")
    for pct in [0, .25, .5, .75, 1.0]:
        y = base_y - pct * max_h
        draw.line((155, y, 900, y), fill="#E5E7EB", width=1)
        draw.text((140, y), f"{pct*100:.0f}%", font=small, fill="#52627A", anchor="rm")

    x0, y0, x1, y1 = 1010, 155, 1520, 745
    draw.rounded_rectangle((x0, y0, x1, y1), radius=14, fill="#FAFCFE", outline="#CBD5E1", width=2)
    draw.text(((x0 + x1) / 2, y0 + 25), "χ²-тест Пирсона", font=head, fill="#172033", anchor="ma")
    lines = [
        f"χ²(1) = {test['chi2']:.2f}",
        f"p-value = {test['p']:.4f}",
        f"Cramér’s V = {test['cramers_v']:.3f}",
        f"Разница = {test['percentage_point_difference']*100:+.1f} п.п.",
        f"Odds ratio = {test['odds_ratio']:.3f}",
    ]
    for idx, line in enumerate(lines):
        draw.text((1060, 250 + idx * 65), line, font=head if idx < 3 else body, fill="#172033")
    conclusion = "Статистически значимая\nсвязь отсутствует" if test["p"] >= 0.05 else "Различие значимо,\nно эффект слабый"
    draw.multiline_text((1060, 595), conclusion, font=head, fill="#14532D", spacing=8)

    draw.rounded_rectangle((80, 790, 1520, 990), radius=14, fill="#FFF4CC", outline="#E8C85A", width=2)
    draw.text((110, 820), "Интерпретация", font=head, fill="#172033")
    draw.multiline_text(
        (110, 865),
        "Во время кризиса дешёвые авто чаще относятся к низкой активности: 52,9% против 47,1%.\n"
        "Cramér’s V = 0,058 указывает на слабую связь. χ² не доказывает причинное влияние цен.",
        font=body, fill="#334155", spacing=10,
    )
    image.save(OUTPUT_DIR / "chi_square_crisis.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clients = pd.read_csv(INPUT_DIR / "clients_demographics_clean.csv", sep=";", decimal=",")
    fuel = pd.read_csv(INPUT_DIR / "fuel_transaction_clean.csv", sep=";", decimal=",")
    fuel["dt"] = pd.to_datetime(fuel["order_datetime"], errors="raise")
    fuel["week"] = fuel["dt"].dt.to_period("W-SUN").dt.start_time

    panel, meta = did.create_panel(clients, fuel)
    client_period = panel.groupby(["client_id", "group", "post"], as_index=False)["liters"].mean()
    client = client_period.pivot(index=["client_id", "group"], columns="post", values="liters").reset_index()
    client.columns = ["client_id", "group", "pre_liters_per_week", "crisis_liters_per_week"]
    client["declined_in_crisis"] = client["crisis_liters_per_week"] < client["pre_liters_per_week"]
    crisis_median = float(client["crisis_liters_per_week"].median())
    client["low_crisis_activity"] = client["crisis_liters_per_week"] <= crisis_median
    client["any_crisis_activity"] = client["crisis_liters_per_week"] > 0

    result = {
        "group_definition": {
            "cheap": f"median car price <= {meta['low_cutoff']:.0f} RUB",
            "expensive": f"median car price >= {meta['high_cutoff']:.0f} RUB",
            "middle_third": "excluded",
        },
        "crisis_period": "2026-06-01 to 2026-08-24",
        "unit": "one row per client",
        "main_outcome": "average positive liters per client-week decreased in crisis versus pre-crisis",
        "decline_test": make_test(client, "declined_in_crisis", "Снизил литры", "Не снизил литры"),
        "crisis_low_activity_cutoff": crisis_median,
        "low_activity_test": make_test(client, "low_crisis_activity", "Низкая активность", "Высокая активность"),
        "any_activity_test": make_test(client, "any_crisis_activity", "Была активность", "Не было активности"),
    }
    client.to_csv(OUTPUT_DIR / "client_crisis_categories.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "chi_square_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for name, key in [("Сократил литры в кризис", "decline_test"), ("Низкая активность в кризис", "low_activity_test"), ("Любая активность в кризис", "any_activity_test")]:
        test = result[key]
        rows.append({"test": name, "chi2": test["chi2"], "p": test["p"], "cramers_v": test["cramers_v"], "cheap_share": test["cheap_share"], "expensive_share": test["expensive_share"], "difference_pp": test["percentage_point_difference"] * 100, "odds_ratio": test["odds_ratio"]})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "chi_square_results.csv", index=False, encoding="utf-8-sig")
    draw_result(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
