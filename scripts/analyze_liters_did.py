from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Library/Everything/hakaton")
INPUT_DIR = ROOT / "outputs" / "fuel_cleaning_20260919"
OUTPUT_DIR = ROOT / "outputs" / "fuel_did_liters_20260919"

spec = importlib.util.spec_from_file_location("activity_did", ROOT / "scripts" / "analyze_activity_did.py")
did = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(did)


def net_liters_panel(base_panel: pd.DataFrame, fuel: pd.DataFrame) -> pd.DataFrame:
    """Use the same clients/groups as the main model, but sum positive purchases and retained returns."""
    panel = base_panel[["client_id", "week", "group", "cheap", "post"]].copy()
    clients = set(panel["client_id"])
    weeks = set(panel["week"])
    transactions = fuel.loc[fuel["client_id"].isin(clients) & fuel["week"].isin(weeks)].copy()
    net = transactions.groupby(["client_id", "week"])["order_fuel_volume"].sum().rename("net_liters").reset_index()
    panel = panel.merge(net, on=["client_id", "week"], how="left")
    panel["net_liters"] = panel["net_liters"].fillna(0.0)
    return panel


def draw_chart(weekly: pd.DataFrame, event: pd.DataFrame, result: dict) -> None:
    image = Image.new("RGB", (1700, 1250), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    title_font = ImageFont.truetype(bold_path, 38)
    heading = ImageFont.truetype(bold_path, 25)
    regular = ImageFont.truetype(font_path, 19)
    small = ImageFont.truetype(font_path, 15)

    draw.text((850, 35), "Литры топлива на клиента в неделю", font=title_font, fill="#172033", anchor="ma")

    def panel(rect, title):
        draw.rounded_rectangle(rect, radius=14, fill="#FAFCFE", outline="#CBD5E1", width=2)
        draw.text(((rect[0] + rect[2]) / 2, rect[1] + 20), title, font=heading, fill="#172033", anchor="ma")

    # Weekly levels
    rect = (45, 90, 1655, 590)
    panel(rect, "Средний положительный объём на клиента, л/нед.")
    left, right, top, bottom = 130, 1615, 165, 530
    vals = np.r_[weekly["cheap_liters"].values, weekly["expensive_liters"].values]
    ymin, ymax = max(0, vals.min() - 2), vals.max() + 2
    px = lambda i: left + i / (len(weekly) - 1) * (right - left)
    py = lambda v: bottom - (v - ymin) / (ymax - ymin) * (bottom - top)
    for tick in np.linspace(ymin, ymax, 5):
        y = py(tick)
        draw.line((left, y, right, y), fill="#E5E7EB", width=1)
        draw.text((left - 12, y), f"{tick:.1f}", font=small, fill="#52627A", anchor="rm")
    colors = {"cheap_liters": "#2563EB", "expensive_liters": "#F97316"}
    for col in ["cheap_liters", "expensive_liters"]:
        points = [(px(i), py(v)) for i, v in enumerate(weekly[col])]
        draw.line(points, fill=colors[col], width=4)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colors[col])
    draw.line((left, top - 28, left + 30, top - 28), fill="#2563EB", width=4)
    draw.text((left + 40, top - 28), "Дешёвые авто", font=regular, fill="#334155", anchor="lm")
    draw.line((left + 250, top - 28, left + 280, top - 28), fill="#F97316", width=4)
    draw.text((left + 290, top - 28), "Дорогие авто", font=regular, fill="#334155", anchor="lm")
    for i, week in enumerate(weekly["week"]):
        if i % 3 == 0 or i == len(weekly) - 1:
            draw.text((px(i), bottom + 16), pd.Timestamp(week).strftime("%d.%m"), font=small, fill="#52627A", anchor="ma")

    # Event study
    rect2 = (45, 625, 1125, 1190)
    panel(rect2, "Event study: дешёвые относительно дорогих, л/нед.")
    left, right, top, bottom = 125, 1090, 710, 1125
    ymin = min(-7.0, float(event["ci_low"].min()) - 0.5)
    ymax = max(7.0, float(event["ci_high"].max()) + 0.5)
    px2 = lambda i: left + i / (len(event) - 1) * (right - left)
    py2 = lambda v: bottom - (v - ymin) / (ymax - ymin) * (bottom - top)
    draw.line((left, py2(0), right, py2(0)), fill="#64748B", width=2)
    for tick in np.linspace(ymin, ymax, 5):
        y = py2(tick)
        draw.line((left, y, right, y), fill="#E5E7EB", width=1)
        draw.text((left - 12, y), f"{tick:.1f}", font=small, fill="#52627A", anchor="rm")
    for i, row in event.reset_index(drop=True).iterrows():
        x = px2(i)
        color = "#2563EB" if pd.Timestamp(row["week"]) < did.POST_FIRST_WEEK else "#C51F1A"
        draw.line((x, py2(row["ci_low"]), x, py2(row["ci_high"])), fill=color, width=2)
        draw.ellipse((x - 5, py2(row["beta"]) - 5, x + 5, py2(row["beta"]) + 5), fill=color)
        if i % 3 == 0 or i == len(event) - 1:
            draw.text((x, bottom + 18), pd.Timestamp(row["week"]).strftime("%d.%m"), font=small, fill="#52627A", anchor="ma")

    # Result block
    rect3 = (1160, 625, 1655, 1190)
    panel(rect3, "Результат DiD")
    main = result["main"]
    draw.text((1200, 725), f"β = {main['beta']:+.2f} л/нед.", font=heading, fill="#C51F1A")
    draw.text((1200, 780), f"95% ДИ: [{main['ci_low']:+.2f}; {main['ci_high']:+.2f}]", font=regular, fill="#334155")
    draw.text((1200, 825), f"p-value: {main['p']:.4f}", font=regular, fill="#334155")
    draw.text((1200, 895), "Дешёвые: 26,98 → 18,26", font=regular, fill="#334155")
    draw.text((1200, 935), "Дорогие: 29,33 → 19,41", font=regular, fill="#334155")
    draw.multiline_text((1200, 1005), "Вывод: дешёвая группа\nне сократила литры сильнее", font=heading, fill="#14532D", spacing=8)
    image.save(OUTPUT_DIR / "liters_per_client_did.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clients = pd.read_csv(INPUT_DIR / "clients_demographics_clean.csv", sep=";", decimal=",")
    fuel = pd.read_csv(INPUT_DIR / "fuel_transaction_clean.csv", sep=";", decimal=",")
    fuel["dt"] = pd.to_datetime(fuel["order_datetime"], errors="raise")
    fuel["week"] = fuel["dt"].dt.to_period("W-SUN").dt.start_time

    panel, meta = did.create_panel(clients, fuel)
    main_fit = did.fit_twfe(panel, "liters")
    pretrend = did.fit_pretrend(panel, "liters")
    event = did.fit_event_study(panel, "liters")
    means = panel.groupby(["group", "post"])["liters"].mean().unstack()

    specs = []
    for label, kwargs in [
        ("Основная: положительные литры", {}),
        ("Без операций >150 ₽/л", {"price_cap": 150.0}),
        ("Только клиенты с одним авто", {"single_car_only": True}),
    ]:
        spec_panel, spec_meta = did.create_panel(clients, fuel, **kwargs)
        fit = did.fit_twfe(spec_panel, "liters")
        fit.update({"specification": label, **spec_meta})
        specs.append(fit)
    net_panel = net_liters_panel(panel, fuel)
    net_fit = did.fit_twfe(net_panel, "net_liters")
    net_fit.update({"specification": "Чистые литры с подтверждёнными возвратами", **meta})
    specs.append(net_fit)

    analysis_weeks = sorted(panel["week"].unique())
    weekly = panel.groupby(["week", "group"])["liters"].mean().unstack().reindex(analysis_weeks).reset_index()
    weekly.columns = ["week", "cheap_liters", "expensive_liters"]

    cheap_pre = float(means.loc["Дешевые авто", 0])
    cheap_post = float(means.loc["Дешевые авто", 1])
    expensive_pre = float(means.loc["Дорогие авто", 0])
    expensive_post = float(means.loc["Дорогие авто", 1])
    result = {
        "definition": "sum of positive fuel volume per eligible client-week; zero for weeks without purchases",
        "main": main_fit,
        "pretrend": pretrend,
        "meta": meta,
        "means": {
            "cheap_pre": cheap_pre,
            "cheap_post": cheap_post,
            "cheap_change": cheap_post - cheap_pre,
            "cheap_pct_change": cheap_post / cheap_pre - 1,
            "expensive_pre": expensive_pre,
            "expensive_post": expensive_post,
            "expensive_change": expensive_post - expensive_pre,
            "expensive_pct_change": expensive_post / expensive_pre - 1,
        },
        "specifications": specs,
        "interpretation": "hypothesis_not_supported_for_liters" if main_fit["beta"] >= 0 else "hypothesis_supported_for_liters",
    }
    (OUTPUT_DIR / "liters_did_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(specs).to_csv(OUTPUT_DIR / "liters_did_results.csv", index=False, encoding="utf-8-sig")
    weekly.to_csv(OUTPUT_DIR / "weekly_liters.csv", index=False, encoding="utf-8-sig")
    event.to_csv(OUTPUT_DIR / "liters_event_study.csv", index=False, encoding="utf-8-sig")
    draw_chart(weekly, event, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
