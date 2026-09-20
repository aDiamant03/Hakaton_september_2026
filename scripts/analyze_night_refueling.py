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
OUTPUT_DIR = ROOT / "outputs" / "night_refueling_20260919"

spec = importlib.util.spec_from_file_location("activity_did", ROOT / "scripts" / "analyze_activity_did.py")
did = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(did)


def chi2_2x2(table: np.ndarray) -> dict:
    table = np.asarray(table, dtype=float)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    value = float(((table - expected) ** 2 / expected).sum())
    return {
        "chi2": value,
        "df": 1,
        "p": math.erfc(math.sqrt(value / 2)),
        "cramers_v": math.sqrt(value / table.sum()),
        "expected_min": float(expected.min()),
    }


def cluster_bootstrap(client_counts: pd.DataFrame, reps: int = 3000, seed: int = 20260919) -> dict:
    rng = np.random.default_rng(seed)
    arrays = {}
    for group in ["Дешевые авто", "Дорогие авто"]:
        g = client_counts.loc[client_counts["group"].eq(group)].set_index("client_id")
        arrays[group] = g[["pre_night", "pre_total", "post_night", "post_total"]].to_numpy(float)

    estimates = []
    for _ in range(reps):
        shares = {}
        for group, arr in arrays.items():
            sample = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
            shares[group] = (sample[2] / sample[3]) - (sample[0] / sample[1])
        estimates.append(shares["Дешевые авто"] - shares["Дорогие авто"])
    estimates = np.asarray(estimates)
    return {
        "reps": reps,
        "ci_low": float(np.quantile(estimates, .025)),
        "ci_high": float(np.quantile(estimates, .975)),
        "p_two_sided": float(min(1.0, 2 * min(np.mean(estimates <= 0), np.mean(estimates >= 0)))),
    }


def draw_chart(result: dict, hourly: pd.DataFrame) -> None:
    image = Image.new("RGB", (1700, 1250), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    font = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    title = ImageFont.truetype(bold, 38)
    head = ImageFont.truetype(bold, 24)
    body = ImageFont.truetype(font, 19)
    small = ImageFont.truetype(font, 15)
    draw.text((850, 38), "Изменился ли паттерн ночных заправок в кризис", font=title, fill="#172033", anchor="ma")
    draw.text((850, 85), "Ночь: 22:00–05:59. Периоды: 23.03–18.05 и 01.06–24.08.2026", font=body, fill="#52627A", anchor="ma")

    # Night shares by group and period
    rect = (50, 135, 1050, 625)
    draw.rounded_rectangle(rect, radius=14, fill="#FAFCFE", outline="#CBD5E1", width=2)
    draw.text(((rect[0] + rect[2]) / 2, rect[1] + 25), "Доля ночных операций", font=head, fill="#172033", anchor="ma")
    shares = result["night_shares"]
    left, top, bottom = 145, 240, 555
    for pct in [0, .05, .10, .15, .20]:
        y = bottom - pct / .20 * (bottom - top)
        draw.line((left, y, 1000, y), fill="#E5E7EB", width=1)
        draw.text((left - 12, y), f"{pct*100:.0f}%", font=small, fill="#52627A", anchor="rm")
    centers = [350, 760]
    colors = ["#2563EB", "#F97316"]
    for center, group, color in zip(centers, ["Дешевые авто", "Дорогие авто"], colors):
        for offset, period in [(-70, "До"), (70, "Кризис")]:
            share = shares[group][period]
            x0 = center + offset - 52
            y0 = bottom - share / .20 * (bottom - top)
            draw.rectangle((x0, y0, x0 + 104, bottom), fill=color if period == "Кризис" else "#9FB9EC")
            draw.text((x0 + 52, y0 - 12), f"{share*100:.2f}%", font=small, fill="#172033", anchor="ms")
            draw.text((x0 + 52, bottom + 16), period, font=small, fill="#52627A", anchor="ma")
        draw.text((center, 610), group, font=head, fill=color, anchor="ma")

    # Result block
    rect2 = (1090, 135, 1650, 625)
    draw.rounded_rectangle(rect2, radius=14, fill="#FAFCFE", outline="#CBD5E1", width=2)
    draw.text(((rect2[0] + rect2[2]) / 2, rect2[1] + 25), "Разница изменений", font=head, fill="#172033", anchor="ma")
    dd = result["difference_in_changes"]
    boot = result["cluster_bootstrap"]
    lines = [
        f"Дешёвые: {dd['cheap_change_pp']:+.2f} п.п.",
        f"Дорогие: {dd['expensive_change_pp']:+.2f} п.п.",
        f"Разница: {dd['did_pp']:+.2f} п.п.",
        f"95% ДИ: [{boot['ci_low']*100:+.2f}; {boot['ci_high']*100:+.2f}]",
        f"p = {boot['p_two_sided']:.3f}",
    ]
    for i, line in enumerate(lines):
        draw.text((1140, 245 + i * 58), line, font=head if i in (2, 4) else body, fill="#172033")
    conclusion = "Группы изменили ночной\nпаттерн одинаково" if boot["p_two_sided"] >= .05 else "Изменение групп\nразличается значимо"
    draw.multiline_text((1140, 520), conclusion, font=head, fill="#14532D", spacing=8)

    # Hourly distribution
    rect3 = (50, 665, 1650, 1170)
    draw.rounded_rectangle(rect3, radius=14, fill="#FAFCFE", outline="#CBD5E1", width=2)
    draw.text(((rect3[0] + rect3[2]) / 2, rect3[1] + 25), "Распределение операций по часу суток", font=head, fill="#172033", anchor="ma")
    left, right, top, bottom = 135, 1610, 755, 1095
    ymax = float(hourly[["До", "Кризис"]].to_numpy().max()) * 1.12
    px = lambda h: left + h / 23 * (right - left)
    py = lambda v: bottom - v / ymax * (bottom - top)
    for tick in np.linspace(0, ymax, 5):
        y = py(tick)
        draw.line((left, y, right, y), fill="#E5E7EB", width=1)
        draw.text((left - 12, y), f"{tick*100:.1f}%", font=small, fill="#52627A", anchor="rm")
    for period, color in [("До", "#64748B"), ("Кризис", "#C51F1A")]:
        points = [(px(int(h)), py(v)) for h, v in zip(hourly.index, hourly[period])]
        draw.line(points, fill=color, width=4)
    for h in range(24):
        if h % 2 == 0:
            draw.text((px(h), bottom + 18), str(h), font=small, fill="#52627A", anchor="ma")
    draw.line((145, 730, 180, 730), fill="#64748B", width=4); draw.text((190, 730), "До", font=body, fill="#334155", anchor="lm")
    draw.line((270, 730, 305, 730), fill="#C51F1A", width=4); draw.text((315, 730), "Кризис", font=body, fill="#334155", anchor="lm")
    image.save(OUTPUT_DIR / "night_refueling_pattern.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clients = pd.read_csv(INPUT_DIR / "clients_demographics_clean.csv", sep=";", decimal=",")
    fuel = pd.read_csv(INPUT_DIR / "fuel_transaction_clean.csv", sep=";", decimal=",")
    fuel["dt"] = pd.to_datetime(fuel["order_datetime"], errors="raise")
    fuel["week"] = fuel["dt"].dt.to_period("W-SUN").dt.start_time
    fuel["hour"] = fuel["dt"].dt.hour
    fuel["night"] = fuel["hour"].ge(22) | fuel["hour"].lt(6)

    base_panel, meta = did.create_panel(clients, fuel)
    group_map = base_panel.drop_duplicates("client_id").set_index("client_id")["group"]
    weeks = set(base_panel["week"])
    tx = fuel.loc[fuel["order_fuel_volume"].gt(0) & fuel["client_id"].isin(group_map.index) & fuel["week"].isin(weeks)].copy()
    tx["group"] = tx["client_id"].map(group_map)
    tx["post"] = tx["week"].ge(did.POST_FIRST_WEEK).astype(int)

    agg = tx.groupby(["client_id", "week"]).agg(total_refuels=("order_id", "size"), night_refuels=("night", "sum")).reset_index()
    panel = base_panel[["client_id", "week", "group", "cheap", "post"]].merge(agg, on=["client_id", "week"], how="left")
    panel[["total_refuels", "night_refuels"]] = panel[["total_refuels", "night_refuels"]].fillna(0.0)
    panel["day_refuels"] = panel["total_refuels"] - panel["night_refuels"]
    night_fit = did.fit_twfe(panel, "night_refuels")
    day_fit = did.fit_twfe(panel, "day_refuels")
    weekly_means = panel.groupby(["group", "post"])[["total_refuels", "night_refuels", "day_refuels"]].mean().reset_index()

    counts = tx.groupby(["group", "post", "night"]).size().unstack(fill_value=0)
    shares = {}
    for group in ["Дешевые авто", "Дорогие авто"]:
        shares[group] = {}
        for post, label in [(0, "До"), (1, "Кризис")]:
            row = counts.loc[(group, post)]
            shares[group][label] = float(row.get(True, 0) / row.sum())

    overall = tx.groupby(["post", "night"]).size().unstack(fill_value=0).reindex(index=[0, 1], columns=[False, True], fill_value=0)
    crisis = tx.loc[tx["post"].eq(1)].groupby(["group", "night"]).size().unstack(fill_value=0).reindex(index=["Дешевые авто", "Дорогие авто"], columns=[False, True], fill_value=0)

    cc = tx.groupby(["client_id", "group", "post"]).agg(night=("night", "sum"), total=("order_id", "size")).reset_index()
    all_clients = pd.DataFrame({"client_id": group_map.index, "group": group_map.values})
    pre = cc.loc[cc.post.eq(0), ["client_id", "night", "total"]].rename(columns={"night": "pre_night", "total": "pre_total"})
    post = cc.loc[cc.post.eq(1), ["client_id", "night", "total"]].rename(columns={"night": "post_night", "total": "post_total"})
    client_counts = all_clients.merge(pre, on="client_id", how="left").merge(post, on="client_id", how="left").fillna(0)
    bootstrap = cluster_bootstrap(client_counts)

    cheap_change = shares["Дешевые авто"]["Кризис"] - shares["Дешевые авто"]["До"]
    expensive_change = shares["Дорогие авто"]["Кризис"] - shares["Дорогие авто"]["До"]
    hourly_counts = tx.groupby(["post", "hour"]).size().unstack(fill_value=0).reindex(index=[0, 1], columns=range(24), fill_value=0)
    hourly = pd.DataFrame({"До": hourly_counts.loc[0] / hourly_counts.loc[0].sum(), "Кризис": hourly_counts.loc[1] / hourly_counts.loc[1].sum()})

    result = {
        "night_definition": "22:00-05:59",
        "sample": meta,
        "periods": {"pre": "2026-03-23 to 2026-05-18", "transition_week": "excluded", "crisis": "2026-06-01 to 2026-08-24"},
        "night_shares": shares,
        "overall_pre_vs_crisis_chi_square": chi2_2x2(overall.to_numpy()),
        "cheap_vs_expensive_during_crisis_chi_square": chi2_2x2(crisis.to_numpy()),
        "difference_in_changes": {
            "cheap_change_pp": cheap_change * 100,
            "expensive_change_pp": expensive_change * 100,
            "did_pp": (cheap_change - expensive_change) * 100,
        },
        "cluster_bootstrap": bootstrap,
        "twfe_night_refuels_per_client_week": night_fit,
        "twfe_day_refuels_per_client_week": day_fit,
        "mean_refuels_per_client_week": weekly_means.to_dict(orient="records"),
        "interpretation": "night_pattern_not_a_differential_mechanism" if bootstrap["p_two_sided"] >= .05 else "night_pattern_differs_by_price_group",
    }
    (OUTPUT_DIR / "night_refueling_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    hourly.to_csv(OUTPUT_DIR / "hourly_distribution.csv", encoding="utf-8-sig")
    client_counts.to_csv(OUTPUT_DIR / "client_night_counts.csv", index=False, encoding="utf-8-sig")
    draw_chart(result, hourly)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
