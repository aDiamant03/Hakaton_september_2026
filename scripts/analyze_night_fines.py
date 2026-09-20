from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/Library/Everything/hakaton")
INPUT_DIR = ROOT / "outputs" / "fuel_cleaning_20260919"
OUTPUT_DIR = ROOT / "outputs" / "night_fines_20260919"
PRE_START = pd.Timestamp("2026-04-01")
POST_START = pd.Timestamp("2026-06-01")
END = pd.Timestamp("2026-09-01")


def chi2_2x2(table: np.ndarray) -> dict:
    table = np.asarray(table, dtype=float)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    value = float(((table - expected) ** 2 / expected).sum())
    return {
        "chi2": value,
        "df": 1,
        "p": math.erfc(math.sqrt(value / 2)),
        "cramers_v": math.sqrt(value / table.sum()),
        "min_expected": float(expected.min()),
    }


def draw_chart(result: dict, monthly: pd.DataFrame) -> None:
    image = Image.new("RGB", (1650, 1080), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    font = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    title = ImageFont.truetype(bold, 38)
    head = ImageFont.truetype(bold, 24)
    body = ImageFont.truetype(font, 19)
    small = ImageFont.truetype(font, 15)
    draw.text((825, 38), "Ночные нарушения ПДД до и во время кризиса", font=title, fill="#172033", anchor="ma")
    draw.text((825, 85), "Ночь: 22:00–05:59. Все очищенные клиенты, включая клиентов без штрафов", font=body, fill="#52627A", anchor="ma")

    # Monthly bars
    rect = (55, 140, 1050, 720)
    draw.rounded_rectangle(rect, radius=14, fill="#FAFCFE", outline="#CBD5E1", width=2)
    draw.text(((rect[0] + rect[2]) / 2, rect[1] + 25), "Ночные штрафы по месяцам", font=head, fill="#172033", anchor="ma")
    vals = monthly["night_fines_per_1000_clients"].to_numpy()
    ymax = vals.max() * 1.18
    left, right, top, bottom = 140, 1005, 245, 640
    for tick in np.linspace(0, ymax, 5):
        y = bottom - tick / ymax * (bottom - top)
        draw.line((left, y, right, y), fill="#E5E7EB", width=1)
        draw.text((left - 12, y), f"{tick:.1f}", font=small, fill="#52627A", anchor="rm")
    xs = np.linspace(220, 925, len(monthly))
    colors = ["#94A3B8", "#94A3B8", "#C51F1A", "#C51F1A", "#C51F1A"]
    for x, (_, row), color in zip(xs, monthly.iterrows(), colors):
        h = row["night_fines_per_1000_clients"] / ymax * (bottom - top)
        draw.rectangle((x - 48, bottom - h, x + 48, bottom), fill=color)
        draw.text((x, bottom - h - 12), f"{row['night_fines_per_1000_clients']:.1f}", font=small, fill="#172033", anchor="ms")
        draw.text((x, bottom + 18), row["month_label"], font=body, fill="#52627A", anchor="ma")
    draw.text((left, 685), "Штрафов на 1 000 клиентов за месяц", font=body, fill="#52627A")

    # Results
    rect2 = (1090, 140, 1595, 720)
    draw.rounded_rectangle(rect2, radius=14, fill="#FAFCFE", outline="#CBD5E1", width=2)
    draw.text(((rect2[0] + rect2[2]) / 2, rect2[1] + 25), "Результаты", font=head, fill="#172033", anchor="ma")
    share = result["night_share_test"]
    rate = result["client_month_rate"]
    lines = [
        f"Доля ночью: {share['pre_share']*100:.2f}% → {share['post_share']*100:.2f}%",
        f"Изменение: {share['difference_pp']:+.2f} п.п.",
        f"χ² p = {share['chi_square']['p']:.3f}",
        "",
        f"На 1 000 клиентов/мес.: {rate['pre_per_1000']:.1f} → {rate['post_per_1000']:.1f}",
        f"Разница: {rate['difference_per_1000']:+.1f}",
        f"95% ДИ: [{rate['ci_low_per_1000']:+.1f}; {rate['ci_high_per_1000']:+.1f}]",
        f"bootstrap p = {rate['p_two_sided']:.3f}",
    ]
    y = 235
    for line in lines:
        draw.text((1130, y), line, font=head if "p =" in line or "На 1" in line else body, fill="#172033")
        y += 54 if line else 24
    draw.multiline_text((1130, 625), "Доля ночных нарушений\nне выросла значимо", font=head, fill="#14532D", spacing=8)

    draw.rounded_rectangle((55, 770, 1595, 1020), radius=14, fill="#FFF4CC", outline="#E8C85A", width=2)
    draw.text((90, 805), "Вывод", font=head, fill="#172033")
    draw.multiline_text(
        (90, 855),
        "По времени штрафов нет убедительных свидетельств роста ночных нарушений во время кризиса.\n"
        "Данные отражают зафиксированные нарушения ПДД, а не ДТП. Изменение числа постановлений также зависит\n"
        "от интенсивности камер, поездок и регионального покрытия.",
        font=body, fill="#334155", spacing=10,
    )
    image.save(OUTPUT_DIR / "night_fines.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clients = pd.read_csv(INPUT_DIR / "clients_demographics_clean.csv", sep=";", decimal=",")
    fines = pd.read_csv(INPUT_DIR / "fines_2026_clean.csv", sep=";", decimal=",")
    fines["dt"] = pd.to_datetime(fines["bill_offence_date"], errors="raise")
    fines = fines.loc[fines["dt"].ge(PRE_START) & fines["dt"].lt(END)].copy()
    fines["post"] = fines["dt"].ge(POST_START)
    fines["night"] = fines["dt"].dt.hour.ge(22) | fines["dt"].dt.hour.lt(6)
    fines["month"] = fines["dt"].dt.month
    client_ids = pd.Index(clients["client_id"].drop_duplicates())
    n_clients = len(client_ids)

    table = fines.groupby(["post", "night"]).size().unstack(fill_value=0).reindex(index=[False, True], columns=[False, True], fill_value=0)
    pre_share = float(table.loc[False, True] / table.loc[False].sum())
    post_share = float(table.loc[True, True] / table.loc[True].sum())
    share_test = {
        "pre_share": pre_share,
        "post_share": post_share,
        "difference_pp": (post_share - pre_share) * 100,
        "observed": table.astype(int).values.tolist(),
        "chi_square": chi2_2x2(table.to_numpy()),
    }

    night_counts = fines.loc[fines["night"]].groupby(["client_id", "post"]).size().unstack(fill_value=0).reindex(client_ids, fill_value=0)
    for col in [False, True]:
        if col not in night_counts.columns:
            night_counts[col] = 0
    pre_rate = night_counts[False].to_numpy(float) / 2.0
    post_rate = night_counts[True].to_numpy(float) / 3.0
    delta = post_rate - pre_rate
    rng = np.random.default_rng(20260919)
    boot = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(4000)])
    rate_result = {
        "clients": n_clients,
        "pre_per_client_month": float(pre_rate.mean()),
        "post_per_client_month": float(post_rate.mean()),
        "pre_per_1000": float(pre_rate.mean() * 1000),
        "post_per_1000": float(post_rate.mean() * 1000),
        "difference_per_1000": float(delta.mean() * 1000),
        "ci_low_per_1000": float(np.quantile(boot, .025) * 1000),
        "ci_high_per_1000": float(np.quantile(boot, .975) * 1000),
        "p_two_sided": float(min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0)))),
    }

    monthly = fines.groupby("month").agg(fines=("bill_id", "nunique"), night_fines=("night", "sum")).reindex(range(4, 9), fill_value=0).reset_index()
    monthly["night_share"] = monthly["night_fines"] / monthly["fines"]
    monthly["night_fines_per_1000_clients"] = monthly["night_fines"] / n_clients * 1000
    monthly["month_label"] = monthly["month"].map({4: "Апр", 5: "Май", 6: "Июн", 7: "Июл", 8: "Авг"})
    monthly.to_csv(OUTPUT_DIR / "monthly_night_fines.csv", index=False, encoding="utf-8-sig")

    result = {
        "definition": {"night": "22:00-05:59", "pre": "2026-04-01 to 2026-05-31", "crisis": "2026-06-01 to 2026-08-31"},
        "coverage": {"clients_including_zero_fines": n_clients, "fine_records": len(fines)},
        "night_share_test": share_test,
        "client_month_rate": rate_result,
        "interpretation": "no_convincing_increase_in_night_violations",
    }
    (OUTPUT_DIR / "night_fines_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_chart(result, monthly)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
