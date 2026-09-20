from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


INPUT_DIR = Path("/Library/Everything/hakaton/outputs/fuel_cleaning_20260919")
OUTPUT_DIR = Path("/Library/Everything/hakaton/outputs/fuel_did_20260919")
PRE_FIRST_WEEK = pd.Timestamp("2026-03-23")
PRE_LAST_WEEK = pd.Timestamp("2026-05-18")
TRANSITION_WEEK = pd.Timestamp("2026-05-25")
POST_FIRST_WEEK = pd.Timestamp("2026-06-01")
POST_LAST_WEEK = pd.Timestamp("2026-08-24")


def two_way_residual(values: np.ndarray) -> np.ndarray:
    """Remove client and week means from a balanced client x week array."""
    return values - values.mean(axis=1, keepdims=True) - values.mean(axis=0, keepdims=True) + values.mean()


def normal_pvalue(t_value: float) -> float:
    return math.erfc(abs(t_value) / math.sqrt(2.0))


def fit_twfe(panel: pd.DataFrame, outcome: str) -> dict[str, float | int]:
    clients = panel["client_id"].drop_duplicates().tolist()
    weeks = panel["week"].drop_duplicates().tolist()
    n_clients, n_weeks = len(clients), len(weeks)
    y = panel.pivot(index="client_id", columns="week", values=outcome).loc[clients, weeks].to_numpy(float)
    cheap = panel.drop_duplicates("client_id").set_index("client_id").loc[clients, "cheap"].to_numpy(float)
    post = np.array([w >= POST_FIRST_WEEK for w in weeks], dtype=float)
    x = cheap[:, None] * post[None, :]
    yr, xr = two_way_residual(y), two_way_residual(x)
    xx = float(np.sum(xr * xr))
    beta = float(np.sum(xr * yr) / xx)
    residual = yr - beta * xr
    cluster_scores = np.sum(xr * residual, axis=1)
    variance = float(np.sum(cluster_scores**2) / (xx**2) * n_clients / (n_clients - 1))
    se = math.sqrt(max(variance, 0.0))
    t_value = beta / se
    p_value = normal_pvalue(t_value)
    return {
        "beta": beta,
        "se": se,
        "t": t_value,
        "p": p_value,
        "ci_low": beta - 1.96 * se,
        "ci_high": beta + 1.96 * se,
        "n_clients": n_clients,
        "n_observations": n_clients * n_weeks,
        "n_weeks": n_weeks,
    }


def fit_pretrend(panel: pd.DataFrame, outcome: str) -> dict[str, float]:
    pre = panel.loc[panel["week"] <= PRE_LAST_WEEK].copy()
    clients = pre["client_id"].drop_duplicates().tolist()
    weeks = pre["week"].drop_duplicates().tolist()
    y = pre.pivot(index="client_id", columns="week", values=outcome).loc[clients, weeks].to_numpy(float)
    cheap = pre.drop_duplicates("client_id").set_index("client_id").loc[clients, "cheap"].to_numpy(float)
    trend = np.arange(len(weeks), dtype=float)
    x = cheap[:, None] * trend[None, :]
    yr, xr = two_way_residual(y), two_way_residual(x)
    xx = float(np.sum(xr * xr))
    beta = float(np.sum(xr * yr) / xx)
    residual = yr - beta * xr
    scores = np.sum(xr * residual, axis=1)
    variance = float(np.sum(scores**2) / (xx**2) * len(clients) / (len(clients) - 1))
    se = math.sqrt(max(variance, 0.0))
    return {"beta_per_week": beta, "se": se, "p": normal_pvalue(beta / se)}


def fit_event_study(panel: pd.DataFrame, outcome: str) -> pd.DataFrame:
    clients = panel["client_id"].drop_duplicates().tolist()
    weeks = panel["week"].drop_duplicates().tolist()
    ref_week = PRE_LAST_WEEK
    event_weeks = [w for w in weeks if w != ref_week]
    y = panel.pivot(index="client_id", columns="week", values=outcome).loc[clients, weeks].to_numpy(float)
    cheap = panel.drop_duplicates("client_id").set_index("client_id").loc[clients, "cheap"].to_numpy(float)
    raw_x = np.zeros((len(clients), len(weeks), len(event_weeks)), dtype=float)
    for j, event_week in enumerate(event_weeks):
        raw_x[:, weeks.index(event_week), j] = cheap
    xr = raw_x - raw_x.mean(axis=1, keepdims=True) - raw_x.mean(axis=0, keepdims=True) + raw_x.mean(axis=(0, 1), keepdims=True)
    yr = two_way_residual(y)
    x2 = xr.reshape(-1, len(event_weeks))
    y2 = yr.reshape(-1)
    beta = np.linalg.lstsq(x2, y2, rcond=None)[0]
    residual = yr - np.einsum("itk,k->it", xr, beta)
    xx_inv = np.linalg.inv(x2.T @ x2)
    meat = np.zeros((len(event_weeks), len(event_weeks)))
    for i in range(len(clients)):
        score = xr[i].T @ residual[i]
        meat += np.outer(score, score)
    covariance = xx_inv @ meat @ xx_inv * len(clients) / (len(clients) - 1)
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    result = []
    for week in weeks:
        if week == ref_week:
            result.append({"week": week, "beta": 0.0, "se": 0.0, "ci_low": 0.0, "ci_high": 0.0, "reference": True})
        else:
            j = event_weeks.index(week)
            result.append(
                {
                    "week": week,
                    "beta": float(beta[j]),
                    "se": float(se[j]),
                    "ci_low": float(beta[j] - 1.96 * se[j]),
                    "ci_high": float(beta[j] + 1.96 * se[j]),
                    "reference": False,
                }
            )
    return pd.DataFrame(result)


def create_panel(
    clients: pd.DataFrame,
    fuel: pd.DataFrame,
    price_cap: float | None = None,
    single_car_only: bool = False,
    quantile: float = 1 / 3,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    weeks_all = pd.date_range(PRE_FIRST_WEEK, POST_LAST_WEEK, freq="7D")
    analysis_weeks = weeks_all[weeks_all != TRANSITION_WEEK]
    positive = fuel.loc[fuel["order_fuel_volume"].gt(0) & fuel["week"].isin(weeks_all)].copy()
    if price_cap is not None:
        positive = positive.loc[positive["order_fuel_price_1liter"].le(price_cap)]
    pre_clients = set(positive.loc[positive["week"].le(PRE_LAST_WEEK), "client_id"])
    car_count = clients.groupby("client_id").size()
    car_price = clients.groupby("client_id")["price"].median()
    eligible = car_price.loc[car_price.index.isin(pre_clients)].dropna()
    if single_car_only:
        eligible = eligible.loc[car_count.reindex(eligible.index).eq(1)]
    low_cutoff = float(eligible.quantile(quantile))
    high_cutoff = float(eligible.quantile(1 - quantile))
    groups = pd.Series(
        np.where(eligible.le(low_cutoff), "Дешевые авто", np.where(eligible.ge(high_cutoff), "Дорогие авто", "Средняя группа")),
        index=eligible.index,
    )
    groups = groups.loc[groups.ne("Средняя группа")]
    index = pd.MultiIndex.from_product([groups.index, analysis_weeks], names=["client_id", "week"])
    panel = pd.DataFrame(index=index).reset_index()
    panel["group"] = panel["client_id"].map(groups)
    panel["cheap"] = panel["group"].eq("Дешевые авто").astype(int)
    panel["post"] = panel["week"].ge(POST_FIRST_WEEK).astype(int)
    transaction_subset = positive.loc[positive["client_id"].isin(groups.index) & positive["week"].isin(analysis_weeks)]
    aggregate = (
        transaction_subset.groupby(["client_id", "week"])
        .agg(
            refuels=("order_id", "size"),
            active_days=("dt", lambda s: s.dt.date.nunique()),
            liters=("order_fuel_volume", "sum"),
        )
        .reset_index()
    )
    panel = panel.merge(aggregate, on=["client_id", "week"], how="left")
    panel[["refuels", "active_days", "liters"]] = panel[["refuels", "active_days", "liters"]].fillna(0.0)
    meta = {
        "low_cutoff": low_cutoff,
        "high_cutoff": high_cutoff,
        "cheap_clients": int(groups.eq("Дешевые авто").sum()),
        "expensive_clients": int(groups.eq("Дорогие авто").sum()),
        "single_car_only": single_car_only,
        "price_cap": price_cap if price_cap is not None else "none",
        "quantile": quantile,
    }
    return panel, meta


def draw_line_panel(draw, rect, title, x_labels, series, colors, y_title=""):
    x0, y0, x1, y1 = rect
    regular = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
    bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    draw.rounded_rectangle(rect, radius=12, outline="#CBD5E1", width=2, fill="#FAFCFE")
    draw.text(((x0 + x1) // 2, y0 + 18), title, font=bold, fill="#111827", anchor="ma")
    left, right, top, bottom = x0 + 82, x1 - 25, y0 + 62, y1 - 55
    all_values = np.concatenate([np.asarray(v, dtype=float) for v in series.values()])
    ymin, ymax = float(np.nanmin(all_values)), float(np.nanmax(all_values))
    pad = (ymax - ymin) * 0.12 or 1.0
    ymin, ymax = max(0.0, ymin - pad), ymax + pad
    def px(i): return left + i / max(len(x_labels) - 1, 1) * (right - left)
    def py(v): return bottom - (v - ymin) / (ymax - ymin) * (bottom - top)
    for tick in np.linspace(ymin, ymax, 5):
        yy = py(tick)
        draw.line((left, yy, right, yy), fill="#E5E7EB", width=1)
        draw.text((left - 8, yy), f"{tick:.2f}", font=small, fill="#64748B", anchor="rm")
    transition_index = x_labels.index("25.05") if "25.05" in x_labels else None
    if transition_index is not None:
        xx = px(transition_index)
        draw.rectangle((xx - 12, top, xx + 12, bottom), fill="#FFF4CC")
        draw.text((xx, top + 4), "переход", font=small, fill="#8A6500", anchor="ma")
    for (name, values), color in zip(series.items(), colors):
        points = [(px(i), py(float(v))) for i, v in enumerate(values)]
        draw.line(points, fill=color, width=4)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
    for i, label in enumerate(x_labels):
        if i % 3 == 0 or label in {"25.05", "01.06"}:
            draw.text((px(i), bottom + 10), label, font=small, fill="#475569", anchor="ma")
    legend_x = left
    for (name, _), color in zip(series.items(), colors):
        draw.line((legend_x, y0 + 44, legend_x + 28, y0 + 44), fill=color, width=4)
        draw.text((legend_x + 35, y0 + 44), name, font=regular, fill="#334155", anchor="lm")
        legend_x += 230
    if y_title:
        draw.text((x0 + 14, (top + bottom) // 2), y_title, font=small, fill="#64748B", anchor="lm")


def draw_event_panel(draw, rect, event: pd.DataFrame):
    x0, y0, x1, y1 = rect
    regular = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 18)
    small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
    bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    draw.rounded_rectangle(rect, radius=12, outline="#CBD5E1", width=2, fill="#FAFCFE")
    draw.text(((x0 + x1) // 2, y0 + 18), "Event study: дешевые относительно дорогих", font=bold, fill="#111827", anchor="ma")
    left, right, top, bottom = x0 + 90, x1 - 25, y0 + 62, y1 - 60
    ymin = float(min(event.ci_low.min(), -0.02)); ymax = float(max(event.ci_high.max(), 0.02)); pad=(ymax-ymin)*.1; ymin-=pad; ymax+=pad
    def px(i): return left + i / max(len(event) - 1, 1) * (right - left)
    def py(v): return bottom - (v - ymin) / (ymax - ymin) * (bottom - top)
    zero_y=py(0); draw.line((left,zero_y,right,zero_y),fill="#64748B",width=2)
    for tick in np.linspace(ymin,ymax,5):
        yy=py(tick); draw.line((left,yy,right,yy),fill="#E5E7EB",width=1); draw.text((left-8,yy),f"{tick:.2f}",font=small,fill="#64748B",anchor="rm")
    post_idx = list(event.week).index(POST_FIRST_WEEK)
    post_x=px(post_idx); draw.line((post_x,top,post_x,bottom),fill="#B91C1C",width=2); draw.text((post_x+5,top+4),"с 1 июня",font=small,fill="#B91C1C",anchor="la")
    for i,row in event.reset_index(drop=True).iterrows():
        xx=px(i); color="#2563EB" if row.week < POST_FIRST_WEEK else "#B91C1C"
        draw.line((xx,py(row.ci_low),xx,py(row.ci_high)),fill=color,width=2)
        draw.ellipse((xx-4,py(row.beta)-4,xx+4,py(row.beta)+4),fill=color)
        if i%3==0 or row.week in {PRE_LAST_WEEK,POST_FIRST_WEEK}:
            draw.text((xx,bottom+12),row.week.strftime("%d.%m"),font=small,fill="#475569",anchor="ma")
    draw.text((left,y0+44),"Точка — разница относительно недели 18.05; линия — 95% ДИ",font=regular,fill="#475569",anchor="lm")


def create_figure(weekly: pd.DataFrame, event: pd.DataFrame, result: dict) -> None:
    image = Image.new("RGB", (1680, 1400), "white")
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 34)
    text_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
    bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    draw.text((840, 28), "Активность владельцев дешевых и дорогих автомобилей", font=title, fill="#111827", anchor="ma")
    labels = weekly["week"].dt.strftime("%d.%m").tolist()
    draw_line_panel(
        draw,
        (45, 90, 1635, 500),
        "Среднее число заправок на клиента в неделю",
        labels,
        {"Дешевые авто": weekly["cheap_refuels"], "Дорогие авто": weekly["expensive_refuels"]},
        ["#2563EB", "#F97316"],
    )
    draw_event_panel(draw, (45, 540, 1635, 980), event)
    draw_line_panel(
        draw,
        (45, 1020, 1060, 1360),
        "Цена топлива по неделям",
        labels,
        {"Медиана": weekly["median_price"], "90-й перцентиль": weekly["p90_price"]},
        ["#059669", "#7C3AED"],
    )
    draw.rounded_rectangle((1090,1020,1635,1360),radius=12,outline="#CBD5E1",width=2,fill="#F8FAFC")
    draw.text((1362,1048),"Итог DiD",font=bold,fill="#111827",anchor="ma")
    main=result["main"]
    draw.text((1120,1095),f"Эффект: {main['beta']:.3f} заправки/нед.",font=bold,fill="#B91C1C")
    draw.text((1120,1140),f"95% ДИ: [{main['ci_low']:.3f}; {main['ci_high']:.3f}]",font=text_font,fill="#334155")
    draw.text((1120,1180),f"p-value: {main['p']:.4g}",font=text_font,fill="#334155")
    draw.text((1120,1220),f"Относительно базы дешевых: {main['effect_pct']*100:.1f}%",font=text_font,fill="#334155")
    draw.text((1120,1265),f"Докризисный тренд p={result['pretrend']['p']:.3f}",font=text_font,fill="#334155")
    draw.text((1120,1310),"Вывод: частота сократилась сильнее",font=bold,fill="#14532D")
    image.save(OUTPUT_DIR / "did_evidence.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clients = pd.read_csv(INPUT_DIR / "clients_demographics_clean.csv", sep=";", decimal=",")
    fuel = pd.read_csv(INPUT_DIR / "fuel_transaction_clean.csv", sep=";", decimal=",")
    fuel["dt"] = pd.to_datetime(fuel["order_datetime"], errors="raise")
    fuel["week"] = fuel["dt"].dt.to_period("W-SUN").dt.start_time

    panel, meta = create_panel(clients, fuel)
    main = fit_twfe(panel, "refuels")
    pretrend = fit_pretrend(panel, "refuels")
    event = fit_event_study(panel, "refuels")
    means = panel.groupby(["group", "post"])[["refuels", "active_days", "liters"]].mean().reset_index()
    cheap_pre = float(means.loc[(means.group.eq("Дешевые авто")) & means.post.eq(0), "refuels"].iloc[0])
    main["effect_pct"] = main["beta"] / cheap_pre

    specifications = []
    for label, kwargs, outcome in [
        ("Основная: число заправок", {}, "refuels"),
        ("Без операций >150 ₽/л", {"price_cap": 150.0}, "refuels"),
        ("Только клиенты с одним авто", {"single_car_only": True}, "refuels"),
        ("Активные дни", {}, "active_days"),
        ("Литры", {}, "liters"),
    ]:
        spec_panel, spec_meta = create_panel(clients, fuel, **kwargs)
        fit = fit_twfe(spec_panel, outcome)
        fit.update({"specification": label, "outcome": outcome, **spec_meta})
        specifications.append(fit)
    pd.DataFrame(specifications).to_csv(OUTPUT_DIR / "did_results.csv", index=False, encoding="utf-8-sig")

    all_weeks = pd.date_range(PRE_FIRST_WEEK, POST_LAST_WEEK, freq="7D")
    graph_panel, _ = create_panel(clients, fuel)
    transition_groups = meta.copy()
    positive = fuel.loc[fuel.order_fuel_volume.gt(0) & fuel.week.isin(all_weeks)].copy()
    car_price = clients.groupby("client_id").price.median()
    pre_clients = set(positive.loc[positive.week.le(PRE_LAST_WEEK),"client_id"])
    eligible = car_price.loc[car_price.index.isin(pre_clients)].dropna()
    groups = pd.Series(np.where(eligible.le(meta["low_cutoff"]),"Дешевые авто",np.where(eligible.ge(meta["high_cutoff"]),"Дорогие авто","Средняя группа")),index=eligible.index)
    groups=groups[groups.ne("Средняя группа")]
    full_index=pd.MultiIndex.from_product([groups.index,all_weeks],names=["client_id","week"])
    full=pd.DataFrame(index=full_index).reset_index(); full["group"]=full.client_id.map(groups)
    counts=positive[positive.client_id.isin(groups.index)].groupby(["client_id","week"]).size().rename("refuels").reset_index()
    full=full.merge(counts,on=["client_id","week"],how="left").fillna({"refuels":0})
    pivot=full.groupby(["week","group"]).refuels.mean().unstack()
    price_week=positive.groupby("week").order_fuel_price_1liter.agg(median_price="median",p90_price=lambda s:s.quantile(.9))
    weekly=pd.DataFrame({"week":all_weeks,"cheap_refuels":pivot["Дешевые авто"].reindex(all_weeks).values,"expensive_refuels":pivot["Дорогие авто"].reindex(all_weeks).values,"median_price":price_week.median_price.reindex(all_weeks).values,"p90_price":price_week.p90_price.reindex(all_weeks).values})
    weekly.to_csv(OUTPUT_DIR / "weekly_series.csv",index=False,encoding="utf-8-sig")
    event.to_csv(OUTPUT_DIR / "event_study.csv",index=False,encoding="utf-8-sig")
    means.to_csv(OUTPUT_DIR / "prepost_means.csv",index=False,encoding="utf-8-sig")

    result = {
        "main": main,
        "pretrend": pretrend,
        "meta": meta,
        "dates": {"pre_first": PRE_FIRST_WEEK.date().isoformat(),"pre_last": PRE_LAST_WEEK.date().isoformat(),"transition": TRANSITION_WEEK.date().isoformat(),"post_first":POST_FIRST_WEEK.date().isoformat(),"post_last":POST_LAST_WEEK.date().isoformat()},
        "means": means.to_dict(orient="records"),
        "specifications": specifications,
        "interpretation": "supported_for_frequency" if main["p"] < .05 and main["beta"] < 0 else "not_supported",
    }
    (OUTPUT_DIR / "did_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    create_figure(weekly,event,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
