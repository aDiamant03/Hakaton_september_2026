# -*- coding: utf-8 -*-
"""Build slide-ready charts and derived variables for the vehicle-age hypothesis."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "wealth_newness_20260919" / "theory_charts"
OUT.mkdir(parents=True, exist_ok=True)

PRE_START, PRE_END, PRE_DAYS = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-25"), 54
POST_START, POST_END, POST_DAYS = pd.Timestamp("2026-06-01"), pd.Timestamp("2026-09-01"), 92
REGION_NAMES = {
    77: "Москва", 50: "Московская область", 78: "Санкт-Петербург", 16: "Татарстан",
    66: "Свердловская область", 54: "Новосибирская область", 47: "Ленинградская область",
    52: "Нижегородская область", 23: "Краснодарский край", 63: "Самарская область",
    74: "Челябинская область", 42: "Кемеровская область", 72: "Тюменская область",
    2: "Башкортостан", 24: "Красноярский край", 55: "Омская область",
}

BG, PANEL, FG, MUTED = "#0B0D0F", "#15191F", "#F4F6FA", "#A9B1BC"
YELLOW, GREEN, RED = "#FFDD2D", "#54E391", "#FF6767"


def period_for_date(s):
    return np.where(s.between(PRE_START, PRE_END, inclusive="left"), "До дефицита",
                    np.where(s.between(POST_START, POST_END, inclusive="left"), "После начала дефицита", None))


def pct(post, pre):
    return (post / pre - 1) * 100


def style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=12)
    for spine in ax.spines.values():
        spine.set_color("#343A43")
    ax.grid(axis="y", color="#31363D", alpha=.55, linewidth=.8)
    ax.set_axisbelow(True)


def load_and_build():
    raw = pd.read_csv(ROOT / "clients_demographics.csv", sep=";", decimal=",")
    raw["signup"] = pd.to_datetime(raw["subscription_creation_date"])
    vehicles = raw.sort_values(["auto_document_id", "signup"]).drop_duplicates("auto_document_id")
    people = vehicles.groupby("client_id").agg(region=("kladr_code", "first"), auto_year=("auto_year", "median"))
    people["vehicle_age"] = 2026 - people["auto_year"]
    people["vehicle_group"] = np.select(
        [people["vehicle_age"].le(5), people["vehicle_age"].ge(11)],
        ["Новые, до 5 лет", "Старые, от 11 лет"], default="Средние, 6–10 лет"
    )

    fines = pd.read_csv(ROOT / "fines_2026.csv", sep=";")
    fines["date"] = pd.to_datetime(fines["bill_offence_date"])
    fines["period"] = period_for_date(fines["date"])
    fines = fines[fines["period"].notna()].join(people[["region", "vehicle_group"]], on="client_id")
    fines = fines[fines["vehicle_group"].isin(["Новые, до 5 лет", "Старые, от 11 лет"])].copy()
    text = fines["offence_short_statement"].fillna("")
    fines["dangerous_flag"] = text.str.contains(
        "40-60|60-80|более чем на 80|красный|телефона|обочине|встречного|пешехода", regex=True
    ).astype(int)
    fines["light_flag"] = 1 - fines["dangerous_flag"]

    clients = people[people["vehicle_group"].isin(["Новые, до 5 лет", "Старые, от 11 лет"])].groupby("vehicle_group").size()
    days = {"До дефицита": PRE_DAYS, "После начала дефицита": POST_DAYS}
    agg = fines.groupby(["vehicle_group", "period"], observed=True).agg(
        all_fines=("bill_id", "size"), dangerous_fines=("dangerous_flag", "sum"), light_fines=("light_flag", "sum")
    ).reset_index()
    agg["clients"] = agg["vehicle_group"].map(clients)
    agg["days"] = agg["period"].map(days)
    agg["dangerous_share_pct"] = agg["dangerous_fines"] / agg["all_fines"] * 100
    agg["light_share_pct"] = agg["light_fines"] / agg["all_fines"] * 100
    agg["dangerous_per_1000_clients_30d"] = agg["dangerous_fines"] / (agg["clients"] * agg["days"] / 30) * 1000
    agg["light_per_1000_clients_30d"] = agg["light_fines"] / (agg["clients"] * agg["days"] / 30) * 1000

    changes = []
    for group, d in agg.groupby("vehicle_group"):
        d = d.set_index("period")
        before, after = d.loc["До дефицита"], d.loc["После начала дефицита"]
        changes.append({
            "vehicle_group": group,
            "dangerous_share_change_pp": after["dangerous_share_pct"] - before["dangerous_share_pct"],
            "dangerous_share_pct_change": pct(after["dangerous_share_pct"], before["dangerous_share_pct"]),
            "dangerous_rate_pct_change": pct(after["dangerous_per_1000_clients_30d"], before["dangerous_per_1000_clients_30d"]),
            "light_rate_pct_change": pct(after["light_per_1000_clients_30d"], before["light_per_1000_clients_30d"]),
        })
    changes = pd.DataFrame(changes)

    regional = fines.groupby(["region", "vehicle_group", "period"], observed=True).agg(
        fine_count=("bill_id", "size"), dangerous_share=("dangerous_flag", "mean")
    ).reset_index()
    wide = regional.pivot(index="region", columns=["vehicle_group", "period"], values=["fine_count", "dangerous_share"])
    wide.columns = ["__".join(map(str, c)) for c in wide.columns]
    wide = wide.reset_index()
    old, new = "Старые, от 11 лет", "Новые, до 5 лет"
    wide["new_change_pp"] = 100 * (wide[f"dangerous_share__{new}__После начала дефицита"] - wide[f"dangerous_share__{new}__До дефицита"])
    wide["old_change_pp"] = 100 * (wide[f"dangerous_share__{old}__После начала дефицита"] - wide[f"dangerous_share__{old}__До дефицита"])
    wide["old_minus_new_gap_pp"] = wide["old_change_pp"] - wide["new_change_pp"]
    wide["region_name"] = wide["region"].map(REGION_NAMES)
    shortage = pd.read_csv(ROOT / "outputs" / "wealth_newness_20260919" / "physical_shortage_regions.csv")
    wide = wide.merge(shortage[["region", "physical_shortage", "liters_pct_change", "tx_pct_change"]], on="region")
    wide["min_group_period_fines"] = wide[[c for c in wide.columns if c.startswith("fine_count__")]].min(axis=1)
    return agg, changes, wide


def chart_share(agg):
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    style(ax)
    x = np.array([0, 1])
    groups = [("Новые, до 5 лет", GREEN, "Новые авто"), ("Старые, от 11 лет", RED, "Старые авто")]
    for group, color, label in groups:
        d = agg[agg["vehicle_group"].eq(group)].set_index("period")
        vals = [d.loc["До дефицита", "dangerous_share_pct"], d.loc["После начала дефицита", "dangerous_share_pct"]]
        ax.plot(x, vals, color=color, linewidth=5, marker="o", markersize=14, label=label)
        for xi, value in zip(x, vals):
            ax.text(xi, value + .14, f"{value:.2f}%".replace(".", ","), color=FG, ha="center", va="bottom", fontsize=18, fontweight="bold")
    ax.set_xlim(-.18, 1.18); ax.set_ylim(4.6, 7.2)
    ax.set_xticks(x, ["До дефицита", "После начала дефицита"], color=FG, fontsize=16)
    ax.set_ylabel("Доля опасных нарушений среди всех штрафов, %", color=MUTED, fontsize=14)
    ax.set_title("У новых автомобилей доля опасных нарушений падает,\nу старых — растёт", loc="left", color=FG, fontsize=23, fontweight="bold", pad=18)
    ax.legend(frameon=False, labelcolor=FG, fontsize=15, loc="upper right")
    ax.text(.01, -.16, "Опасные: скорость >40 км/ч, красный свет, телефон, обочина, встречная полоса, пешеход.  DDD: OR=0,712; p=0,000230.", transform=ax.transAxes, color=MUTED, fontsize=11)
    fig.tight_layout(rect=[0.04,.08,.98,.96])
    fig.savefig(OUT / "01_dangerous_share_prepost.png", dpi=180, facecolor=BG)
    plt.close(fig)


def chart_rates(changes):
    order = ["Новые, до 5 лет", "Старые, от 11 лет"]
    d = changes.set_index("vehicle_group").loc[order]
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    style(ax)
    y = np.arange(2); h = .28
    dangerous = d["dangerous_rate_pct_change"].to_numpy(); light = d["light_rate_pct_change"].to_numpy()
    ax.barh(y + h/1.7, dangerous, height=h, color=RED, label="Опасные нарушения")
    ax.barh(y - h/1.7, light, height=h, color=YELLOW, alpha=.92, label="Лёгкие нарушения")
    ax.axvline(0, color=FG, linewidth=1)
    for yy, value in zip(y + h/1.7, dangerous):
        ax.text(value + (1 if value >= 0 else -1), yy, f"{value:+.0f}%", color=FG, va="center", ha="left" if value >= 0 else "right", fontsize=17, fontweight="bold")
    for yy, value in zip(y - h/1.7, light):
        ax.text(value + 1, yy, f"{value:+.0f}%", color=FG, va="center", ha="left", fontsize=17, fontweight="bold")
    ax.set_yticks(y, ["Новые авто\nдо 5 лет", "Старые авто\nот 11 лет"], color=FG, fontsize=16)
    ax.invert_yaxis(); ax.set_xlim(-8, 34)
    ax.set_xlabel("Изменение числа штрафов на 1 000 клиентов за 30 дней, %", color=MUTED, fontsize=14, labelpad=18)
    ax.set_title("У новых авто растут только лёгкие штрафы;\nу старых растут и опасные", loc="left", color=FG, fontsize=23, fontweight="bold", pad=18)
    ax.legend(frameon=False, labelcolor=FG, fontsize=14, loc="lower right")
    fig.text(.18, .035, "Нормировка на 30 дней устраняет разную длину периодов: 54 дня до дефицита и 92 дня после.", color=MUTED, fontsize=11)
    fig.tight_layout(rect=[0.04,.15,.98,.96])
    fig.savefig(OUT / "02_fine_rate_change.png", dpi=180, facecolor=BG)
    plt.close(fig)


def chart_regions(regional):
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    style(ax)
    x = regional["physical_shortage"].to_numpy(); y = regional["old_minus_new_gap_pp"].to_numpy()
    ax.scatter(x, y, s=130, c=YELLOW, edgecolors=BG, linewidths=1.3, zorder=3)
    slope, intercept, r, p, _ = stats.linregress(x, y)
    line_x = np.linspace(x.min()-.08, x.max()+.08, 100)
    ax.plot(line_x, intercept+slope*line_x, color=FG, linewidth=2.4, alpha=.85)
    for _, row in regional.iterrows():
        short = {"Московская область":"МО","Ленинградская область":"ЛО","Нижегородская область":"НО",
                 "Свердловская область":"Свердл.","Новосибирская область":"Новосиб.","Краснодарский край":"Краснодар",
                 "Красноярский край":"Красноярск","Челябинская область":"Челябинск","Кемеровская область":"Кемерово",
                 "Самарская область":"Самара","Тюменская область":"Тюмень","Омская область":"Омск",
                 "Санкт-Петербург":"СПб"}.get(row["region_name"], row["region_name"])
        ax.annotate(short, (row["physical_shortage"], row["old_minus_new_gap_pp"]), xytext=(6,6), textcoords="offset points", color=MUTED, fontsize=10)
    ax.axhline(0, color="#5A616B", linewidth=1)
    ax.set_xlabel("Индекс физического дефицита региона, стандартные отклонения", color=MUTED, fontsize=14)
    ax.set_ylabel("Разрыв старые − новые в изменении доли опасных нарушений, п.п.", color=MUTED, fontsize=14)
    ax.set_title("Чем сильнее дефицит, тем больше разрыв\nмежду старыми и новыми автомобилями", loc="left", color=FG, fontsize=23, fontweight="bold", pad=18)
    ax.text(.02,.94,f"r = {r:.2f}   p = {p:.4f}".replace(".",","), transform=ax.transAxes, color=FG, fontsize=17, fontweight="bold")
    ax.text(.01,-.15,"Положительное значение: у старых автомобилей опасная доля ухудшилась сильнее, чем у новых. 16 регионов; график описательный, основной тест — DDD.", transform=ax.transAxes, color=MUTED, fontsize=11)
    fig.tight_layout(rect=[0.04,.09,.98,.96])
    fig.savefig(OUT / "03_shortage_dose_response.png", dpi=180, facecolor=BG)
    plt.close(fig)


def save_tables(agg, changes, regional):
    agg.to_csv(OUT / "chart_data_prepost.csv", index=False)
    changes.to_csv(OUT / "chart_data_changes.csv", index=False)
    regional.to_csv(OUT / "chart_data_regions.csv", index=False)
    dictionary = pd.DataFrame([
        ["vehicle_age", "клиент", "2026 − auto_year", "Возраст основного автомобиля"],
        ["vehicle_group", "клиент", "новые: age≤5; старые: age≥11", "Сравниваем контрастные группы"],
        ["period", "штраф/транзакция", "до: 01.04–24.05; после: 01.06–31.08", "Единый cutoff 1 июня"],
        ["dangerous_flag", "штраф", "1 для >40 км/ч, красного, телефона, обочины, встречной полосы, пешехода", "Характер нарушения"],
        ["light_flag", "штраф", "1 − dangerous_flag", "Остальные нарушения"],
        ["dangerous_share_pct", "группа×период", "100 × dangerous_fines / all_fines", "Структура штрафов"],
        ["dangerous_per_1000_clients_30d", "группа×период", "dangerous_fines / (clients × days/30) × 1000", "Сопоставимая частота при разной длине периодов"],
        ["light_per_1000_clients_30d", "группа×период", "light_fines / (clients × days/30) × 1000", "Сопоставимая частота лёгких штрафов"],
        ["physical_shortage", "регион", "mean[z(−Δ% литров), z(−Δ% транзакций)]", "Сила физического дефицита без цены"],
        ["new_change_pp", "регион", "share_new_after − share_new_before", "Изменение опасной доли у новых"],
        ["old_change_pp", "регион", "share_old_after − share_old_before", "Изменение опасной доли у старых"],
        ["old_minus_new_gap_pp", "регион", "old_change_pp − new_change_pp", "Насколько старые ухудшились сильнее новых"],
    ], columns=["variable", "level", "formula", "purpose"])
    dictionary.to_csv(OUT / "derived_variables_dictionary.csv", index=False)


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans", "axes.unicode_minus":False})
    agg, changes, regional = load_and_build()
    save_tables(agg, changes, regional)
    chart_share(agg)
    chart_rates(changes)
    chart_regions(regional)
    rho, rho_p = stats.spearmanr(regional["physical_shortage"], regional["old_minus_new_gap_pp"])
    print(agg.to_string(index=False))
    print("\nChanges\n", changes.to_string(index=False))
    print("\nRegional Spearman", rho, rho_p)


if __name__ == "__main__":
    main()
