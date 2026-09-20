from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path("/Library/Everything/hakaton")
OUT = ROOT / "outputs" / "clean_correlation_matrix_20260920"
OUT.mkdir(parents=True, exist_ok=True)

DEMOGRAPHICS = ROOT / "outputs/fuel_cleaning_20260919/clients_demographics_clean.csv"
TRANSACTIONS = ROOT / "outputs/fuel_cleaning_20260919/fuel_transaction_clean.csv"
FINES = ROOT / "outputs/fuel_cleaning_20260919/fines_2026_clean.csv"


def number(series):
    return pd.to_numeric(
        series.astype("string").str.replace(" ", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def rank_corr(frame):
    return frame.rank(method="average").corr(method="pearson")


def main():
    demo = pd.read_csv(DEMOGRAPHICS, sep=";", encoding="utf-8-sig")
    tx = pd.read_csv(TRANSACTIONS, sep=";", encoding="utf-8-sig")
    fines = pd.read_csv(FINES, sep=";", encoding="utf-8-sig")

    demo["Стоимость автомобиля"] = number(demo["price"])
    demo["Возраст автомобиля"] = 2026 - pd.to_numeric(demo["auto_year"], errors="coerce")
    fine_history_cols = [
        "april_2025_fines", "may_2025_fines", "jun_2025_fines", "jul_2025_fines", "aug_2025_fines",
        "fines_last_6_month", "fines_last_12_month", "fines_last_24_month", "fines_last_36_month",
    ]
    for col in fine_history_cols:
        demo[col] = number(demo[col]).fillna(0)

    tx["Объём топлива, л"] = number(tx["order_fuel_volume"])
    tx["Цена топлива, ₽/л"] = number(tx["order_fuel_price_1liter"])
    tx["Стоимость заправки, ₽"] = tx["Объём топлива, л"] * tx["Цена топлива, ₽/л"]
    tx_agg = tx.groupby("client_id", as_index=False).agg(
        **{
            "Количество заправок": ("order_id", "nunique"),
            "Объём топлива, л": ("Объём топлива, л", "sum"),
            "Средний объём заправки, л": ("Объём топлива, л", "mean"),
            "Средняя цена топлива, ₽/л": ("Цена топлива, ₽/л", "mean"),
            "Расходы на топливо, ₽": ("Стоимость заправки, ₽", "sum"),
            "Средний чек заправки, ₽": ("Стоимость заправки, ₽", "mean"),
        }
    )

    fines["Сумма штрафа, ₽"] = number(fines["total_fine_amount"])
    fines_agg = fines.groupby("client_id", as_index=False).agg(
        **{
            "Количество штрафов 2026": ("bill_id", "nunique"),
            "Сумма штрафов 2026, ₽": ("Сумма штрафа, ₽", "sum"),
        }
    )

    data = demo[["client_id", "Возраст автомобиля", "Стоимость автомобиля"] + fine_history_cols].copy()
    data = data.merge(tx_agg, on="client_id", how="left").merge(fines_agg, on="client_id", how="left")
    numeric_cols = [c for c in data.columns if c != "client_id"]
    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    data = data[data["Количество заправок"] > 0].copy()

    rename = {
        "fines_last_6_month": "Штрафы за 6 мес.",
        "fines_last_12_month": "Штрафы за 12 мес.",
        "fines_last_24_month": "Штрафы за 24 мес.",
        "fines_last_36_month": "Штрафы за 36 мес.",
    }
    data = data.rename(columns=rename)
    labels = [
        "Возраст авто, лет", "Стоимость авто, ₽", "Штрафы за 6 мес.", "Штрафы за 12 мес.",
        "Штрафы за 24 мес.", "Штрафы за 36 мес.", "Заправки, шт.", "Топливо, л",
        "Средний объём, л", "Средняя цена, ₽/л", "Траты на топливо, ₽", "Средний чек, ₽",
        "Штрафы 2026, шт.", "Сумма штрафов 2026, ₽",
    ]
    mapping = dict(zip(
        ["Возраст автомобиля", "Стоимость автомобиля", "Штрафы за 6 мес.", "Штрафы за 12 мес.",
         "Штрафы за 24 мес.", "Штрафы за 36 мес.", "Количество заправок", "Объём топлива, л",
         "Средний объём заправки, л", "Средняя цена топлива, ₽/л", "Расходы на топливо, ₽",
         "Средний чек заправки, ₽", "Количество штрафов 2026", "Сумма штрафов 2026, ₽"], labels
    ))
    selected = data[list(mapping)].rename(columns=mapping)

    corr = rank_corr(selected).loc[labels, labels]
    corr.to_csv(OUT / "матрица_корреляций_spearman.csv", sep=";", decimal=",", encoding="utf-8-sig")
    selected.to_csv(OUT / "клиентские_показатели_для_матрицы.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titleweight"] = "bold"
    fig, ax = plt.subplots(figsize=(23, 18), dpi=180)
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")
    cmap = LinearSegmentedColormap.from_list("corr", ["#2166AC", "#67A9CF", "#F7F7F7", "#EF8A62", "#B2182B"], N=256)
    im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(labels)), labels, rotation=48, ha="right", rotation_mode="anchor", fontsize=10)
    ax.set_yticks(range(len(labels)), labels, fontsize=10)
    ax.tick_params(length=0, pad=8)
    ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=1.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.iloc[i, j]
            color = "white" if abs(val) >= 0.55 else "#243447"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Коэффициент Спирмена", fontsize=11, labelpad=12)
    cbar.ax.tick_params(labelsize=10)
    ax.set_title(
        "Матрица корреляций очищенных данных\n"
        f"Уровень клиента · Spearman · n = {len(selected):,} клиентов".replace(",", " "),
        fontsize=20, loc="left", pad=24, color="#182230",
    )
    fig.text(
        0.02, 0.018,
        "Синие оттенки — обратная связь, красные — прямая; чем насыщеннее цвет, тем сильнее связь. "
        "Технические идентификаторы и даты исключены.",
        fontsize=10.5, color="#52606D",
    )
    plt.subplots_adjust(left=0.27, right=0.92, top=0.88, bottom=0.22)
    fig.savefig(OUT / "матрица_корреляций_очищенные_данные.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = pd.DataFrame({
        "Показатель": ["Клиенты в матрице", "Исходный clean-файл клиентов", "Исходный clean-файл транзакций", "Исходный clean-файл штрафов"],
        "Значение": [len(selected), DEMOGRAPHICS.name, TRANSACTIONS.name, FINES.name],
    })
    summary.to_csv(OUT / "описание.csv", sep=";", index=False, encoding="utf-8-sig")
    print(f"saved: {OUT / 'матрица_корреляций_очищенные_данные.png'}")
    print(f"clients: {len(selected)}")
    print(corr.round(2).iloc[:4, :4])


if __name__ == "__main__":
    main()
