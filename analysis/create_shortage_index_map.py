"""Map the standardized regional fuel-purchase decline index S."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon, Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "wealth_newness_clean_20260920" / "region_shortage_clean_full.csv"
GEO = ROOT / "resources" / "geodata" / "candidates" / "russia_subjects_github.json"
OUT = ROOT / "outputs" / "wealth_newness_clean_20260920"

REGION_NAMES = {
    2: "Республика Башкортостан",
    16: "Республика Татарстан (Татарстан)",
    23: "Краснодарский край",
    24: "Красноярский край",
    42: "Кемеровская область",
    47: "Ленинградская область",
    50: "Московская область",
    52: "Нижегородская область",
    54: "Новосибирская область",
    55: "Омская область",
    63: "Самарская область",
    66: "Свердловская область",
    72: "Тюменская область",
    74: "Челябинская область",
    77: "город федерального значения Москва",
    78: "город федерального значения Санкт-Петербург",
}


def main():
    data = pd.read_csv(DATA)
    values = {REGION_NAMES[int(row.region)]: float(row.shortage_std) for row in data.itertuples()}
    geo = json.loads(GEO.read_text())

    cmap = LinearSegmentedColormap.from_list(
        "shortage", ["#9CCAF0", "#EDF3F7", "#F4C07A", "#C74C3C"]
    )
    limit = max(abs(min(values.values())), abs(max(values.values())))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    found = set()

    for feature in geo["features"]:
        name = feature["properties"].get("NL_NAME_1")
        value = values.get(name)
        color = cmap(norm(value)) if value is not None else "#E8EDF1"
        if value is not None:
            found.add(name)
        geometry = feature["geometry"]
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for polygon in polygons:
            coords = np.asarray(polygon[0])
            ax.add_patch(
                Polygon(coords, closed=True, facecolor=color, edgecolor="white", linewidth=0.45)
            )

    # Федеральные города почти не видны в масштабе страны.
    for name, short, lon, lat in [
        ("город федерального значения Москва", "Москва", 37.62, 55.75),
        ("город федерального значения Санкт-Петербург", "Санкт-Петербург", 30.32, 59.94),
    ]:
        value = values[name]
        ax.scatter(lon, lat, s=95, color=cmap(norm(value)), edgecolor="#17324D", linewidth=0.8, zorder=5)
        ax.annotate(short, (lon, lat), xytext=(8, 8), textcoords="offset points", fontsize=10, color="#17324D")

    ax.set_xlim(19, 116)
    ax.set_ylim(43, 73)
    ax.set_aspect(1.7)
    ax.axis("off")

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.028, pad=0.025)
    colorbar.set_label("Индекс S, стандартные отклонения", fontsize=12)
    colorbar.ax.tick_params(labelsize=10)

    legend = [
        Patch(facecolor=cmap(norm(1.25)), edgecolor="none", label="Сильнее среднего падение топлива"),
        Patch(facecolor=cmap(norm(-1.0)), edgecolor="none", label="Слабее среднего падение топлива"),
        Patch(facecolor="#E8EDF1", edgecolor="none", label="Регион вне выборки"),
    ]
    ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.01, 0.01), frameon=False, fontsize=11)

    fig.suptitle("Региональный индекс падения покупок топлива S", x=0.06, y=0.965,
                 ha="left", fontsize=22, fontweight="bold", color="#17324D")
    fig.text(0.06, 0.91,
             "S = среднее стандартизированного падения литров и числа заправок после 1 июня 2026 года",
             fontsize=12, color="#526F8B")
    fig.text(0.06, 0.055,
             "Важно: S измеряет падение покупок в данных, а не подтверждённый физический остаток топлива на АЗС.",
             fontsize=10.5, color="#526F8B")
    fig.subplots_adjust(left=0.02, right=0.94, top=0.88, bottom=0.11)

    assert found == set(values), f"Не найдены регионы: {set(values) - found}"
    fig.savefig(OUT / "shortage_index_s_map.png", dpi=200, facecolor="white")
    fig.savefig(OUT / "shortage_index_s_map.svg", facecolor="white")
    plt.close(fig)
    print(f"Saved map for {len(found)} regions")


if __name__ == "__main__":
    main()
