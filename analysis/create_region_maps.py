from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
GEOJSON = ROOT / "resources" / "geodata" / "candidates" / "russia_subjects_github.json"
REGIONS_CSV = ROOT / "working_data" / "client_month_analysis" / "region_summary_quality.csv"
OUT = ROOT / "figures" / "client_month_analysis"
OUT.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 1800, 980
MAP_LEFT, MAP_TOP, MAP_WIDTH, MAP_HEIGHT = 60, 130, 1240, 760
PANEL_LEFT = 1350
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

NAME_MAP = {
    "Москва": "город федерального значения Москва",
    "Санкт-Петербург": "город федерального значения Санкт-Петербург",
    "Республика Татарстан": "Республика Татарстан (Татарстан)",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD_PATH if bold else FONT_PATH, size)


def blend(left: tuple[int, int, int], right: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(round(a + (b - a) * t) for a, b in zip(left, right))


def sequential(value: float, low: float, high: float, colors: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    if high <= low:
        return colors[-1]
    t = max(0.0, min(1.0, (value - low) / (high - low)))
    scaled = t * (len(colors) - 1)
    index = min(len(colors) - 2, int(scaled))
    return blend(colors[index], colors[index + 1], scaled - index)


def diverging(value: float, low: float, high: float) -> tuple[int, int, int]:
    if value <= 0:
        return blend((37, 99, 235), (248, 250, 252), (value - low) / (0 - low) if low < 0 else 1)
    return blend((248, 250, 252), (220, 38, 38), value / high if high > 0 else 1)


def project(lon: float, lat: float) -> tuple[float, float]:
    if lon < 0:
        lon += 360
    x = MAP_LEFT + (lon - 18) / (192 - 18) * MAP_WIDTH
    # A light pseudo-Mercator transform makes high-latitude shapes less compressed.
    lat = max(-85, min(85, lat))
    merc = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    merc_min = math.log(math.tan(math.pi / 4 + math.radians(40) / 2))
    merc_max = math.log(math.tan(math.pi / 4 + math.radians(83) / 2))
    y = MAP_TOP + (merc_max - merc) / (merc_max - merc_min) * MAP_HEIGHT
    return x, y


def polygon_rings(geometry: dict):
    if geometry["type"] == "Polygon":
        yield geometry["coordinates"]
    elif geometry["type"] == "MultiPolygon":
        yield from geometry["coordinates"]


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width: int, line_height: int, fill, fnt):
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=fnt)[2] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += line_height
    return y


def render_map(metric: str, title: str, subtitle: str, filename: str, palette: str, value_format) -> Path:
    regions = pd.read_csv(REGIONS_CSV)
    values = {NAME_MAP.get(row.registration_region, row.registration_region): float(getattr(row, metric))
              for row in regions.itertuples() if pd.notna(getattr(row, metric))}
    with GEOJSON.open(encoding="utf-8") as stream:
        geo = json.load(stream)

    image = Image.new("RGB", (WIDTH, HEIGHT), "#F8FAFC")
    draw = ImageDraw.Draw(image)
    draw.text((60, 35), title, font=font(34, True), fill="#111827")
    draw.text((60, 82), subtitle, font=font(19), fill="#475569")

    numeric = list(values.values())
    if palette == "diverging":
        low, high = min(numeric), max(numeric)
    else:
        low, high = min(numeric), max(numeric)

    for feature in geo["features"]:
        name = feature["properties"].get("NL_NAME_1")
        value = values.get(name)
        if value is None:
            fill = (226, 232, 240)
        elif palette == "fines":
            fill = sequential(value, low, high, [(254, 249, 195), (251, 191, 36), (220, 38, 38)])
        elif palette == "price":
            fill = sequential(value, low, high, [(204, 251, 241), (20, 184, 166), (15, 118, 110)])
        else:
            fill = diverging(value, low, high)

        for rings in polygon_rings(feature["geometry"]):
            outer = [project(point[0], point[1]) for point in rings[0]]
            if len(outer) >= 3:
                draw.polygon(outer, fill=fill, outline="#FFFFFF", width=1)
            for hole in rings[1:]:
                hole_points = [project(point[0], point[1]) for point in hole]
                if len(hole_points) >= 3:
                    draw.polygon(hole_points, fill="#F8FAFC")

    # Border around plotting area.
    draw.rounded_rectangle(
        (MAP_LEFT - 8, MAP_TOP - 8, MAP_LEFT + MAP_WIDTH + 8, MAP_TOP + MAP_HEIGHT + 8),
        radius=16, outline="#CBD5E1", width=2
    )

    draw.text((PANEL_LEFT, 145), "Шкала", font=font(22, True), fill="#111827")
    legend_y = 190
    for i in range(220):
        t = i / 219
        v = low + (high - low) * t
        if palette == "fines":
            color = sequential(v, low, high, [(254, 249, 195), (251, 191, 36), (220, 38, 38)])
        elif palette == "price":
            color = sequential(v, low, high, [(204, 251, 241), (20, 184, 166), (15, 118, 110)])
        else:
            color = diverging(v, low, high)
        draw.line((PANEL_LEFT + i, legend_y, PANEL_LEFT + i, legend_y + 24), fill=color, width=2)
    draw.text((PANEL_LEFT, legend_y + 34), value_format(low), font=font(16), fill="#475569")
    high_text = value_format(high)
    high_width = draw.textbbox((0, 0), high_text, font=font(16))[2]
    draw.text((PANEL_LEFT + 220 - high_width, legend_y + 34), high_text, font=font(16), fill="#475569")
    draw.rectangle((PANEL_LEFT, legend_y + 75, PANEL_LEFT + 24, legend_y + 99), fill="#E2E8F0")
    draw.text((PANEL_LEFT + 36, legend_y + 75), "Нет данных", font=font(16), fill="#475569")

    draw.text((PANEL_LEFT, 340), "Регионы в выборке", font=font(22, True), fill="#111827")
    ranking = regions.loc[regions[metric].notna(), ["registration_region", metric]].sort_values(metric, ascending=False)
    y = 385
    for rank, row in enumerate(ranking.head(10).itertuples(index=False), start=1):
        name, value = row
        draw.text((PANEL_LEFT, y), f"{rank}.", font=font(16, True), fill="#64748B")
        y = draw_wrapped(draw, (PANEL_LEFT + 30, y), str(name), 265, 20, "#1F2937", font(16))
        draw.text((PANEL_LEFT + 30, y + 1), value_format(float(value)), font=font(16, True), fill="#0F766E")
        y += 32
        if y > 820:
            break

    draw.text((60, 928), "Серым показаны субъекты, отсутствующие в клиентской выборке.", font=font(16), fill="#64748B")
    draw.text((PANEL_LEFT, 900), "Источник границ: OSM/GADM, репозиторий rnekrasov-msk/geojson", font=font(13), fill="#64748B")

    path = OUT / filename
    image.save(path, format="PNG", optimize=True)
    return path


def main() -> None:
    outputs = [
        render_map(
            "fines_per_client_2026",
            "Штрафы 2026: распределение по региону регистрации",
            "Количество штрафов за апрель–август на одного клиента; цвет относится к региону регистрации клиента",
            "map_fines_per_client_2026.png",
            "fines",
            lambda x: f"{x:.2f}",
        ),
        render_map(
            "fuel_price_weighted_standard_band",
            "Робастная цена топлива по региону регистрации",
            "Средневзвешенная цена положительных покупок в диапазоне 50–130 ₽/л, апрель–август 2026",
            "map_fuel_price_standard.png",
            "price",
            lambda x: f"{x:.2f} ₽/л",
        ),
        render_map(
            "fines_change_pct",
            "Изменение числа штрафов: 2026 к 2025",
            "Сопоставлены одинаковые месяцы апрель–август; синий — снижение, красный — рост",
            "map_fines_change_2026_vs_2025.png",
            "diverging",
            lambda x: f"{x:+.1%}",
        ),
    ]
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
