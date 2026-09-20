from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GEOJSON = ROOT / "resources" / "geodata" / "candidates" / "russia_subjects_github.json"
OUT_DIR = Path("/Users/adiamant_03/.codex/visualizations/2026/09/19/01a0b9de-040c-74b1-8506-a899e1bc5768")
OUT_HTML = OUT_DIR / "fuel-price-change-percent-by-region.html"
OUT_CSV = ROOT / "working_data" / "preanalysis" / "region_fuel_price_change_percent.csv"

NAME_MAP = {
    "Москва": "город федерального значения Москва",
    "Санкт-Петербург": "город федерального значения Санкт-Петербург",
    "Республика Татарстан": "Республика Татарстан (Татарстан)",
}


def simplify_line(points: list, tolerance: float = 0.08) -> list:
    if len(points) <= 3:
        return points
    x1, y1 = points[0][:2]
    x2, y2 = points[-1][:2]
    denominator = math.hypot(x2 - x1, y2 - y1)
    best_distance, best_index = -1.0, 0
    for index, point in enumerate(points[1:-1], start=1):
        x, y = point[:2]
        distance = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1) / denominator if denominator else math.hypot(x-x1, y-y1)
        if distance > best_distance:
            best_distance, best_index = distance, index
    if best_distance > tolerance:
        return simplify_line(points[:best_index+1], tolerance)[:-1] + simplify_line(points[best_index:], tolerance)
    return [points[0], points[-1]]


def simplify_geometry(geometry: dict) -> dict:
    def ring(points: list) -> list:
        result = simplify_line(points)
        if result[0] != result[-1]:
            result.append(result[0])
        return [[round(p[0], 3), round(p[1], 3)] for p in result] if len(result) >= 4 else []
    if geometry["type"] == "Polygon":
        coordinates = [result for points in geometry["coordinates"] if (result := ring(points))]
    else:
        coordinates = []
        for polygon in geometry["coordinates"]:
            rings = [result for points in polygon if (result := ring(points))]
            if rings:
                coordinates.append(rings)
    return {"type": geometry["type"], "coordinates": coordinates}


def main() -> None:
    fuel = pd.read_csv(ROOT / "fuel_transaction.csv", sep=";", decimal=",", parse_dates=["order_datetime"])
    clients = pd.read_csv(ROOT / "working_data" / "client_profile.csv", usecols=["client_id", "registration_region"])
    data = fuel.merge(clients, on="client_id", how="inner")
    data = data.loc[
        data["order_datetime"].between("2026-04-01", "2026-08-31 23:59:59.999")
        & data["order_fuel_volume"].gt(0)
        & data["order_fuel_price_1liter"].between(50, 130)
        & data["registration_region"].ne("Другой/неизвестный")
    ].copy()
    data["period"] = data["order_datetime"].dt.month.map(
        lambda month: "До кризиса" if month <= 5 else "Во время кризиса"
    )
    summary = (
        data.groupby(["registration_region", "period"], as_index=False)
        .agg(
            median_price=("order_fuel_price_1liter", "median"),
            transactions=("order_id", "nunique"),
            clients=("client_id", "nunique"),
        )
    )
    wide = summary.pivot(index="registration_region", columns="period", values="median_price")
    details = summary.pivot(index="registration_region", columns="period")
    result = pd.DataFrame(index=wide.index)
    result["spring_median_rub_l"] = wide["До кризиса"]
    result["summer_median_rub_l"] = wide["Во время кризиса"]
    result["change_pct"] = (result["summer_median_rub_l"] / result["spring_median_rub_l"] - 1) * 100
    result["spring_transactions"] = details["transactions"]["До кризиса"]
    result["summer_transactions"] = details["transactions"]["Во время кризиса"]
    result = result.reset_index()
    result.to_csv(OUT_CSV, index=False)

    values = {}
    for row in result.itertuples(index=False):
        geo_name = NAME_MAP.get(row.registration_region, row.registration_region)
        values[geo_name] = {
            "region": row.registration_region,
            "spring": round(float(row.spring_median_rub_l), 2),
            "summer": round(float(row.summer_median_rub_l), 2),
            "changePct": round(float(row.change_pct), 2),
            "springTransactions": int(row.spring_transactions),
            "summerTransactions": int(row.summer_transactions),
        }

    geo = json.loads(GEOJSON.read_text(encoding="utf-8"))
    features = []
    for feature in geo["features"]:
        features.append({
            "type": "Feature",
            "properties": {"name": feature["properties"].get("NL_NAME_1", "")},
            "geometry": simplify_geometry(feature["geometry"]),
        })
    embedded = json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, separators=(",", ":"))
    embedded_values = json.dumps(values, ensure_ascii=False, separators=(",", ":"))

    fragment = f'''<div id="fuel-price-change-map">
  <h2>Подорожание топлива после начала кризиса</h2>
  <div class="viz-row legend-row" aria-label="Шкала изменения медианной цены">
    <span class="text-muted">+1,5%</span><span class="gradient" aria-hidden="true"></span><span class="text-muted">+3,0%</span>
    <span class="missing-key" aria-hidden="true"></span><span class="text-muted">нет данных</span>
  </div>
  <div class="map"></div>
  <div class="tooltip" role="tooltip" hidden></div>
  <div class="text-small text-muted note">Изменение медианной цены: июнь–август против апреля–мая 2026. Положительный объём, цена 50–130 ₽/л; регион регистрации клиента.</div>
</div>
<style>
#fuel-price-change-map {{ position:relative; width:100%; color:var(--foreground); }}
#fuel-price-change-map h2 {{ margin:0 0 8px; }}
#fuel-price-change-map .legend-row {{ gap:8px; justify-content:center; margin-bottom:6px; }}
#fuel-price-change-map .gradient {{ width:200px; height:12px; background:linear-gradient(90deg,var(--muted),var(--viz-series-2)); }}
#fuel-price-change-map .missing-key {{ width:12px; height:12px; background:var(--muted); opacity:.45; }}
#fuel-price-change-map .map {{ max-width:760px; margin:0 auto; }}
#fuel-price-change-map svg {{ display:block; width:100%; height:auto; overflow:visible; }}
#fuel-price-change-map path {{ stroke:var(--background); stroke-width:.45; vector-effect:non-scaling-stroke; }}
#fuel-price-change-map path.has-data {{ cursor:pointer; }}
#fuel-price-change-map path.has-data:focus {{ stroke:var(--ring); stroke-width:2; outline:none; }}
#fuel-price-change-map .tooltip {{ position:absolute; pointer-events:none; z-index:5; padding:8px 10px; background:var(--popover); color:var(--popover-foreground); border:1px solid var(--border); border-radius:6px; box-shadow:0 4px 16px color-mix(in srgb,var(--foreground) 18%,transparent); max-width:280px; }}
#fuel-price-change-map .tooltip strong {{ font-weight:500; }}
#fuel-price-change-map .note {{ margin-top:6px; text-align:center; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const root = document.getElementById('fuel-price-change-map');
  const geo = {embedded};
  const values = {embedded_values};
  const tooltip = root.querySelector('.tooltip');
  const fmt = new Intl.NumberFormat('ru-RU');
  const getColor = (value) => {{
    if (value == null) return 'color-mix(in srgb, var(--muted) 45%, transparent)';
    const t = Math.max(0, Math.min(1, (value - 1.5) / 1.5));
    return `color-mix(in srgb, var(--viz-series-2) ${{Math.round(45 + 55*t)}}%, var(--muted))`;
  }};
  const show = (event, item) => {{
    tooltip.innerHTML = `<strong>${{item.region}}</strong><br><strong>+${{item.changePct.toFixed(2).replace('.', ',')}}%</strong><br><span class="text-muted">Весна: ${{item.spring.toFixed(2).replace('.', ',')}} ₽/л · ${{fmt.format(item.springTransactions)}} транзакций<br>Лето: ${{item.summer.toFixed(2).replace('.', ',')}} ₽/л · ${{fmt.format(item.summerTransactions)}} транзакций</span>`;
    tooltip.hidden = false;
    const box = root.getBoundingClientRect();
    tooltip.style.left = `${{Math.min(event.clientX - box.left + 12, box.width - 270)}}px`;
    tooltip.style.top = `${{event.clientY - box.top + 12}}px`;
  }};
  const hide = () => {{ tooltip.hidden = true; }};
  function draw(container) {{
    const width = Math.max(300, container.clientWidth);
    const height = Math.round(width * .55);
    container.replaceChildren();
    const svg = d3.select(container).append('svg').attr('viewBox', `0 0 ${{width}} ${{height}}`).attr('role','img')
      .attr('aria-label', 'Карта процентного изменения медианной цены топлива: лето к весне 2026 года');
    svg.append('title').text('Подорожание медианной цены топлива по регионам, лето к весне 2026 года');
    const projection = d3.geoMercator().fitExtent([[6,6],[width-6,height-6]], geo);
    const path = d3.geoPath(projection);
    svg.selectAll('path').data(geo.features).join('path')
      .attr('d', path)
      .attr('fill', d => getColor(values[d.properties.name]?.changePct))
      .attr('class', d => values[d.properties.name] ? 'has-data' : null)
      .attr('tabindex', d => values[d.properties.name] ? 0 : null)
      .attr('aria-label', d => {{ const x=values[d.properties.name]; return x ? `${{x.region}}: подорожание на ${{x.changePct}} процента` : null; }})
      .on('pointerenter pointermove', (event,d) => {{ const x=values[d.properties.name]; if(x) show(event,x); }})
      .on('pointerleave', hide)
      .on('focus', (event,d) => {{ const x=values[d.properties.name]; if(x) {{ const r=event.currentTarget.getBoundingClientRect(); show({{clientX:r.left+r.width/2,clientY:r.top+r.height/2}},x); }} }})
      .on('blur', hide);
  }}
  const observer = new ResizeObserver(entries => entries.forEach(e => draw(e.target)));
  observer.observe(root.querySelector('.map'));
}})();
</script>
'''
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(fragment, encoding="utf-8")
    print(OUT_HTML)
    print(OUT_CSV)
    print(result.sort_values("change_pct").to_string(index=False))


if __name__ == "__main__":
    main()
