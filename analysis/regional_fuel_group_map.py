"""Categorical map of the pre-period fuel-activity groups used in the heterogeneity model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'clean_research_full_20260920'
TABLES = OUT / 'tables'
CHARTS = OUT / 'charts'
CHARTS.mkdir(exist_ok=True)

HIGH = '#2850F0'
LOW = '#A9D0FF'
OUTSIDE = '#EEF3F8'
INK = '#102A66'
MUTED = '#526F8B'

NAME_MAP = {
    'Москва': 'город федерального значения Москва',
    'Санкт-Петербург': 'город федерального значения Санкт-Петербург',
    'Татарстан': 'Республика Татарстан (Татарстан)',
    'Башкортостан': 'Республика Башкортостан',
}


def main():
    baseline = pd.read_csv(TABLES / 'regional_grouping_baselines.csv')
    cutoff = pd.read_csv(TABLES / 'regional_grouping_cutoffs.csv').query(
        "scheme == 'Предпериодная топливная активность'"
    ).median_cutoff.iloc[0]
    names = {
        77: 'Москва', 50: 'Московская область', 78: 'Санкт-Петербург', 16: 'Татарстан',
        66: 'Свердловская область', 54: 'Новосибирская область', 47: 'Ленинградская область',
        52: 'Нижегородская область', 23: 'Краснодарский край', 63: 'Самарская область',
        74: 'Челябинская область', 42: 'Кемеровская область', 72: 'Тюменская область',
        2: 'Башкортостан', 24: 'Красноярский край', 55: 'Омская область',
    }
    activity = {
        NAME_MAP.get(names[int(r.region)], names[int(r.region)]): ('high' if r.pre_liters > cutoff else 'low')
        for r in baseline.itertuples()
    }
    geo = json.loads((ROOT / 'resources' / 'geodata' / 'candidates' / 'russia_subjects_github.json').read_text())
    fig, ax = plt.subplots(figsize=(13.333, 7.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    found = set()
    for feature in geo['features']:
        name = feature['properties'].get('NL_NAME_1')
        group = activity.get(name)
        if group:
            found.add(name)
        color = HIGH if group == 'high' else LOW if group == 'low' else OUTSIDE
        geom = feature['geometry']
        polygons = [geom['coordinates']] if geom['type'] == 'Polygon' else geom['coordinates']
        for polygon in polygons:
            coords = np.asarray(polygon[0])
            ax.add_patch(Polygon(coords, closed=True, facecolor=color, edgecolor='white', linewidth=.55))
    # Federal cities are too small at national scale; markers prevent them disappearing into oblast contours.
    for label, lon, lat, group in [('Москва',37.62,55.75,'high'), ('Санкт-Петербург',30.32,59.94,'low')]:
        ax.scatter(lon, lat, s=88, color=HIGH if group == 'high' else LOW, edgecolor=INK, linewidth=.7, zorder=5)
        ax.annotate(label, (lon,lat), xytext=(8,8), textcoords='offset points', color=INK, fontsize=10, weight='bold')
    ax.set_xlim(19,116)
    ax.set_ylim(43,73)
    ax.set_aspect(1.7)
    ax.axis('off')
    legend = [
        Patch(facecolor=HIGH, edgecolor='none', label='Высокая активность: 8 регионов'),
        Patch(facecolor=LOW, edgecolor='none', label='Низкая активность: 8 регионов'),
        Patch(facecolor=OUTSIDE, edgecolor='none', label='Вне выборки'),
    ]
    ax.legend(handles=legend, loc='lower left', bbox_to_anchor=(.01,.01), ncol=3, frameon=False,
              fontsize=12, handlelength=1.5, columnspacing=1.6)
    # The methodological note is placed in the slide footer, outside the chart.
    # Keeping the export self-contained prevents it overlapping the map in Figma.
    fig.subplots_adjust(left=.03, right=.98, top=.98, bottom=.06)
    assert found == set(activity), f'Не найдены на карте: {set(activity) - found}'
    fig.savefig(CHARTS / '31_map_fuel_activity_groups.png', dpi=220, facecolor='white')
    fig.savefig(CHARTS / '31_map_fuel_activity_groups.svg', facecolor='white')
    plt.close(fig)
    print('Saved categorical map for', len(found), 'regions.')


if __name__ == '__main__':
    main()
