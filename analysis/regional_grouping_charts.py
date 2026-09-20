"""Slide-ready charts for the preregional-fuel-activity heterogeneity result."""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'clean_research_full_20260920'
TABLES = OUT / 'tables'
CHARTS = OUT / 'charts'
CHARTS.mkdir(exist_ok=True)

BLUE = '#1769C2'
NAVY = '#123F78'
SKY = '#74BBEC'
PALE = '#E8F4FE'
INK = '#133453'
MUTED = '#526F8B'
GRID = '#DCEAF5'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.titlesize': 19,
    'axes.titleweight': 'bold', 'axes.titlecolor': INK, 'axes.labelcolor': INK,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#C3D6E6', 'grid.color': GRID, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white',
})

NAMES = {
    77: 'Москва', 50: 'Московская область', 78: 'Санкт-Петербург', 16: 'Татарстан',
    66: 'Свердловская область', 54: 'Новосибирская область', 47: 'Ленинградская область',
    52: 'Нижегородская область', 23: 'Краснодарский край', 63: 'Самарская область',
    74: 'Челябинская область', 42: 'Кемеровская область', 72: 'Тюменская область',
    2: 'Башкортостан', 24: 'Красноярский край', 55: 'Омская область',
}


def save(fig, name):
    fig.tight_layout(rect=[0.0, 0.06, 1.0, .94])
    fig.savefig(CHARTS / f'{name}.png', dpi=220)
    fig.savefig(CHARTS / f'{name}.svg')
    plt.close(fig)


def tidy(ax, ygrid=True):
    if ygrid:
        ax.grid(axis='y', alpha=.85)
        ax.set_axisbelow(True)


def model(formula, x):
    return smf.ols(formula, x, missing='raise').fit(
        cov_type='cluster', cov_kwds={'groups': x.region}, use_t=True
    )


def main():
    yy = pd.read_pickle(ROOT / 'tmp' / 'clean_research_full' / 'fines.pkl')
    baseline = pd.read_csv(TABLES / 'regional_grouping_baselines.csv').set_index('region')
    cutoff = pd.read_csv(TABLES / 'regional_grouping_cutoffs.csv').query(
        "scheme == 'Предпериодная топливная активность'"
    ).median_cutoff.iloc[0]
    high_key, low_key = 'Выше медианы активности', 'Ниже медианы активности'
    group = pd.Series(np.where(baseline.pre_liters > cutoff, high_key, low_key), index=baseline.index)
    x = yy.copy()
    x['fuel_group'] = x.region.map(group)

    # 1. The audience-facing observed DiD: actual shares by cohort and period in each group.
    levels = x.groupby(['fuel_group', 'new', 'post']).agg(
        fines=('dangerous', 'size'), dangerous_share=('dangerous', 'mean')
    ).reset_index()
    levels['dangerous_share'] *= 100
    labels = {0: 'До шока', 1: 'После шока'}
    colors = {1: BLUE, 0: SKY}
    cohorts = {1: 'Новые авто (0–5 лет)', 0: 'Старые авто (11+ лет)'}
    effect = pd.read_csv(TABLES / 'regional_grouping_models.csv')
    effect = effect.query("scheme == 'Предпериодная топливная активность'").set_index('group')
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), sharey=True)
    for ax, g in zip(axes, [high_key, low_key]):
        d = levels[levels.fuel_group == g]
        for new in [1, 0]:
            q = d[d.new == new].sort_values('post')
            ax.plot(q.post, q.dangerous_share, color=colors[new], lw=3, marker='o', ms=9, label=cohorts[new])
            for r in q.itertuples():
                ax.annotate(f'{r.dangerous_share:.1f}%', (r.post, r.dangerous_share), xytext=(0, 11 if new else -20),
                            textcoords='offset points', ha='center', fontsize=11, color=colors[new], weight='bold')
        r = effect.loc[g]
        ax.set_title('Высокая предпериодная\nтопливная активность' if g == high_key else 'Низкая предпериодная\nтопливная активность', loc='left', pad=13)
        ax.set_xticks([0, 1], [labels[0], labels[1]])
        ax.set_xlim(-.18, 1.18)
        ax.text(.5, .05, f'Оценка модели: {r.beta_pp:.2f} п.п.\n95% ДИ [{r.ci_low_pp:.2f}; {r.ci_high_pp:.2f}]',
                transform=ax.transAxes, ha='center', va='bottom', fontsize=11,
                bbox=dict(facecolor=PALE, edgecolor='none', boxstyle='round,pad=.5'))
        tidy(ax)
    axes[0].set_ylabel('Опасные нарушения, % всех штрафов')
    axes[0].legend(loc='upper left', frameon=False)
    fig.suptitle('Динамика опасной доли до и после шока',
                 x=.02, ha='left', y=.992, fontsize=19, weight='bold', color=INK)
    fig.text(.02, .012, 'Группы заданы по литрам на клиента до шока: апрель–май 2026. Периоды: до — 1 апреля–24 мая; после — 1 июня–31 августа.', fontsize=9.5, color=MUTED)
    save(fig, '28_fuel_activity_did')

    # 2. Compact forest plot: the adjusted estimates and their direct comparison.
    d = effect.loc[[high_key, low_key]].reset_index()
    fig, ax = plt.subplots(figsize=(11.2, 5.7))
    y = np.array([1, 0])
    bar_colors = [NAVY, SKY]
    ax.errorbar(d.beta_pp, y, xerr=np.array([d.beta_pp - d.ci_low_pp, d.ci_high_pp - d.beta_pp]),
                fmt='none', ecolor=SKY, elinewidth=3, capsize=5, zorder=2)
    ax.scatter(d.beta_pp, y, s=130, color=bar_colors, zorder=3)
    ax.axvline(0, color=MUTED, ls='--', lw=1.4)
    for yi, r in zip(y, d.itertuples()):
        p_label = 'p<0,001' if r.p_value < .001 else f'p={r.p_value:.3f}'
        ax.text(r.ci_high_pp + .17, yi, f'{r.beta_pp:.2f} п.п.  ({p_label})', va='center', fontsize=12, color=INK)
    ax.set_yticks(y, ['Высокая предпериодная\nтопливная активность (8 регионов)', 'Низкая предпериодная\nтопливная активность (8 регионов)'])
    ax.set_xlabel('Изменение опасной доли у новых авто относительно старых, п.п.')
    ax.set_xlim(-4.25, 1.35)
    ax.set_ylim(-.55, 1.55)
    tidy(ax, False)
    ax.set_title('Эффект существенно сильнее при высокой\nтопливной активности', loc='left', pad=18)
    fig.text(.02, .03, 'Различие оценок между группами: p=0,0157; FDR q=0,0313. Модель: регион × период, регион × возраст авто и клиентские контроли.', fontsize=10, color=MUTED)
    save(fig, '29_fuel_activity_forest')

    # 3. Continuous specification: no reliance on the median boundary.
    x = yy.merge(baseline[['pre_liters']], left_on='region', right_index=True)
    x['fuel_z'] = (x.pre_liters - x.pre_liters.mean()) / x.pre_liters.std(ddof=0)
    controls = 'log_price + price_missing + log_fines25 + cars + C(gender) + C(age_type)'
    m = model('dangerous ~ C(region)*post + C(region)*new + post_new + post_new:fuel_z + ' + controls, x)
    b0, b1 = m.params['post_new'], m.params['post_new:fuel_z']
    cov = m.cov_params().loc[['post_new', 'post_new:fuel_z'], ['post_new', 'post_new:fuel_z']].to_numpy()
    grid_liters = np.linspace(baseline.pre_liters.min(), baseline.pre_liters.max(), 100)
    mean, sd = x.pre_liters.mean(), x.pre_liters.std(ddof=0)
    gz = (grid_liters - mean) / sd
    pred = b0 + b1 * gz
    v = np.vstack([np.ones_like(gz), gz]).T
    se = np.sqrt(np.einsum('ij,jk,ik->i', v, cov, v))
    crit = st.t.ppf(.975, x.region.nunique() - 1)
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.fill_between(grid_liters, (pred - crit * se) * 100, (pred + crit * se) * 100, color=PALE, label='95% доверительный интервал')
    ax.plot(grid_liters, pred * 100, color=NAVY, lw=3, label='Предсказанный эффект модели')
    ax.axvline(cutoff, color=BLUE, ls='--', lw=1.8, label='Медиана: 108 л')
    ax.axhline(0, color=MUTED, ls='--', lw=1.2)
    for r in baseline.itertuples():
        ax.scatter(r.pre_liters, -4.08, s=38, color=NAVY if r.pre_liters > cutoff else SKY, edgecolor='white', zorder=4)
    ax.set_ylim(-4.65, 1.5)
    ax.set_xlabel('Предпериодная топливная активность, л / клиент / 30 дней')
    ax.set_ylabel('Предсказанный DiD опасной доли, п.п.')
    ax.set_title('Чем выше исходная топливная активность, тем сильнее эффект', loc='left', pad=18)
    ax.legend(frameon=False, loc='upper right')
    tidy(ax)
    fig.text(.02, .018, 'Точки на нижней шкале — 16 регионов.\nВзаимодействие Post × New × топливная активность: −0,97 п.п. на 1 SD; p=0,033; wild-cluster p=0,0036.', fontsize=10, color=MUTED)
    save(fig, '30_fuel_activity_continuous')

    # Data behind the observed slide chart for reviewers and deck tables.
    levels.assign(period=levels.post.map({0: 'pre', 1: 'post'})).to_csv(TABLES / 'regional_grouping_observed_did.csv', index=False)
    print('Generated 3 charts and observed group-period data.')


if __name__ == '__main__':
    main()
