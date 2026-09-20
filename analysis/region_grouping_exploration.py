"""Pre-specified, interpretable regional groupings for the clean 2026 study.

All cut-offs that use project data are defined only from pre-period or 2025 values.
This file reports every candidate split; it does not select groups by the outcome.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'clean_research_full_20260920' / 'tables'

FD = {
    77: 'Центральный', 50: 'Центральный',
    78: 'Северо-Западный', 47: 'Северо-Западный',
    2: 'Приволжский', 16: 'Приволжский', 63: 'Приволжский', 52: 'Приволжский',
    66: 'Уральский', 74: 'Уральский', 72: 'Уральский',
    54: 'Сибирский', 55: 'Сибирский', 42: 'Сибирский', 24: 'Сибирский',
    23: 'Южный',
}
METRO = {77, 50, 78, 47}


def fit(formula, data):
    return smf.ols(formula, data, missing='raise').fit(
        cov_type='cluster', cov_kwds={'groups': data.region}, use_t=True
    )


def group_effects(data, grouping, label, rationale):
    """Estimate one pooled DiD for each externally defined group."""
    x = data.copy()
    x['grp'] = grouping.loc[x.region].to_numpy()
    groups = list(pd.Series(x.grp.unique()).sort_values())
    counts = x.groupby('grp').agg(
        regions=('region', 'nunique'), fines=('dangerous', 'size'),
        clients=('client_id', 'nunique')
    )
    controls = 'log_price + price_missing + log_fines25 + cars + C(gender) + C(age_type)'
    # Group-specific DiD, retaining region-by-period and region-by-cohort fixed levels.
    model = fit('dangerous ~ C(region)*post + C(region)*new + C(grp):post_new + ' + controls, x)
    params = model.params
    cov = model.cov_params()
    rows = []
    for g in groups:
        term = f'C(grp)[{g}]:post_new'
        beta = params[term]
        se = np.sqrt(cov.loc[term, term])
        crit = st.t.ppf(.975, x.region.nunique() - 1)
        rows.append({
            'scheme': label, 'rationale': rationale, 'group': g,
            'regions': int(counts.loc[g, 'regions']), 'fine_rows': int(counts.loc[g, 'fines']),
            'clients': int(counts.loc[g, 'clients']), 'beta_pp': beta * 100,
            'se_pp': se * 100, 'ci_low_pp': (beta - crit * se) * 100,
            'ci_high_pp': (beta + crit * se) * 100,
            'p_value': 2 * st.t.sf(abs(beta / se), x.region.nunique() - 1),
        })
    # Wald test: equality of effects across groups.
    terms = [f'C(grp)[{g}]:post_new' for g in groups]
    if len(terms) > 1:
        R = np.zeros((len(terms) - 1, len(params)))
        for j in range(1, len(terms)):
            R[j - 1, list(params.index).index(terms[j])] = 1
            R[j - 1, list(params.index).index(terms[0])] = -1
        p_het = float(model.f_test(R).pvalue)
    else:
        p_het = np.nan
    for row in rows:
        row['heterogeneity_p'] = p_het
    return rows


def median_split(regional_value, low_name, high_name):
    med = regional_value.median()
    return regional_value.map(lambda v: high_name if v > med else low_name), med


def wild_cluster_pvalue(model, groups, term, B=4999):
    """Restricted wild cluster bootstrap-t p-value for one coefficient."""
    X = model.model.exog
    y = model.model.endog
    j = model.model.exog_names.index(term)
    X0 = np.delete(X, j, axis=1)
    b0 = np.insert(np.linalg.lstsq(X0, y, rcond=None)[0], j, 0)
    e0 = y - X @ b0
    A = np.linalg.pinv(X.T @ X)
    clusters = np.unique(groups)
    scores = np.array([X[groups == g].T @ e0[groups == g] for g in clusters])
    xx = np.array([X[groups == g].T @ X[groups == g] for g in clusters])
    rng = np.random.default_rng(260920)
    weights = rng.choice([-1.0, 1.0], (B, len(clusters)))
    delta = weights @ scores @ A
    q = np.einsum('k,gkl->gl', A[j], xx)
    projected = weights * (scores @ A[j])[None, :] - delta @ q.T
    n, k = X.shape
    se = np.sqrt((projected ** 2).sum(1) * (len(clusters) / (len(clusters) - 1)) * ((n - 1) / (n - k)))
    t_boot = delta[:, j] / se
    t_observed = float(model.tvalues[term])
    return float((1 + np.sum(np.abs(t_boot) >= abs(t_observed))) / (B + 1))


def main():
    yy = pd.read_pickle(ROOT / 'tmp' / 'clean_research_full' / 'fines.pkl')
    changes = pd.read_pickle(ROOT / 'tmp' / 'clean_research_full' / 'client_changes.pkl')
    regions = pd.Index(sorted(yy.region.unique()), name='region')
    fd = pd.Series(FD).reindex(regions)
    if fd.isna().any():
        raise ValueError(f'Missing federal district: {fd[fd.isna()].index.tolist()}')
    designs = [
        ('Федеральный округ', fd,
         'Административно и экономически связанная макрорегиональная единица; группировка задана вне исхода.'),
        ('Агломерация', pd.Series({r: 'Столичная агломерация' if r in METRO else 'Другие регионы' for r in regions}),
         'Москва–МО и Санкт-Петербург–Ленобласть имеют плотную дорожную сеть, иной общественный транспорт и интенсивность камер.'),
    ]
    pre = changes.groupby('region').agg(
        pre_liters=('liters_30_pre', 'mean'),
        pre_transactions=('transactions_30_pre', 'mean'),
        pre_all_fines=('all_rate_pre', 'mean'),
        pre_fines25=('fines25', 'mean'),
        pre_price=('price', 'median'),
        pre_multi_car=('cars', lambda x: (x > 1).mean()),
    )
    z = lambda s: (s - s.mean()) / s.std(ddof=0)
    pre['road_exposure_index'] = (z(pre.pre_liters) + z(pre.pre_all_fines)) / 2
    candidates = [
        ('Предпериодная топливная активность', 'pre_liters', 'Ниже медианы активности', 'Выше медианы активности',
         'Средние литры на клиента в апреле–мае 2026: прокси исходной автомобильной зависимости, измеренный до шока.'),
        ('Предпериодная частота штрафов', 'pre_all_fines', 'Ниже медианы контроля', 'Выше медианы контроля',
         'Все штрафы на клиента в апреле–мае 2026: прокси дорожной экспозиции и интенсивности контроля, измеренный до шока.'),
        ('Уровень цен автомобилей', 'pre_price', 'Ниже медианы цены', 'Выше медианы цены',
         'Медианная заявленная цена автомобиля в регионе: прокси платёжеспособности и различий в автопарке.'),
        ('Мультиавтомобильность', 'pre_multi_car', 'Ниже медианы владения', 'Выше медианы владения',
         'Доля клиентов с несколькими автомобилями: прокси возможности заменить автомобиль и гибкости мобильности.'),
        ('Предпериодная дорожная экспозиция', 'road_exposure_index', 'Ниже медианы экспозиции', 'Выше медианы экспозиции',
         'Среднее двух стандартизированных предпериодных показателей: литры на клиента и все штрафы на клиента. Это заранее измеренный индикатор интенсивности автомобильной мобильности и наблюдаемости на дороге.'),
    ]
    cutoffs = []
    for scheme, variable, low, high, rationale in candidates:
        group, cutoff = median_split(pre[variable], low, high)
        designs.append((scheme, group, rationale))
        cutoffs.append({'scheme': scheme, 'variable': variable, 'median_cutoff': cutoff, 'low_group': low, 'high_group': high})
    all_rows = []
    for label, grouping, rationale in designs:
        all_rows.extend(group_effects(yy, grouping, label, rationale))
    results = pd.DataFrame(all_rows)
    results['significant_5pct'] = results.p_value < .05
    # The five 2-group heterogeneity tests form one clearly labelled exploratory family.
    two_group = results.groupby('scheme').filter(lambda d: len(d) == 2).copy()
    scheme_p = two_group.groupby('scheme', as_index=False).heterogeneity_p.first()
    scheme_p['heterogeneity_fdr_bh'] = multipletests(scheme_p.heterogeneity_p, method='fdr_bh')[1]
    results = results.merge(scheme_p[['scheme', 'heterogeneity_fdr_bh']], on='scheme', how='left')
    results.to_csv(OUT / 'regional_grouping_models.csv', index=False)
    pre.reset_index().to_csv(OUT / 'regional_grouping_baselines.csv', index=False)
    pd.DataFrame(cutoffs).to_csv(OUT / 'regional_grouping_cutoffs.csv', index=False)
    # Continuous specifications use the same pre-period variables and avoid a median split.
    x = yy.merge(pre[['pre_liters', 'pre_all_fines', 'road_exposure_index']], left_on='region', right_index=True)
    controls = 'log_price + price_missing + log_fines25 + cars + C(gender) + C(age_type)'
    continuous = []
    for var, title in [
        ('pre_liters', 'Предпериодные литры'),
        ('pre_all_fines', 'Предпериодная частота штрафов'),
        ('road_exposure_index', 'Индекс дорожной экспозиции'),
    ]:
        x[var + '_z'] = z(x[var])
        m = fit('dangerous ~ C(region)*post + C(region)*new + post_new + post_new:' + var + '_z + ' + controls, x)
        term = 'post_new:' + var + '_z'
        beta, se = m.params[term], m.bse[term]
        crit = st.t.ppf(.975, x.region.nunique() - 1)
        continuous.append({
            'specification': title, 'term': term, 'beta_pp_per_sd': beta * 100,
            'ci_low_pp': (beta - crit * se) * 100, 'ci_high_pp': (beta + crit * se) * 100,
            'p_value': 2 * st.t.sf(abs(beta / se), x.region.nunique() - 1),
            'wild_cluster_p': wild_cluster_pvalue(m, x.region.to_numpy(), term),
        })
    continuous = pd.DataFrame(continuous)
    continuous['p_fdr_bh'] = multipletests(continuous.p_value, method='fdr_bh')[1]
    continuous.to_csv(OUT / 'regional_grouping_continuous.csv', index=False)
    print(results[['scheme', 'group', 'regions', 'beta_pp', 'ci_low_pp', 'ci_high_pp', 'p_value', 'heterogeneity_p']].to_string(index=False))
    print('\nCUT-OFFS')
    print(pd.DataFrame(cutoffs).to_string(index=False))
    print('\nCONTINUOUS')
    print(continuous.to_string(index=False))


if __name__ == '__main__':
    main()
