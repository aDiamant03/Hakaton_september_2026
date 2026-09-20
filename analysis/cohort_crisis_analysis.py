# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "cohort_crisis_20260919"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY_CUTOFF = pd.Timestamp("2026-05-25")
ALT_CUTOFF = pd.Timestamp("2026-06-01")
OBS_END = pd.Timestamp("2026-09-01")
PRE_START = pd.Timestamp("2026-04-01")
PRE_END = pd.Timestamp("2026-05-25")
POST_START = pd.Timestamp("2026-06-01")
POST_END = pd.Timestamp("2026-09-01")
PRE_DAYS = (PRE_END - PRE_START).days
POST_DAYS = (POST_END - POST_START).days


def first_valid(s: pd.Series):
    z = s.dropna()
    return z.iloc[0] if len(z) else np.nan


def parse_engine(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    text = s.fillna("").astype(str)
    liters = pd.to_numeric(text.str.extract(r"^\s*([0-9]+(?:[.,][0-9]+)?)")[0].str.replace(",", "."), errors="coerce")
    hp = pd.to_numeric(text.str.extract(r"\(([0-9]+(?:[.,][0-9]+)?)\s*л\.с\.")[0].str.replace(",", "."), errors="coerce")
    return liters, hp


def load_and_prepare():
    clients = pd.read_csv(ROOT / "clients_demographics.csv", sep=";", decimal=",")
    fuel = pd.read_csv(ROOT / "fuel_transaction.csv", sep=";", decimal=",")
    fines = pd.read_csv(ROOT / "fines_2026.csv", sep=";")

    clients["signup"] = pd.to_datetime(clients["subscription_creation_date"], errors="coerce")
    fuel["date"] = pd.to_datetime(fuel["order_datetime"], errors="coerce")
    fines["date"] = pd.to_datetime(fines["bill_offence_date"], errors="coerce")

    liters, hp = parse_engine(clients["engine_type"])
    clients["engine_liters"] = liters
    clients["horsepower"] = hp

    # The assignment says one row per vehicle, but 144 document IDs repeat and
    # 137 of those groups conflict on at least one field. Keep one deterministic
    # record per vehicle before aggregating to client, so duplicated vehicles do
    # not double-count the 2025 fine history.
    vehicles = clients.sort_values(["auto_document_id", "signup", "client_id"]).drop_duplicates("auto_document_id", keep="first")

    monthly_2025 = [f"{m}_2025_fines" for m in ["april", "may", "jun", "jul", "aug"]]
    vehicles["fines_2025_5m"] = vehicles[monthly_2025].sum(axis=1)

    profile = vehicles.groupby("client_id", sort=False).agg(
        signup_min=("signup", "min"),
        signup_max=("signup", "max"),
        gender=("gender", first_valid),
        age=("age_type_code", first_valid),
        region=("kladr_code", first_valid),
        autos=("auto_document_id", "nunique"),
        auto_year=("auto_year", "median"),
        price=("price", "median"),
        engine_liters=("engine_liters", "median"),
        horsepower=("horsepower", "median"),
        primary_brand=("auto_mark", first_valid),
        primary_color=("color", first_valid),
        fines_2025_5m=("fines_2025_5m", "sum"),
        fines_last_12=("fines_last_12_month", "sum"),
        brands=("auto_mark", "nunique"),
    )
    profile["signup_inconsistent"] = profile["signup_min"].ne(profile["signup_max"])
    profile["vehicle_age"] = 2026 - profile["auto_year"]
    profile["log_price"] = np.log1p(profile["price"])
    profile["log_fines_2025"] = np.log1p(profile["fines_2025_5m"])
    profile["male"] = profile["gender"].eq("M").astype(float)
    profile["age_41plus"] = profile["age"].isin(["A40", "A60", "A80"]).astype(float)

    raw_fuel_n = len(fuel)
    bad_volume = ~fuel["order_fuel_volume"].between(0.01, 120, inclusive="both")
    bad_price = ~fuel["order_fuel_price_1liter"].between(30, 150, inclusive="both")
    out_of_window_fuel = ~fuel["date"].between(PRE_START, OBS_END, inclusive="left")
    fuel_clean = fuel.loc[~bad_volume & ~bad_price & ~out_of_window_fuel].copy()
    fuel_clean["spend"] = fuel_clean["order_fuel_volume"] * fuel_clean["order_fuel_price_1liter"]

    out_of_window_fines = ~fines["date"].between(PRE_START, OBS_END, inclusive="left")
    fines_clean = fines.loc[~out_of_window_fines].copy()

    quality = [
        ("clients_rows", len(clients)),
        ("unique_clients", clients["client_id"].nunique()),
        ("unique_auto_documents", clients["auto_document_id"].nunique()),
        ("repeated_auto_document_rows", int(clients["auto_document_id"].duplicated().sum())),
        ("clients_with_inconsistent_signup_dates", int(profile["signup_inconsistent"].sum())),
        ("missing_gender_pct", clients["gender"].isna().mean() * 100),
        ("missing_price_pct", clients["price"].isna().mean() * 100),
        ("missing_color_pct", clients["color"].isna().mean() * 100),
        ("fuel_rows_raw", raw_fuel_n),
        ("fuel_bad_volume_rows", int(bad_volume.sum())),
        ("fuel_bad_price_rows", int(bad_price.sum())),
        ("fuel_rows_used_apr_aug", len(fuel_clean)),
        ("fine_rows_raw", len(fines)),
        ("fine_rows_used_apr_aug", len(fines_clean)),
        ("fuel_max_date", fuel["date"].max().isoformat()),
        ("fine_max_date", fines["date"].max().isoformat()),
    ]

    signup_map = profile["signup_min"]
    fine_with_signup = fines.join(signup_map.rename("signup"), on="client_id")
    fuel_with_signup = fuel.join(signup_map.rename("signup"), on="client_id")
    quality.extend([
        ("fines_before_signup_pct", 100 * (fine_with_signup["date"] < fine_with_signup["signup"]).mean()),
        ("fuel_before_signup_pct", 100 * (fuel_with_signup["date"] < fuel_with_signup["signup"]).mean()),
    ])
    return clients, fuel, fines, fuel_clean, fines_clean, profile, pd.DataFrame(quality, columns=["check", "value"])


def aggregate_periods(fuel: pd.DataFrame, fines: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    idx = profile.index
    result = profile.copy()
    for label, start, end, days in [
        ("pre", PRE_START, PRE_END, PRE_DAYS),
        ("post", POST_START, POST_END, POST_DAYS),
    ]:
        x = fuel[fuel["date"].between(start, end, inclusive="left")]
        a = x.groupby("client_id").agg(
            fuel_tx=("order_id", "size"),
            fuel_liters=("order_fuel_volume", "sum"),
            fuel_spend=("spend", "sum"),
            active_days=("date", lambda z: z.dt.normalize().nunique()),
        ).reindex(idx).fillna(0)
        a[f"fuel_tx_rate_{label}"] = a["fuel_tx"] * 30 / days
        a[f"fuel_liters_rate_{label}"] = a["fuel_liters"] * 30 / days
        a[f"fuel_price_vw_{label}"] = np.where(a["fuel_liters"] > 0, a["fuel_spend"] / a["fuel_liters"], np.nan)
        for col in [f"fuel_tx_rate_{label}", f"fuel_liters_rate_{label}", f"fuel_price_vw_{label}"]:
            result[col] = a[col]

        y = fines[fines["date"].between(start, end, inclusive="left")]
        counts = y.groupby("client_id").size().reindex(idx).fillna(0)
        result[f"fines_count_{label}"] = counts
        result[f"fines_rate_{label}"] = counts * 30 / days

    for stem in ["fuel_tx_rate", "fuel_liters_rate", "fuel_price_vw", "fines_rate"]:
        result[f"delta_{stem}"] = result[f"{stem}_post"] - result[f"{stem}_pre"]
    result["fuel_active_both"] = result[["fuel_price_vw_pre", "fuel_price_vw_post"]].notna().all(axis=1)
    return result


def standardized_difference(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (b.mean() - a.mean()) / pooled if pooled else np.nan


def cohort_frame(panel: pd.DataFrame, cutoff: pd.Timestamp, window: int, signup_col: str = "signup_min") -> pd.DataFrame:
    lo = cutoff - pd.Timedelta(days=window)
    hi = cutoff + pd.Timedelta(days=window)
    d = panel[panel[signup_col].between(lo, hi, inclusive="left")].copy()
    d["post_signup"] = d[signup_col].ge(cutoff).astype(int)
    d["running_day"] = (d[signup_col] - cutoff).dt.days
    return d


def balance_table(d: pd.DataFrame, cutoff_label: str) -> pd.DataFrame:
    metrics = {
        "male_share": "male",
        "age_41plus_share": "age_41plus",
        "vehicles_per_client": "autos",
        "vehicle_age_years": "vehicle_age",
        "vehicle_price_rub": "price",
        "engine_liters": "engine_liters",
        "horsepower": "horsepower",
        "fines_2025_apr_aug": "fines_2025_5m",
        "any_fine_2025_apr_aug": "fines_2025_5m",
        "fuel_tx_per_30d_pre": "fuel_tx_rate_pre",
        "fuel_liters_per_30d_pre": "fuel_liters_rate_pre",
        "fuel_price_pre": "fuel_price_vw_pre",
        "fuel_tx_per_30d_post": "fuel_tx_rate_post",
        "fuel_liters_per_30d_post": "fuel_liters_rate_post",
        "fuel_price_post": "fuel_price_vw_post",
        "fines_per_30d_pre_2026": "fines_rate_pre",
        "fines_per_30d_post_2026": "fines_rate_post",
    }
    rows = []
    for label, col in metrics.items():
        x = d[col].copy()
        if label.startswith("any_fine"):
            x = x.gt(0).astype(float)
        pre = x[d["post_signup"].eq(0)]
        post = x[d["post_signup"].eq(1)]
        test = stats.ttest_ind(post.dropna(), pre.dropna(), equal_var=False)
        rows.append({
            "cutoff": cutoff_label,
            "metric": label,
            "pre_n": pre.notna().sum(),
            "post_n": post.notna().sum(),
            "pre_mean": pre.mean(),
            "post_mean": post.mean(),
            "difference_post_minus_pre": post.mean() - pre.mean(),
            "standardized_difference": standardized_difference(pre, post),
            "p_value_welch": test.pvalue,
        })
    tab = pd.DataFrame(rows)
    tab["p_value_fdr"] = multipletests(tab["p_value_welch"].fillna(1), method="fdr_bh")[1]
    return tab


def categorical_tests(d: pd.DataFrame, cutoff_label: str) -> pd.DataFrame:
    rows = []
    for col in ["gender", "age", "region", "primary_brand", "primary_color"]:
        tab = pd.crosstab(d[col].fillna("Missing"), d["post_signup"])
        chi2, p, dof, _ = stats.chi2_contingency(tab)
        n = tab.to_numpy().sum()
        denom = max(1, min(tab.shape) - 1)
        v = math.sqrt(chi2 / (n * denom))
        rows.append({"cutoff": cutoff_label, "variable": col, "chi2": chi2, "dof": dof, "p_value": p, "cramers_v": v, "n": n})
    out = pd.DataFrame(rows)
    out["p_value_fdr"] = multipletests(out["p_value"], method="fdr_bh")[1]
    return out


BASE_CONTROLS = "C(gender) + C(age) + C(region) + vehicle_age + log_price + autos + engine_liters + horsepower"


def extract_result(fit, term: str, outcome: str, model: str, transform: str = "level") -> dict:
    beta = float(fit.params[term])
    se = float(fit.bse[term])
    p = float(fit.pvalues[term])
    lo, hi = map(float, fit.conf_int().loc[term])
    if transform == "exp":
        beta, lo, hi = np.exp([beta, lo, hi])
    return {
        "outcome": outcome,
        "model": model,
        "term": term,
        "estimate": beta,
        "std_error_log_or_level": se,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p,
        "scale": "IRR" if transform == "exp" else "absolute difference",
        "n": int(fit.nobs),
    }


def fit_models(d: pd.DataFrame, spec_label: str, adjust_signup_trend: bool = False) -> pd.DataFrame:
    rows = []
    trend = " + running_day + post_signup:running_day" if adjust_signup_trend else ""
    model_suffix = "with separate signup-date slopes" if adjust_signup_trend else "demographic/vehicle controls"
    # Robust Poisson pseudo-ML targets the conditional mean of a count and uses
    # sandwich standard errors, so inference does not require equidispersion.
    # Since the outcome is in 2025, this is a selection signal, not an effect of
    # the 2026 crisis.
    formula_ppml = f"fines_2025_5m ~ post_signup + {BASE_CONTROLS}{trend}"
    ppml = smf.glm(formula_ppml, data=d, family=sm.families.Poisson()).fit(cov_type="HC3")
    rows.append(extract_result(ppml, "post_signup", "2025 fines (Apr-Aug count)", f"Robust Poisson: {spec_label}; {model_suffix}", "exp"))

    any_data = d.assign(any_fine_2025=d["fines_2025_5m"].gt(0).astype(int))
    logit = smf.logit(f"any_fine_2025 ~ post_signup + {BASE_CONTROLS}{trend}", data=any_data).fit(disp=False, cov_type="HC0", maxiter=200)
    rows.append(extract_result(logit, "post_signup", "Any 2025 fine (Apr-Aug)", f"Logit: {spec_label}; {model_suffix}", "exp"))

    for outcome, label in [
        ("delta_fuel_tx_rate", "Change in fuel transactions per 30d"),
        ("delta_fuel_liters_rate", "Change in liters per 30d"),
        ("delta_fuel_price_vw", "Change in volume-weighted rub/l"),
        ("delta_fines_rate", "Change in 2026 fines per 30d"),
    ]:
        use = d.copy()
        if outcome == "delta_fuel_price_vw":
            use = use[use["fuel_active_both"]]
        fit = smf.ols(f"{outcome} ~ post_signup + {BASE_CONTROLS}{trend}", data=use).fit(cov_type="HC3")
        rows.append(extract_result(fit, "post_signup", label, f"OLS change: {spec_label}; {model_suffix}"))

    # Count-model DiD with all clients and explicit exposure offsets.
    long = pd.concat([
        d.assign(period_post=0, count=d["fines_count_pre"], exposure=PRE_DAYS),
        d.assign(period_post=1, count=d["fines_count_post"], exposure=POST_DAYS),
    ], ignore_index=False).reset_index(names="client_id")
    long["offset"] = np.log(long["exposure"] / 30)
    gee = GEE.from_formula(
        f"count ~ period_post * post_signup + {BASE_CONTROLS}",
        groups="client_id",
        data=long,
        family=sm.families.Poisson(),
        cov_struct=Exchangeable(),
        offset=long["offset"],
    ).fit()
    rows.append(extract_result(gee, "period_post:post_signup", "2026 fine-rate DiD", f"Poisson GEE: {spec_label}", "exp"))
    return pd.DataFrame(rows)


def fine_mix_models(fines: pd.DataFrame, d: pd.DataFrame, spec_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["post_signup", "gender", "age", "region", "vehicle_age", "log_price", "autos", "engine_liters", "horsepower"]
    x = fines[fines["date"].between(POST_START, POST_END, inclusive="left")].join(d[cols], on="client_id").dropna(subset=["post_signup"])
    x["seatbelt_or_phone"] = x["offence_short_statement"].str.contains("ремень|телефона", case=False, regex=True).astype(int)
    x["speed_20_40"] = x["offence_short_statement"].eq("Превышение скорости на 20-40 км/ч").astype(int)
    rows = []
    for outcome, label in [("seatbelt_or_phone", "Seatbelt or phone share among 2026 fines"), ("speed_20_40", "20-40 km/h speeding share among 2026 fines")]:
        fit = GEE.from_formula(
            f"{outcome} ~ post_signup + {BASE_CONTROLS}",
            groups="client_id", data=x, family=sm.families.Binomial(), cov_struct=Exchangeable()
        ).fit()
        rows.append(extract_result(fit, "post_signup", label, f"Binomial GEE fine mix: {spec_label}", "exp"))
    shares = x.groupby("post_signup").agg(
        fines=("bill_id", "size"),
        clients=("client_id", "nunique"),
        seatbelt_or_phone_share=("seatbelt_or_phone", "mean"),
        speed_20_40_share=("speed_20_40", "mean"),
    ).reset_index()
    return pd.DataFrame(rows), shares


def monthly_dynamics(fuel: pd.DataFrame, fines: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    cohort = d[["post_signup"]]
    months = pd.period_range("2026-04", "2026-08", freq="M")
    idx = pd.MultiIndex.from_product([cohort.index, months], names=["client_id", "month"])
    panel = pd.DataFrame(index=idx).reset_index().join(cohort, on="client_id")

    fx = fuel.copy()
    fx["month"] = fx["date"].dt.to_period("M")
    fa = fx.groupby(["client_id", "month"]).agg(fuel_tx=("order_id", "size"), fuel_liters=("order_fuel_volume", "sum"), spend=("spend", "sum"))
    fa["fuel_price"] = fa["spend"] / fa["fuel_liters"]
    panel = panel.join(fa[["fuel_tx", "fuel_liters", "fuel_price"]], on=["client_id", "month"])
    panel[["fuel_tx", "fuel_liters"]] = panel[["fuel_tx", "fuel_liters"]].fillna(0)

    fn = fines.copy()
    fn["month"] = fn["date"].dt.to_period("M")
    fc = fn.groupby(["client_id", "month"]).size().rename("fines")
    panel = panel.join(fc, on=["client_id", "month"])
    panel["fines"] = panel["fines"].fillna(0)

    days = panel["month"].dt.days_in_month
    panel["fuel_tx_30d"] = panel["fuel_tx"] * 30 / days
    panel["fuel_liters_30d"] = panel["fuel_liters"] * 30 / days
    panel["fines_30d"] = panel["fines"] * 30 / days
    out = panel.groupby(["month", "post_signup"]).agg(
        clients=("client_id", "nunique"),
        fuel_tx_30d=("fuel_tx_30d", "mean"),
        fuel_liters_30d=("fuel_liters_30d", "mean"),
        fuel_price=("fuel_price", "mean"),
        fines_30d=("fines_30d", "mean"),
    ).reset_index()
    out["month"] = out["month"].astype(str)
    return out


def make_figures(monthly: pd.DataFrame, balance: pd.DataFrame):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = {0: "#666666", 1: "#FFCC00"}
    labels = {0: "Регистрация до 25 мая", 1: "Регистрация с 25 мая"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for post in [0, 1]:
        z = monthly[monthly["post_signup"].eq(post)]
        x = np.arange(len(z))
        for ax, col, title in [
            (axes[0, 0], "fuel_tx_30d", "Транзакции топлива на 30 дней"),
            (axes[0, 1], "fuel_liters_30d", "Литры на 30 дней"),
            (axes[1, 0], "fuel_price", "Средняя цена, руб./л (активные клиенты)"),
            (axes[1, 1], "fines_30d", "Штрафы на 30 дней"),
        ]:
            ax.plot(x, z[col], marker="o", linewidth=2.2, color=colors[post], label=labels[post])
            ax.set_xticks(x, z["month"])
            ax.set_title(title, loc="left", fontweight="bold")
            ax.axvline(1.5, color="#CC3333", linestyle="--", alpha=.7)
            ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Локальные когорты ±90 дней вокруг 25 мая 2026", fontsize=15, fontweight="bold")
    fig.savefig(OUT / "cohort_monthly_dynamics.png", dpi=180)
    plt.close(fig)

    show = balance[balance["metric"].isin([
        "male_share", "age_41plus_share", "vehicle_age_years", "vehicle_price_rub",
        "fines_2025_apr_aug", "fuel_tx_per_30d_pre", "fuel_liters_per_30d_pre"
    ])].copy().iloc[::-1]
    labels_ru = {
        "male_share": "Мужчины",
        "age_41plus_share": "Возраст 41+",
        "vehicle_age_years": "Возраст автомобиля",
        "vehicle_price_rub": "Цена автомобиля",
        "fines_2025_apr_aug": "Штрафы апрель–август 2025",
        "fuel_tx_per_30d_pre": "Транзакции топлива до кризиса",
        "fuel_liters_per_30d_pre": "Литры до кризиса",
    }
    y = np.arange(len(show))
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    ax.barh(y, show["standardized_difference"], color=np.where(show["p_value_fdr"] < .05, "#FFCC00", "#999999"))
    ax.axvline(0, color="black", linewidth=.8)
    ax.axvline(.1, color="#CC3333", linestyle="--", alpha=.5)
    ax.axvline(-.1, color="#CC3333", linestyle="--", alpha=.5)
    ax.set_yticks(y, [labels_ru[x] for x in show["metric"]])
    ax.set_xlabel("Стандартизированная разница: после − до")
    ax.set_title("Баланс локальных когорт вокруг 25 мая", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=.2)
    fig.savefig(OUT / "cohort_balance_smd.png", dpi=180)
    plt.close(fig)


def build_report(primary: pd.DataFrame, balance: pd.DataFrame, robustness: pd.DataFrame, quality: pd.DataFrame, monthly: pd.DataFrame, fine_mix: pd.DataFrame, fine_shares: pd.DataFrame, categorical: pd.DataFrame):
    def b(metric):
        return balance.loc[balance.metric.eq(metric)].iloc[0]
    def m(outcome):
        return primary.loc[primary.outcome.eq(outcome)].iloc[0]
    fmtp = lambda x: "<0.001" if x < .001 else f"{x:.3f}"
    trend_robust = robustness[(robustness["spec"].eq("trend_sensitivity"))]
    def tr(outcome):
        return trend_robust.loc[trend_robust.outcome.eq(outcome)].iloc[0]
    def fm(outcome):
        return fine_mix.loc[fine_mix.outcome.eq(outcome)].iloc[0]
    fs0 = fine_shares.loc[fine_shares.post_signup.eq(0)].iloc[0]
    fs1 = fine_shares.loc[fine_shares.post_signup.eq(1)].iloc[0]
    report = f"""# Когорты регистрации вокруг топливного кризиса 2026

## Главный результат

Основная выборка — клиенты, зарегистрировавшиеся в сервисе штрафов в окне ±90 дней вокруг 25 мая 2026. До cutoff: {int(b('fines_2025_apr_aug').pre_n):,} клиентов; после: {int(b('fines_2025_apr_aug').post_n):,}. В отличие от наивного сравнения всех старых и новых клиентов, локальное окно уменьшает смешение с многолетним стажем использования сервиса.

Когорты почти не различаются по полу и возрасту: доля мужчин {b('male_share').pre_mean:.1%} против {b('male_share').post_mean:.1%}, доля 41+ {b('age_41plus_share').pre_mean:.1%} против {b('age_41plus_share').post_mean:.1%}. Но у зарегистрировавшихся после начала кризиса уже в апреле–августе 2025 было больше штрафов: {b('fines_2025_apr_aug').pre_mean:.2f} против {b('fines_2025_apr_aug').post_mean:.2f} на клиента. Robust Poisson с демографическими и автомобильными контролями дает IRR {m('2025 fines (Apr-Aug count)').estimate:.2f} (95% ДИ {m('2025 fines (Apr-Aug count)').ci_low:.2f}–{m('2025 fines (Apr-Aug count)').ci_high:.2f}, p {fmtp(m('2025 fines (Apr-Aug count)').p_value)}).

После кризиса новые подписчики не просто чаще имели прошлые штрафы. Их топливная активность менялась иначе. Скорректированная разница когорт в изменении частоты заправок составила {m('Change in fuel transactions per 30d').estimate:+.2f} транзакции на 30 дней (95% ДИ {m('Change in fuel transactions per 30d').ci_low:+.2f}…{m('Change in fuel transactions per 30d').ci_high:+.2f}, p {fmtp(m('Change in fuel transactions per 30d').p_value)}), а в объеме — {m('Change in liters per 30d').estimate:+.1f} л/30 дней (95% ДИ {m('Change in liters per 30d').ci_low:+.1f}…{m('Change in liters per 30d').ci_high:+.1f}, p {fmtp(m('Change in liters per 30d').p_value)}). Для цены эффект равен {m('Change in volume-weighted rub/l').estimate:+.2f} руб./л среди активных в обоих периодах (p {fmtp(m('Change in volume-weighted rub/l').p_value)}).

Ключевая проверка устойчивости меняет вывод: если разрешить показателям плавно меняться с календарной датой регистрации по обе стороны cutoff, разрыв по истории штрафов становится IRR {tr('2025 fines (Apr-Aug count)').estimate:.2f} (p {fmtp(tr('2025 fines (Apr-Aug count)').p_value)}), по частоте топлива {tr('Change in fuel transactions per 30d').estimate:+.2f} (p {fmtp(tr('Change in fuel transactions per 30d').p_value)}), по литрам {tr('Change in liters per 30d').estimate:+.1f} (p {fmtp(tr('Change in liters per 30d').p_value)}). Значит, описательная разница когорт реальна, но нет устойчивого свидетельства резкого перелома именно в конце мая: она совместима с плавным трендом состава новых регистраций.

Для штрафов 2026 взаимодействие «после кризиса × новая когорта» в Poisson GEE дает IRR {m('2026 fine-rate DiD').estimate:.2f} (95% ДИ {m('2026 fine-rate DiD').ci_low:.2f}–{m('2026 fine-rate DiD').ci_high:.2f}, p {fmtp(m('2026 fine-rate DiD').p_value)}). Это сильная ассоциация, но не чистый причинный эффект кризиса: дата подписки может менять наблюдаемость штрафов и отражать одновременный онбординг в экосистему.

Состав штрафов тоже отличается. Среди штрафов июня–августа нарушения «ремень или телефон» занимают {fs0.seatbelt_or_phone_share:.1%} у старой локальной когорты и {fs1.seatbelt_or_phone_share:.1%} у новой; скорректированный кластерный binomial GEE дает OR {fm('Seatbelt or phone share among 2026 fines').estimate:.2f} (95% ДИ {fm('Seatbelt or phone share among 2026 fines').ci_low:.2f}–{fm('Seatbelt or phone share among 2026 fines').ci_high:.2f}, p {fmtp(fm('Seatbelt or phone share among 2026 fines').p_value)}). Доля обычных превышений 20–40 км/ч, наоборот, ниже: {fs0.speed_20_40_share:.1%} против {fs1.speed_20_40_share:.1%}, OR {fm('20-40 km/h speeding share among 2026 fines').estimate:.2f} (p {fmtp(fm('20-40 km/h speeding share among 2026 fines').p_value)}). Это условный состав уже полученных штрафов, а не вероятность нарушения для всех водителей.

## Проверяемая гипотеза

Проверяемая гипотеза состоит из двух конкурирующих механизмов. H1: топливный шок усилил самоотбор интенсивных водителей в сервис штрафов. H0/альтернатива: видимый разрыв — плавный тренд онбординга и наблюдаемости, а не кризисный скачок. На текущих данных H1 поддерживается простым когортным сравнением, но не выдерживает контроль календарного тренда; для штрафов 2026 остается сильный скачок, однако он особенно уязвим к изменению наблюдаемости. Предсказание для новых данных: если H1 верна, то индивидуальный скачок цены при сохранении литража должен повышать вероятность регистрации в следующие 14 дней даже после учета даты, региона и маркетинговых контактов; затем он должен предсказывать штрафы на 1 000 литров в следующие 60–90 дней.

## Модель и устойчивость

- Основной дизайн: локальные когорты ±90 дней вокруг 25 мая; контроли пола, возраста, региона, возраста и цены автомобиля, числа авто, объема двигателя и мощности. Это оценка ассоциации, а не regression discontinuity, потому что cutoff не назначает регистрацию.
- История штрафов 2025: robust Poisson PML со sandwich-ошибками для счетной переменной; «любой штраф» отдельно проверен логитом.
- Изменения топлива: разность нормированных на 30 дней показателей между 1 апреля–24 мая и 1 июня–31 августа; HC3-ошибки. Цена считается объемно-взвешенно и только у активных в обоих периодах.
- Штрафы 2026: Poisson GEE с клиентскими кластерами и offset по длине периода; нули сохранены.
- Проверки: cutoff 1 июня, окна ±60/±120 дней, максимальная вместо минимальной даты подписки, а также отдельные линейные тренды даты регистрации по обе стороны cutoff. Полная таблица находится в `robustness.csv`. Именно трендовая проверка показывает, что топливные различия не являются устойчивым скачком на cutoff.

## Policy implication для Т-Банка

Не использовать признак «зарегистрировался после 25 мая» как самостоятельный causal/risk feature: он смешивает календарный тренд и наблюдаемость. Вместо этого запустить рандомизированный триггер «топливо → защита от штрафов» для клиентов, у которых за 14 дней одновременно (1) выросла персональная цена литра и (2) не снизились литры или частота заправок. Половине случайно показывать пакет: подключение уведомлений/автоплатежа штрафов, повышенный кэшбэк на топливо и короткий safety-nudge. Основные метрики: конверсия, просрочки/скидка 50%, удержание топлива, штрафы на 1 000 литров; guardrails — маржинальность и отсутствие роста рискованного вождения. Эксперимент отделит полезный таргетинг от простой корреляции.

## Ограничения и альтернативные объяснения

- Дата регистрации не рандомна. Эффект может быть вызван маркетинговой кампанией, сезонностью, сменой автомобиля, ростом пробега, работой в такси/доставке или общим онбордингом в продукты Т-Банка.
- В данных нет пробега, типа топлива, АЗС, профессии и экспозиции в километрах. Литры — лишь прокси мобильности; цена смешивает регион, марку топлива и выбор АЗС.
- {int(float(quality.loc[quality.check.eq('clients_with_inconsistent_signup_dates'),'value'].iloc[0])):,} клиентов имеют несколько дат подписки по автомобилям. Основной расчет использует минимальную, а максимальная проверена отдельно.
- {float(quality.loc[quality.check.eq('fines_before_signup_pct'),'value'].iloc[0]):.2f}% штрафов 2026 и {float(quality.loc[quality.check.eq('fuel_before_signup_pct'),'value'].iloc[0]):.2f}% топливных операций датированы раньше первой подписки. Это может быть ретроспективная подгрузка или независимая продуктовая история, поэтому штрафы 2026 особенно уязвимы к смещению наблюдаемости.
- Исходные файлы доходят до сентября, хотя описание обещает апрель–август; расчеты ограничены 31 августа. Отрицательные/аномальные литры и экстремальные цены исключены по заранее заданным физически правдоподобным границам; чувствительность к очистке следует дополнительно проверить при наличии справочника видов топлива.
- Локальный дизайн поддерживает сравнимость около cutoff, но не доказывает параллельные тренды и не переносится автоматически на давних клиентов.

## Файлы результатов

- `cohort_balance.csv` — сырые средние, SMD, Welch p-value и FDR.
- `model_results_primary.csv` — основные модели.
- `robustness.csv` — альтернативные cutoff/окна/определение даты подписки.
- `monthly_dynamics.csv` — данные для графиков.
- `fine_mix_models.csv` и `fine_mix_shares.csv` — различия типов штрафов с клиентской кластеризацией.
- `categorical_tests.csv` — χ² и Cramér's V для пола, возраста, региона, марки и цвета.
- `cohort_monthly_dynamics.png` и `cohort_balance_smd.png` — визуализации.
- `data_quality.csv` — проверки качества и правила отбора.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")


def main():
    clients, fuel_raw, fines_raw, fuel, fines, profile, quality = load_and_prepare()
    panel = aggregate_periods(fuel, fines, profile)

    primary_data = cohort_frame(panel, PRIMARY_CUTOFF, 90, "signup_min")
    alt_data = cohort_frame(panel, ALT_CUTOFF, 90, "signup_min")
    primary_balance = balance_table(primary_data, "2026-05-25 ±90d")
    alt_balance = balance_table(alt_data, "2026-06-01 ±90d")
    balance = pd.concat([primary_balance, alt_balance], ignore_index=True)
    categorical = pd.concat([
        categorical_tests(primary_data, "2026-05-25 ±90d"),
        categorical_tests(alt_data, "2026-06-01 ±90d"),
    ], ignore_index=True)

    primary_models = fit_models(primary_data, "cutoff 2026-05-25, ±90d, earliest signup", adjust_signup_trend=False)
    robustness_parts = [primary_models.assign(spec="primary")]
    robustness_parts.append(fit_models(primary_data, "cutoff 2026-05-25, ±90d, earliest signup", adjust_signup_trend=True).assign(spec="trend_sensitivity"))
    for cutoff, window, signup_col, label in [
        (PRIMARY_CUTOFF, 60, "signup_min", "2026-05-25 ±60d earliest"),
        (PRIMARY_CUTOFF, 120, "signup_min", "2026-05-25 ±120d earliest"),
        (ALT_CUTOFF, 90, "signup_min", "2026-06-01 ±90d earliest"),
        (PRIMARY_CUTOFF, 90, "signup_max", "2026-05-25 ±90d latest"),
    ]:
        alt = cohort_frame(panel, cutoff, window, signup_col)
        robustness_parts.append(fit_models(alt, label, adjust_signup_trend=False).assign(spec=label))
        if cutoff == ALT_CUTOFF and window == 90 and signup_col == "signup_min":
            robustness_parts.append(fit_models(alt, label, adjust_signup_trend=True).assign(spec=f"{label} trend sensitivity"))
    robustness = pd.concat(robustness_parts, ignore_index=True)
    fine_mix, fine_shares = fine_mix_models(fines, primary_data, "cutoff 2026-05-25, ±90d, earliest signup")

    monthly = monthly_dynamics(fuel, fines, primary_data)
    make_figures(monthly, primary_balance)

    primary_balance.to_csv(OUT / "cohort_balance.csv", index=False)
    alt_balance.to_csv(OUT / "cohort_balance_cutoff_june1.csv", index=False)
    primary_models.to_csv(OUT / "model_results_primary.csv", index=False)
    robustness.to_csv(OUT / "robustness.csv", index=False)
    categorical.to_csv(OUT / "categorical_tests.csv", index=False)
    fine_mix.to_csv(OUT / "fine_mix_models.csv", index=False)
    fine_shares.to_csv(OUT / "fine_mix_shares.csv", index=False)
    monthly.to_csv(OUT / "monthly_dynamics.csv", index=False)
    quality.to_csv(OUT / "data_quality.csv", index=False)
    primary_data.reset_index().to_csv(OUT / "analysis_client_level.csv", index=False)
    build_report(primary_models, primary_balance, robustness, quality, monthly, fine_mix, fine_shares, categorical)

    summary = {
        "output_dir": str(OUT),
        "primary_cohort_n_pre": int((primary_data.post_signup == 0).sum()),
        "primary_cohort_n_post": int((primary_data.post_signup == 1).sum()),
        "primary_results": primary_models.to_dict(orient="records"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
