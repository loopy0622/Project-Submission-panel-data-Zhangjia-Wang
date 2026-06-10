import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats


PROJECT = Path(__file__).resolve().parent.parent
PANEL_FILE = PROJECT / "data/processed/market_day_panel.csv"
SEED = 20260608


def tidy_result(name, result, variables):
    """Convert selected model coefficients into a consistent long-format table."""
    rows = []
    for variable in variables:
        rows.append({
            "model": name,
            "variable": variable,
            "coefficient": result.params.get(variable, np.nan),
            "std_error": result.bse.get(variable, np.nan),
            "p_value": result.pvalues.get(variable, np.nan),
            "nobs": int(result.nobs),
        })
    return rows


def clustered_ols(y, x, groups):
    """Fit OLS while allowing regression errors to be correlated within groups."""
    return sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})


def fixed_effects(df, yvar, xvars):
    """Estimate contract fixed effects using the within transformation.

    Demeaning each variable by its contract-specific mean removes all time-invariant
    contract characteristics. Standard errors remain clustered at the event level.
    """
    transformed = df[[yvar, *xvars]].sub(df.groupby("ticker")[[yvar, *xvars]].transform("mean"))
    return clustered_ols(transformed[yvar], transformed[xvars], df["event_ticker"])


def random_effects(df, yvar, xvars):
    """Estimate a feasible random-effects model by contract-level quasi-demeaning.

    Unlike fixed effects, random effects uses both within- and between-contract
    variation and requires the contract effect to be uncorrelated with regressors.
    """
    x = sm.add_constant(df[xvars], has_constant="add")
    pooled = sm.OLS(df[yvar], x).fit()
    fe = fixed_effects(df, yvar, xvars)
    n, groups, k = len(df), df["ticker"].nunique(), len(xvars)
    sigma_e2 = np.sum(np.square(fe.resid)) / max(1, n - groups - k)
    means = df.groupby("ticker")[[yvar, *xvars]].mean()
    between = sm.OLS(means[yvar], sm.add_constant(means[xvars], has_constant="add")).fit()
    tbar = df.groupby("ticker").size().mean()
    sigma_u2 = max(0.0, np.var(between.resid, ddof=len(xvars) + 1) - sigma_e2 / tbar)
    counts = df.groupby("ticker").size()
    theta = 1 - np.sqrt(sigma_e2 / (sigma_e2 + counts * sigma_u2))
    group_means = df.groupby("ticker")[[yvar, *xvars]].transform("mean")
    th = df["ticker"].map(theta).to_numpy()
    y_star = df[yvar] - th * group_means[yvar]
    x_star = pd.DataFrame({"const": 1 - th}, index=df.index)
    for variable in xvars:
        x_star[variable] = df[variable] - th * group_means[variable]
    result = clustered_ols(y_star, x_star, df["event_ticker"])
    result.sigma_e2 = sigma_e2
    result.sigma_u2 = sigma_u2
    return result


def hausman(fe, re, xvars):
    """Test whether FE and RE slope estimates differ systematically."""
    diff = fe.params[xvars].to_numpy() - re.params[xvars].to_numpy()
    covariance = fe.cov_params().loc[xvars, xvars].to_numpy() - re.cov_params().loc[xvars, xvars].to_numpy()
    statistic = float(diff @ np.linalg.pinv(covariance) @ diff)
    return statistic, float(stats.chi2.sf(max(0, statistic), len(xvars)))


def wald_test(result, restrictions, targets):
    """Conduct a joint Wald test that named coefficients equal target values."""
    names = list(result.params.index)
    r_matrix = np.zeros((len(restrictions), len(names)))
    for row, variable in enumerate(restrictions):
        r_matrix[row, names.index(variable)] = 1.0
    difference = r_matrix @ result.params.to_numpy() - np.asarray(targets)
    covariance = r_matrix @ result.cov_params().to_numpy() @ r_matrix.T
    statistic = float(difference @ np.linalg.pinv(covariance) @ difference)
    return statistic, float(stats.chi2.sf(max(0, statistic), len(restrictions)))


def horizon_logit(df, horizons):
    """Estimate calibration regressions using one observation per contract and horizon.

    At each requested horizon, the closest available trading day on or before that
    distance from settlement is selected. This prevents frequently traded contracts
    from contributing more observations within a fixed-horizon regression.
    """
    rows = []
    for horizon in horizons:
        candidates = df[df["time_to_close_days"] >= horizon].copy()
        candidates["distance"] = candidates["time_to_close_days"] - horizon
        sample = candidates.sort_values("distance").groupby("ticker", as_index=False).first()
        sample = sample[(sample.last_price > 0) & (sample.last_price < 1)].copy()
        sample["logit_price"] = np.log(sample.last_price / (1 - sample.last_price))
        result = sm.GLM(
            sample["outcome"], sm.add_constant(sample[["logit_price"]]),
            family=sm.families.Binomial(),
        ).fit(cov_type="cluster", cov_kwds={"groups": sample["event_ticker"]})
        beta_se = result.bse["logit_price"]
        joint_stat, joint_p = wald_test(
            result, ["const", "logit_price"], [0.0, 1.0]
        )
        rows.append({
            "horizon_days": horizon,
            "contracts": len(sample),
            "alpha": result.params["const"],
            "alpha_se": result.bse["const"],
            "alpha_p_equal_0": result.pvalues["const"],
            "beta": result.params["logit_price"],
            "beta_se": beta_se,
            "beta_p_equal_1": 2 * stats.norm.sf(abs((result.params["logit_price"] - 1) / beta_se)),
            "joint_wald_alpha0_beta1": joint_stat,
            "joint_p_alpha0_beta1": joint_p,
        })
    return pd.DataFrame(rows)


def simulation(repetitions=250, n_units=120, t_periods=8):
    """Illustrate bias when regressors correlate with an unobserved unit effect.

    The data-generating process favors fixed effects because x is deliberately
    correlated with alpha. Pooled OLS and random effects violate their orthogonality
    assumptions in this design.
    """
    rng = np.random.default_rng(SEED)
    estimates = {"Pooled OLS": [], "Fixed Effects": [], "Random Effects": []}
    beta = 0.35
    for _ in range(repetitions):
        alpha = rng.normal(size=n_units)
        rows = []
        for i in range(n_units):
            x = 0.65 * alpha[i] + rng.normal(size=t_periods)
            y = alpha[i] + beta * x + rng.normal(size=t_periods)
            rows.extend((i, j, y[j], x[j]) for j in range(t_periods))
        d = pd.DataFrame(rows, columns=["ticker", "time", "y", "x"])
        pooled = sm.OLS(d.y, sm.add_constant(d[["x"]])).fit()
        demeaned = d[["y", "x"]].sub(d.groupby("ticker")[["y", "x"]].transform("mean"))
        fe = sm.OLS(demeaned.y, demeaned[["x"]]).fit()
        d["event_ticker"] = d["ticker"]
        re = random_effects(d, "y", ["x"])
        estimates["Pooled OLS"].append(pooled.params["x"])
        estimates["Fixed Effects"].append(fe.params["x"])
        estimates["Random Effects"].append(re.params["x"])
    rows = []
    for method, values in estimates.items():
        values = np.asarray(values)
        rows.append({
            "method": method, "true_beta": beta, "mean_estimate": values.mean(),
            "bias": values.mean() - beta, "rmse": np.sqrt(np.mean((values - beta) ** 2)),
            "sd": values.std(ddof=1), "repetitions": repetitions,
        })
    return pd.DataFrame(rows)


def make_figures(df):
    """Display the forecast-error trend and pooled daily-price calibration figures."""
    sns.set_theme(style="whitegrid", context="paper")

    # A reproducible subsample keeps the LOWESS figure computationally manageable.
    sampled = df.sample(min(60000, len(df)), random_state=SEED)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.regplot(data=sampled, x="time_to_close_days", y="absolute_forecast_error",
                scatter_kws={"alpha": 0.08, "s": 5}, lowess=True, line_kws={"color": "#b2182b"}, ax=ax)
    ax.set(xlabel="Days to close", ylabel="Absolute forecast error",
           title="Forecast error declines as contracts approach resolution")
    fig.tight_layout()

    # Compare average prices with realized settlement frequencies in ten price bins.
    bins = pd.cut(df.last_price, np.linspace(0, 1, 11), include_lowest=True)
    calibration = df.assign(price_bin=bins).groupby("price_bin", observed=True).agg(
        mean_price=("last_price", "mean"), outcome_rate=("outcome", "mean"), observations=("outcome", "size")
    ).reset_index()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(calibration.mean_price, calibration.outcome_rate, "o-", color="#2166ac", label="Observed")
    ax.set(xlabel="Mean daily last price", ylabel="Settlement frequency", xlim=(0, 1), ylim=(0, 1),
           title="Daily-price calibration")
    ax.legend()
    fig.tight_layout()
    plt.show()
    return calibration


def display_table(title, table):
    """Print a labeled table without writing files to disk."""
    print(f"\n{title}\n{'=' * len(title)}")
    print(table.to_string(index=False))


def main():
    """Run the full empirical workflow and display all results."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    df = pd.read_csv(PANEL_FILE)
    df["event_ticker"] = df.event_ticker.fillna(df.ticker)
    df["time_to_close_30d"] = df.time_to_close_days / 30
    df["log_daily_volume"] = np.log1p(df.daily_contract_volume)
    df["log_trade_count"] = np.log1p(df.daily_trade_count)

    # Main linear specifications: pooled OLS, contract FE, and random effects.
    xvars = ["time_to_close_30d", "log_daily_volume", "log_trade_count"]
    x = sm.add_constant(df[xvars], has_constant="add")
    pooled = clustered_ols(df.absolute_forecast_error, x, df.event_ticker)
    fe = fixed_effects(df, "absolute_forecast_error", xvars)
    re = random_effects(df, "absolute_forecast_error", xvars)
    hstat, hp = hausman(fe, re, xvars)

    linear_rows = []
    linear_rows += tidy_result("Pooled OLS", pooled, ["const", *xvars])
    linear_rows += tidy_result("Contract FE", fe, xvars)
    linear_rows += tidy_result("Random Effects", re, ["const", *xvars])

    # Dynamic FE is descriptive robustness only: including lagged AFE can create
    # Nickell bias because the within-transformed lag correlates with the error term.
    dynamic = df.sort_values(["ticker", "date"]).copy()
    dynamic["lag_afe"] = dynamic.groupby("ticker").absolute_forecast_error.shift()
    dynamic = dynamic.dropna(subset=["lag_afe"])
    dynamic_vars = ["lag_afe", *xvars]
    dynamic_fe = fixed_effects(dynamic, "absolute_forecast_error", dynamic_vars)
    linear_rows += tidy_result("Dynamic Contract FE", dynamic_fe, dynamic_vars)
    linear = pd.DataFrame(linear_rows)

    # Prices at exactly zero or one are excluded because their logits are undefined.
    bern = df[(df.last_price > 0) & (df.last_price < 1)].copy()
    bern["logit_price"] = np.log(bern.last_price / (1 - bern.last_price))
    bern["price_time_interaction"] = bern.logit_price * bern.time_to_close_30d
    bern_vars = ["logit_price", "time_to_close_30d", "price_time_interaction", "log_daily_volume", "log_trade_count"]
    gee = sm.GEE(
        bern.outcome, sm.add_constant(bern[bern_vars], has_constant="add"),
        groups=bern.ticker, family=sm.families.Binomial(),
        # A contract FE logit is unidentified because settlement outcome is constant
        # within contract. Population-averaged GEE instead uses repeated daily prices.
        # An exchangeable working correlation is numerically degenerate here, so the
        # independence structure is paired with contract-cluster-robust GEE covariance.
        cov_struct=sm.cov_struct.Independence(),
    ).fit(maxiter=100)
    gee_rows = tidy_result("Bernoulli panel GEE", gee, ["const", *bern_vars])
    gee_table = pd.DataFrame(gee_rows)
    # Perfect calibration requires more than a price slope of one: the intercept,
    # time effects, interaction, and activity coefficients must also satisfy the null.
    beta1_stat, beta1_p = wald_test(gee, ["logit_price"], [1.0])
    full_stat, full_p = wald_test(
        gee,
        ["const", "logit_price", "time_to_close_30d", "price_time_interaction", "log_daily_volume", "log_trade_count"],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )
    calibration_tests = pd.DataFrame([
        {
            "test": "Price slope equals one",
            "null_hypothesis": "b=1",
            "wald_statistic": beta1_stat,
            "df": 1,
            "p_value": beta1_p,
        },
        {
            "test": "Time-invariant perfect calibration",
            "null_hypothesis": "a=0, b=1, c=0, d=0, activity coefficients=0",
            "wald_statistic": full_stat,
            "df": 6,
            "p_value": full_p,
        },
    ])
    horizons = horizon_logit(df, [30, 14, 7, 3, 1])

    simulation_df = simulation()
    descriptive = df[[
        "absolute_forecast_error", "last_price", "time_to_close_days",
        "daily_contract_volume", "daily_trade_count", "duration_days", "total_contract_volume",
    ]].describe().T

    summary = {
        "panel_rows": len(df),
        "contracts": int(df.ticker.nunique()),
        "events": int(df.event_ticker.nunique()),
        "date_min": str(df.date.min()),
        "date_max": str(df.date.max()),
        "hausman_statistic": hstat,
        "hausman_p_value": hp,
        "pooled_time_to_close_coef": pooled.params["time_to_close_30d"],
        "fe_time_to_close_coef": fe.params["time_to_close_30d"],
        "re_time_to_close_coef": re.params["time_to_close_30d"],
        "dynamic_lag_afe_coef": dynamic_fe.params["lag_afe"],
        "gee_logit_price_coef": gee.params["logit_price"],
        "gee_time_to_close_coef": gee.params["time_to_close_30d"],
        "gee_price_time_interaction_coef": gee.params["price_time_interaction"],
        "gee_beta_p_equal_1": beta1_p,
        "gee_perfect_calibration_joint_wald": full_stat,
        "gee_perfect_calibration_joint_p": full_p,
        "gee_dependence_parameter": 0.0 if gee.cov_struct.dep_params is None else float(np.asarray(gee.cov_struct.dep_params).reshape(-1)[0]),
    }

    display_table("Descriptive statistics", descriptive.reset_index(names="variable"))
    display_table("Linear panel models", linear)
    display_table("Bernoulli panel GEE", gee_table)
    display_table("Bernoulli calibration tests", calibration_tests)
    display_table("Fixed-horizon calibration", horizons)
    display_table("Linear-panel finite-sample simulation", simulation_df)
    display_table("Analysis summary", pd.DataFrame([summary]))
    calibration_bins = make_figures(df)
    display_table("Calibration-plot bins", calibration_bins)


if __name__ == "__main__":
    main()
