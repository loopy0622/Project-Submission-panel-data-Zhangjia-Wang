import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


SEED = 20260609


def logistic(value):
    """Map a linear predictor to a probability."""
    return 1.0 / (1.0 + np.exp(-value))


def run_simulation(repetitions=300, n_contracts=200, t_periods=30, rho=0.35):
    """Run repeated samples and summarize slope accuracy and 95% CI coverage."""
    rng = np.random.default_rng(SEED)
    true_beta = 1.0
    results = {"Pooled logit": [], "Bernoulli GEE": []}

    # This fixed identifier vector assigns T repeated observations to each contract.
    contract = np.repeat(np.arange(n_contracts), t_periods)
    for repetition in range(1, repetitions + 1):
        x = rng.normal(size=n_contracts * t_periods)
        probability = logistic(true_beta * x)

        # The shared Gaussian shock creates within-contract dependence. Transforming
        # the latent normal to a uniform variable preserves the intended Bernoulli
        # marginal probability for every observation.
        shared_shock = rng.normal(size=n_contracts)
        idiosyncratic_shock = rng.normal(size=n_contracts * t_periods)
        latent_normal = (
            np.sqrt(rho) * shared_shock[contract]
            + np.sqrt(1.0 - rho) * idiosyncratic_shock
        )
        correlated_uniform = norm.cdf(latent_normal)
        outcome = (correlated_uniform < probability).astype(int)
        design = sm.add_constant(x)

        # Pooled logit treats observations as independent. GEE explicitly groups
        # repeated observations by contract and reports robust sandwich covariance.
        pooled = sm.GLM(outcome, design, family=sm.families.Binomial()).fit()
        gee = sm.GEE(
            outcome,
            design,
            groups=contract,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit(maxiter=100)
        for label, fitted in (("Pooled logit", pooled), ("Bernoulli GEE", gee)):
            estimate = float(fitted.params[1])
            se = float(fitted.bse[1])
            results[label].append({
                "estimate": estimate,
                "se": se,
                "covered": abs(estimate - true_beta) <= 1.96 * se,
            })

        if repetition % 50 == 0:
            print(f"completed {repetition}/{repetitions} repetitions", flush=True)

    rows = []
    for estimator, result_rows in results.items():
        values = np.asarray([row["estimate"] for row in result_rows])
        standard_errors = np.asarray([row["se"] for row in result_rows])
        rows.append({
            "estimator": estimator,
            "mean_beta": values.mean(),
            "bias": values.mean() - true_beta,
            "rmse": np.sqrt(np.mean((values - true_beta) ** 2)),
            "mean_standard_error": standard_errors.mean(),
            "coverage_95": np.mean([row["covered"] for row in result_rows]),
            "true_beta": true_beta,
            "contracts": n_contracts,
            "periods": t_periods,
            "repetitions": repetitions,
            "within_contract_rho": rho,
        })
    return pd.DataFrame(rows)


def main():
    """Run the simulation and display the comparison table."""
    results = run_simulation()
    print("\nBernoulli finite-sample simulation")
    print("==================================")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
