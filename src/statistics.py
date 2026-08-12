"""
statistics.py — Statistical tests and change-point detection.

Reads:
  reports/tables/linguistic_metrics.csv
  reports/tables/semantic_similarity_stats.csv
Writes:
  reports/tables/statistical_results.csv
  reports/tables/changepoint_analysis.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.config import (
    TABLES_DIR,
    ALPHA,
    RANDOM_SEED,
    MULTIPLE_CORRECTION,
    CHANGEPOINT_MIN_YEAR,
    CHANGEPOINT_MAX_YEAR,
)
from src.utils import get_logger

logger = get_logger(__name__)

np.random.seed(RANDOM_SEED)


# ─── Effect-size uncertainty ──────────────────────────────────────────────────

def bootstrap_effect_size_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = RANDOM_SEED,
) -> tuple[float | None, float | None]:
    """
    Percentile bootstrap CI for the rank-biserial correlation.

    A p-value alone says nothing about the magnitude or precision of a
    difference. Reporting an interval makes it visible when an effect is
    'significant' but small and poorly constrained — which matters here,
    because large corpora make trivial differences significant.
    """
    if len(a) < 5 or len(b) < 5:
        return None, None

    rng = np.random.default_rng(seed)
    n_a, n_b = len(a), len(b)
    stats_boot = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sa = rng.choice(a, size=n_a, replace=True)
        sb = rng.choice(b, size=n_b, replace=True)
        try:
            u, _ = stats.mannwhitneyu(sa, sb, alternative="two-sided")
        except ValueError:
            stats_boot[i] = np.nan
            continue
        stats_boot[i] = 1 - (2 * u) / (n_a * n_b)

    stats_boot = stats_boot[~np.isnan(stats_boot)]
    if stats_boot.size == 0:
        return None, None

    lo = float(np.percentile(stats_boot, 100 * alpha / 2))
    hi = float(np.percentile(stats_boot, 100 * (1 - alpha / 2)))
    return round(lo, 4), round(hi, 4)


# ─── Period comparison tests ──────────────────────────────────────────────────

def compare_periods(
    df: pd.DataFrame,
    metric_col: str,
    period_a: tuple[int, int],
    period_b: tuple[int, int],
) -> dict:
    """
    Mann-Whitney U test comparing metric_col between two year ranges.
    Returns: statistic, p-value, effect_size (rank-biserial correlation), medians.
    """
    a_data = df[df["year"].between(*period_a)][metric_col].dropna()
    b_data = df[df["year"].between(*period_b)][metric_col].dropna()

    if len(a_data) < 5 or len(b_data) < 5:
        return {
            "metric":     metric_col,
            "period_a":   f"{period_a[0]}-{period_a[1]}",
            "period_b":   f"{period_b[0]}-{period_b[1]}",
            "n_a":        len(a_data),
            "n_b":        len(b_data),
            "median_a":   None,
            "median_b":   None,
            "statistic":  None,
            "p_value":    None,
            "effect_size": None,
        }

    u_stat, p_val = stats.mannwhitneyu(a_data, b_data, alternative="two-sided")

    # Rank-biserial correlation as effect size
    n_a, n_b = len(a_data), len(b_data)
    r = 1 - (2 * u_stat) / (n_a * n_b)

    ci_low, ci_high = bootstrap_effect_size_ci(a_data.values, b_data.values)

    return {
        "metric":      metric_col,
        "period_a":    f"{period_a[0]}-{period_a[1]}",
        "period_b":    f"{period_b[0]}-{period_b[1]}",
        "n_a":         n_a,
        "n_b":         n_b,
        "median_a":    float(a_data.median()),
        "median_b":    float(b_data.median()),
        "median_diff": round(float(b_data.median() - a_data.median()), 6),
        "statistic":   float(u_stat),
        "p_value":     float(p_val),
        "effect_size": round(r, 4),
        "effect_ci_low":  ci_low,
        "effect_ci_high": ci_high,
    }


def run_period_tests(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    Compare baseline (2015-2019) vs AI-era (2024-2026) for all metrics.
    Returns a DataFrame of test results.
    """
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        result = compare_periods(df, metric, (2015, 2019), (2024, 2026))
        rows.append(result)

    results_df = pd.DataFrame(rows)

    # Multiple-test correction
    valid_p = results_df["p_value"].dropna()
    if len(valid_p) > 0:
        corrected = multipletests(valid_p, alpha=ALPHA, method=MULTIPLE_CORRECTION)
        results_df.loc[results_df["p_value"].notna(), "p_corrected"] = corrected[1]
        results_df["significant"] = results_df["p_corrected"] < ALPHA
    else:
        results_df["p_corrected"] = None
        results_df["significant"] = False

    return results_df


# ─── Change-point detection ───────────────────────────────────────────────────

def fit_segmented_regression(
    years: np.ndarray,
    values: np.ndarray,
    breakpoint: int,
) -> dict:
    """
    Fit a piecewise linear regression with a single breakpoint.
    Model: y = a1 + b1*t  for t < breakpoint
           y = a2 + b2*t  for t >= breakpoint
    Returns dict with slopes, intercepts, and R².
    """
    mask_before = years < breakpoint
    mask_after  = years >= breakpoint

    x_before = years[mask_before]
    y_before = values[mask_before]
    x_after  = years[mask_after]
    y_after  = values[mask_after]

    if len(x_before) < 2 or len(x_after) < 2:
        return {"valid": False}

    slope_before, intercept_before, r_before, _, _ = stats.linregress(x_before, y_before)
    slope_after,  intercept_after,  r_after,  _, _ = stats.linregress(x_after,  y_after)

    # Overall R² as proportion of variance explained
    y_pred = np.concatenate([
        intercept_before + slope_before * x_before,
        intercept_after  + slope_after  * x_after,
    ])
    y_actual = np.concatenate([y_before, y_after])
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "valid":           True,
        "breakpoint":      breakpoint,
        "slope_before":    round(slope_before, 6),
        "slope_after":     round(slope_after, 6),
        "intercept_before": round(intercept_before, 6),
        "intercept_after":  round(intercept_after, 6),
        "r2":              round(r2, 4),
        "slope_change":    round(slope_after - slope_before, 6),
    }


def search_changepoint(yearly_df: pd.DataFrame, metric: str) -> dict:
    """
    Search for the best breakpoint year for a given metric using segmented regression.
    Returns the breakpoint with the highest R².
    """
    if metric not in yearly_df.columns:
        return {"metric": metric, "valid": False}

    df = yearly_df[["year", metric]].dropna()
    if len(df) < 6:
        return {"metric": metric, "valid": False}

    years  = df["year"].values
    values = df[metric].values

    best = None
    for bp in range(CHANGEPOINT_MIN_YEAR, CHANGEPOINT_MAX_YEAR + 1):
        fit = fit_segmented_regression(years, values, bp)
        if not fit.get("valid"):
            continue
        if best is None or fit["r2"] > best["r2"]:
            best = fit

    if best is None:
        return {"metric": metric, "valid": False}

    # A segmented model has more free parameters than a straight line, so it can
    # never fit worse. Searching for max R² therefore ALWAYS returns some
    # breakpoint. Report the single-line baseline alongside it so the reader can
    # judge whether the break explains anything beyond a simple linear trend.
    slope_lin, intercept_lin, r_lin, p_lin, _ = stats.linregress(years, values)
    r2_linear = float(r_lin ** 2)

    best["metric"]         = metric
    best["r2_linear"]      = round(r2_linear, 4)
    best["r2_improvement"] = round(best["r2"] - r2_linear, 4)
    best["linear_slope"]   = round(float(slope_lin), 6)
    best["linear_p_value"] = round(float(p_lin), 6)
    best["n_years"]        = int(len(years))
    return best


def run_changepoint_analysis(yearly_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Run changepoint search on yearly aggregated metrics."""
    rows = []
    for metric in metrics:
        result = search_changepoint(yearly_df, metric)
        if result.get("valid"):
            rows.append(result)
    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_statistical_analysis() -> None:
    ling_path = TABLES_DIR / "linguistic_metrics.csv"
    sim_path  = TABLES_DIR / "semantic_similarity_stats.csv"

    if not ling_path.exists():
        logger.error("linguistic_metrics.csv not found. Run linguistic_metrics.py first.")
        sys.exit(1)

    ling_df = pd.read_csv(ling_path)
    logger.info("Loaded linguistic metrics: %d rows", len(ling_df))

    # Article-level metrics to test
    test_metrics = [
        "mattr", "ttr", "root_ttr", "hapax_ratio", "func_word_ratio",
        "avg_sentence_len", "med_sentence_len",
        "flesch_reading_ease", "flesch_kincaid_grade",
    ]

    logger.info("Running period-comparison tests (2015-2019 vs 2024-2026) …")
    results = run_period_tests(ling_df, test_metrics)
    results.to_csv(TABLES_DIR / "statistical_results.csv", index=False)
    logger.info("Saved statistical_results.csv")

    # Print significant results
    sig = results[results.get("significant", False) == True]
    if not sig.empty:
        print("\n=== STATISTICALLY SIGNIFICANT DIFFERENCES ===")
        print(sig[["metric","median_a","median_b","p_corrected","effect_size"]].to_string(index=False))
    else:
        print("\nNo statistically significant differences detected after correction.")

    # Changepoint analysis on yearly aggregates
    yearly_path = TABLES_DIR / "linguistic_metrics_by_year.csv"
    if yearly_path.exists():
        yearly_df = pd.read_csv(yearly_path)
        yearly_metrics = [c for c in yearly_df.columns
                          if c.endswith("_median") and any(m in c for m in test_metrics)]

        logger.info("Running changepoint analysis …")
        cp_df = run_changepoint_analysis(yearly_df, yearly_metrics)
        cp_df.to_csv(TABLES_DIR / "changepoint_analysis.csv", index=False)
        logger.info("Saved changepoint_analysis.csv")

        if not cp_df.empty:
            print("\n=== CHANGEPOINT ANALYSIS (best breakpoints) ===")
            print(cp_df[["metric","breakpoint","slope_before","slope_after","slope_change","r2"]].to_string(index=False))

    # Similarity trend test (if available)
    #
    # NOTE: semantic_similarity_stats.csv holds ONE ROW PER YEAR (~12 rows), not
    # one row per article. A Mann-Whitney U on 5 vs 3 yearly aggregates is both
    # underpowered and inappropriate (the unit of analysis is the year, and the
    # values are already means). A monotonic trend test across years is the
    # correct tool at this level of aggregation.
    if sim_path.exists():
        sim_df = pd.read_csv(sim_path).dropna(subset=["year", "mean_sim"])
        print("\n=== SEMANTIC SIMILARITY TREND (year-level, n = %d years) ===" % len(sim_df))

        if len(sim_df) < 4:
            print("  Too few years to test a trend.")
        else:
            rho, p_rho = stats.spearmanr(sim_df["year"], sim_df["mean_sim"])
            tau, p_tau = stats.kendalltau(sim_df["year"], sim_df["mean_sim"])
            trend = pd.DataFrame([{
                "n_years":         int(len(sim_df)),
                "first_year":      int(sim_df["year"].min()),
                "last_year":       int(sim_df["year"].max()),
                "spearman_rho":    round(float(rho), 4),
                "spearman_p":      round(float(p_rho), 6),
                "kendall_tau":     round(float(tau), 4),
                "kendall_p":       round(float(p_tau), 6),
                "mean_sim_first":  round(float(sim_df.iloc[0]["mean_sim"]), 5),
                "mean_sim_last":   round(float(sim_df.iloc[-1]["mean_sim"]), 5),
            }])
            trend.to_csv(TABLES_DIR / "similarity_trend.csv", index=False)
            logger.info("Saved similarity_trend.csv")
            print(trend.to_string(index=False))
            print("  Interpretation: a monotonic trend in within-year similarity is a")
            print("  TEMPORAL ASSOCIATION only. It does not identify a cause.")


if __name__ == "__main__":
    run_statistical_analysis()
