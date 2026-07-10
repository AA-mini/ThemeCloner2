# rppca_walkforward.py
# V2 NEW MODULE -- walk-forward / rolling RP-PCA estimation.
#
# WHY THIS EXISTS (V1 -> V2 change #2 of 2):
#   V1 fits RP-PCA once, on the full sample, on returns that are forced to be
#   mean-zero (see residualize.py). V2's first change (residualize.py's
#   keep_alpha=True) restores real cross-sectional mean-return information so
#   RP-PCA's premium-reward penalty has something to work with -- but doing
#   that on a single full-sample fit would be a genuine look-ahead problem:
#   the mean return of a stock over 2018-2026 obviously "knows" about its
#   2024-2026 performance when you use it to build a portfolio in 2019.
#
#   This module is the fix: refit RP-PCA repeatedly, at each rebalance date,
#   using ONLY data up to that date (expanding window) or a fixed trailing
#   window (rolling window). Nothing at or after the rebalance date is ever
#   visible to the fit used to score that period.
#
# WHAT IS REUSED FROM V1 UNCHANGED:
#   The actual RP-PCA math (src/rppca.py: rppca(), fit_rppca()) is untouched
#   and imported directly. There was no reason to rewrite a solver that
#   already works -- this module is purely about WHEN it gets called and on
#   WHAT SLICE of data, not HOW the eigenproblem is solved.

import numpy as np
import pandas as pd

from src.rppca import fit_rppca


# -------------------- 1. Build a rebalance schedule --------------------

def make_rebalance_dates(index: pd.DatetimeIndex,
                          freq: str = "Q",
                          min_window_weeks: int = 104) -> list:
    """
    Builds a list of rebalance dates from a weekly return index.

    freq             : pandas offset alias for rebalance frequency.
                        'QE' (quarterly) is the recommended default -- frequent
                        enough to let a theme's factor identity actually drift
                        and be observed (the instability question raised
                        against V1), infrequent enough to keep the walk-forward
                        backtest computationally tractable.
    min_window_weeks : the first rebalance date is only included once at
                        least this many weeks of history exist before it.
                        104 weeks (2 years) is a reasonable floor for a
                        15-factor RP-PCA fit on a ~2,500-stock cross-section;
                        tune down if the covariance universe is smaller.

    Returns a list of pd.Timestamp rebalance dates, in order, each of which
    has enough trailing history to fit on.
    """
    all_period_ends = pd.date_range(index[0], index[-1], freq=freq)
    rebal = []
    for d in all_period_ends:
        trailing = index[index <= d]
        if len(trailing) >= min_window_weeks:
            rebal.append(trailing[-1])   # snap to the actual last trading week <= d
    return rebal


# -------------------- 2. Walk-forward RP-PCA fit --------------------

def rolling_rppca_fit(residual_returns: pd.DataFrame,
                       rebalance_dates: list,
                       K: int = 15,
                       gamma: float = 10.0,
                       window: str = "expanding",
                       rolling_window_weeks: int = 208,
                       verbose: bool = True) -> dict:
    """
    Fits RP-PCA at every rebalance date, using only data strictly up to and
    including that date.

    Parameters
    ----------
    residual_returns     : T x N DataFrame. MUST be built with
                            residualize_returns(..., keep_alpha=True) --
                            passing V1-style mean-zero residuals here defeats
                            the entire point of V2 (RP-PCA's penalty term
                            would again have no real mu to reward).
    rebalance_dates       : output of make_rebalance_dates()
    K, gamma              : same meaning as in rppca.fit_rppca()
    window                : 'expanding' (use all history from the start of
                             residual_returns up to the rebalance date -- more
                             stable estimates, standard choice for this kind
                             of backtest) or 'rolling' (use only the trailing
                             rolling_window_weeks -- more adaptive to regime
                             change, noisier with fewer weeks)
    rolling_window_weeks  : only used when window='rolling'

    Returns
    -------
    dict {rebalance_date -> rppca_result}, where rppca_result is exactly the
    same dict shape fit_rppca() already returns (loadings, factors, eigvals,
    sr_is, sr_oos, gamma, K) -- so every downstream V1 module that already
    knows how to consume an rppca_result (theme_dna.py, projection.py) works
    against each period's fit with zero changes.
    """
    results = {}
    for i, d in enumerate(rebalance_dates):
        if window == "expanding":
            window_data = residual_returns.loc[:d]
        elif window == "rolling":
            all_idx = residual_returns.index
            end_pos = all_idx.get_indexer([d], method="pad")[0]
            start_pos = max(0, end_pos - rolling_window_weeks + 1)
            window_data = residual_returns.iloc[start_pos:end_pos + 1]
        else:
            raise ValueError(f"window must be 'expanding' or 'rolling', got {window!r}")

        if verbose:
            print(f"\n[{i+1}/{len(rebalance_dates)}] fitting RP-PCA as of {d.date()} "
                  f"({window}, {window_data.shape[0]} weeks x {window_data.shape[1]} stocks)")

        result = fit_rppca(window_data, K=K, gamma=gamma, run_oos=False)
        results[d] = result

    return results


# -------------------- 3. Factor-identity drift diagnostic --------------------

def factor_drift_report(walkforward_results: dict, top_n_stocks: int = 8) -> pd.DataFrame:
    """
    THE DIRECT ANSWER TO "if I ran this at my desk tomorrow, could I keep
    rotating in and out of these positions" -- quantifies how much a
    factor's economic identity actually drifts from one rebalance to the
    next, rather than asserting stability from a single snapshot.

    For each consecutive pair of rebalance dates, measures the cosine
    similarity between factor_k's loading vector this period and factor_k's
    loading vector last period, restricted to stocks present in both fits.
    A similarity near 1.0 means "this is still the same economic factor."
    A similarity near 0 (or negative) means the factor's identity has
    rotated -- exactly the risk flagged in the V1 conversation about
    re-underwriting positions every rebalance.

    Returns a long DataFrame: rebalance_date, factor, cosine_similarity_to_prior.
    """
    dates = sorted(walkforward_results.keys())
    rows = []
    for prev_d, curr_d in zip(dates[:-1], dates[1:]):
        prev_load = walkforward_results[prev_d]["loadings"]
        curr_load = walkforward_results[curr_d]["loadings"]
        common_stocks = prev_load.index.intersection(curr_load.index)
        if len(common_stocks) < 20:
            continue

        for factor in curr_load.columns:
            if factor not in prev_load.columns:
                continue
            v_prev = prev_load.loc[common_stocks, factor].values
            v_curr = curr_load.loc[common_stocks, factor].values
            n_prev, n_curr = np.linalg.norm(v_prev), np.linalg.norm(v_curr)
            sim = float(np.dot(v_prev, v_curr) / (n_prev * n_curr)) if n_prev > 0 and n_curr > 0 else np.nan
            rows.append({"rebalance_date": curr_d, "factor": factor,
                         "cosine_similarity_to_prior": round(sim, 3)})

    report = pd.DataFrame(rows)
    if not report.empty:
        summary = report.groupby("factor")["cosine_similarity_to_prior"].agg(["mean", "min"])
        print("\nfactor identity drift (cosine similarity to the SAME factor_k, prior period):")
        print(summary.round(3).to_string())
        print("  near 1.0 = stable identity, safe to hold across rebalances")
        print("  near 0 or negative = factor has rotated, re-underwrite before re-sizing")
    return report
