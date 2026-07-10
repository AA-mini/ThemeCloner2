# backtest_v2.py
# V2 NEW MODULE -- the walk-forward backtest loop.
#
# WHAT THIS FIXES:
# V1's "out-of-sample" validation (projection.py: validate_against_etf) is
# out-of-sample only in the CROSS-SECTIONAL sense: target-universe stocks
# were never used to fit the factor model. But the factor model, the theme
# fingerprints, AND the candidate scores are all still computed on the FULL
# 2018-2026 sample at once. A candidate's score in 2019 already "knows"
# about its 2024-2026 behaviour. V1's paper is explicit that this is a
# limitation, not an oversight (Section 3.6 / 5.5).
#
# This module closes that gap. At every rebalance date:
#   1. RP-PCA is fit using ONLY trailing covariance-universe data
#      (rppca_walkforward.rolling_rppca_fit, called by the orchestrator here
#      per-period -- see run_walkforward_backtest below)
#   2. Theme fingerprints are built using ONLY trailing ETF data
#   3. Candidates are scored using ONLY trailing target-universe data
#   4. The resulting basket is then held for exactly one forward period, and
#      its REALIZED return over that period (data the scoring step never saw)
#      is recorded.
# Steps 1-3 all reuse existing, unchanged V1 functions (fit_rppca,
# label_theme_factors, rank_candidates) -- only the SLICING of what data each
# step is allowed to see is new. Step 4 is the genuinely new piece: a proper
# walk-forward equity curve, not a single-period snapshot.

import numpy as np
import pandas as pd

from src.theme_dna import label_theme_factors
from src.projection import synthetic_theme_returns, score_universe_v2
from src.scoring import rank_candidates


def run_walkforward_backtest(cov_returns_resid_keepalpha: pd.DataFrame,
                              etf_returns_resid: pd.DataFrame,
                              target_returns_resid: pd.DataFrame,
                              target_returns_raw: pd.DataFrame,
                              etf_config: pd.DataFrame,
                              rebalance_dates: list,
                              K: int = 15,
                              gamma: float = 10.0,
                              window: str = "expanding",
                              rolling_window_weeks: int = 208,
                              top_n: int = 30,
                              min_score: float = 0.0,
                              top_factors: int = 3,
                              verbose: bool = True) -> dict:
    """
    Runs the full point-in-time walk-forward backtest.

    Parameters
    ----------
    cov_returns_resid_keepalpha : T x N covariance-universe returns,
        residualized with residualize_returns(..., keep_alpha=True).
        This is what RP-PCA is fit on at each rebalance date -- MUST be
        keep_alpha=True or V2's entire premise (giving RP-PCA's penalty a
        real mu to reward) does not hold.
    etf_returns_resid : T x N_etf, standard V1 residualization (keep_alpha
        does not matter here -- see projection.py's V2 section docstring
        for why).
    target_returns_resid : T x N_target, standard V1 residualization. Used
        ONLY for scoring (synthetic-return correlation), sliced to
        trailing-only at each rebalance date.
    target_returns_raw : T x N_target, ACTUAL (non-residualized) target
        universe returns. Used ONLY to compute realized forward basket
        returns after a rebalance -- this is the real-world return an
        investor would have earned, not a factor-model construct.
    etf_config : etfs.csv mapping (ticker -> theme)
    rebalance_dates : output of rppca_walkforward.make_rebalance_dates()
    K, gamma, window, rolling_window_weeks : passed to the per-period RP-PCA fit
    top_n, min_score : passed to rank_candidates at each rebalance
    top_factors : passed to label_theme_factors at each rebalance

    Returns
    -------
    dict with:
        period_baskets   : {rebalance_date -> {theme -> [tickers]}}
        forward_returns  : pd.DataFrame, index=rebalance_date, columns=theme,
                            values = realized equal-weighted forward return
                            of that period's basket over the NEXT rebalance
                            interval (out of sample in both the cross-
                            sectional AND temporal sense)
        equity_curves    : pd.DataFrame, cumulative growth of $1 per theme,
                            stitched across all rebalance periods
        factor_drift     : output of rppca_walkforward.factor_drift_report(),
                            included here so one function call gives you
                            both the backtest AND the stability diagnostic
    """
    from src.rppca_walkforward import rolling_rppca_fit, factor_drift_report

    themes = etf_config["theme"].unique().tolist()
    period_baskets = {}
    forward_return_rows = []

    for i, d in enumerate(rebalance_dates):
        if verbose:
            print(f"\n{'='*70}\nrebalance {i+1}/{len(rebalance_dates)}: {d.date()}\n{'='*70}")

        # -------------------- 1. point-in-time RP-PCA fit --------------------
        if window == "expanding":
            cov_window = cov_returns_resid_keepalpha.loc[:d]
        else:
            idx = cov_returns_resid_keepalpha.index
            end_pos = idx.get_indexer([d], method="pad")[0]
            start_pos = max(0, end_pos - rolling_window_weeks + 1)
            cov_window = cov_returns_resid_keepalpha.iloc[start_pos:end_pos + 1]

        wf_result = rolling_rppca_fit(cov_returns_resid_keepalpha, [d],
                                       K=K, gamma=gamma, window=window,
                                       rolling_window_weeks=rolling_window_weeks,
                                       verbose=verbose)[d]

        # -------------------- 2. point-in-time fingerprints --------------------
        etf_window = etf_returns_resid.loc[:d]
        dna = label_theme_factors(wf_result, etf_window, etf_config,
                                   top_factors=top_factors)

        # -------------------- 3. point-in-time scoring --------------------
        target_score_window = target_returns_resid.loc[:d]
        synth = synthetic_theme_returns(wf_result, dna["theme_factors"])
        scores = score_universe_v2(target_score_window, synth)
        ranked = rank_candidates(scores, top_n=top_n, min_score=min_score)

        baskets_this_period = {}
        for theme in themes:
            tickers = ranked[ranked["theme"] == theme]["ticker"].tolist()
            baskets_this_period[theme] = tickers
        period_baskets[d] = baskets_this_period

        # -------------------- 4. genuinely forward realized return --------------------
        next_d = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else target_returns_raw.index[-1]
        fwd_mask = (target_returns_raw.index > d) & (target_returns_raw.index <= next_d)
        fwd_returns = target_returns_raw.loc[fwd_mask]

        row = {"rebalance_date": d}
        for theme, tickers in baskets_this_period.items():
            avail = [t for t in tickers if t in fwd_returns.columns]
            if not avail:
                row[theme] = np.nan
                continue
            period_ret = float((1 + fwd_returns[avail].mean(axis=1)).prod() - 1)
            row[theme] = period_ret
        forward_return_rows.append(row)

    forward_returns = pd.DataFrame(forward_return_rows).set_index("rebalance_date")
    equity_curves = (1 + forward_returns.fillna(0)).cumprod()

    drift = factor_drift_report(
        rolling_rppca_fit(cov_returns_resid_keepalpha, rebalance_dates,
                           K=K, gamma=gamma, window=window,
                           rolling_window_weeks=rolling_window_weeks,
                           verbose=False)
    )

    if verbose:
        print(f"\n{'='*70}\nWALK-FORWARD BACKTEST COMPLETE: {len(rebalance_dates)} rebalances\n{'='*70}")
        print("\nfinal cumulative growth of $1, by theme:")
        print(equity_curves.iloc[-1].round(3).to_string())

    return {
        "period_baskets": period_baskets,
        "forward_returns": forward_returns,
        "equity_curves": equity_curves,
        "factor_drift": drift,
    }


def summarize_backtest(forward_returns: pd.DataFrame, annualise_periods: float = 4.0) -> pd.DataFrame:
    """
    Per-theme summary stats from the walk-forward forward_returns panel.
    annualise_periods: number of rebalances per year (4.0 for quarterly).
    """
    rows = []
    for theme in forward_returns.columns:
        r = forward_returns[theme].dropna()
        if len(r) < 2:
            continue
        mean_ann = r.mean() * annualise_periods
        vol_ann = r.std() * np.sqrt(annualise_periods)
        sharpe = mean_ann / vol_ann if vol_ann > 0 else np.nan
        hit_rate = float((r > 0).mean())
        rows.append({
            "theme": theme,
            "n_periods": len(r),
            "mean_ret_ann": round(mean_ann, 3),
            "vol_ann": round(vol_ann, 3),
            "sharpe": round(sharpe, 3),
            "hit_rate": round(hit_rate, 3),
        })
    result = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print("\nwalk-forward backtest summary (genuinely out-of-sample, per rebalance period):")
    print(result.to_string(index=False))
    return result
