# momentum_test.py
#
# Direct test of the industry critique: "is thematic investing just momentum?"
#
# We distinguish three claims and test the strongest one:
#   1. theme membership == momentum ranking?        (cross-sectional overlap)
#   2. theme returns == momentum returns?            (return attribution)
#   3. theme scores predict returns ONLY via momentum? (incremental predictive)
#
# Claim 3 is the referee-proof version: does a stock's theme score predict its
# forward return AFTER controlling for its OWN momentum? If yes, theme carries
# information momentum does not.
#
# THREE TESTS (full battery):
#   A. Independent double sort (momentum quintile x theme quintile) -> forward
#      return grid. If theme spread survives WITHIN momentum buckets, theme != momentum.
#   B. Fama-MacBeth cross-sectional regressions: weekly regress forward return on
#      theme score, momentum, and both jointly. Theme coef surviving in the joint
#      model = incremental predictive power. Newey-West t-stats.
#   C. Spanning regression: regress the theme long-short portfolio return on the
#      momentum factor (+ FF5). Significant alpha = not spanned by momentum.
#
# CRITICAL DESIGN NOTE:
#   Momentum here is each STOCK's OWN trailing 12-1 month return (standard
#   Jegadeesh-Titman: cumulative return from t-12mo to t-1mo, skipping the most
#   recent month to avoid 1-month reversal). This is STOCK-LEVEL momentum,
#   computed in the target universe -- it is a DIFFERENT object from the FF
#   momentum FACTOR (a long-short portfolio return) used in residualization.
#   Conflating them would be a hole; we keep them strictly separate.

import numpy as np
import pandas as pd


# -------------------- 1. Stock-level momentum signal --------------------

def compute_momentum(returns: pd.DataFrame,
                      lookback_weeks: int = 52,
                      skip_weeks: int = 4) -> pd.DataFrame:
    """
    Jegadeesh-Titman 12-1 momentum, weekly version.

    For each week t and each stock, momentum = cumulative return over the window
    [t - lookback_weeks, t - skip_weeks). Skipping the most recent ~4 weeks
    (one month) avoids the well-known short-term reversal effect.

    Parameters
    ----------
    returns       : T x N weekly simple returns
    lookback_weeks: formation window length (52 = ~12 months)
    skip_weeks    : weeks skipped at the recent end (4 = ~1 month)

    Returns
    -------
    momentum : T x N DataFrame of trailing momentum (NaN for early weeks)
    """
    # log returns for clean compounding
    logret = np.log1p(returns)
    # rolling sum over the full lookback, then subtract the skipped tail
    cum_full = logret.rolling(lookback_weeks).sum()
    cum_skip = logret.rolling(skip_weeks).sum()
    mom_log  = cum_full - cum_skip
    momentum = np.expm1(mom_log)        # back to simple cumulative return
    return momentum


# -------------------- 2. Forward returns --------------------

def compute_forward_returns(returns: pd.DataFrame,
                            horizon_weeks: int = 4) -> pd.DataFrame:
    """
    Forward cumulative return over the next `horizon_weeks`, aligned to the
    decision date t (so row t holds the return realized over (t, t+horizon]).
    """
    logret = np.log1p(returns)
    fwd_log = logret.rolling(horizon_weeks).sum().shift(-horizon_weeks)
    return np.expm1(fwd_log)


# -------------------- 3. Test A: independent double sort --------------------

def double_sort(theme_score: pd.Series,
                momentum: pd.DataFrame,
                fwd_returns: pd.DataFrame,
                n_quintiles: int = 5) -> dict:
    """
    Independent double sort on momentum and theme score.

    theme_score is STATIC per stock (the cosine similarity to one theme's
    fingerprint). momentum and fwd_returns are time-varying (T x N).

    For each week, assign each stock to a momentum quintile and a theme-score
    quintile, then average forward returns within each (mom, theme) cell across
    all stocks and weeks. The key question: within a momentum quintile (a row),
    does the forward return rise across theme quintiles (columns)?

    Returns
    -------
    dict with:
      grid          : 5x5 DataFrame, mean forward return per (mom Q, theme Q)
      theme_spread_within_mom : for each mom quintile, (theme Q5 - theme Q1)
      mom_spread_within_theme : for each theme quintile, (mom Q5 - mom Q1)
      theme_spread_overall    : avg theme Q5-Q1 spread across mom buckets
    """
    common_stocks = theme_score.index.intersection(momentum.columns)
    ts = theme_score.loc[common_stocks]

    # theme quintile is static (one score per stock)
    theme_q = pd.qcut(ts.rank(method="first"), n_quintiles, labels=False)

    # accumulate forward returns per (mom_q, theme_q) cell
    sums   = np.zeros((n_quintiles, n_quintiles))
    counts = np.zeros((n_quintiles, n_quintiles))

    dates = momentum.index.intersection(fwd_returns.index)
    for dt in dates:
        mom_row = momentum.loc[dt, common_stocks]
        fwd_row = fwd_returns.loc[dt, common_stocks]
        valid = mom_row.notna() & fwd_row.notna()
        if valid.sum() < n_quintiles * 2:
            continue
        mom_q = pd.qcut(mom_row[valid].rank(method="first"),
                        n_quintiles, labels=False)
        for stock in mom_row[valid].index:
            mi = int(mom_q[stock])
            ti = int(theme_q[stock])
            sums[mi, ti]   += fwd_row[stock]
            counts[mi, ti] += 1

    grid = pd.DataFrame(np.where(counts > 0, sums / np.maximum(counts, 1), np.nan),
                        index=[f"mom_Q{i+1}" for i in range(n_quintiles)],
                        columns=[f"theme_Q{i+1}" for i in range(n_quintiles)])

    theme_spread_within_mom = grid.iloc[:, -1] - grid.iloc[:, 0]
    mom_spread_within_theme = grid.iloc[-1, :] - grid.iloc[0, :]

    return {
        "grid": grid,
        "theme_spread_within_mom": theme_spread_within_mom,
        "mom_spread_within_theme": mom_spread_within_theme,
        "theme_spread_overall": float(theme_spread_within_mom.mean()),
        "mom_spread_overall": float(mom_spread_within_theme.mean()),
    }


# -------------------- 4. Test B: Fama-MacBeth regressions --------------------

def _newey_west_tstat(series: np.ndarray, lags: int = 4) -> float:
    """t-stat of the mean of a time series with Newey-West HAC correction."""
    x = series[~np.isnan(series)]
    T = len(x)
    if T < 10:
        return np.nan
    mean = x.mean()
    demeaned = x - mean
    gamma0 = (demeaned @ demeaned) / T
    var = gamma0
    for L in range(1, min(lags, T - 1) + 1):
        w = 1 - L / (lags + 1)
        cov = (demeaned[L:] @ demeaned[:-L]) / T
        var += 2 * w * cov
    se = np.sqrt(var / T)
    return float(mean / se) if se > 0 else np.nan


def fama_macbeth(theme_score: pd.Series,
                 momentum: pd.DataFrame,
                 fwd_returns: pd.DataFrame,
                 standardize: bool = True) -> pd.DataFrame:
    """
    Fama-MacBeth cross-sectional regressions of forward return on:
      (i)   theme score alone
      (ii)  momentum alone
      (iii) theme + momentum jointly

    Each week we run the cross-sectional regression, collect coefficients, then
    average over time with Newey-West t-stats. The decisive comparison is the
    theme coefficient in (i) vs (iii): if it survives the addition of momentum,
    theme has incremental predictive power.

    standardize: z-score the regressors cross-sectionally each week so coefs
                 are comparable in units of "per 1 std of signal".
    """
    common = theme_score.index.intersection(momentum.columns)
    ts_static = theme_score.loc[common]

    coefs_theme_only = []
    coefs_mom_only   = []
    coefs_joint_t    = []   # theme coef in joint model
    coefs_joint_m    = []   # momentum coef in joint model

    dates = momentum.index.intersection(fwd_returns.index)
    for dt in dates:
        mom = momentum.loc[dt, common]
        fwd = fwd_returns.loc[dt, common]
        valid = mom.notna() & fwd.notna()
        if valid.sum() < 20:
            continue

        y = fwd[valid].values
        t = ts_static[valid].values
        m = mom[valid].values

        if standardize:
            t = (t - t.mean()) / (t.std() + 1e-10)
            m = (m - m.mean()) / (m.std() + 1e-10)

        # (i) theme only
        coefs_theme_only.append(np.polyfit(t, y, 1)[0])
        # (ii) momentum only
        coefs_mom_only.append(np.polyfit(m, y, 1)[0])
        # (iii) joint: OLS y on [1, t, m]
        X = np.column_stack([np.ones_like(t), t, m])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        coefs_joint_t.append(beta[1])
        coefs_joint_m.append(beta[2])

    def summarize(arr):
        a = np.array(arr)
        return (float(np.nanmean(a)) if len(a) else np.nan,
                _newey_west_tstat(a))

    rows = []
    for name, arr in [("theme (alone)", coefs_theme_only),
                      ("momentum (alone)", coefs_mom_only),
                      ("theme (joint)", coefs_joint_t),
                      ("momentum (joint)", coefs_joint_m)]:
        mean, tstat = summarize(arr)
        rows.append({"regressor": name,
                     "avg_coef": round(mean, 5) if mean == mean else np.nan,
                     "nw_tstat": round(tstat, 2) if tstat == tstat else np.nan,
                     "n_weeks": len(arr)})
    return pd.DataFrame(rows)


# -------------------- 5. Test C: spanning regression --------------------

def spanning_regression(theme_ls_return: pd.Series,
                        factors: pd.DataFrame,
                        mom_col: str = "MOM") -> dict:
    """
    Regress the theme long-short portfolio return on the momentum factor
    (and any other factor columns provided, e.g. FF5). A significant intercept
    (alpha) means the theme portfolio is NOT spanned by momentum + controls.

    theme_ls_return : T-series, top-minus-bottom theme-score portfolio return
    factors         : T x K factor returns (must include mom_col)
    """
    common = theme_ls_return.index.intersection(factors.index)
    y = theme_ls_return.loc[common].values
    F = factors.loc[common]

    use_cols = [c for c in F.columns if c != "RF"]
    X = np.column_stack([np.ones(len(common)), F[use_cols].values])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    resid = y - X @ beta
    T, k = X.shape
    sigma2 = (resid @ resid) / (T - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    tstats = beta / se

    names = ["alpha"] + use_cols
    out = {n: {"coef": round(float(b), 5), "tstat": round(float(t), 2)}
           for n, b, t in zip(names, beta, tstats)}
    out["alpha_ann"] = round(float(beta[0] * 52), 4)   # weekly -> annual
    out["r_squared"] = round(float(1 - resid.var() / y.var()), 3)
    return out


def build_theme_ls_return(theme_score: pd.Series,
                          returns: pd.DataFrame,
                          top_frac: float = 0.2) -> pd.Series:
    """
    Build a top-minus-bottom theme-score long-short weekly return series.
    Long the top `top_frac` of stocks by theme score, short the bottom,
    equal-weighted, rebalanced implicitly (static membership by score).
    """
    common = theme_score.index.intersection(returns.columns)
    ts = theme_score.loc[common].dropna()
    n = len(ts)
    k = max(1, int(n * top_frac))
    longs  = ts.nlargest(k).index
    shorts = ts.nsmallest(k).index
    long_ret  = returns[longs].mean(axis=1)
    short_ret = returns[shorts].mean(axis=1)
    return (long_ret - short_ret).dropna()


# -------------------- 6. Orchestrator --------------------

def run_momentum_battery(scores: pd.DataFrame,
                         tgt_returns: pd.DataFrame,
                         factors: pd.DataFrame,
                         horizon_weeks: int = 4,
                         verbose: bool = True) -> dict:
    """
    Runs the full battery (A double sort, B Fama-MacBeth, C spanning) for every
    theme, plus a pooled summary. Returns a dict of results per theme.

    scores      : target stocks x themes (cosine similarity), from score_universe
    tgt_returns : weeks x target stocks
    factors     : weeks x factors (must include 'MOM' for spanning test)
    """
    momentum = compute_momentum(tgt_returns)
    fwd      = compute_forward_returns(tgt_returns, horizon_weeks=horizon_weeks)

    results = {}
    for theme in scores.columns:
        ts = scores[theme].dropna()
        if len(ts) < 50:
            continue

        A = double_sort(ts, momentum, fwd)
        B = fama_macbeth(ts, momentum, fwd)
        ls = build_theme_ls_return(ts, tgt_returns)
        C = spanning_regression(ls, factors) if "MOM" in factors.columns else None

        results[theme] = {"double_sort": A, "fama_macbeth": B, "spanning": C}

        if verbose:
            theme_joint = B[B["regressor"] == "theme (joint)"].iloc[0]
            print(f"\n{'='*60}\n{theme}\n{'='*60}")
            print(f"  A. theme spread within momentum buckets: "
                  f"{A['theme_spread_overall']*100:+.2f}% (per {horizon_weeks}wk)")
            print(f"  B. theme coef (joint w/ momentum): {theme_joint['avg_coef']:+.5f} "
                  f"(NW t = {theme_joint['nw_tstat']})")
            if C:
                print(f"  C. spanning alpha (ann): {C['alpha_ann']*100:+.2f}%  "
                      f"(t = {C['alpha']['tstat']}), "
                      f"momentum beta = {C.get('MOM', {}).get('coef', 'n/a')}")

    return results


def summarize_battery(results: dict, horizon_weeks: int = 4) -> pd.DataFrame:
    """Compact cross-theme summary table of the three tests."""
    rows = []
    for theme, r in results.items():
        A = r["double_sort"]
        Bdf = r["fama_macbeth"]
        theme_joint = Bdf[Bdf["regressor"] == "theme (joint)"].iloc[0]
        theme_alone = Bdf[Bdf["regressor"] == "theme (alone)"].iloc[0]
        C = r["spanning"]
        rows.append({
            "theme": theme,
            "A_theme_spread_in_mom": round(A["theme_spread_overall"] * 100, 2),
            "B_theme_coef_alone":    theme_alone["avg_coef"],
            "B_theme_t_alone":       theme_alone["nw_tstat"],
            "B_theme_coef_joint":    theme_joint["avg_coef"],
            "B_theme_t_joint":       theme_joint["nw_tstat"],
            "C_alpha_ann_%":         round(C["alpha_ann"] * 100, 2) if C else np.nan,
            "C_alpha_t":             C["alpha"]["tstat"] if C else np.nan,
            "C_mom_beta":            C.get("MOM", {}).get("coef", np.nan) if C else np.nan,
        })
    return pd.DataFrame(rows)


# -------------------- 7. Horizon robustness --------------------

def horizon_robustness(scores: pd.DataFrame,
                       tgt_returns: pd.DataFrame,
                       factors: pd.DataFrame,
                       horizons: list = None) -> pd.DataFrame:
    """
    Re-runs the momentum battery across multiple forward-return horizons to
    confirm that a theme's momentum-independence is not an artifact of one
    horizon choice. Momentum effects are horizon-sensitive, so a result that
    holds at 1, 4, and 12 weeks is far more defensible than one at 4 weeks alone.

    For each horizon and theme we record the key statistic from each test:
      A: theme spread within momentum buckets (%)
      B: joint Fama-MacBeth theme t-stat (the decisive number)
      C: spanning alpha t-stat

    Returns a tidy DataFrame: one row per (theme, horizon).
    """
    if horizons is None:
        horizons = [1, 4, 12]

    momentum = compute_momentum(tgt_returns)   # momentum signal is horizon-independent
    rows = []

    for h in horizons:
        fwd = compute_forward_returns(tgt_returns, horizon_weeks=h)
        for theme in scores.columns:
            ts = scores[theme].dropna()
            if len(ts) < 50:
                continue
            A = double_sort(ts, momentum, fwd)
            B = fama_macbeth(ts, momentum, fwd)
            theme_joint = B[B["regressor"] == "theme (joint)"].iloc[0]
            ls = build_theme_ls_return(ts, tgt_returns)
            C = spanning_regression(ls, factors) if "MOM" in factors.columns else None

            rows.append({
                "theme": theme,
                "horizon_wk": h,
                "A_spread_%": round(A["theme_spread_overall"] * 100, 2),
                "B_joint_t": theme_joint["nw_tstat"],
                "C_alpha_t": C["alpha"]["tstat"] if C else np.nan,
            })

    result = pd.DataFrame(rows)
    return result.sort_values(["theme", "horizon_wk"]).reset_index(drop=True)


def robustness_verdict(robust_df: pd.DataFrame, t_threshold: float = 2.0) -> pd.DataFrame:
    """
    Summarises horizon robustness into a per-theme verdict: does the theme's
    momentum-independence (joint Fama-MacBeth t-stat) hold across ALL horizons,
    SOME, or NONE.
    """
    rows = []
    for theme, g in robust_df.groupby("theme"):
        n_sig = (g["B_joint_t"].abs() >= t_threshold).sum()
        n_tot = len(g)
        n_sig_pos = ((g["B_joint_t"] >= t_threshold)).sum()
        if n_sig_pos == n_tot:
            verdict = "ROBUST (all horizons)"
        elif n_sig_pos >= 1:
            verdict = "PARTIAL (some horizons)"
        elif (g["B_joint_t"] <= -t_threshold).any():
            verdict = "NEGATIVE (momentum-contaminated)"
        else:
            verdict = "NULL (no independent signal)"
        rows.append({
            "theme": theme,
            "n_horizons_significant": f"{n_sig_pos}/{n_tot}",
            "min_joint_t": round(g["B_joint_t"].min(), 2),
            "max_joint_t": round(g["B_joint_t"].max(), 2),
            "verdict": verdict,
        })
    return pd.DataFrame(rows)
