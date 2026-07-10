# projection.py
#
# Projects target universe stocks into the covariance universe factor space
# and scores them against theme fingerprints.
#
# Architecture:
#   - Factor space is fixed by RP-PCA on the covariance universe (rppca_result)
#   - Each target stock is projected into that space via OLS regression
#     of its return series on the K factor time series
#   - Cosine similarity between each stock's factor coordinates and each
#     theme's fingerprint gives the thematic exposure score
#
# Critically: the covariance universe and target universe are DIFFERENT.
#   Covariance universe = broad (ACWI proxy) -- defines the factor space
#   Target universe     = small caps (Russell proxy) -- what we score
#
# A stock scores high if its residual return structure resembles the theme
# factor's direction in the shared latent space -- even if the market has
# not yet labeled it as thematic.

import numpy as np
import pandas as pd
import os


# -------------------- 1. Project target universe --------------------

def project_universe(target_returns: pd.DataFrame,
                      rppca_result: dict) -> pd.DataFrame:
    """
    Projects each stock in the target universe into the covariance factor space.

    Method: OLS regression of each stock's weekly returns on the K factor
    time series extracted from the covariance universe. The regression betas
    (K-vector per stock) are its coordinates in the factor space.

    This is valid even though the stocks weren't in the RP-PCA estimation --
    we're asking "how much does this stock co-move with each latent factor?"
    which is just a time-series regression.

    Returns
    -------
    projections : pd.DataFrame (target stocks x K factors)
    """

    factors_df = rppca_result["factors"]   # T x K

    # align on common dates
    common = target_returns.index.intersection(factors_df.index)
    if len(common) < 52:
        raise ValueError(f"only {len(common)} common weeks -- check date alignment")

    print(f"\nprojecting {target_returns.shape[1]} target stocks into factor space")
    print(f"  common dates: {len(common)} weeks")

    F = factors_df.loc[common].values                    # T x K
    X = target_returns.loc[common].fillna(0).values      # T x N_target

    # OLS: betas = (F'F)^{-1} F'X   shape: K x N_target
    K       = F.shape[1]
    FtF_inv = np.linalg.inv(F.T @ F + 1e-8 * np.eye(K))
    betas   = FtF_inv @ F.T @ X                         # K x N_target

    projections = pd.DataFrame(
        betas.T,
        index   = target_returns.columns,
        columns = rppca_result["factors"].columns
    )

    print(f"  projections: {projections.shape[0]} stocks x {projections.shape[1]} factors")
    return projections


# -------------------- 2. Score against theme fingerprints --------------------

def score_universe(projections: pd.DataFrame,
                    theme_factors: dict) -> pd.DataFrame:
    """
    Computes cosine similarity between each stock's factor coordinates
    and each theme's fingerprint vector.

    Score interpretation:
      +1.0  = stock loads identically to the theme (pure thematic)
       0.0  = orthogonal to the theme (no exposure)
      -1.0  = opposite direction (counter-thematic)

    In practice, genuine thematic small caps score 0.4-0.8.
    Scores above 0.8 may indicate the stock is already well-known as thematic.

    Returns
    -------
    scores : pd.DataFrame (target stocks x themes)
    """

    themes = list(theme_factors.keys())
    scores = pd.DataFrame(index=projections.index, columns=themes, dtype=float)

    proj_norms = np.linalg.norm(projections.values, axis=1)
    proj_norms = np.where(proj_norms == 0, 1e-10, proj_norms)

    for theme, dna in theme_factors.items():
        dna_norm = np.linalg.norm(dna)
        if dna_norm == 0:
            print(f"  WARNING: zero-norm fingerprint for '{theme}' -- skipping")
            scores[theme] = np.nan
            continue

        dots           = projections.values @ dna
        scores[theme]  = dots / (proj_norms * dna_norm)

    print(f"\nscores: {scores.shape[0]} stocks x {scores.shape[1]} themes")
    print(f"  range: [{scores.min().min():.3f}, {scores.max().max():.3f}]")
    return scores


# -------------------- 3. OOS validation: do picks co-move with ETF? --------------------

def validate_against_etf(scores: pd.DataFrame,
                           target_returns: pd.DataFrame,
                           etf_returns: pd.DataFrame,
                           etf_config: pd.DataFrame,
                           top_n: int = 30,
                           min_score: float = 0.4) -> pd.DataFrame:
    """
    Validates candidate stocks by checking whether their equal-weighted
    return correlates with the theme ETF going forward.

    This is the clean OOS test that doesn't require constituent holdings:
    we built our candidates using return co-movement structure (RP-PCA),
    and we validate using a completely separate signal (ETF correlation).

    For each theme:
      - Take top_n scoring stocks as candidates
      - Compute equal-weighted candidate portfolio return
      - Correlate with the theme's ETF returns
      - High correlation = candidates genuinely track the theme

    PAPER NOTE: this is the primary validation metric. A high R² between
    the candidate portfolio and the ETF (without the candidates being in
    the ETF) is evidence that RP-PCA found genuine thematic exposure.

    Returns a summary DataFrame with correlation per theme.
    """

    from src.scoring import rank_candidates

    # get candidates per theme
    ranked = rank_candidates(scores, top_n=top_n, min_score=min_score)

    results = []
    for theme in scores.columns:
        theme_candidates = ranked[ranked["theme"] == theme]["ticker"].tolist()
        if not theme_candidates:
            continue

        # equal-weighted candidate portfolio
        cand_avail = [t for t in theme_candidates if t in target_returns.columns]
        if not cand_avail:
            continue
        portfolio_ret = target_returns[cand_avail].mean(axis=1)

        # ETFs for this theme
        theme_etfs = etf_config[etf_config["theme"] == theme]["ticker"].tolist()
        etf_avail  = [e for e in theme_etfs if e in etf_returns.columns]
        if not etf_avail:
            continue
        etf_ret = etf_returns[etf_avail].mean(axis=1)

        # align and correlate
        common = portfolio_ret.index.intersection(etf_ret.index)
        if len(common) < 12:
            continue

        corr = float(portfolio_ret.loc[common].corr(etf_ret.loc[common]))

        # -------------------- beta decomposition --------------------
        # correlation says they move TOGETHER; beta says by how MUCH.
        # regress candidate portfolio return on ETF return:
        #   beta < 1  -> candidate is lower-amplitude than the ETF (the usual case:
        #                ETF is cap-weighted toward mega-cap winners the small-cap
        #                basket structurally excludes, plus idiosyncratic/survivorship
        #                drag in the equal-weighted small-cap cross-section)
        #   alpha     -> annualised intercept: does the basket out/underperform the
        #                ETF after accounting for its beta exposure?
        p = portfolio_ret.loc[common].values
        e = etf_ret.loc[common].values
        var_e = np.var(e)
        beta  = float(np.cov(p, e)[0, 1] / var_e) if var_e > 0 else np.nan
        alpha_weekly = float(p.mean() - beta * e.mean())
        alpha_ann    = alpha_weekly * 52   # annualised intercept

        # cumulative return gap (level difference over the window)
        cum_port = float(np.prod(1 + p) - 1)
        cum_etf  = float(np.prod(1 + e) - 1)

        # tracking error / tracking difference (weekly diff, annualised)
        diff = p - e
        tracking_error = float(diff.std() * np.sqrt(52))
        tracking_diff  = float(diff.mean() * 52)

        results.append({
            "theme":           theme,
            "n_candidates":    len(cand_avail),
            "etf_correlation": round(corr, 3),
            "beta_to_etf":     round(beta, 3),
            "alpha_ann":       round(alpha_ann, 3),
            "tracking_error":  round(tracking_error, 3),
            "tracking_diff":   round(tracking_diff, 3),
            "cum_ret_basket":  round(cum_port, 3),
            "cum_ret_etf":     round(cum_etf, 3),
            "candidates":      cand_avail,
        })

    validation = pd.DataFrame(results)
    if not validation.empty:
        print("\nOOS validation (correlation + beta decomposition):")
        cols = ["theme", "n_candidates", "etf_correlation", "beta_to_etf",
                "alpha_ann", "tracking_error", "tracking_diff", "cum_ret_basket", "cum_ret_etf"]
        print(validation[cols].to_string(index=False))
        print("\nreading the table:")
        print("  high corr + beta<1  = basket tracks the theme but at lower amplitude")
        print("                        (mega-cap exclusion + small-cap idio/survivorship drag)")
        print("  alpha_ann           = basket return net of its beta-scaled ETF exposure")
    return validation


# -------------------- V2 NEW: synthetic-factor-return scoring --------------------
#
# WHY THIS EXISTS:
# score_universe() above (V1, unchanged) uses cosine similarity between a
# stock's K-vector of factor loadings and the theme fingerprint. Cosine
# similarity normalizes both vectors to unit length before comparing --
# by construction it captures DIRECTION only. A stock whose factor exposure
# is 90% idiosyncratic noise and 10% genuine theme signal can score identically
# to a stock that is cleanly, mostly explained by the theme's factors, as long
# as the *direction* of whatever signal exists happens to match. This was
# diagnosed as a likely driver of negative realized basket alpha in V1
# (mean candidate R² against the K-factor model was only ~0.06-0.08).
#
# THE FIX: instead of comparing loading VECTORS, build the theme's synthetic
# factor-implied RETURN SERIES (the fingerprint-weighted combination of the
# K factor return time series), then correlate each candidate's ACTUAL return
# series against that synthetic series directly. This one number captures:
#   - direction (sign of the correlation, same job cosine similarity did)
#   - magnitude (a stock that is mostly idiosyncratic noise will show a LOW
#     correlation to the synthetic series even if its residual loading
#     direction matched, because the noise dilutes the correlation -- this
#     is exactly the information cosine similarity discards)
# This replaces cosine similarity; it does not need to be combined with a
# separate R²-weighting step the way V1's post-hoc fix did, because magnitude
# is now part of the score itself, not bolted on afterward.

def synthetic_theme_returns(rppca_result: dict, theme_factors: dict) -> pd.DataFrame:
    """
    Builds each theme's synthetic factor-implied return series: the K factor
    return time series, combined using the theme's fingerprint as weights.

    synthetic_return_t = sum_k( fingerprint[k] * factor_return[k, t] )

    Returns
    -------
    pd.DataFrame, T x n_themes, one synthetic return column per theme.
    """
    factors_df = rppca_result["factors"]   # T x K
    out = {}
    for theme, fp in theme_factors.items():
        fp = np.asarray(fp)
        out[theme] = factors_df.values @ fp
    return pd.DataFrame(out, index=factors_df.index)


def score_universe_v2(target_returns: pd.DataFrame,
                       synthetic_returns: pd.DataFrame) -> pd.DataFrame:
    """
    V2 scoring: correlation of each target-universe stock's actual return
    series with each theme's synthetic factor-implied return series.

    This is the direct replacement for score_universe()'s cosine similarity.
    Same output shape (stocks x themes) so every downstream V1 function that
    consumes a "scores" DataFrame (rank_candidates, validate_against_etf)
    works unchanged against V2 scores.

    Returns
    -------
    scores : pd.DataFrame (target stocks x themes), values in [-1, 1]
    """
    common = target_returns.index.intersection(synthetic_returns.index)
    if len(common) < 52:
        raise ValueError(f"only {len(common)} common weeks -- check date alignment")

    R = target_returns.loc[common]
    S = synthetic_returns.loc[common]

    scores = pd.DataFrame(index=R.columns, columns=S.columns, dtype=float)
    for theme in S.columns:
        s = S[theme].values
        s_centered = s - s.mean()
        s_norm = np.linalg.norm(s_centered)
        if s_norm == 0:
            scores[theme] = np.nan
            continue
        for stock in R.columns:
            r = R[stock].fillna(0).values
            r_centered = r - r.mean()
            r_norm = np.linalg.norm(r_centered)
            if r_norm == 0:
                scores.loc[stock, theme] = 0.0
                continue
            scores.loc[stock, theme] = float(np.dot(r_centered, s_centered) / (r_norm * s_norm))

    print(f"\nV2 scores (synthetic-factor-return correlation): "
          f"{scores.shape[0]} stocks x {scores.shape[1]} themes")
    print(f"  range: [{scores.min().min():.3f}, {scores.max().max():.3f}]")
    return scores


# -------------------- V2 NEW: candidate-level null / placebo test --------------------
#
# WHY THIS EXISTS:
# Neither V1's rank_candidates() nor V2's score_universe_v2() has a "there is
# nothing here" detector -- both will always return a top-N regardless of
# whether the target universe contains any genuine thematic exposure at all.
# This directly tests that: for each theme, compares the actual top-N
# candidates' scores against a null distribution built from randomly drawn
# stocks. If the real top-N isn't clearly above where random stocks land,
# the "discovery" for that theme is not distinguishable from noise.

def candidate_null_test(scores: pd.DataFrame,
                         top_n: int = 30,
                         n_null_draws: int = 500,
                         random_state: int = 0) -> pd.DataFrame:
    """
    For each theme, compares the mean score of the actual top_n candidates
    against the distribution of mean scores from n_null_draws random draws
    of top_n stocks from the same universe.

    Returns
    -------
    DataFrame: theme, actual_top_n_mean_score, null_p50, null_p95, null_p99,
               percentile_of_actual (where the real top-N sits in the null
               distribution -- >99 means the real discovery clearly beats
               chance; well below that is a genuine warning sign, not proof
               of failure, since a theme can be real but weak).
    """
    rng = np.random.default_rng(random_state)
    n_stocks = scores.shape[0]
    rows = []

    for theme in scores.columns:
        theme_scores = scores[theme].dropna()
        if len(theme_scores) < top_n:
            continue

        actual_top = theme_scores.sort_values(ascending=False).head(top_n).mean()

        null_means = []
        vals = theme_scores.values
        for _ in range(n_null_draws):
            draw = rng.choice(vals, size=top_n, replace=False)
            null_means.append(draw.mean())
        null_means = np.array(null_means)

        pct = float((null_means < actual_top).mean() * 100)

        rows.append({
            "theme": theme,
            "actual_top_n_mean_score": round(float(actual_top), 4),
            "null_p50": round(float(np.percentile(null_means, 50)), 4),
            "null_p95": round(float(np.percentile(null_means, 95)), 4),
            "null_p99": round(float(np.percentile(null_means, 99)), 4),
            "percentile_of_actual": round(pct, 1),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        print(f"\ncandidate-level null test (top_{top_n} mean score vs. {n_null_draws} random draws):")
        print(result.to_string(index=False))
        print("  percentile_of_actual near 100 = clearly beats chance")
        print("  percentile_of_actual well below 99 = cannot rule out noise for this theme")
    return result


# -------------------- 4. Save --------------------

def save_projections(projections: pd.DataFrame, scores: pd.DataFrame,
                      out_dir: str = None):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "data")
    os.makedirs(out_dir, exist_ok=True)

    for df, name in [(projections, "projections.csv"), (scores, "scores.csv")]:
        path = os.path.join(out_dir, name)
        df.to_csv(path)
        print(f"saved: {path}")
