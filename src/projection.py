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
