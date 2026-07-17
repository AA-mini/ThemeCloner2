# theme_dna.py
#
# Extracts theme factor loadings from the RP-PCA covariance universe,
# then labels which factors correspond to which themes using ETF returns.
#
# Architecture (residualized version):
#   1. Returns in the COVARIANCE UNIVERSE are residualized against FF5+momentum
#      -- removes market, size, value, profitability, investment, momentum
#   2. RP-PCA runs on the RESIDUALS (not raw returns)
#      -- whatever co-movement survives cannot be explained by standard factors
#   3. ETF returns are ALSO residualized against the same FF5+momentum
#      -- ensures the regression in step 4 is in matched coordinates
#   4. Each residualized ETF return is regressed on the K RP-PCA factors
#      -- the factor(s) that best explain the residualized ETF = theme factor(s)
#   5. Per theme, the fingerprint is the centroid of ETF factor weights
#      across all ETFs in that theme (ensemble purification)
#
# PAPER POSITIONING -- "isnt thematic just momentum?":
#   By residualizing against momentum BEFORE looking for thematic structure,
#   we directly address the common industry critique that thematic investing
#   is just momentum with a story. Any theme factor that survives is by
#   construction orthogonal to momentum -- it captures shared cross-sectional
#   co-movement that the momentum factor cannot explain.
#
# PAPER NOTE:
#   R² of the ETF regression on the RESIDUALIZED factor space will be LOWER
#   than on raw factors (typically 0.30-0.50 vs 0.70-0.90). This is intentional:
#   we have stripped out the market beta and growth co-movement that inflated
#   the raw R². What remains is the genuinely thematic signal.

import numpy as np
import pandas as pd
from src.rppca import fit_rppca


# -------------------- 1. Fit RP-PCA on covariance universe --------------------

def fit_covariance_universe(cov_returns: pd.DataFrame,
                             K: int = 10,
                             gamma: float = 10.0) -> dict:
    """
    Runs RP-PCA on the broad covariance universe.

    IMPORTANT: cov_returns should already be RESIDUALIZED against FF5+momentum
    before being passed in. See residualize.py for the residualization step.
    Running RP-PCA on raw returns picks up market beta and momentum as
    "themes" (the TWST-everywhere problem) -- residualization isolates
    the genuinely thematic co-movement.

    K=10 is the default; the eigenvalue scree plot will show how many
    factors are genuinely above the noise floor.

    Returns the full RP-PCA result dict from rppca.fit_rppca().
    """
    print(f"\nfitting RP-PCA on covariance universe: "
          f"{cov_returns.shape[1]} stocks x {cov_returns.shape[0]} weeks")
    print(f"  K={K}, gamma={gamma}")
    print(f"  this may take a minute with a large cross-section...")

    result = fit_rppca(cov_returns, K=K, gamma=gamma,
                        run_oos=True, annualise=52.0)
    return result


# -------------------- 2. Label theme factors using ETF returns --------------------

def label_theme_factors(rppca_result: dict,
                          etf_returns: pd.DataFrame,
                          etf_config: pd.DataFrame,
                          top_factors: int = 3,
                          min_etf_r2: float = 0.0) -> dict:
    """
    Identifies which RP-PCA factors correspond to which themes by regressing
    each ETF's return series on the extracted factor time series.

    For each ETF: R² tells us how much of the ETF's variance is explained
    by the factor space. The factor(s) with highest regression coefficient
    (in absolute terms) are the most thematic for that ETF.

    For each theme: average the factor weight vectors across all ETFs in
    the theme -- this is the ensemble purification step that removes
    individual ETF construction biases.

    Parameters
    ----------
    rppca_result  : output from fit_covariance_universe()
    etf_returns   : weekly ETF return series (Bloomberg ticker columns)
    etf_config    : etfs.csv dataframe with ticker/theme mapping
    top_factors   : how many factors per ETF to consider when building
                    the theme fingerprint (default 3 -- avoids noise from
                    factors with near-zero coefficients)
    min_etf_r2    : V2 ADDITION. ETFs whose own R² against the K-factor
                    model falls below this threshold are excluded from the
                    fingerprint centroid entirely -- a weak, noisy ETF fit
                    (one the factor model barely explains) should not get
                    equal say in defining the theme alongside a clean one.
                    Default 0.0 reproduces V1 behaviour (no ETF screening).
                    A theme left with zero surviving ETFs after this filter
                    is skipped with a warning, same as the existing
                    no-ETFs-available case.

    Returns
    -------
    dict with:
        theme_factors : {theme -> K-vector} -- theme fingerprint in factor space
        etf_r2        : {etf_ticker -> float} -- R² of each ETF on all factors
        etf_weights   : {etf_ticker -> K-vector} -- regression weights per ETF
        purity        : {theme -> float} -- cosine similarity across ETFs in theme
        etfs_dropped  : {theme -> [tickers]} -- ETFs excluded by min_etf_r2, for
                        transparency (V2 addition)
    """

    factors_df = rppca_result["factors"]   # T x K factor time series

    # align dates
    common = etf_returns.index.intersection(factors_df.index)
    if len(common) < 52:
        raise ValueError(f"only {len(common)} common weeks between ETFs and "
                          f"covariance universe factors -- check date alignment")

    F = factors_df.loc[common].values          # T x K
    E = etf_returns.loc[common].fillna(0)      # T x N_etfs

    K = F.shape[1]
    FtF_inv = np.linalg.inv(F.T @ F + 1e-8 * np.eye(K))

    etf_weights = {}
    etf_r2      = {}

    print(f"\nlabeling theme factors using {E.shape[1]} ETFs:")
    print(f"  {'ETF':20s}  {'R²':>6s}  {'top factor weights'}")
    print(f"  {'-'*60}")

    for etf in E.columns:
        e = E[etf].values                          # T-vector
        w = FtF_inv @ F.T @ e                      # K-vector of regression weights
        e_hat = F @ w
        ss_res = np.sum((e - e_hat) ** 2)
        ss_tot = np.sum((e - e.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        etf_weights[etf] = w
        etf_r2[etf]      = float(r2)

        # show top 3 factor weights for inspection
        top_idx = np.argsort(np.abs(w))[::-1][:3]
        top_str = ", ".join([f"f{i+1}:{w[i]:+.3f}" for i in top_idx])
        print(f"  {etf:20s}  {r2:6.3f}  {top_str}")

    # -------------------- build theme fingerprint per theme --------------------
    # for each ETF: zero out all but the top_factors loadings
    # (avoids blending in noise from weakly-loaded factors)
    # then average across ETFs in the same theme

    theme_factors = {}
    purity        = {}
    etfs_dropped  = {}
    themes        = etf_config["theme"].unique()

    print(f"\nbuilding theme fingerprints:")
    for theme in themes:
        theme_etfs = etf_config[etf_config["theme"] == theme]["ticker"].tolist()
        available  = [e for e in theme_etfs if e in etf_weights]

        if not available:
            print(f"  WARNING: '{theme}' has no ETFs with factor weights -- skipping")
            continue

        # V2: screen out weak-fitting ETFs before they enter the fingerprint
        weak = [e for e in available if etf_r2[e] < min_etf_r2]
        strong = [e for e in available if etf_r2[e] >= min_etf_r2]
        etfs_dropped[theme] = weak
        if weak:
            print(f"  {theme}: dropping {len(weak)} weak ETF(s) below min_etf_r2="
                  f"{min_etf_r2:.2f}: {[(e, round(etf_r2[e],3)) for e in weak]}")
        if not strong:
            print(f"  WARNING: '{theme}' has no ETFs surviving min_etf_r2={min_etf_r2:.2f} -- skipping")
            continue
        available = strong

        # sparse weight vectors: keep only top_factors non-zero entries
        sparse_weights = []
        for etf in available:
            w     = etf_weights[etf].copy()
            top_k = np.argsort(np.abs(w))[::-1][:top_factors]
            mask  = np.zeros(K)
            mask[top_k] = 1
            sparse_weights.append(w * mask)

        fp_matrix = np.stack(sparse_weights)   # (n_etfs x K)
        centroid  = fp_matrix.mean(axis=0)     # K-vector

        # purity: mean cosine similarity of each ETF fingerprint to centroid
        sims = []
        for fp in fp_matrix:
            n1, n2 = np.linalg.norm(fp), np.linalg.norm(centroid)
            sims.append(float(np.dot(fp, centroid) / (n1 * n2))
                         if n1 > 0 and n2 > 0 else 0.0)

        theme_factors[theme] = centroid
        purity[theme]        = float(np.mean(sims))

        avg_r2 = np.mean([etf_r2[e] for e in available])
        print(f"  {theme}: {len(available)} ETFs, "
              f"avg R²={avg_r2:.3f}, purity={purity[theme]:.3f}")

    return {
        "theme_factors": theme_factors,
        "etf_r2":        etf_r2,
        "etf_weights":   etf_weights,
        "purity":        purity,
        "etfs_dropped":  etfs_dropped,
    }


# -------------------- 3. Convenience wrapper --------------------

def build_theme_dna(cov_returns: pd.DataFrame,
                     etf_returns: pd.DataFrame,
                     etf_config: pd.DataFrame,
                     K: int = 10,
                     gamma: float = 10.0,
                     top_factors: int = 3) -> dict:
    """
    Full pipeline from raw returns to theme fingerprints.
    Called by the master notebook.

    Returns a dict with:
        rppca_result  : full RP-PCA output on covariance universe
        theme_factors : {theme -> K-vector}
        etf_r2        : {etf -> R²}
        purity        : {theme -> purity score}
    """
    rppca_result = fit_covariance_universe(cov_returns, K=K, gamma=gamma)
    dna_result   = label_theme_factors(rppca_result, etf_returns,
                                        etf_config, top_factors=top_factors)
    return {**dna_result, "rppca_result": rppca_result}


# -------------------- 4. Factor interpretation diagnostic --------------------

def interpret_factors(rppca_result: dict,
                       etf_returns_resid: pd.DataFrame,
                       etf_config: pd.DataFrame,
                       cov_returns_raw: pd.DataFrame = None) -> pd.DataFrame:
    """
    Answers the question: "what does each RP-PCA factor actually represent?"

    For each factor (PC1, PC2, ...), regresses every theme's ETF returns on
    that single factor and reports the R². This tells us whether a factor
    is thematically interpretable (high R² for one theme, low for others)
    or diffuse (similar low R² across all themes = not a clean theme axis).

    This is the diagnostic that tests the regime hypothesis:
      - If PC1 has high R² for AI themes and low for everything else,
        the dominant axis IS the AI regime
      - If PC2, PC3... each light up a distinct theme, the secondary axes
        ARE thematically interpretable (fintech, defense etc.)
      - If PC2+ show similar diffuse R² across all themes, the secondary
        structure is statistical noise, not clean themes

    Parameters
    ----------
    rppca_result      : output from fit_rppca on residualized covariance universe
    etf_returns_resid : RESIDUALIZED ETF returns (same residualization as universe)
    etf_config        : etfs.csv mapping
    cov_returns_raw   : optional, unused here but kept for signature stability

    Returns
    -------
    DataFrame indexed by theme, columns = factor_1..factor_K, values = R²
    of that theme's average ETF return on that single factor.
    """

    factors_df = rppca_result["factors"]   # T x K
    K          = factors_df.shape[1]

    # align dates
    common = etf_returns_resid.index.intersection(factors_df.index)
    F = factors_df.loc[common]
    E = etf_returns_resid.loc[common]

    themes = etf_config["theme"].unique()
    rows   = {}

    for theme in themes:
        theme_etfs = etf_config[etf_config["theme"] == theme]["ticker"].tolist()
        avail      = [e for e in theme_etfs if e in E.columns]
        if not avail:
            continue

        # average ETF return for the theme
        theme_ret = E[avail].mean(axis=1).values

        # regress theme return on EACH factor individually, record R²
        r2_per_factor = []
        for k in range(K):
            f = F.iloc[:, k].values
            # single-factor R² = correlation^2
            corr = np.corrcoef(theme_ret, f)[0, 1]
            r2_per_factor.append(corr ** 2)

        rows[theme] = r2_per_factor

    result = pd.DataFrame(rows,
                          index=[f"factor_{k+1}" for k in range(K)]).T

    # add a column showing which factor each theme loads on most
    result["dominant_factor"] = result.idxmax(axis=1)
    result["max_r2"]          = result.drop(columns="dominant_factor").max(axis=1)

    return result


# -------------------- 5. Label factors by top-loading stocks --------------------

def label_factors_by_stocks(rppca_result: dict,
                             top_n: int = 12) -> dict:
    """
    For each RP-PCA factor, returns the stocks with the largest absolute
    loadings -- both the positive tail and negative tail separately.

    This is how we interpret what a factor actually represents. A factor
    whose top-loading stocks are all semiconductor names IS a semiconductor
    factor, regardless of what the eigenvalue ordering says.

    Pairs with interpret_factors(): that tells us which factor a theme loads
    on; this tells us what that factor actually contains. Together they
    confirm (or refute) that the framework found genuine economic structure.

    Returns
    -------
    dict {factor_name -> {"positive": [(ticker, loading), ...],
                          "negative": [(ticker, loading), ...]}}
    """
    loadings = rppca_result["loadings"]   # stocks x K

    result = {}
    for factor in loadings.columns:
        col = loadings[factor].sort_values(ascending=False)
        positive = [(t, round(v, 4)) for t, v in col.head(top_n).items()]
        negative = [(t, round(v, 4)) for t, v in col.tail(top_n).items()][::-1]
        result[factor] = {"positive": positive, "negative": negative}

    return result


def print_factor_labels(factor_labels: dict, top_n: int = 10):
    """Pretty-prints the top-loading stocks for each factor."""
    for factor, sides in factor_labels.items():
        print(f"\n{'='*55}")
        print(f"{factor}")
        print(f"{'='*55}")
        pos = ", ".join(f"{t}({v:+.3f})" for t, v in sides["positive"][:top_n])
        neg = ", ".join(f"{t}({v:+.3f})" for t, v in sides["negative"][:top_n])
        print(f"  + tail: {pos}")
        print(f"  - tail: {neg}")


# -------------------- 6. Gamma sweep on theme detectability --------------------

def sweep_gamma_theme_detection(cov_residuals: pd.DataFrame,
                                 etf_residuals: pd.DataFrame,
                                 etf_config: pd.DataFrame,
                                 gammas: list = None,
                                 K: int = 10) -> pd.DataFrame:
    """
    The standard gamma sweep maximises factor Sharpe ratio. But on residualized
    returns that's near-zero and uninformative (we stripped the priced factors).

    What we actually care about: does raising gamma make THEMES more detectable?
    RP-PCA's gamma penalty is designed to surface weak high-Sharpe factors --
    and our themes ARE weak factors (max R² ~0.09). So higher gamma might pull
    them up and make them more separable.

    This sweep measures, for each gamma, the average best-factor R² across themes
    (i.e. how strongly the most-aligned factor explains each theme). If theme
    detectability rises with gamma, that's evidence the themes are exactly the
    weak factors RP-PCA was built to find.

    Returns DataFrame: gamma, mean_best_r2, max_best_r2, per-theme columns.
    """
    from src.rppca import fit_rppca

    if gammas is None:
        gammas = [-1, 0, 5, 10, 20, 50, 100]

    themes = etf_config["theme"].unique()
    rows   = []

    for g in gammas:
        # fit RP-PCA at this gamma
        res = fit_rppca(cov_residuals, K=K, gamma=g, run_oos=False)
        interp = interpret_factors(res, etf_residuals, etf_config)

        best_r2 = interp["max_r2"]   # best factor R² per theme
        row = {"gamma": g,
               "mean_best_r2": round(best_r2.mean(), 4),
               "max_best_r2":  round(best_r2.max(), 4),
               "min_best_r2":  round(best_r2.min(), 4)}
        rows.append(row)
        print(f"  gamma={g:>5}: mean theme R²={row['mean_best_r2']:.4f}, "
              f"best={row['max_best_r2']:.4f}, worst={row['min_best_r2']:.4f}")

    return pd.DataFrame(rows)
