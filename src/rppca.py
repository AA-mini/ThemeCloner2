# 02_rppca.py
# Core RP-PCA implementation -- ported from the working rppca_openAP.py
# in the original course project, so the math here is battle-tested.
#
# RP-PCA (Lettau & Pelger 2020) differs from standard PCA by adding a
# penalty term gamma * (cross-sectional mean)^2 to the objective. This
# tilts extracted factors toward high Sharpe ratio directions rather than
# pure variance. For thematic investing this matters: we want factors that
# are economically meaningful, not just statistically large.
#
# Key outputs:
#   loadings   -- N x K matrix, how much each asset loads on each factor
#   factors    -- T x K matrix, the factor return time series
#   gamma_opt  -- the gamma value that maximised OOS Sharpe (if sweep run)

import numpy as np
import pandas as pd
from numpy.linalg import eigh


# -------------------- 1. Core RP-PCA solver --------------------

def rppca(X: np.ndarray, K: int, gamma: float = 10.0):
    """
    Fits RP-PCA to a return matrix X.

    X      : T x N array of excess returns (time x assets)
    K      : number of factors to extract
    gamma  : penalty weight on the cross-sectional mean.
              gamma=0  -> standard PCA
              gamma=10 -> Lettau-Pelger default, good for most applications

    Returns
    -------
    loadings : N x K  (each column is one factor's asset loadings)
    factors  : T x K  (each column is one factor's return time series)
    eigvals  : K      (eigenvalues, largest first -- useful for scree plots)
    """

    T, N = X.shape

    # -------------------- build the RP-PCA objective matrix --------------------
    # standard PCA maximises X'X / T  (the sample covariance)
    # RP-PCA maximises  X'X / T  +  gamma * mu * mu'
    # where mu = X.mean(axis=0) is the vector of mean returns
    # adding gamma * mu*mu' up-weights directions with high risk premia

    mu  = X.mean(axis=0)                    # N-vector of mean returns
    cov = X.T @ X / T                       # N x N second moment matrix
    M   = cov + gamma * np.outer(mu, mu)    # RP-PCA objective matrix

    # -------------------- eigen-decomposition --------------------
    # eigh is faster and more stable than eig for symmetric matrices
    # returns eigenvalues in ascending order -- we reverse to get largest first
    eigvals_all, eigvecs_all = eigh(M)
    idx      = np.argsort(eigvals_all)[::-1]
    eigvals  = eigvals_all[idx][:K]
    loadings = eigvecs_all[:, idx][:, :K]   # N x K

    # -------------------- recover factor time series --------------------
    # factors are the projection of returns onto the loading vectors
    # result is T x K
    factors = X @ loadings

    return loadings, factors, eigvals


# -------------------- 2. In-sample Sharpe --------------------

def compute_sr(factors: np.ndarray, annualise: float = 52.0) -> float:
    """
    Computes the combined annualised Sharpe ratio of a set of factors
    using the maximum Sharpe ratio portfolio (tangency portfolio weights).

    annualise : 52 for weekly, 12 for monthly
    """
    means = factors.mean(axis=0)
    cov   = np.cov(factors.T)

    if factors.shape[1] == 1:
        sr = (means[0] / np.sqrt(cov)) * np.sqrt(annualise)
        return float(sr)

    # tangency portfolio SR = sqrt(mu' * Sigma^{-1} * mu)
    try:
        cov_inv = np.linalg.inv(cov + 1e-8 * np.eye(cov.shape[0]))
        sr_sq   = float(means @ cov_inv @ means) * annualise
        return np.sqrt(max(sr_sq, 0))
    except np.linalg.LinAlgError:
        return np.nan


# -------------------- 3. Out-of-sample rolling evaluation --------------------

def rppca_oos(X: np.ndarray, K: int, gamma: float = 10.0,
              train_frac: float = 0.6, annualise: float = 52.0) -> dict:
    """
    Evaluates RP-PCA out-of-sample using a simple expanding window.

    We estimate loadings on the first train_frac of data, then apply them
    to the remaining holdout period. This mimics how the model would be
    used live -- loadings are fixed from the training window.

    Returns a dict with IS and OOS Sharpe ratios for RP-PCA and PCA.
    """

    T = X.shape[0]
    split = int(T * train_frac)

    X_train = X[:split]
    X_test  = X[split:]

    # -------------------- fit on training data --------------------
    load_rp, _, _ = rppca(X_train, K=K, gamma=gamma)
    load_pc, _, _ = rppca(X_train, K=K, gamma=0.0)   # gamma=0 is standard PCA

    # -------------------- apply loadings to test data --------------------
    factors_rp_oos = X_test  @ load_rp
    factors_pc_oos = X_test  @ load_pc
    factors_rp_is  = X_train @ load_rp
    factors_pc_is  = X_train @ load_pc

    return {
        "rppca_is":  compute_sr(factors_rp_is,  annualise),
        "pca_is":    compute_sr(factors_pc_is,   annualise),
        "rppca_oos": compute_sr(factors_rp_oos,  annualise),
        "pca_oos":   compute_sr(factors_pc_oos,  annualise),
    }


# -------------------- 4. Gamma sweep --------------------

def sweep_gamma(X: np.ndarray, K: int, gammas: list = None,
                annualise: float = 52.0) -> pd.DataFrame:
    """
    Runs RP-PCA across a range of gamma values and records the in-sample SR.
    Used to pick the best gamma and to produce the SR-vs-gamma plot for the paper.

    This is a direct port of the gamma sweep logic from rppca_openAP.py
    which already worked cleanly -- no reason to change the approach.
    """

    if gammas is None:
        gammas = [-1, 0, 1, 2, 5, 10, 15, 20, 30, 50]

    rows = []
    for g in gammas:
        _, factors, _ = rppca(X, K=K, gamma=g)
        sr = compute_sr(factors, annualise)
        rows.append({"gamma": g, "K": K, "sr": sr})

    return pd.DataFrame(rows)


# -------------------- 5. Number-of-factors test (Onatski 2010) --------------------

def onatski_criterion(X: np.ndarray, gamma: float = 10.0,
                       K_max: int = 10, delta: float = 0.1) -> int:
    """
    Uses the Onatski (2010) eigenvalue difference test to select K.
    The idea: genuine systematic factors produce eigenvalues well-separated
    from each other and from the noise bulk. When consecutive differences
    fall below a threshold, we've hit the noise floor.

    This is the same approach used in the original replicate notebook --
    kept here because it produced sensible results there.
    """

    T, N = X.shape
    mu   = X.mean(axis=0)
    cov  = X.T @ X / T
    M    = cov + gamma * np.outer(mu, mu)

    eigvals = np.sort(np.linalg.eigvalsh(M))[::-1]
    diffs   = np.diff(eigvals[:K_max + 1])   # K_max consecutive differences
    diffs   = np.abs(diffs)

    # find where the drop becomes small relative to the first difference
    threshold = delta * diffs[0]
    K_selected = 1
    for i, d in enumerate(diffs):
        if d >= threshold:
            K_selected = i + 1

    return K_selected


# -------------------- 6. Fit and return full results --------------------

def fit_rppca(X: pd.DataFrame, K: int = 5, gamma: float = 10.0,
              run_oos: bool = True, annualise: float = 52.0) -> dict:
    """
    Convenience wrapper -- takes a pandas DataFrame, fits RP-PCA,
    and returns everything the downstream modules need.

    X         : T x N DataFrame of excess returns
    K         : number of factors
    gamma     : RP-PCA penalty (10 is the Lettau-Pelger default)
    run_oos   : whether to also compute OOS Sharpe stats

    Returns a dict with:
        loadings     -- pd.DataFrame (assets x factors)
        factors      -- pd.DataFrame (dates x factors)
        eigvals      -- np.array
        sr_is        -- in-sample Sharpe
        sr_oos       -- OOS Sharpe (None if run_oos=False)
        gamma        -- gamma used
        K            -- K used
    """

    Xmat = X.fillna(0).values   # fill any residual NaNs conservatively

    loadings_arr, factors_arr, eigvals = rppca(Xmat, K=K, gamma=gamma)

    factor_cols = [f"factor_{k+1}" for k in range(K)]

    loadings_df = pd.DataFrame(loadings_arr, index=X.columns,   columns=factor_cols)
    factors_df  = pd.DataFrame(factors_arr,  index=X.index,     columns=factor_cols)

    sr_is  = compute_sr(factors_arr, annualise)
    sr_oos = None
    if run_oos:
        oos_stats = rppca_oos(Xmat, K=K, gamma=gamma, annualise=annualise)
        sr_oos    = oos_stats["rppca_oos"]

    print(f"RP-PCA fitted: K={K}, gamma={gamma}")
    print(f"  in-sample SR:      {sr_is:.3f}")
    if sr_oos is not None:
        print(f"  out-of-sample SR:  {sr_oos:.3f}")
    print(f"  top eigenvalues:   {np.round(eigvals, 1)}")

    return {
        "loadings": loadings_df,
        "factors":  factors_df,
        "eigvals":  eigvals,
        "sr_is":    sr_is,
        "sr_oos":   sr_oos,
        "gamma":    gamma,
        "K":        K,
    }
