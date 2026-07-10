# residualize.py
#
# Strips known risk factors (FF5 + momentum) from return panels before
# running RP-PCA. The point: ensure that what RP-PCA picks up as "thematic"
# co-movement is genuinely orthogonal to known factor exposures and not
# just market beta, size, value, profitability, investment, or momentum
# dressed up as a theme.
#
# THEORETICAL POSITIONING (for paper):
#   The common industry critique of thematic investing is that it's just
#   momentum with a story. Residualizing against momentum BEFORE looking
#   for thematic structure directly addresses this concern -- whatever
#   shared co-movement survives the residualization cannot be explained
#   by either FF5 risk exposures or by the momentum factor.
#
#   This sharpens the definition of "theme": not sustained outperformance
#   (which momentum already captures), but structured cross-sectional
#   co-movement around a shared economic driver, independent of known
#   factor premia.
#
# PAPER NOTE:
#   FF5 + momentum factors are pulled from Kenneth French's data library
#   via pandas-datareader -- free, daily/weekly available, monthly is the
#   most reliable cadence so we monthlify the weekly stock returns for
#   the regression and then strip back to weekly.
#   Alternative: use the daily FF factors and resample. We use monthly
#   for cleaner alignment and because Newey-West adjusted std errors
#   on monthly are more standard in the literature.

import os
import warnings
import pandas as pd
import numpy as np
import pandas_datareader.data as web

warnings.filterwarnings("ignore")


# -------------------- 1. Pull FF5 + momentum from Kenneth French --------------------

def get_ff_factors(start: str = "2017-01-01", end: str = None,
                    freq: str = "weekly") -> pd.DataFrame:
    """
    Downloads Fama-French 5 factors plus the momentum (MOM) factor
    from Kenneth French's data library.

    factors returned: Mkt-RF, SMB, HML, RMW, CMA, MOM, RF
    units: decimal weekly returns

    PAPER NOTE: French's library publishes daily, weekly, and monthly.
    Weekly is the natural match for our pipeline frequency (W-FRI close).
    """

    if freq == "weekly":
        ff5_name = "F-F_Research_Data_5_Factors_2x3_daily"
        mom_name = "F-F_Momentum_Factor_daily"
        resample_to = "W-FRI"
    else:
        ff5_name = "F-F_Research_Data_5_Factors_2x3"
        mom_name = "F-F_Momentum_Factor"
        resample_to = None

    print(f"\npulling FF5 + momentum from Kenneth French library ({freq})...")

    # FF5
    ff5_raw = web.DataReader(ff5_name, "famafrench",
                              start=start, end=end)[0]
    ff5 = ff5_raw / 100   # convert from % to decimal

    # momentum
    try:
        mom_raw = web.DataReader(mom_name, "famafrench", start=start, end=end)[0]
        mom = mom_raw / 100
        mom.columns = ["MOM"]
    except Exception as e:
        print(f"  momentum fetch failed via pandas_datareader ({e})")
        print("  falling back to direct download from Ken French's data library...")
        try:
            import requests, zipfile, io, re
            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"
            r = requests.get(url, timeout=30)
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            raw_text = zf.read(zf.namelist()[0]).decode("latin-1")
            print("  first 20 raw lines (for diagnosis):")
            for ln in raw_text.splitlines()[:20]:
                print(f"    {ln!r}")
            rows = []
            for ln in raw_text.splitlines():
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) >= 2 and re.match(r"^\d{8}$", parts[0]) and parts[1] != "":
                    rows.append(parts[:2])
            if not rows:
                raise ValueError("no valid date rows found in downloaded file")
            mom = pd.DataFrame(rows, columns=["date", "MOM"])
            mom["date"] = pd.to_datetime(mom["date"], format="%Y%m%d")
            mom = mom.set_index("date")
            mom["MOM"] = pd.to_numeric(mom["MOM"], errors="coerce") / 100
            mom = mom.dropna()
            mom = mom.loc[start:end] if end else mom.loc[start:]
            print(f"  momentum recovered via direct download: {len(mom)} obs")
        except Exception as e2:
            print(f"  direct download also failed ({e2}) -- proceeding with FF5 only")
            mom = pd.DataFrame()

    # combine
    if not mom.empty:
        ff = ff5.join(mom, how="inner")
    else:
        ff = ff5

    # resample daily -> weekly (Friday close, compound returns)
    if resample_to:
        ff.index = pd.to_datetime(ff.index)
        # compound daily to weekly: (1+r1)(1+r2)... - 1
        ff = (1 + ff).resample(resample_to).prod() - 1

    print(f"  factors: {ff.columns.tolist()}")
    print(f"  date range: {ff.index[0].date()} to {ff.index[-1].date()}")
    print(f"  observations: {len(ff)}")
    return ff


# -------------------- 2. Residualize returns against factor model --------------------

def residualize_returns(returns: pd.DataFrame,
                          factors: pd.DataFrame,
                          factor_cols: list = None,
                          keep_alpha: bool = False) -> pd.DataFrame:
    """
    For each stock, regresses its excess returns on the factor model
    and returns the residuals. The residuals are what RP-PCA then
    operates on -- co-movement here cannot be explained by the factor model.

    Parameters
    ----------
    returns      : T x N DataFrame of stock returns (already in excess of RF, ideally)
    factors      : T x K DataFrame from get_ff_factors()
    factor_cols  : which columns of factors to use as regressors. Default:
                   all except RF (we strip RF off the returns separately).
    keep_alpha   : V2 ADDITION. See note below. Default False reproduces the
                   exact V1 behaviour (residuals mean-zero by construction).

    Returns
    -------
    residuals : T x N DataFrame, same shape as input returns.
                If keep_alpha=False (V1 default): each column has mean
                approximately zero by construction -- this is an automatic
                property of OLS with an intercept term, not a separate
                demeaning step. If keep_alpha=True (V2): each column's mean
                equals that stock's estimated regression intercept (its
                factor-neutral alpha), NOT zero.

    Notes
    -----
    We use OLS over the full sample, not rolling. Rationale: rolling
    residualization introduces look-ahead concerns and adds noise.
    Full-sample residuals are what the cross-sectional literature uses
    (e.g. Lettau-Pelger themselves residualize this way).

    V2 NOTE -- why keep_alpha exists and why it is NOT "don't residualize":
    V1's residuals are mean-zero in the estimation sample because the
    regression includes an intercept and OLS residuals from an intercept
    regression always sum to zero. That is precisely the information RP-PCA's
    penalty term needs (see rppca.py: gamma * mu @ mu.T, where mu is the
    cross-sectional mean return vector) -- with mu forced to ~0, the penalty
    has nothing to reward, and RP-PCA collapses to ordinary PCA (confirmed
    empirically in the V1 paper, Section 3.3).
    keep_alpha=True does NOT skip residualization and does NOT reintroduce
    raw market/style beta. Stocks are still fully orthogonalized against
    Mkt-RF/SMB/HML/RMW/CMA/MOM -- the factor-neutral character V1 relies on
    for the "not momentum in disguise" argument is fully preserved. The ONLY
    change is that the estimated intercept (alpha) is added back on top of
    the factor-neutral residual, so the residual's mean reflects that stock's
    own average abnormal return instead of being forced to zero. This gives
    RP-PCA's premium-reward term real cross-sectional variation in mu to
    exploit, at the cost of reintroducing standard in-sample look-ahead in mu
    itself -- which is why V2 also requires walk-forward re-estimation
    (rppca_walkforward.py): mu must only ever be estimated on data prior to
    the period being scored.
    """

    if factor_cols is None:
        factor_cols = [c for c in factors.columns if c != "RF"]

    # align dates
    common = returns.index.intersection(factors.index)
    if len(common) < 52:
        raise ValueError(f"only {len(common)} common dates -- check alignment")

    print(f"\nresidualizing {returns.shape[1]} stocks against {len(factor_cols)} factors"
          f"{' (keep_alpha=True, V2)' if keep_alpha else ''}")
    print(f"  factor cols: {factor_cols}")
    print(f"  common dates: {len(common)} weeks")

    R = returns.loc[common]
    F = factors.loc[common, factor_cols]

    # -------------------- subtract RF if available --------------------
    if "RF" in factors.columns:
        rf = factors.loc[common, "RF"]
        R  = R.subtract(rf, axis=0)
        print(f"  excess returns computed (subtracted RF)")

    # -------------------- vectorized OLS for all stocks at once --------------------
    # add intercept column to factors
    F_mat = np.column_stack([np.ones(len(F)), F.values])  # T x (K+1)
    R_mat = R.fillna(0).values                              # T x N

    # betas = (F'F)^{-1} F'R   shape: (K+1) x N   -- row 0 is the intercept (alpha)
    FtF_inv = np.linalg.inv(F_mat.T @ F_mat + 1e-10 * np.eye(F_mat.shape[1]))
    betas   = FtF_inv @ F_mat.T @ R_mat                    # (K+1) x N
    alpha   = betas[0, :]                                   # N-vector, per-stock intercept

    # residuals = R - F @ betas   (V1 behaviour: this includes subtracting the
    # intercept, which is exactly what forces the mean to zero)
    residuals = R_mat - F_mat @ betas                       # T x N

    if keep_alpha:
        # V2: add the intercept back so the residual's mean = alpha, not 0.
        # Still orthogonal to every factor column (that part of betas is
        # untouched) -- only the demeaning effect of the intercept term
        # is reversed.
        residuals = residuals + alpha[np.newaxis, :]

    resid_df = pd.DataFrame(residuals, index=R.index, columns=R.columns)

    # -------------------- diagnostics --------------------
    avg_r2 = _compute_avg_r2(R_mat, residuals)
    print(f"  average R² across stocks: {avg_r2:.3f}")
    print(f"  (high R² = factor model explains a lot, low residual variance left)")
    if keep_alpha:
        print(f"  mean |alpha| across stocks (annualised): {np.mean(np.abs(alpha)) * 52:.3f}")
        print(f"  (this is the cross-sectional mean-return signal RP-PCA's penalty now sees)")

    return resid_df


def _compute_avg_r2(returns_mat: np.ndarray, residuals_mat: np.ndarray) -> float:
    """Average R² across stocks: 1 - var(residual) / var(returns)."""
    var_ret   = np.var(returns_mat, axis=0)
    var_resid = np.var(residuals_mat, axis=0)
    r2_per_stock = 1 - var_resid / (var_ret + 1e-10)
    return float(np.mean(r2_per_stock))


# -------------------- 3. Convenience wrapper --------------------

def residualize_universe(returns: pd.DataFrame,
                          start: str = None,
                          end: str = None,
                          factor_cols: list = None,
                          keep_alpha: bool = False) -> dict:
    """
    Full pipeline: pull FF5+momentum, align to return panel, return residuals.

    keep_alpha : V2 addition, passed straight through to residualize_returns().
                 See that function's docstring for what this does and why.
                 Note standardize_residuals() (below) only divides by std and
                 never subtracts a mean, so a keep_alpha=True residual's
                 preserved cross-sectional mean survives standardization
                 intact (rescaled, not removed).

    Returns a dict with:
        residuals : T x N residualized return DataFrame  (feed this to RP-PCA)
        factors   : T x K factor DataFrame used for the regression
        avg_r2    : float, average R² of the factor model across stocks
    """
    if start is None:
        start = str(returns.index[0].date())
    if end is None:
        end = str(returns.index[-1].date())

    factors  = get_ff_factors(start=start, end=end, freq="weekly")
    resid_df = residualize_returns(returns, factors, factor_cols=factor_cols,
                                    keep_alpha=keep_alpha)

    common  = returns.index.intersection(factors.index)
    R_mat   = returns.loc[common].fillna(0).values
    if "RF" in factors.columns:
        rf = factors.loc[common, "RF"]
        R_mat = R_mat - rf.values[:, None]

    avg_r2 = _compute_avg_r2(R_mat, resid_df.values)

    return {
        "residuals": resid_df,
        "factors":   factors,
        "avg_r2":    avg_r2,
    }


# -------------------- 4. Standardize residuals (cross-sectional vol normalization) --------------------

def standardize_residuals(residuals: pd.DataFrame,
                           winsorize_pct: float = 0.01) -> pd.DataFrame:
    """
    Standardizes each stock's residual return series to unit variance,
    after winsorizing extreme weekly observations.

    Why this matters for PCA/RP-PCA:
      Without standardization, a single high-volatility stock (a crypto miner,
      a biotech that 5x'd on trial data) has enormous residual variance and
      dominates the top principal components -- you get "the most volatile
      micro-cap" as factor 1 instead of an economic theme. Standardizing to
      unit variance puts every stock on equal footing so factors reflect
      shared co-movement direction, not raw amplitude.

    Steps:
      1. Winsorize each stock's residuals at the winsorize_pct / (1-winsorize_pct)
         quantiles (default 1%/99%) to clip extreme single-week blow-ups
      2. Divide each stock's series by its standard deviation (unit vol)

    PAPER NOTE: standardizing residuals before PCA is standard practice in
    the statistical-factor literature -- it makes the analysis a correlation
    (not covariance) decomposition, which is appropriate when we care about
    the STRUCTURE of co-movement rather than its amplitude.

    Returns
    -------
    standardized residuals, same shape, each column with unit std
    """
    R = residuals.copy()

    # -------------------- winsorize per stock --------------------
    lo = R.quantile(winsorize_pct)
    hi = R.quantile(1 - winsorize_pct)
    R  = R.clip(lower=lo, upper=hi, axis=1)

    # -------------------- standardize to unit vol --------------------
    stds = R.std()
    stds = stds.replace(0, 1e-10)   # avoid div by zero for dead stocks
    R    = R.divide(stds, axis=1)

    print(f"  standardized {R.shape[1]} stocks to unit vol "
          f"(winsorized at {winsorize_pct*100:.0f}%/{(1-winsorize_pct)*100:.0f}%)")
    return R


# -------------------- 5. Commodity factor controls --------------------

def get_commodity_factors(start: str, end: str = None,
                           freq: str = "W-FRI") -> pd.DataFrame:
    """
    Pulls liquid commodity-sector ETF returns to use as additional
    residualization controls, targeting the commodity-beta contamination
    seen in Clean Energy (gold miners) and Agribusiness (base metals / ag).

    PAPER NOTE: there is no single canonical commodity risk-factor library
    analogous to Fama-French. We proxy the commodity factor structure of
    Bakshi, Gao & Rossi (2019) using liquid US-listed commodity-sector ETFs.
    These are investable index returns, not academic long-short factors, but
    they span the commodity sectors (energy, industrial metals, precious
    metals, agriculture) that drive the contamination.

    ETFs used:
        DBC  -- broad commodity (PowerShares DB Commodity)
        DBE  -- energy
        DBB  -- industrial/base metals (copper, aluminium, zinc)
        GLD  -- gold (precious metals -- catches the Clean Energy gold-miner issue)
        DBA  -- agriculture (catches the Agribusiness farm-commodity issue)

    All US-listed, so no timing/FX issues. Returns weekly log returns.
    """
    import yfinance as yf

    commodity_etfs = ["DBC", "DBE", "DBB", "GLD", "DBA"]
    print(f"\npulling commodity controls: {commodity_etfs}")

    raw = yf.download(commodity_etfs, start=start, end=end,
                       auto_adjust=True, progress=False)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    weekly = px.resample(freq).last()
    rets   = np.log(weekly / weekly.shift(1)).dropna(how="all")

    # rename with a COMM_ prefix to distinguish from FF factors
    rets.columns = [f"COMM_{c}" for c in rets.columns]

    print(f"  commodity factors: {rets.columns.tolist()}")
    print(f"  observations: {len(rets)}")
    return rets


def get_combined_factors(start: str, end: str = None,
                          freq: str = "weekly",
                          include_commodity: bool = False) -> pd.DataFrame:
    """
    Convenience: pulls FF5+momentum, optionally joins commodity controls.

    include_commodity: if True, adds DBC/DBE/DBB/GLD/DBA as extra columns.
                       Use this to test whether stripping commodity beta
                       rescues Clean Energy / Agribusiness baskets.

    Returns combined factor DataFrame aligned on common dates.
    """
    ff = get_ff_factors(start=start, end=end, freq=freq)

    if not include_commodity:
        return ff

    resample_to = "W-FRI" if freq == "weekly" else None
    comm = get_commodity_factors(start=start, end=end,
                                  freq=resample_to or "W-FRI")

    # align on common dates
    combined = ff.join(comm, how="inner")
    print(f"\ncombined factors: {combined.columns.tolist()}")
    print(f"  common observations: {len(combined)}")
    return combined
