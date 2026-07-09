# theme_diagnostics.py
#
# UPSTREAM diagnostic: characterises whether a "theme" is a genuine economic
# covariance cluster, and whether the factor it loads on is SHARED with other
# themes (which causes basket contamination/bleed).
#
# THREE-CASE TAXONOMY (the refined paper framing):
#   1. COHERENT + DISTINCT  : concentrated on one factor that is its own
#                             (semis/AI infra) -> clean, correct baskets
#   2. COHERENT + SHARED    : concentrated on one factor, but that factor is
#                             shared with other themes (clean energy loads on
#                             a commodity factor also used by mining themes)
#                             -> clean-looking but WRONG baskets (contaminated)
#   3. DIFFUSE              : spread across many factors (defense, agribusiness
#                             as a loose grab-bag) -> no single cluster to find
#
# Concentration alone (coherence) cannot separate case 1 from case 2 --
# Clean Energy is highly concentrated yet produces gold-miner baskets because
# its factor is shared. We therefore report BOTH:
#   - coherence     : how concentrated the theme's fingerprint is
#   - distinctiveness: how UNIQUELY the theme owns its dominant factor
#
# COHERENCE = average of three normalised concentration metrics on the
#   theme's factor-R² profile (concentration ratio, Herfindahl, effective-n).
#
# DISTINCTIVENESS = 1 - (share of the theme's dominant factor that is also
#   claimed by other themes). High = the theme owns its factor. Low = many
#   themes pile onto the same factor (bleed).
#
# NULL (fixed): random baskets are now built from ACTUAL stock residual
#   returns (sample N real stocks, average their returns), giving a fair
#   like-for-like baseline. The previous version used noiseless factor
#   reconstructions which were artificially concentrated and inflated the null.

import numpy as np
import pandas as pd


# -------------------- 1. Per-series factor R² profile --------------------

def _series_factor_r2(ret: np.ndarray, F: np.ndarray) -> np.ndarray:
    """R² of a return series on each individual factor. Returns K-vector."""
    K = F.shape[1]
    r2 = np.empty(K)
    for k in range(K):
        c = np.corrcoef(ret, F[:, k])[0, 1]
        r2[k] = 0.0 if np.isnan(c) else c ** 2
    return r2


# -------------------- 2. Concentration metrics --------------------

def _concentration_ratio(r2: np.ndarray) -> float:
    total = r2.sum()
    return float(r2.max() / total) if total > 0 else 0.0


def _herfindahl(r2: np.ndarray) -> float:
    total = r2.sum()
    if total <= 0:
        return 0.0
    shares = r2 / total
    return float((shares ** 2).sum())


def _effective_factors_concentration(r2: np.ndarray) -> float:
    total = r2.sum()
    K = len(r2)
    if total <= 0 or K <= 1:
        return 0.0
    shares = r2 / total
    shares = shares[shares > 0]
    entropy = -(shares * np.log(shares)).sum()
    eff_n   = np.exp(entropy)
    return float((K - eff_n) / (K - 1))


def _coherence_from_r2(r2: np.ndarray) -> float:
    """Combined coherence = average of the three concentration metrics."""
    return (_concentration_ratio(r2)
            + _herfindahl(r2)
            + _effective_factors_concentration(r2)) / 3


# -------------------- 3. Main coherence + distinctiveness --------------------

def compute_coherence(rppca_result: dict,
                       etf_residuals: pd.DataFrame,
                       etf_config: pd.DataFrame,
                       cov_residuals: pd.DataFrame = None,
                       run_null: bool = True,
                       n_null: int = 300,
                       null_basket_size: int = 30,
                       seed: int = 42) -> pd.DataFrame:
    """
    Computes coherence AND distinctiveness for every theme.

    Parameters
    ----------
    rppca_result     : fit_rppca output on residualized covariance universe
    etf_residuals    : residualized ETF returns (theme labels)
    etf_config       : etfs.csv mapping
    cov_residuals    : residualized stock returns (REQUIRED for a valid null --
                       random baskets are drawn from these actual returns)
    run_null         : compare each theme to random real-stock baskets
    n_null           : number of random baskets
    null_basket_size : stocks per random basket

    Returns
    -------
    DataFrame indexed by theme with coherence, distinctiveness, the raw
    metrics, null baseline, and a three-case verdict.
    """

    factors_df = rppca_result["factors"]
    K          = factors_df.shape[1]

    common = etf_residuals.index.intersection(factors_df.index)
    F = factors_df.loc[common].values
    E = etf_residuals.loc[common]

    rng = np.random.default_rng(seed)

    # -------------------- per-theme R² profiles --------------------
    # type lookup: theme / sector / subsector (controls are calibration anchors)
    if "type" in etf_config.columns:
        type_lookup = (etf_config.drop_duplicates("theme")
                       .set_index("theme")["type"].to_dict())
    else:
        type_lookup = {}

    theme_r2     = {}     # theme -> K-vector of R²
    theme_domfac = {}     # theme -> index of dominant factor
    for theme in etf_config["theme"].unique():
        avail = [e for e in etf_config[etf_config["theme"] == theme]["ticker"]
                 if e in E.columns]
        if not avail:
            continue
        ret = E[avail].mean(axis=1).values
        r2  = _series_factor_r2(ret, F)
        theme_r2[theme]     = r2
        theme_domfac[theme] = int(np.argmax(r2))

    # -------------------- distinctiveness --------------------
    # for each theme's dominant factor, how much of that factor's "claim"
    # belongs to this theme vs others. We use each theme's R² ON its dominant
    # factor, and compare to other themes' R² on the SAME factor.
    distinctiveness = {}
    for theme, dom in theme_domfac.items():
        own_r2 = theme_r2[theme][dom]
        # sum of all themes' R² on this same dominant factor
        others_r2 = sum(theme_r2[t][dom] for t in theme_r2)
        distinctiveness[theme] = float(own_r2 / others_r2) if others_r2 > 0 else 0.0

    # -------------------- fixed null: random REAL-stock baskets --------------------
    null_mean = null_p95 = np.nan
    if run_null and cov_residuals is not None:
        Rc = cov_residuals.loc[cov_residuals.index.intersection(common)]
        Rc = Rc.reindex(common).fillna(0)
        stock_mat = Rc.values            # T x N_stocks
        n_stocks  = stock_mat.shape[1]
        null_scores = []
        for _ in range(n_null):
            idx = rng.choice(n_stocks,
                             size=min(null_basket_size, n_stocks),
                             replace=False)
            basket_ret = stock_mat[:, idx].mean(axis=1)   # real avg return
            r2 = _series_factor_r2(basket_ret, F)
            null_scores.append(_coherence_from_r2(r2))
        null_scores = np.array(null_scores)
        null_mean = float(null_scores.mean())
        null_p95  = float(np.percentile(null_scores, 95))
    elif run_null and cov_residuals is None:
        print("  WARNING: run_null=True but cov_residuals not provided -- "
              "skipping null (pass cov_residuals for a valid baseline)")
        run_null = False

    # -------------------- count how many themes share each dominant factor --------------------
    factor_share_count = {}
    for theme, dom in theme_domfac.items():
        factor_share_count[dom] = factor_share_count.get(dom, 0) + 1

    # -------------------- assemble --------------------
    # verdict keys off DISTINCTIVENESS and factor-sharing, NOT coherence-vs-null.
    # coherence measures concentration (random baskets are also concentrated, so
    # beating the null on coherence is the wrong test). what separates clean from
    # contaminated baskets is whether a theme UNIQUELY OWNS its dominant factor.
    rows = {}
    for theme in theme_r2:
        r2  = theme_r2[theme]
        dom = theme_domfac[theme]
        coh = _coherence_from_r2(r2)
        dist = distinctiveness[theme]
        n_sharing = factor_share_count[dom]

        # verdict: does the theme own a distinct factor?
        if n_sharing == 1 and dist >= 0.30:
            verdict = "OWNS FACTOR (clean)"
        elif n_sharing == 1:
            verdict = "OWNS FACTOR (weak)"
        elif dist >= 0.30:
            verdict = "SHARED (contested lead)"
        else:
            verdict = "SHARED (contaminated)"

        rows[theme] = {
            "coherence":       round(coh, 3),
            "distinctiveness": round(dist, 3),
            "conc_ratio":      round(_concentration_ratio(r2), 3),
            "herfindahl":      round(_herfindahl(r2), 3),
            "eff_factor_conc": round(_effective_factors_concentration(r2), 3),
            "dom_factor":      f"factor_{dom+1}",
            "dom_factor_r2":   round(float(r2[dom]), 3),
            "type":            type_lookup.get(theme, "theme"),
            "n_themes_on_factor": n_sharing,
            "null_p95":        round(null_p95, 3) if run_null else np.nan,
            "verdict":         verdict,
        }

    result = pd.DataFrame(rows).T
    result = result.sort_values(["distinctiveness", "coherence"], ascending=False)
    return result


# -------------------- 4. Report --------------------

def print_coherence_report(coherence_df: pd.DataFrame):
    print("=" * 74)
    print("THEME DIAGNOSTIC  (coherence + distinctiveness)")
    print("=" * 74)
    print("\ncoherence       = how concentrated the theme's factor fingerprint is")
    print("distinctiveness = how uniquely the theme OWNS its dominant factor")
    print("  COHERENT+DISTINCT = clean correct baskets")
    print("  COHERENT+SHARED   = clean-looking but contaminated (factor shared)")
    print("  DIFFUSE           = no single cluster -> taxonomic label\n")

    if "null_p95" in coherence_df.columns and coherence_df["null_p95"].notna().any():
        print(f"random real-stock null: p95 = {coherence_df['null_p95'].iloc[0]}\n")

    has_type = "type" in coherence_df.columns
    if has_type and (coherence_df["type"] != "theme").any():
        anchors = coherence_df[coherence_df["type"] != "theme"]
        themes  = coherence_df[coherence_df["type"] == "theme"]
        print("CALIBRATION ANCHORS (sectors/subsectors -- the coherence ceiling):")
        for theme, r in anchors.iterrows():
            print(f"  {theme:26s}  coh={r['coherence']:.3f}  "
                  f"dist={r['distinctiveness']:.3f}  dom={r['dom_factor']:9s}")
        print("\nTHEMES (compare against the anchors above):")
        iter_df = themes
    else:
        iter_df = coherence_df

    for theme, r in iter_df.iterrows():
        share = f"({int(r['n_themes_on_factor'])} themes)" if r['n_themes_on_factor'] > 1 else "(unique)"
        print(f"  {theme:26s}  coh={r['coherence']:.3f}  "
              f"dist={r['distinctiveness']:.3f}  "
              f"dom={r['dom_factor']:9s} {share:12s}  [{r['verdict']}]")

    # flag factor-sharing explicitly
    print("\nfactor-sharing map (themes piling on the same factor = bleed risk):")
    by_factor = {}
    for theme, r in coherence_df.iterrows():
        by_factor.setdefault(r["dom_factor"], []).append(theme)
    for fac, themes in sorted(by_factor.items()):
        if len(themes) > 1:
            print(f"  {fac}: SHARED by {', '.join(themes)}")
        else:
            print(f"  {fac}: unique to {themes[0]}")


# -------------------- 5. K-sweep: how does coherence change with factor count? --------------------

def coherence_k_sweep(cov_residuals: pd.DataFrame,
                      etf_residuals: pd.DataFrame,
                      etf_config: pd.DataFrame,
                      k_values: list = None,
                      gamma: float = 0.0) -> pd.DataFrame:
    """
    Re-fits RP-PCA at several K values and recomputes each label's coherence and
    distinctiveness. Tells us whether the coherence ordering (and especially the
    sector-vs-theme pattern) is robust to K or an artifact of too few factors.

    At low K, factor-sharing is mechanically forced (more labels than factors), so
    everything looks contaminated. As K rises, genuine clusters should separate and
    own distinct factors. The key questions:
      - do the GICS sectors eventually own distinct factors (become the ceiling)?
      - or do cross-sector themes (Water, AI Infra) remain more distinct than the
        sectors they span, even with ample factors (the regime-coherence result)?

    Returns a tidy DataFrame: one row per (label, K) with coherence, distinctiveness,
    dominant factor, n_themes_on_factor, and type.
    """
    from src.rppca import fit_rppca

    if k_values is None:
        k_values = [10, 15, 20, 25]

    rows = []
    for K in k_values:
        res = fit_rppca(cov_residuals, K=K, gamma=gamma, run_oos=False)
        coh = compute_coherence(res, etf_residuals, etf_config,
                                cov_residuals=cov_residuals,
                                run_null=False)   # skip null for speed in sweep
        for label, r in coh.iterrows():
            rows.append({
                "label": label,
                "K": K,
                "type": r.get("type", "theme"),
                "coherence": r["coherence"],
                "distinctiveness": r["distinctiveness"],
                "dom_factor": r["dom_factor"],
                "n_on_factor": r["n_themes_on_factor"],
            })
        print(f"  K={K}: fitted and scored {len(coh)} labels")

    return pd.DataFrame(rows)


def summarize_k_sweep(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivots the K-sweep into a label x K matrix of distinctiveness, with the type
    and the mean distinctiveness across K for ranking. Useful as a paper table.
    """
    pivot = sweep_df.pivot_table(index="label", columns="K",
                                  values="distinctiveness")
    types = sweep_df.drop_duplicates("label").set_index("label")["type"]
    pivot.insert(0, "type", types)
    pivot["mean_dist"] = pivot.drop(columns="type").mean(axis=1)
    return pivot.sort_values("mean_dist", ascending=False)
