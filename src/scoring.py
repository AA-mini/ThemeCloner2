# 05_scoring.py
# Takes the raw cosine similarity scores and turns them into
# clean, investable ranked candidate lists per theme.
#
# Also runs the robustness check we discussed: compares the ETF-space
# approach (our primary method) with the universe-native RP-PCA approach.
# Stocks that rank highly in BOTH methods are the high-confidence candidates.
# Stocks that only appear in one method are flagged for closer inspection --
# these are often the most interesting mispricing stories.
#
# Output:
#   ranked_candidates.csv   -- top N stocks per theme with scores and metadata
#   robustness_check.csv    -- agreement between the two methods

import numpy as np
import pandas as pd
import os

from src.rppca import fit_rppca
from src.projection import score_universe


# -------------------- 1. Rank candidates per theme --------------------

def rank_candidates(scores: pd.DataFrame,
                     top_n: int = 30,
                     min_score: float = 0.3) -> pd.DataFrame:
    """
    Ranks stocks by their cosine similarity score for each theme,
    applies a minimum score cutoff, and returns the top N per theme.

    top_n     : how many candidates to return per theme
    min_score : minimum cosine similarity to be considered a match
                (0.3 is a reasonable starting point -- tune based on results)

    Returns a long-format DataFrame: one row per (theme, stock) pair.
    """

    rows = []
    for theme in scores.columns:
        theme_scores = scores[theme].dropna().sort_values(ascending=False)
        theme_scores = theme_scores[theme_scores >= min_score].head(top_n)

        for rank, (ticker, score) in enumerate(theme_scores.items(), start=1):
            rows.append({
                "theme":  theme,
                "rank":   rank,
                "ticker": ticker,
                "score":  round(score, 4),
            })

    ranked = pd.DataFrame(rows)
    print(f"\nranked candidates summary:")
    print(ranked.groupby("theme").size().rename("candidates").to_string())
    return ranked


# -------------------- 1b. V2: screen by score AND R², variable basket size --------------------
#
# WHY THIS EXISTS: rank_candidates (above) already technically supports a
# variable basket size (fewer than top_n survive if min_score is strict), but
# in practice min_score was often set permissively (e.g. -1.0 or 0.0) purely
# to force exactly top_n candidates every time -- which is precisely the
# "forcing 30 stocks" concern raised against V1. This function makes the two
# floors explicit and separate: a score floor (direction match) AND an R²
# floor (how much of the candidate's own variance the factor model actually
# explains, from projection.compute_candidate_r2). top_n becomes a genuine
# CAP, not a target -- a theme can legitimately end up with 6 candidates or
# 30, depending on how many names clear both bars.

def screen_and_rank_candidates(scores: pd.DataFrame,
                                candidate_r2: pd.Series,
                                top_n: int = 30,
                                min_score: float = 0.3,
                                min_r2: float = 0.15,
                                weight_by: str = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    scores       : stocks x themes score DataFrame (cosine similarity or the
                   V2 synthetic-return correlation -- either works)
    candidate_r2 : pd.Series from projection.compute_candidate_r2(), how much
                   of each stock's OWN variance the K-factor model explains
    top_n        : maximum candidates per theme (a cap, not a target)
    min_score    : minimum score to be considered at all
    min_r2       : minimum candidate R² to be considered at all -- this is
                   the new, explicit noise floor. Inspect the distribution of
                   candidate_r2 before picking a value; V1's post-hoc
                   diagnostic found mean candidate R² around 0.06-0.08, so a
                   floor anywhere from 0.10-0.20 is a reasonable starting
                   range to actually bite.
    weight_by    : None (equal-weight, default), 'score', or 'r2' -- if set,
                   also returns each candidate's normalized weight in that
                   theme's basket (proportional to score or R², renormalized
                   to sum to 1 within the theme). Equal-weighting is still
                   what most of this codebase reports by default; this is
                   for testing the weighted alternative explicitly.

    Returns
    -------
    Long DataFrame: theme, rank, ticker, score, candidate_r2, and (if
    weight_by is set) weight.
    """
    rows = []
    for theme in scores.columns:
        theme_scores = scores[theme].dropna()
        eligible = theme_scores[theme_scores >= min_score]
        eligible = eligible[eligible.index.map(lambda t: candidate_r2.get(t, -1) >= min_r2)]
        eligible = eligible.sort_values(ascending=False).head(top_n)

        if weight_by == 'score':
            w = eligible.clip(lower=0)
        elif weight_by == 'r2':
            w = candidate_r2.reindex(eligible.index).clip(lower=0)
        else:
            w = None
        if w is not None and w.sum() > 0:
            w = w / w.sum()

        for rank, (ticker, score) in enumerate(eligible.items(), start=1):
            row = {"theme": theme, "rank": rank, "ticker": ticker,
                   "score": round(score, 4),
                   "candidate_r2": round(float(candidate_r2.get(ticker, np.nan)), 4)}
            if w is not None:
                row["weight"] = round(float(w.get(ticker, 0.0)), 4)
            rows.append(row)

    ranked = pd.DataFrame(rows)
    print(f"\nscreened + ranked candidates summary (min_score={min_score}, min_r2={min_r2}, cap={top_n}):")
    if not ranked.empty:
        print(ranked.groupby("theme").size().rename("candidates").to_string())
    else:
        print("  (no candidates survived both floors for any theme -- floors may be too strict)")
    return ranked


def weights_from_ranked(ranked: pd.DataFrame) -> dict:
    """
    Convenience: turns a screen_and_rank_candidates() output (with a
    'weight' column) into the {theme -> pd.Series(ticker -> weight)} dict
    format validate_against_etf's candidate_weights parameter expects.
    Returns {} if ranked has no 'weight' column (i.e. weight_by was None).
    """
    if "weight" not in ranked.columns:
        return {}
    out = {}
    for theme, g in ranked.groupby("theme"):
        out[theme] = g.set_index("ticker")["weight"]
    return out


# -------------------- 2. Robustness check -- two-method agreement --------------------

def robustness_check(universe_returns: pd.DataFrame,
                      scores_etf_space: pd.DataFrame,
                      etf_config: pd.DataFrame,
                      K: int = 5,
                      gamma: float = 10.0,
                      top_n: int = 50) -> pd.DataFrame:
    """
    Runs the alternative method (RP-PCA on the universe directly) and
    compares its top-N candidates to the ETF-space method's top-N.

    Method 1 (primary):   RP-PCA on ETF panel -> project universe stocks in
    Method 2 (robustness): RP-PCA on universe -> project ETF theme DNA in

    Stocks that appear in BOTH top-N lists are HIGH CONFIDENCE candidates.
    Stocks only in Method 1 top-N but not Method 2: the market prices them
      as thematic but their return structure is different from large caps.
    Stocks only in Method 2: universe-native co-movement without ETF resemblance.

    This comparison IS a finding for the paper, not just a sanity check.
    """

    print("\nrunning robustness check (universe-native RP-PCA)...")

    # -------------------- fit RP-PCA on the universe itself --------------------
    universe_result = fit_rppca(universe_returns, K=K, gamma=gamma,
                                 run_oos=False, annualise=52.0)

    # -------------------- build theme DNA in universe factor space --------------------
    # we need ETF returns aligned to the universe factor time series
    # project ETF returns onto universe factors (same OLS approach as 04_projection.py)
    from src.projection import project_universe

    # we need etf_returns here -- caller must pass it or we reload
    # for now we return the universe loadings so caller can do the comparison
    universe_loadings = universe_result["loadings"]   # (universe stocks x K)

    # -------------------- get top-N per theme from each method --------------------
    # method 1 rankings (already computed, passed in)
    m1_ranked = rank_candidates(scores_etf_space, top_n=top_n, min_score=0.0)

    rows = []
    for theme in scores_etf_space.columns:
        m1_top = set(m1_ranked[m1_ranked["theme"] == theme]["ticker"].tolist())

        # for method 2 we need to score universe stocks against the theme in universe space
        # we flag this as "needs ETF returns" -- handled in the notebook
        rows.append({
            "theme":          theme,
            "m1_candidates":  len(m1_top),
            "note":           "pass etf_returns to complete method 2 comparison"
        })

    return pd.DataFrame(rows), universe_result


def compare_methods(m1_ranked: pd.DataFrame,
                     m2_ranked: pd.DataFrame) -> pd.DataFrame:
    """
    Given ranked candidate lists from both methods, produces a comparison table.
    Called from the notebook once both method results are available.
    """

    themes = m1_ranked["theme"].unique()
    rows   = []

    for theme in themes:
        m1_set = set(m1_ranked[m1_ranked["theme"] == theme]["ticker"])
        m2_set = set(m2_ranked[m2_ranked["theme"] == theme]["ticker"])

        both      = m1_set & m2_set   # high confidence
        m1_only   = m1_set - m2_set   # ETF-space only (potential mispricing story)
        m2_only   = m2_set - m1_set   # universe-native only

        rows.append({
            "theme":             theme,
            "high_confidence":   len(both),
            "etf_space_only":    len(m1_only),
            "universe_only":     len(m2_only),
            "agreement_rate":    len(both) / max(len(m1_set | m2_set), 1),
            "hc_tickers":        sorted(both),
            "m1_only_tickers":   sorted(m1_only),
        })

    result = pd.DataFrame(rows)
    print("\nmethod agreement summary:")
    print(result[["theme", "high_confidence", "etf_space_only",
                   "universe_only", "agreement_rate"]].to_string(index=False))
    return result


# -------------------- 3. Save final outputs --------------------

def save_outputs(ranked: pd.DataFrame,
                  robustness: pd.DataFrame = None,
                  out_dir: str = None):

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    ranked_path = os.path.join(out_dir, "ranked_candidates.csv")
    ranked.to_csv(ranked_path, index=False)
    print(f"saved: {ranked_path}")

    if robustness is not None:
        rob_path = os.path.join(out_dir, "robustness_check.csv")
        robustness.to_csv(rob_path, index=False)
        print(f"saved: {rob_path}")
