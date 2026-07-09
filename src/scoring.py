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
