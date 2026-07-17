"""Forward evaluation tools for the revised ThemeCloner walk forward process.

The primary diagnostics measure whether a score formed at rebalance date t
predicts realized thematic exposure during the next holding period. Raw return
tracking remains available as a secondary diagnostic only.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_EPS = 1e-12


def _beta_and_correlation(
    asset: pd.Series,
    benchmark: pd.Series,
    min_obs: int = 8,
) -> Tuple[float, float, int]:
    """Return OLS beta, correlation and number of common observations."""

    panel = pd.concat([asset.rename("asset"), benchmark.rename("benchmark")], axis=1).dropna()
    nobs = len(panel)
    if nobs < min_obs:
        return np.nan, np.nan, nobs

    x = panel["benchmark"].to_numpy(dtype=float)
    y = panel["asset"].to_numpy(dtype=float)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    x_var = float(np.dot(x_centered, x_centered))
    y_var = float(np.dot(y_centered, y_centered))
    if x_var <= _EPS or y_var <= _EPS:
        return np.nan, np.nan, nobs

    covariance = float(np.dot(x_centered, y_centered))
    beta = covariance / x_var
    correlation = covariance / np.sqrt(x_var * y_var)
    return float(beta), float(correlation), nobs


def build_forward_theme_benchmarks(
    forward_factors: pd.DataFrame,
    forward_etf_residuals: pd.DataFrame,
    theme_references: Mapping[str, dict],
) -> Dict[str, dict]:
    """Build factor synthetic and residual ETF benchmarks for each theme."""

    benchmarks: Dict[str, dict] = {}
    for theme, reference in theme_references.items():
        factor_cols = reference["factor_columns"]
        if not set(factor_cols).issubset(forward_factors.columns):
            continue

        medoid_beta = np.asarray(reference["medoid_beta"], dtype=float)
        synthetic = pd.Series(
            forward_factors.loc[:, factor_cols].to_numpy(dtype=float) @ medoid_beta,
            index=forward_factors.index,
            name=f"{theme}_synthetic",
        )

        available_etfs = [
            ticker for ticker in reference["tickers"] if ticker in forward_etf_residuals.columns
        ]
        etf_blend = (
            forward_etf_residuals.loc[:, available_etfs].mean(axis=1).rename(f"{theme}_etf_blend")
            if available_etfs
            else pd.Series(dtype=float, name=f"{theme}_etf_blend")
        )

        benchmarks[str(theme)] = {
            "synthetic": synthetic,
            "etf_blend": etf_blend,
            "etfs": available_etfs,
        }
    return benchmarks


def _equal_weight_series(returns: pd.DataFrame, tickers: Sequence[str]) -> pd.Series:
    available = [ticker for ticker in tickers if ticker in returns.columns]
    if not available:
        return pd.Series(index=returns.index, dtype=float)
    return returns.loc[:, available].mean(axis=1)


def _rank_ic(scores: pd.Series, exposure: pd.Series) -> float:
    panel = pd.concat([scores.rename("score"), exposure.rename("exposure")], axis=1).dropna()
    if len(panel) < 5 or panel["score"].nunique() < 2 or panel["exposure"].nunique() < 2:
        return np.nan
    return float(panel["score"].corr(panel["exposure"], method="spearman"))


def _top_bottom_spread(scores: pd.Series, exposure: pd.Series, quantiles: int = 5) -> float:
    panel = pd.concat([scores.rename("score"), exposure.rename("exposure")], axis=1).dropna()
    if len(panel) < max(10, quantiles * 2) or panel["score"].nunique() < quantiles:
        return np.nan
    try:
        panel["bucket"] = pd.qcut(panel["score"], quantiles, labels=False, duplicates="drop")
    except ValueError:
        return np.nan
    if panel["bucket"].nunique() < 2:
        return np.nan
    bucket_means = panel.groupby("bucket", observed=True)["exposure"].mean()
    return float(bucket_means.iloc[-1] - bucket_means.iloc[0])


def evaluate_forward_period(
    scores: pd.DataFrame,
    baskets: Mapping[str, dict],
    forward_target_residuals: pd.DataFrame,
    forward_target_raw: pd.DataFrame,
    benchmarks: Mapping[str, dict],
    rebalance_date: pd.Timestamp,
    min_forward_obs: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all requested forward exposure metrics for one period."""

    summary_rows = []
    detail_rows = []

    for theme, benchmark_set in benchmarks.items():
        theme_scores = scores.loc[scores["theme"] == theme].copy()
        if theme_scores.empty:
            continue

        synthetic = benchmark_set["synthetic"]
        etf_blend = benchmark_set["etf_blend"]

        for row in theme_scores.itertuples(index=False):
            ticker = row.ticker
            if ticker not in forward_target_residuals.columns:
                continue
            stock = forward_target_residuals[ticker]
            syn_beta, syn_corr, syn_nobs = _beta_and_correlation(
                stock, synthetic, min_obs=min_forward_obs
            )
            etf_beta, etf_corr, etf_nobs = _beta_and_correlation(
                stock, etf_blend, min_obs=min_forward_obs
            )
            detail_rows.append(
                {
                    "rebalance_date": pd.Timestamp(rebalance_date),
                    "theme": theme,
                    "ticker": ticker,
                    "rank_score": row.rank_score,
                    "penalized_distance": row.penalized_distance,
                    "eligible": row.eligible,
                    "selected": ticker in baskets.get(theme, {}).get("tickers", []),
                    "forward_synthetic_beta": syn_beta,
                    "forward_synthetic_corr": syn_corr,
                    "forward_etf_beta": etf_beta,
                    "forward_etf_corr": etf_corr,
                    "forward_obs": max(syn_nobs, etf_nobs),
                }
            )

        theme_detail = pd.DataFrame(
            [row for row in detail_rows if row["theme"] == theme and row["rebalance_date"] == pd.Timestamp(rebalance_date)]
        )
        selected_tickers = baskets.get(theme, {}).get("tickers", [])
        selected_detail = theme_detail.loc[theme_detail["selected"]] if not theme_detail.empty else pd.DataFrame()

        residual_basket = _equal_weight_series(forward_target_residuals, selected_tickers)
        raw_basket = _equal_weight_series(forward_target_raw, selected_tickers)
        basket_syn_beta, basket_syn_corr, _ = _beta_and_correlation(
            residual_basket, synthetic, min_obs=min_forward_obs
        )
        basket_etf_beta, basket_etf_corr, _ = _beta_and_correlation(
            residual_basket, etf_blend, min_obs=min_forward_obs
        )
        raw_period_return = (
            float((1.0 + raw_basket.dropna()).prod() - 1.0) if raw_basket.notna().any() else np.nan
        )

        if theme_detail.empty:
            rank_ic_beta = rank_ic_corr = spread_beta = spread_corr = np.nan
        else:
            indexed = theme_detail.set_index("ticker")
            rank_ic_beta = _rank_ic(indexed["rank_score"], indexed["forward_synthetic_beta"])
            rank_ic_corr = _rank_ic(indexed["rank_score"], indexed["forward_synthetic_corr"])
            spread_beta = _top_bottom_spread(
                indexed["rank_score"], indexed["forward_synthetic_beta"]
            )
            spread_corr = _top_bottom_spread(
                indexed["rank_score"], indexed["forward_synthetic_corr"]
            )

        selected_mean_beta = (
            float(selected_detail["forward_synthetic_beta"].mean())
            if not selected_detail.empty
            else np.nan
        )
        selected_mean_corr = (
            float(selected_detail["forward_synthetic_corr"].mean())
            if not selected_detail.empty
            else np.nan
        )
        exposure_hit_rate = (
            float((selected_detail["forward_synthetic_beta"] > 0).mean())
            if not selected_detail.empty
            else np.nan
        )

        summary_rows.append(
            {
                "rebalance_date": pd.Timestamp(rebalance_date),
                "theme": theme,
                "n_selected": len(selected_tickers),
                "forward_rank_ic_beta": rank_ic_beta,
                "forward_rank_ic_corr": rank_ic_corr,
                "top_bottom_beta_spread": spread_beta,
                "top_bottom_corr_spread": spread_corr,
                "selected_mean_forward_beta": selected_mean_beta,
                "selected_mean_forward_corr": selected_mean_corr,
                "exposure_hit_rate": exposure_hit_rate,
                "basket_synthetic_beta": basket_syn_beta,
                "basket_synthetic_corr": basket_syn_corr,
                "basket_residual_etf_beta": basket_etf_beta,
                "basket_residual_etf_corr": basket_etf_corr,
                "raw_basket_period_return": raw_period_return,
                "n_benchmark_etfs": len(benchmark_set.get("etfs", [])),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def basket_correlation_metric(
    tickers: Sequence[str],
    forward_target_residuals: pd.DataFrame,
    benchmark: pd.Series,
    min_forward_obs: int = 8,
) -> float:
    """Metric used for the placebo distribution."""

    basket = _equal_weight_series(forward_target_residuals, tickers)
    _, correlation, _ = _beta_and_correlation(basket, benchmark, min_obs=min_forward_obs)
    return correlation


def empirical_placebo_pvalue(actual: float, placebo_values: Sequence[float]) -> float:
    """One sided empirical p value where larger exposure is better."""

    values = np.asarray(placebo_values, dtype=float)
    values = values[np.isfinite(values)]
    if not np.isfinite(actual) or len(values) == 0:
        return np.nan
    return float((1 + np.sum(values >= actual)) / (1 + len(values)))


def summarize_forward_evaluation(evaluation: pd.DataFrame) -> pd.DataFrame:
    """Aggregate period metrics by theme using medians and positive hit rates."""

    if evaluation.empty:
        return pd.DataFrame()

    grouped = evaluation.groupby("theme", sort=False)
    summary = grouped.agg(
        periods=("rebalance_date", "nunique"),
        median_rank_ic=("forward_rank_ic_corr", "median"),
        mean_rank_ic=("forward_rank_ic_corr", "mean"),
        median_top_bottom_spread=("top_bottom_corr_spread", "median"),
        median_selected_forward_corr=("selected_mean_forward_corr", "median"),
        median_basket_etf_corr=("basket_residual_etf_corr", "median"),
        median_exposure_hit_rate=("exposure_hit_rate", "median"),
        median_placebo_pvalue=("placebo_pvalue", "median"),
    )
    summary["rank_ic_positive_rate"] = grouped["forward_rank_ic_corr"].apply(
        lambda x: float((x.dropna() > 0).mean()) if x.notna().any() else np.nan
    )
    return summary.reset_index()


def make_semantic_review_sample(
    scores: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
    top_k: int = 10,
    controls_per_candidate: int = 1,
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Create a blinded economic relevance review sheet and a private key.

    ``metadata`` may be indexed by ticker and contain ``sector`` and
    ``market_cap``. When supplied, controls are drawn from the same sector and
    nearest market capitalization. Otherwise, controls are sampled from the
    middle of the score distribution.
    """

    rng = np.random.default_rng(random_state)
    metadata = metadata.copy() if metadata is not None else pd.DataFrame()
    rows = []

    for theme, group in scores.groupby("theme", sort=False):
        ranked = group.sort_values("penalized_distance")
        candidates = ranked.loc[ranked["eligible"]].head(top_k)
        candidate_set = set(candidates["ticker"])
        pool = ranked.loc[~ranked["ticker"].isin(candidate_set)].copy()

        for candidate in candidates.itertuples(index=False):
            rows.append(
                {
                    "theme": theme,
                    "ticker": candidate.ticker,
                    "model_group": "candidate",
                    "model_rank": getattr(candidate, "rank", np.nan),
                    "rank_score": candidate.rank_score,
                }
            )

            for _ in range(controls_per_candidate):
                if pool.empty:
                    break
                if not metadata.empty and candidate.ticker in metadata.index:
                    candidate_meta = metadata.loc[candidate.ticker]
                    eligible_pool = pool.copy()
                    if "sector" in metadata.columns and pd.notna(candidate_meta.get("sector")):
                        same_sector = [
                            ticker
                            for ticker in eligible_pool["ticker"]
                            if ticker in metadata.index
                            and metadata.loc[ticker].get("sector") == candidate_meta.get("sector")
                        ]
                        if same_sector:
                            eligible_pool = eligible_pool.loc[eligible_pool["ticker"].isin(same_sector)]
                    if "market_cap" in metadata.columns and pd.notna(candidate_meta.get("market_cap")):
                        eligible_pool = eligible_pool.assign(
                            cap_distance=eligible_pool["ticker"].map(
                                lambda ticker: abs(
                                    np.log(max(float(metadata.loc[ticker, "market_cap"]), 1.0))
                                    - np.log(max(float(candidate_meta["market_cap"]), 1.0))
                                )
                                if ticker in metadata.index
                                and pd.notna(metadata.loc[ticker, "market_cap"])
                                else np.inf
                            )
                        ).sort_values("cap_distance")
                        control_row = eligible_pool.iloc[0]
                    else:
                        control_row = eligible_pool.iloc[int(rng.integers(len(eligible_pool)))]
                else:
                    middle = pool.iloc[len(pool) // 4 : max(len(pool) // 4 + 1, 3 * len(pool) // 4)]
                    control_row = middle.iloc[int(rng.integers(len(middle)))] if not middle.empty else pool.iloc[0]

                control_ticker = control_row["ticker"]
                rows.append(
                    {
                        "theme": theme,
                        "ticker": control_ticker,
                        "model_group": "control",
                        "model_rank": control_row.get("rank", np.nan),
                        "rank_score": control_row["rank_score"],
                    }
                )
                pool = pool.loc[pool["ticker"] != control_ticker]

    key = pd.DataFrame(rows).drop_duplicates(["theme", "ticker", "model_group"]).reset_index(drop=True)
    key.insert(0, "review_id", [f"R{i:04d}" for i in range(1, len(key) + 1)])

    if not metadata.empty:
        for column in ["company_name", "sector", "market_cap"]:
            if column in metadata.columns:
                key[column] = key["ticker"].map(metadata[column])

    review_columns = ["review_id", "theme", "ticker"]
    review_columns += [
        column for column in ["company_name", "sector", "market_cap"] if column in key.columns
    ]
    review = key.loc[:, review_columns].sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    review["relevance_score"] = np.nan
    review["reviewer_notes"] = ""
    return {"review_sheet": review, "review_key": key}


def evaluate_semantic_review(
    completed_review: pd.DataFrame,
    review_key: pd.DataFrame,
    positive_threshold: float = 2.0,
) -> pd.DataFrame:
    """Calculate precision and enrichment from completed blinded reviews."""

    merged = completed_review.merge(review_key, on=["review_id", "theme", "ticker"], how="inner")
    merged["is_relevant"] = merged["relevance_score"] >= positive_threshold

    rows = []
    for theme, group in merged.groupby("theme", sort=False):
        candidate = group.loc[group["model_group"] == "candidate"]
        control = group.loc[group["model_group"] == "control"]
        candidate_precision = float(candidate["is_relevant"].mean()) if len(candidate) else np.nan
        control_precision = float(control["is_relevant"].mean()) if len(control) else np.nan
        enrichment = (
            candidate_precision / control_precision
            if np.isfinite(candidate_precision) and control_precision > 0
            else np.nan
        )
        rows.append(
            {
                "theme": theme,
                "candidate_precision": candidate_precision,
                "control_precision": control_precision,
                "enrichment": enrichment,
                "candidate_mean_relevance": float(candidate["relevance_score"].mean())
                if len(candidate)
                else np.nan,
                "control_mean_relevance": float(control["relevance_score"].mean())
                if len(control)
                else np.nan,
                "n_candidates_reviewed": len(candidate),
                "n_controls_reviewed": len(control),
            }
        )
    return pd.DataFrame(rows)
