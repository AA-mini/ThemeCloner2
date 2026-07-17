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

        # Build the forward synthetic benchmark from all surviving theme ETFs.
        # Matching still compares candidates with each ETF separately. The mean
        # here is an explicit equal weight consensus benchmark for evaluation.
        reference_betas = np.asarray(reference["betas"], dtype=float)
        synthetic_components = (
            forward_factors.loc[:, factor_cols].to_numpy(dtype=float)
            @ reference_betas.T
        )
        synthetic = pd.Series(
            np.mean(synthetic_components, axis=1),
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


def _panel_beta_and_correlation(
    returns: pd.DataFrame,
    benchmark: pd.Series,
    min_obs: int = 8,
) -> pd.DataFrame:
    """Vectorized beta and correlation for many assets against one benchmark."""

    columns = returns.columns.tolist()
    output = pd.DataFrame(
        index=columns,
        columns=["beta", "correlation", "nobs"],
        dtype=float,
    )
    common = returns.index.intersection(benchmark.index)
    if len(common) == 0 or not columns:
        return output

    x = benchmark.reindex(common).to_numpy(dtype=float)
    y = returns.reindex(common).to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(x)[:, None]
    n = valid.sum(axis=0)
    x0 = np.where(valid, x[:, None], 0.0)
    y0 = np.where(valid, y, 0.0)
    x_mean = np.divide(
        x0.sum(axis=0),
        n,
        out=np.full(len(columns), np.nan),
        where=n > 0,
    )
    y_mean = np.divide(
        y0.sum(axis=0),
        n,
        out=np.full(len(columns), np.nan),
        where=n > 0,
    )
    x_centered = np.where(valid, x[:, None] - x_mean[None, :], 0.0)
    y_centered = np.where(valid, y - y_mean[None, :], 0.0)
    covariance = np.sum(x_centered * y_centered, axis=0)
    x_var = np.sum(x_centered * x_centered, axis=0)
    y_var = np.sum(y_centered * y_centered, axis=0)

    usable = (n >= min_obs) & (x_var > _EPS) & (y_var > _EPS)
    beta = np.full(len(columns), np.nan)
    correlation = np.full(len(columns), np.nan)
    beta[usable] = covariance[usable] / x_var[usable]
    correlation[usable] = covariance[usable] / np.sqrt(
        x_var[usable] * y_var[usable]
    )
    output["beta"] = beta
    output["correlation"] = correlation
    output["nobs"] = n
    return output


def evaluate_forward_period(
    scores: pd.DataFrame,
    baskets: Mapping[str, dict],
    forward_target_residuals: pd.DataFrame,
    forward_target_raw: pd.DataFrame,
    benchmarks: Mapping[str, dict],
    rebalance_date: pd.Timestamp,
    min_forward_obs: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate forward exposure metrics for one holding period."""

    summary_rows = []
    detail_frames = []

    for theme, benchmark_set in benchmarks.items():
        theme_scores = scores.loc[scores["theme"] == theme].copy()
        if "base_eligible" in theme_scores.columns:
            theme_scores = theme_scores.loc[theme_scores["base_eligible"]]
        if theme_scores.empty:
            continue

        tickers = [
            ticker
            for ticker in theme_scores["ticker"]
            if ticker in forward_target_residuals.columns
        ]
        if not tickers:
            continue
        theme_scores = theme_scores.set_index("ticker").loc[tickers]

        synthetic = benchmark_set["synthetic"]
        etf_blend = benchmark_set["etf_blend"]
        syn_metrics = _panel_beta_and_correlation(
            forward_target_residuals.loc[:, tickers],
            synthetic,
            min_obs=min_forward_obs,
        )
        etf_metrics = _panel_beta_and_correlation(
            forward_target_residuals.loc[:, tickers],
            etf_blend,
            min_obs=min_forward_obs,
        )

        selected_tickers = baskets.get(theme, {}).get("tickers", [])
        detail = pd.DataFrame(
            {
                "rebalance_date": pd.Timestamp(rebalance_date),
                "theme": theme,
                "ticker": tickers,
                "rank_score": theme_scores["rank_score"].to_numpy(dtype=float),
                "penalized_distance": theme_scores["penalized_distance"].to_numpy(dtype=float),
                "eligible": theme_scores["eligible"].to_numpy(dtype=bool),
                "selected": [ticker in selected_tickers for ticker in tickers],
                "forward_synthetic_beta": syn_metrics.loc[tickers, "beta"].to_numpy(dtype=float),
                "forward_synthetic_corr": syn_metrics.loc[tickers, "correlation"].to_numpy(dtype=float),
                "forward_etf_beta": etf_metrics.loc[tickers, "beta"].to_numpy(dtype=float),
                "forward_etf_corr": etf_metrics.loc[tickers, "correlation"].to_numpy(dtype=float),
                "forward_obs": np.maximum(
                    syn_metrics.loc[tickers, "nobs"].to_numpy(dtype=float),
                    etf_metrics.loc[tickers, "nobs"].to_numpy(dtype=float),
                ),
            }
        )
        for optional_column in [
            "candidate_r2",
            "candidate_adjusted_r2",
            "consensus_cosine",
            "placebo_rank_pvalue",
            "theme_distance_rank",
            "n_eligible_themes",
        ]:
            if optional_column in theme_scores.columns:
                detail[optional_column] = theme_scores[optional_column].to_numpy()
        detail_frames.append(detail)

        selected_detail = detail.loc[detail["selected"]]
        residual_basket = _equal_weight_series(
            forward_target_residuals,
            selected_tickers,
        )
        raw_basket = _equal_weight_series(
            forward_target_raw,
            selected_tickers,
        )
        basket_syn_beta, basket_syn_corr, _ = _beta_and_correlation(
            residual_basket,
            synthetic,
            min_obs=min_forward_obs,
        )
        basket_etf_beta, basket_etf_corr, _ = _beta_and_correlation(
            residual_basket,
            etf_blend,
            min_obs=min_forward_obs,
        )
        raw_period_return = (
            float((1.0 + raw_basket.dropna()).prod() - 1.0)
            if raw_basket.notna().any()
            else np.nan
        )

        indexed = detail.set_index("ticker")
        rank_ic_beta = _rank_ic(
            indexed["rank_score"],
            indexed["forward_synthetic_beta"],
        )
        rank_ic_corr = _rank_ic(
            indexed["rank_score"],
            indexed["forward_synthetic_corr"],
        )
        spread_beta = _top_bottom_spread(
            indexed["rank_score"],
            indexed["forward_synthetic_beta"],
        )
        spread_corr = _top_bottom_spread(
            indexed["rank_score"],
            indexed["forward_synthetic_corr"],
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
                "n_base_eligible": len(detail),
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

    details = (
        pd.concat(detail_frames, ignore_index=True)
        if detail_frames
        else pd.DataFrame()
    )
    return pd.DataFrame(summary_rows), details

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
        
        eligibility_col = (
            "base_eligible"
            if "base_eligible" in ranked.columns
            else "eligible"
        )

        candidates = ranked.loc[
            ranked[eligibility_col].fillna(False)
        ].head(top_k)
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
    
    review = (
        key.reindex(columns=review_columns)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    
    review["relevance_score"] = np.nan
    review["reviewer_notes"] = ""
    return {"review_sheet": review, "review_key": key}


def evaluate_semantic_review(
    completed_review: pd.DataFrame,
    review_key: pd.DataFrame,
    positive_threshold: float = 2.0,
) -> pd.DataFrame:
    """Calculate precision and enrichment using only completed reviews."""

    merged = completed_review.merge(
        review_key, on=["review_id", "theme", "ticker"], how="inner"
    )
    merged["relevance_score"] = pd.to_numeric(
        merged["relevance_score"], errors="coerce"
    )

    invalid = (
        merged["relevance_score"].notna()
        & ~merged["relevance_score"].between(0, 3)
    )
    if invalid.any():
        raise ValueError("relevance_score must be between 0 and 3.")

    merged["reviewed"] = merged["relevance_score"].notna()
    merged["is_relevant"] = np.where(
        merged["reviewed"],
        merged["relevance_score"] >= positive_threshold,
        np.nan,
    )

    rows = []
    for theme, group in merged.groupby("theme", sort=False):
        candidate_all = group.loc[group["model_group"] == "candidate"]
        control_all = group.loc[group["model_group"] == "control"]
        candidate = candidate_all.loc[candidate_all["reviewed"]]
        control = control_all.loc[control_all["reviewed"]]

        candidate_precision = (
            float(candidate["is_relevant"].mean()) if len(candidate) else np.nan
        )
        control_precision = (
            float(control["is_relevant"].mean()) if len(control) else np.nan
        )
        enrichment = (
            candidate_precision / control_precision
            if np.isfinite(candidate_precision)
            and np.isfinite(control_precision)
            and control_precision > 0
            else np.nan
        )

        rows.append(
            {
                "theme": theme,
                "candidate_precision": candidate_precision,
                "control_precision": control_precision,
                "enrichment": enrichment,
                "candidate_mean_relevance": (
                    float(candidate["relevance_score"].mean())
                    if len(candidate)
                    else np.nan
                ),
                "control_mean_relevance": (
                    float(control["relevance_score"].mean())
                    if len(control)
                    else np.nan
                ),
                "n_candidates_reviewed": len(candidate),
                "n_controls_reviewed": len(control),
                "n_candidates_total": len(candidate_all),
                "n_controls_total": len(control_all),
            }
        )
    return pd.DataFrame(rows)

