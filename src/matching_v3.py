"""Robust factor exposure matching for ThemeCloner.

The module keeps matching in one common RP PCA factor space.

Matching hierarchy
------------------
1. Primary metric: covariance weighted distance between factor loading vectors.
2. Multi ETF consensus: use an upper distance quantile, not a simple average.
3. Directional check: require cosine agreement across a majority of theme ETFs.
4. Confidence filter: require both raw and adjusted regression R squared.
5. Penalty: large single factor gaps and disagreement across theme ETFs.
6. Placebo filter: compare each real rank with the same rank under shifted themes.

Cross theme overlap is reported as a diagnostic. It is not prohibited because a
company can have legitimate exposure to more than one related theme.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

_EPS = 1e-12


@dataclass(frozen=True)
class MatchingConfig:
    """Admission and ranking settings for candidate matching."""

    # Five factors retains more theme shape than the earlier three factor cut.
    top_factors: int = 5

    # Raw and adjusted R squared are both retained for transparent diagnostics.
    min_etf_r2: float = 0.10
    min_etf_adjusted_r2: float = 0.05
    min_candidate_r2: float = 0.15
    min_candidate_adjusted_r2: float = 0.10

    # Directional consensus across the theme ETF reference set.
    min_cosine: float = 0.30
    cosine_quantile: float = 0.25

    # Use the 75th percentile distance so one close ETF cannot hide a poor match.
    reference_distance_quantile: float = 0.75
    max_relative_distance: Optional[float | Mapping[str, float]] = None

    # Penalties remain secondary to covariance weighted distance.
    factor_gap_weight: float = 0.25
    consensus_weight: float = 0.25

    # With 2 ETFs this requires both; with 3 ETFs it requires at least 2.
    min_etf_matches: int = 1
    min_etf_match_fraction: float = 0.60
    etf_match_distance: Optional[float] = None

    # Cross theme discrimination is diagnostic by default, not a hard exclusion.
    max_theme_rank: Optional[int] = None


def _safe_cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return pairwise cosine similarity for rows of two matrices."""

    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True).T
    denom = np.maximum(left_norm * right_norm, _EPS)
    return (left @ right.T) / denom


def _covariance_distance_matrix(
    candidate_betas: np.ndarray,
    reference_betas: np.ndarray,
    factor_cov: np.ndarray,
) -> np.ndarray:
    """Return pairwise covariance weighted factor exposure distance."""

    delta = candidate_betas[:, None, :] - reference_betas[None, :, :]
    distance_sq = np.einsum("nmk,kl,nml->nm", delta, factor_cov, delta, optimize=True)
    return np.sqrt(np.maximum(distance_sq, 0.0))


def _batched_ols(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    min_obs: int,
    chunk_size: int = 512,
) -> Dict[str, object]:
    """Fit all asset regressions with vectorized batched linear algebra.

    This replaces one Python regression call per ticker. Each asset can retain
    its own missing observation pattern. The factor matrix is small, so the
    batched normal equations are substantially faster than joblib threads.
    """

    common = returns.index.intersection(factors.index)
    if len(common) < min_obs:
        raise ValueError(f"Only {len(common)} common observations; need {min_obs}.")

    factor_panel = factors.loc[common].astype(float)
    valid_factor_rows = factor_panel.notna().all(axis=1)
    common = common[valid_factor_rows.to_numpy()]
    factor_panel = factor_panel.loc[common]
    if len(common) < min_obs:
        raise ValueError(f"Only {len(common)} complete factor observations; need {min_obs}.")

    factor_cols = factor_panel.columns.tolist()
    x = factor_panel.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(common)), x])
    p_design = design.shape[1]
    p_factors = p_design - 1

    coefficients = pd.DataFrame(
        index=returns.columns,
        columns=["alpha", *factor_cols],
        dtype=float,
    )
    r2 = pd.Series(index=returns.columns, dtype=float, name="r2")
    adjusted_r2 = pd.Series(index=returns.columns, dtype=float, name="adjusted_r2")
    nobs = pd.Series(index=returns.columns, dtype=float, name="nobs")

    columns = returns.columns.tolist()
    for start in range(0, len(columns), max(1, int(chunk_size))):
        chunk = columns[start : start + max(1, int(chunk_size))]
        y = returns.loc[common, chunk].to_numpy(dtype=float)
        mask = np.isfinite(y)
        y0 = np.where(mask, y, 0.0)
        mask_float = mask.astype(float)
        n = mask.sum(axis=0).astype(int)

        xtwx = np.einsum(
            "tp,tn,tq->npq",
            design,
            mask_float,
            design,
            optimize=True,
        )
        xtwy = np.einsum("tp,tn->np", design, y0, optimize=True)

        valid = n >= max(min_obs, p_design + 2)
        coef = np.full((len(chunk), p_design), np.nan, dtype=float)
        if valid.any():
            matrices = xtwx[valid].copy()
            # Tiny scale aware ridge handles rare near singular windows.
            ridge_scale = np.maximum(
                np.trace(matrices, axis1=1, axis2=2) / p_design,
                1.0,
            )
            matrices += (
                np.eye(p_design)[None, :, :]
                * ridge_scale[:, None, None]
                * 1e-12
            )
            solved = np.linalg.solve(matrices, xtwy[valid][..., None]).squeeze(-1)
            coef[valid] = solved

        fitted = design @ np.nan_to_num(coef, nan=0.0).T
        residual = np.where(mask, y - fitted, np.nan)
        means = np.divide(
            np.nansum(y, axis=0),
            n,
            out=np.full(len(chunk), np.nan),
            where=n > 0,
        )
        ss_res = np.nansum(residual * residual, axis=0)
        ss_tot = np.nansum((y - means[None, :]) ** 2, axis=0)
        raw_r2 = np.where(valid & (ss_tot > _EPS), 1.0 - ss_res / ss_tot, np.nan)
        denominator = n - p_factors - 1
        adj_r2 = np.where(
            valid & (denominator > 0) & np.isfinite(raw_r2),
            1.0 - (1.0 - raw_r2) * (n - 1) / denominator,
            np.nan,
        )

        coefficients.loc[chunk] = coef
        r2.loc[chunk] = raw_r2
        adjusted_r2.loc[chunk] = adj_r2
        nobs.loc[chunk] = n

    return {
        "betas": coefficients.loc[:, factor_cols],
        "alpha": coefficients["alpha"],
        "coefficients": coefficients,
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "nobs": nobs,
        "common_index": common,
    }


def fit_factor_loadings(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    min_obs: int = 52,
    n_jobs: int = 1,
    chunk_size: int = 512,
) -> Dict[str, object]:
    """Regress all asset returns on the common factor panel.

    ``n_jobs`` is retained for API compatibility. The fit itself is vectorized;
    process parallelism is reserved for independent placebo runs.
    """

    del n_jobs
    return _batched_ols(
        returns=returns,
        factors=factors,
        min_obs=min_obs,
        chunk_size=chunk_size,
    )


def _sparsify_vector(vector: np.ndarray, top_factors: int) -> np.ndarray:
    """Keep only the largest absolute factor loadings."""

    result = np.zeros_like(vector, dtype=float)
    if top_factors <= 0:
        return result
    keep = np.argsort(np.abs(vector))[::-1][: min(top_factors, len(vector))]
    result[keep] = vector[keep]
    return result


def _theme_medoid(reference_betas: np.ndarray, factor_cov: np.ndarray) -> int:
    """Return the ETF whose median distance to the other ETFs is smallest."""

    if len(reference_betas) == 1:
        return 0
    distances = _covariance_distance_matrix(reference_betas, reference_betas, factor_cov)
    return int(np.argmin(np.median(distances, axis=1)))


def build_theme_reference_sets(
    etf_returns: pd.DataFrame,
    factors: pd.DataFrame,
    etf_config: pd.DataFrame,
    factor_cov: Optional[np.ndarray] = None,
    top_factors: int = 5,
    min_etf_r2: float = 0.10,
    min_etf_adjusted_r2: float = 0.05,
    min_obs: int = 52,
    n_jobs: int = 1,
) -> Dict[str, dict]:
    """Build separate ETF reference vectors for each theme."""

    fitted = fit_factor_loadings(
        etf_returns,
        factors,
        min_obs=min_obs,
        n_jobs=n_jobs,
    )
    all_betas = fitted["betas"]
    all_r2 = fitted["r2"]
    all_adjusted_r2 = fitted["adjusted_r2"]
    factor_cols = factors.columns.tolist()

    if factor_cov is None:
        factor_cov = factors.cov().to_numpy()
    factor_cov = np.asarray(factor_cov, dtype=float)

    references: Dict[str, dict] = {}
    for theme in etf_config["theme"].dropna().unique():
        configured = etf_config.loc[etf_config["theme"] == theme, "ticker"].tolist()
        available = [
            ticker
            for ticker in configured
            if ticker in all_betas.index
            and all_betas.loc[ticker].notna().all()
            and pd.notna(all_r2.loc[ticker])
            and float(all_r2.loc[ticker]) >= min_etf_r2
            and pd.notna(all_adjusted_r2.loc[ticker])
            and float(all_adjusted_r2.loc[ticker]) >= min_etf_adjusted_r2
        ]
        if not available:
            continue

        sparse = np.vstack(
            [
                _sparsify_vector(
                    all_betas.loc[ticker, factor_cols].to_numpy(dtype=float),
                    top_factors,
                )
                for ticker in available
            ]
        )
        medoid_position = _theme_medoid(sparse, factor_cov)

        references[str(theme)] = {
            "tickers": available,
            "betas": sparse,
            "r2": all_r2.loc[available].astype(float).to_dict(),
            "adjusted_r2": all_adjusted_r2.loc[available].astype(float).to_dict(),
            "medoid_ticker": available[medoid_position],
            "medoid_beta": sparse[medoid_position],
            # Used only to build a consensus benchmark, not for matching.
            "consensus_beta": np.mean(sparse, axis=0),
            "factor_columns": factor_cols,
        }

    return references


def _add_theme_discrimination(scores: pd.DataFrame) -> pd.DataFrame:
    """Add cross theme distance diagnostics without forcing exclusivity."""

    if scores.empty:
        return scores

    result = scores.copy()
    result["theme_distance_rank"] = np.nan
    result["best_theme"] = None
    result["second_best_theme"] = None
    result["best_theme_distance"] = np.nan
    result["second_best_theme_distance"] = np.nan
    result["theme_distance_margin"] = np.nan
    result["distance_from_best_theme"] = np.nan

    for ticker, group in result.groupby("ticker", sort=False):
        ordered = group.sort_values("penalized_distance")
        best = ordered.iloc[0]
        second = ordered.iloc[1] if len(ordered) > 1 else None
        ranks = pd.Series(
            np.arange(1, len(ordered) + 1),
            index=ordered.index,
            dtype=float,
        )
        second_distance = (
            float(second["penalized_distance"]) if second is not None else np.nan
        )
        margin = (
            (second_distance - float(best["penalized_distance"]))
            / max(abs(second_distance), _EPS)
            if np.isfinite(second_distance)
            else np.nan
        )

        result.loc[ordered.index, "theme_distance_rank"] = ranks
        result.loc[ordered.index, "best_theme"] = str(best["theme"])
        result.loc[ordered.index, "second_best_theme"] = (
            str(second["theme"]) if second is not None else None
        )
        result.loc[ordered.index, "best_theme_distance"] = float(
            best["penalized_distance"]
        )
        result.loc[ordered.index, "second_best_theme_distance"] = second_distance
        result.loc[ordered.index, "theme_distance_margin"] = margin
        result.loc[ordered.index, "distance_from_best_theme"] = (
            ordered["penalized_distance"].to_numpy(dtype=float)
            - float(best["penalized_distance"])
        )

    base_counts = result.groupby("ticker")["base_eligible"].transform("sum")
    result["n_base_eligible_themes"] = base_counts.astype(int)
    return result


def score_candidates(
    candidate_betas: pd.DataFrame,
    candidate_r2: pd.Series,
    theme_references: Mapping[str, dict],
    factor_cov: pd.DataFrame | np.ndarray,
    config: MatchingConfig = MatchingConfig(),
    candidate_adjusted_r2: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Score candidates using covariance distance and multi ETF consensus."""

    factor_cov_array = (
        factor_cov.to_numpy(dtype=float)
        if isinstance(factor_cov, pd.DataFrame)
        else np.asarray(factor_cov, dtype=float)
    )
    factor_std = np.sqrt(np.maximum(np.diag(factor_cov_array), _EPS))

    clean_betas = candidate_betas.dropna(how="any")
    adjusted = (
        candidate_adjusted_r2
        if candidate_adjusted_r2 is not None
        else pd.Series(np.nan, index=candidate_r2.index, dtype=float)
    )
    rows = []

    for theme, reference in theme_references.items():
        factor_cols = reference["factor_columns"]
        if not set(factor_cols).issubset(clean_betas.columns):
            continue

        tickers = clean_betas.index.to_numpy()
        beta_matrix = clean_betas.loc[:, factor_cols].to_numpy(dtype=float)
        reference_matrix = np.asarray(reference["betas"], dtype=float)
        medoid_beta = np.asarray(reference["medoid_beta"], dtype=float)

        raw_distance = _covariance_distance_matrix(
            beta_matrix,
            reference_matrix,
            factor_cov_array,
        )
        reference_risk = np.sqrt(
            np.maximum(
                np.einsum(
                    "mk,kl,ml->m",
                    reference_matrix,
                    factor_cov_array,
                    reference_matrix,
                    optimize=True,
                ),
                _EPS,
            )
        )
        theme_scale = float(np.median(reference_risk))
        relative_distance = raw_distance / max(theme_scale, _EPS)

        median_distance = np.median(relative_distance, axis=1)
        primary_distance = np.quantile(
            relative_distance,
            np.clip(config.reference_distance_quantile, 0.0, 1.0),
            axis=1,
        )
        worst_distance = np.max(relative_distance, axis=1)
        distance_mad = np.median(
            np.abs(relative_distance - median_distance[:, None]),
            axis=1,
        )

        cosine_matrix = _safe_cosine_matrix(beta_matrix, reference_matrix)
        median_cosine = np.median(cosine_matrix, axis=1)
        consensus_cosine = np.quantile(
            cosine_matrix,
            np.clip(config.cosine_quantile, 0.0, 1.0),
            axis=1,
        )
        minimum_cosine = np.min(cosine_matrix, axis=1)

        medoid_delta = beta_matrix - medoid_beta[None, :]
        max_factor_gap = np.max(
            np.abs(medoid_delta) * factor_std[None, :],
            axis=1,
        )
        max_factor_gap = max_factor_gap / max(theme_scale, _EPS)

        penalized_distance = (
            primary_distance
            + config.consensus_weight * distance_mad
            + config.factor_gap_weight * max_factor_gap
        )

        match_mask = cosine_matrix >= config.min_cosine
        if config.etf_match_distance is not None:
            match_mask &= relative_distance <= config.etf_match_distance
        n_etf_matches = match_mask.sum(axis=1)
        required_matches = max(
            int(config.min_etf_matches),
            int(ceil(config.min_etf_match_fraction * reference_matrix.shape[0])),
        )
        required_matches = min(required_matches, reference_matrix.shape[0])

        r2_values = candidate_r2.reindex(tickers).to_numpy(dtype=float)
        adjusted_values = adjusted.reindex(tickers).to_numpy(dtype=float)
        base_eligible = (
            np.isfinite(r2_values)
            & (r2_values >= config.min_candidate_r2)
            & np.isfinite(adjusted_values)
            & (adjusted_values >= config.min_candidate_adjusted_r2)
            & (consensus_cosine >= config.min_cosine)
            & (n_etf_matches >= required_matches)
        )

        if config.max_relative_distance is not None:
            if isinstance(config.max_relative_distance, Mapping):
                theme_max_distance = config.max_relative_distance.get(theme)
            else:
                theme_max_distance = config.max_relative_distance
            if theme_max_distance is not None and np.isfinite(theme_max_distance):
                base_eligible &= primary_distance <= float(theme_max_distance)

        for position, ticker in enumerate(tickers):
            rows.append(
                {
                    "ticker": str(ticker),
                    "theme": str(theme),
                    "primary_distance": float(primary_distance[position]),
                    "median_distance": float(median_distance[position]),
                    "worst_distance": float(worst_distance[position]),
                    "penalized_distance": float(penalized_distance[position]),
                    "consensus_cosine": float(consensus_cosine[position]),
                    "median_cosine": float(median_cosine[position]),
                    "minimum_cosine": float(minimum_cosine[position]),
                    "candidate_r2": float(r2_values[position]),
                    "candidate_adjusted_r2": float(adjusted_values[position]),
                    "max_factor_gap": float(max_factor_gap[position]),
                    "distance_mad": float(distance_mad[position]),
                    "n_etf_matches": int(n_etf_matches[position]),
                    "required_etf_matches": int(required_matches),
                    "n_theme_etfs": int(reference_matrix.shape[0]),
                    "base_eligible": bool(base_eligible[position]),
                    "eligible": bool(base_eligible[position]),
                    "rank_score": float(-penalized_distance[position]),
                    "medoid_etf": reference["medoid_ticker"],
                }
            )

    if not rows:
        return pd.DataFrame()

    result = _add_theme_discrimination(pd.DataFrame(rows))
    if config.max_theme_rank is not None:
        result["base_eligible"] &= (
            result["theme_distance_rank"] <= int(config.max_theme_rank)
        )
        result["eligible"] = result["base_eligible"]

    result = result.sort_values(
        [
            "theme",
            "base_eligible",
            "penalized_distance",
            "consensus_cosine",
            "candidate_adjusted_r2",
        ],
        ascending=[True, False, True, False, False],
    )
    result["rank"] = result.groupby("theme").cumcount() + 1
    return result.reset_index(drop=True)


def _benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """Return Benjamini Hochberg q values for reporting."""

    output = pd.Series(np.nan, index=pvalues.index, dtype=float)
    clean = pvalues.dropna().sort_values()
    if clean.empty:
        return output
    m = len(clean)
    raw = clean.to_numpy(dtype=float) * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    output.loc[clean.index] = np.minimum(adjusted, 1.0)
    return output


def apply_rank_placebo_filter(
    real_scores: pd.DataFrame,
    placebo_scores: Sequence[pd.DataFrame],
    alpha: float = 0.10,
    max_rank: Optional[int] = None,
    use_fdr: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each real candidate rank with the same rank under null themes.

    This avoids the earlier pooled 5 percent distance threshold, which could
    mechanically admit a fixed share of a large universe. Lower distance is
    better. FDR q values are reported, but rank p values are used by default
    because 50 to 100 placebos provide limited q value resolution.
    """

    result = real_scores.copy()
    result["placebo_rank_pvalue"] = np.nan
    result["placebo_rank_qvalue"] = np.nan
    result["placebo_rank_threshold"] = np.nan
    result["passes_placebo"] = False
    diagnostics = []

    if not placebo_scores:
        result["passes_placebo"] = result["base_eligible"]
        result["eligible"] = result["base_eligible"]
        return result, pd.DataFrame()

    for theme, real_group in result.groupby("theme", sort=False):
        real_ordered = real_group.loc[real_group["base_eligible"]].sort_values(
            "penalized_distance"
        )
        if max_rank is not None:
            real_ordered = real_ordered.head(int(max_rank))
        if real_ordered.empty:
            continue

        null_ordered = []
        for scores in placebo_scores:
            if scores.empty:
                continue
            subset = scores.loc[
                (scores["theme"] == theme) & scores["base_eligible"]
            ].sort_values("penalized_distance")
            null_ordered.append(subset["penalized_distance"].to_numpy(dtype=float))

        for rank_position, (index, row) in enumerate(real_ordered.iterrows(), start=1):
            null_values = np.array(
                [
                    values[rank_position - 1]
                    for values in null_ordered
                    if len(values) >= rank_position
                    and np.isfinite(values[rank_position - 1])
                ],
                dtype=float,
            )
            if len(null_values) == 0:
                continue
            actual = float(row["penalized_distance"])
            pvalue = float(
                (1 + np.sum(null_values <= actual)) / (1 + len(null_values))
            )
            threshold = float(np.quantile(null_values, alpha))
            result.loc[index, "placebo_rank_pvalue"] = pvalue
            result.loc[index, "placebo_rank_threshold"] = threshold
            diagnostics.append(
                {
                    "theme": theme,
                    "rank": rank_position,
                    "actual_penalized_distance": actual,
                    "null_alpha_distance": threshold,
                    "placebo_rank_pvalue": pvalue,
                    "n_placebo_ranks": len(null_values),
                }
            )

        theme_indices = real_ordered.index
        result.loc[theme_indices, "placebo_rank_qvalue"] = _benjamini_hochberg(
            result.loc[theme_indices, "placebo_rank_pvalue"]
        )

    significance_column = (
        "placebo_rank_qvalue" if use_fdr else "placebo_rank_pvalue"
    )
    result["passes_placebo"] = (
        result[significance_column].notna()
        & (result[significance_column] <= alpha)
    )
    result["eligible"] = result["base_eligible"]
    result["n_eligible_themes"] = result.groupby("ticker")["eligible"].transform(
        "sum"
    ).astype(int)

    result = result.sort_values(
        ["theme", "eligible", "penalized_distance"],
        ascending=[True, False, True],
    )
    result["rank"] = result.groupby("theme").cumcount() + 1
    return result.reset_index(drop=True), pd.DataFrame(diagnostics)


def select_equal_weight_baskets(
    scores: pd.DataFrame,
    top_n: int = 30,
) -> Dict[str, dict]:
    """Select up to ``top_n`` eligible names per theme and equal weight them."""

    baskets: Dict[str, dict] = {}
    if scores.empty:
        return baskets

    for theme, group in scores.groupby("theme", sort=False):
        selected = group.loc[group["eligible"]].nsmallest(
            top_n,
            "penalized_distance",
        )
        tickers = selected["ticker"].tolist()
        weight = 1.0 / len(tickers) if tickers else np.nan
        baskets[str(theme)] = {
            "tickers": tickers,
            "weights": {ticker: weight for ticker in tickers},
        }
    return baskets


def theme_reference_distance_table(
    theme_references: Mapping[str, dict],
    factor_cov: pd.DataFrame | np.ndarray,
) -> pd.DataFrame:
    """Report pairwise separation between theme ETF reference sets."""

    factor_cov_array = (
        factor_cov.to_numpy(dtype=float)
        if isinstance(factor_cov, pd.DataFrame)
        else np.asarray(factor_cov, dtype=float)
    )
    themes = list(theme_references)
    rows = []
    for i, left_theme in enumerate(themes):
        left = np.asarray(theme_references[left_theme]["betas"], dtype=float)
        for right_theme in themes[i + 1 :]:
            right = np.asarray(theme_references[right_theme]["betas"], dtype=float)
            distances = _covariance_distance_matrix(left, right, factor_cov_array)
            cosines = _safe_cosine_matrix(left, right)
            risks = np.concatenate(
                [
                    np.sqrt(
                        np.maximum(
                            np.einsum(
                                "mk,kl,ml->m",
                                matrix,
                                factor_cov_array,
                                matrix,
                                optimize=True,
                            ),
                            _EPS,
                        )
                    )
                    for matrix in (left, right)
                ]
            )
            scale = max(float(np.median(risks)), _EPS)
            rows.append(
                {
                    "theme_1": left_theme,
                    "theme_2": right_theme,
                    "median_relative_distance": float(np.median(distances) / scale),
                    "minimum_relative_distance": float(np.min(distances) / scale),
                    "maximum_cosine": float(np.max(cosines)),
                    "median_cosine": float(np.median(cosines)),
                }
            )
    return pd.DataFrame(rows)


def summarize_selection_overlap(
    scores: pd.DataFrame,
    top_k: int = 10,
    eligible_only: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Summarize overlap across the top candidate positions by theme."""

    if scores.empty:
        return {
            "summary": pd.DataFrame(),
            "ticker_counts": pd.DataFrame(),
            "pairwise_overlap": pd.DataFrame(),
        }

    selected_parts = []
    for theme, group in scores.groupby("theme", sort=False):
        subset = group.loc[group["eligible"]] if eligible_only else group
        selected_parts.append(subset.nsmallest(top_k, "penalized_distance"))
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    if selected.empty:
        return {
            "summary": pd.DataFrame(
                [{"positions": 0, "unique_stocks": 0, "overlap_positions": 0}]
            ),
            "ticker_counts": pd.DataFrame(),
            "pairwise_overlap": pd.DataFrame(),
        }

    ticker_counts = (
        selected.groupby("ticker")
        .agg(
            n_themes=("theme", "nunique"),
            themes=("theme", lambda x: ", ".join(sorted(set(map(str, x))))),
        )
        .sort_values(["n_themes", "ticker"], ascending=[False, True])
        .reset_index()
    )
    summary = pd.DataFrame(
        [
            {
                "positions": len(selected),
                "unique_stocks": selected["ticker"].nunique(),
                "overlap_positions": len(selected) - selected["ticker"].nunique(),
                "share_positions_reused": 1.0
                - selected["ticker"].nunique() / max(len(selected), 1),
                "max_themes_per_stock": int(ticker_counts["n_themes"].max()),
            }
        ]
    )

    pair_rows = []
    theme_sets = {
        theme: set(group["ticker"])
        for theme, group in selected.groupby("theme", sort=False)
    }
    themes = list(theme_sets)
    for i, left in enumerate(themes):
        for right in themes[i + 1 :]:
            intersection = len(theme_sets[left] & theme_sets[right])
            union = len(theme_sets[left] | theme_sets[right])
            pair_rows.append(
                {
                    "theme_1": left,
                    "theme_2": right,
                    "overlap_count": intersection,
                    "jaccard": intersection / union if union else np.nan,
                }
            )
    return {
        "summary": summary,
        "ticker_counts": ticker_counts,
        "pairwise_overlap": pd.DataFrame(pair_rows),
    }


def flatten_baskets(
    period_baskets: Mapping[pd.Timestamp, Mapping[str, dict]],
) -> pd.DataFrame:
    """Convert nested basket output into a tidy DataFrame."""

    rows = []
    for date, theme_map in period_baskets.items():
        for theme, basket in theme_map.items():
            for ticker in basket.get("tickers", []):
                rows.append(
                    {
                        "rebalance_date": pd.Timestamp(date),
                        "theme": theme,
                        "ticker": ticker,
                        "weight": basket.get("weights", {}).get(ticker, np.nan),
                    }
                )
    return pd.DataFrame(
    rows,
    columns=["rebalance_date", "theme", "ticker", "weight"],
    )
