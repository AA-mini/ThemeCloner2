"""Robust factor exposure matching for ThemeCloner.

This module is intentionally additive. It does not alter the existing V1 or V2
projection functions. The revised walk forward notebook imports it explicitly.

Matching hierarchy
------------------
1. Primary metric: covariance weighted distance between factor loading vectors.
2. Secondary check: cosine similarity.
3. Confidence filter: candidate regression R squared.
4. Penalty: the largest single factor mismatch, plus disagreement across ETFs.

A theme is represented by its surviving ETF loading vectors, not by a simple
average. Candidate distance is measured against every ETF and aggregated with
the median. This avoids one extreme ETF loading offsetting a poor match to
another ETF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

_EPS = 1e-12


@dataclass(frozen=True)
class MatchingConfig:
    """Admission and ranking settings for candidate matching."""

    top_factors: int = 3
    min_etf_r2: float = 0.10
    min_candidate_r2: float = 0.10
    min_cosine: float = 0.20
    max_relative_distance: Optional[float | Mapping[str, float]] = None
    factor_gap_weight: float = 0.25
    consensus_weight: float = 0.25
    min_etf_matches: int = 1
    etf_match_distance: Optional[float] = None


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
    """Pairwise covariance weighted distance.

    Distance between candidate i and reference j is
    sqrt((beta_i - beta_j)' Sigma_F (beta_i - beta_j)).
    """

    delta = candidate_betas[:, None, :] - reference_betas[None, :, :]
    distance_sq = np.einsum("nmk,kl,nml->nm", delta, factor_cov, delta)
    return np.sqrt(np.maximum(distance_sq, 0.0))


def fit_factor_loadings(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    min_obs: int = 52,
) -> Dict[str, object]:
    """Regress each asset return on the common factor return panel.

    The regression includes an intercept. Betas are returned in the same factor
    order as ``factors``. Missing observations are handled asset by asset.
    """

    common = returns.index.intersection(factors.index)
    if len(common) < min_obs:
        raise ValueError(f"Only {len(common)} common observations; need {min_obs}.")

    factor_panel = factors.loc[common].astype(float)
    factor_cols = factor_panel.columns.tolist()
    betas = pd.DataFrame(index=returns.columns, columns=factor_cols, dtype=float)
    r2 = pd.Series(index=returns.columns, dtype=float, name="r2")
    alpha = pd.Series(index=returns.columns, dtype=float, name="alpha")
    nobs = pd.Series(index=returns.columns, dtype=float, name="nobs")

    for ticker in returns.columns:
        y = returns.loc[common, ticker].astype(float)
        valid = y.notna() & factor_panel.notna().all(axis=1)
        n = int(valid.sum())
        nobs.loc[ticker] = n
        if n < min_obs:
            continue

        x = factor_panel.loc[valid].to_numpy()
        yv = y.loc[valid].to_numpy()
        design = np.column_stack([np.ones(n), x])
        coef, *_ = np.linalg.lstsq(design, yv, rcond=None)
        fitted = design @ coef
        residual = yv - fitted
        ss_res = float(np.dot(residual, residual))
        centered = yv - yv.mean()
        ss_tot = float(np.dot(centered, centered))

        alpha.loc[ticker] = float(coef[0])
        betas.loc[ticker] = coef[1:]
        r2.loc[ticker] = 1.0 - ss_res / ss_tot if ss_tot > _EPS else np.nan

    return {
        "betas": betas,
        "r2": r2,
        "alpha": alpha,
        "nobs": nobs,
    }


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
    top_factors: int = 3,
    min_etf_r2: float = 0.10,
    min_obs: int = 52,
) -> Dict[str, dict]:
    """Build robust multi ETF reference sets for each theme.

    Each qualifying ETF remains a separate reference vector. The function also
    identifies a medoid ETF for the single factor mismatch penalty. No simple
    average is used for the candidate match.
    """

    fitted = fit_factor_loadings(etf_returns, factors, min_obs=min_obs)
    all_betas = fitted["betas"]
    all_r2 = fitted["r2"]
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
        ]
        if not available:
            continue

        sparse = np.vstack(
            [
                _sparsify_vector(all_betas.loc[ticker, factor_cols].to_numpy(dtype=float), top_factors)
                for ticker in available
            ]
        )
        medoid_position = _theme_medoid(sparse, factor_cov)

        references[str(theme)] = {
            "tickers": available,
            "betas": sparse,
            "r2": all_r2.loc[available].astype(float).to_dict(),
            "medoid_ticker": available[medoid_position],
            "medoid_beta": sparse[medoid_position],
            "factor_columns": factor_cols,
        }

    return references


def score_candidates(
    candidate_betas: pd.DataFrame,
    candidate_r2: pd.Series,
    theme_references: Mapping[str, dict],
    factor_cov: pd.DataFrame | np.ndarray,
    config: MatchingConfig = MatchingConfig(),
) -> pd.DataFrame:
    """Score candidate stocks using the ordered matching framework.

    Ranking is lexicographic in spirit:
    * candidates must pass the R squared, cosine and optional distance filters;
    * eligible names are ranked by penalized covariance distance;
    * cosine similarity and R squared act as tie breakers.

    ``rank_score`` is the negative penalized distance so that larger is better.
    """

    factor_cov_array = (
        factor_cov.to_numpy(dtype=float)
        if isinstance(factor_cov, pd.DataFrame)
        else np.asarray(factor_cov, dtype=float)
    )
    factor_std = np.sqrt(np.maximum(np.diag(factor_cov_array), _EPS))

    clean_betas = candidate_betas.dropna(how="any")
    rows = []

    for theme, reference in theme_references.items():
        factor_cols = reference["factor_columns"]
        available_cols = [column for column in factor_cols if column in clean_betas.columns]
        if len(available_cols) != len(factor_cols):
            continue

        tickers = clean_betas.index.to_numpy()
        beta_matrix = clean_betas.loc[:, factor_cols].to_numpy(dtype=float)
        reference_matrix = np.asarray(reference["betas"], dtype=float)
        medoid_beta = np.asarray(reference["medoid_beta"], dtype=float)

        raw_distance = _covariance_distance_matrix(beta_matrix, reference_matrix, factor_cov_array)
        reference_risk = np.sqrt(
            np.maximum(
                np.einsum("mk,kl,ml->m", reference_matrix, factor_cov_array, reference_matrix),
                _EPS,
            )
        )
        theme_scale = float(np.median(reference_risk))
        relative_distance = raw_distance / max(theme_scale, _EPS)

        primary_distance = np.median(relative_distance, axis=1)
        distance_mad = np.median(
            np.abs(relative_distance - primary_distance[:, None]), axis=1
        )
        cosine_matrix = _safe_cosine_matrix(beta_matrix, reference_matrix)
        median_cosine = np.median(cosine_matrix, axis=1)
        minimum_cosine = np.min(cosine_matrix, axis=1)

        medoid_delta = beta_matrix - medoid_beta[None, :]
        max_factor_gap = np.max(np.abs(medoid_delta) * factor_std[None, :], axis=1)
        max_factor_gap = max_factor_gap / max(theme_scale, _EPS)

        penalized_distance = (
            primary_distance
            + config.consensus_weight * distance_mad
            + config.factor_gap_weight * max_factor_gap
        )

        if config.etf_match_distance is not None:
            n_etf_matches = (relative_distance <= config.etf_match_distance).sum(axis=1)
        else:
            n_etf_matches = (cosine_matrix >= config.min_cosine).sum(axis=1)

        r2_values = candidate_r2.reindex(tickers).to_numpy(dtype=float)
        eligible = (
            np.isfinite(r2_values)
            & (r2_values >= config.min_candidate_r2)
            & (median_cosine >= config.min_cosine)
            & (n_etf_matches >= config.min_etf_matches)
        )
        if config.max_relative_distance is not None:
            if isinstance(config.max_relative_distance, Mapping):
                theme_max_distance = config.max_relative_distance.get(theme)
            else:
                theme_max_distance = config.max_relative_distance
            if theme_max_distance is not None and np.isfinite(theme_max_distance):
                eligible &= primary_distance <= float(theme_max_distance)

        for position, ticker in enumerate(tickers):
            rows.append(
                {
                    "ticker": str(ticker),
                    "theme": str(theme),
                    "primary_distance": float(primary_distance[position]),
                    "penalized_distance": float(penalized_distance[position]),
                    "median_cosine": float(median_cosine[position]),
                    "minimum_cosine": float(minimum_cosine[position]),
                    "candidate_r2": float(r2_values[position]),
                    "max_factor_gap": float(max_factor_gap[position]),
                    "distance_mad": float(distance_mad[position]),
                    "n_etf_matches": int(n_etf_matches[position]),
                    "n_theme_etfs": int(reference_matrix.shape[0]),
                    "eligible": bool(eligible[position]),
                    "rank_score": float(-penalized_distance[position]),
                    "medoid_etf": reference["medoid_ticker"],
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "theme",
                "primary_distance",
                "penalized_distance",
                "median_cosine",
                "minimum_cosine",
                "candidate_r2",
                "max_factor_gap",
                "distance_mad",
                "n_etf_matches",
                "n_theme_etfs",
                "eligible",
                "rank_score",
                "medoid_etf",
                "rank",
            ]
        )

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["theme", "eligible", "penalized_distance", "median_cosine", "candidate_r2"],
        ascending=[True, False, True, False, False],
    )
    result["rank"] = result.groupby("theme").cumcount() + 1
    return result.reset_index(drop=True)


def select_equal_weight_baskets(
    scores: pd.DataFrame,
    top_n: int = 30,
) -> Dict[str, dict]:
    """Select up to ``top_n`` eligible names per theme and equal weight them."""

    baskets: Dict[str, dict] = {}
    if scores.empty:
        return baskets

    for theme, group in scores.groupby("theme", sort=False):
        selected = group.loc[group["eligible"]].nsmallest(top_n, "penalized_distance")
        tickers = selected["ticker"].tolist()
        weight = 1.0 / len(tickers) if tickers else np.nan
        baskets[str(theme)] = {
            "tickers": tickers,
            "weights": {ticker: weight for ticker in tickers},
        }
    return baskets


def flatten_baskets(period_baskets: Mapping[pd.Timestamp, Mapping[str, dict]]) -> pd.DataFrame:
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
    return pd.DataFrame(rows)
