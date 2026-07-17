"""Point in time ThemeCloner walk forward pipeline.

The module is additive and leaves the existing V1 and V2 code untouched. It
moves residualization, winsorization and volatility scaling inside each
rebalance window, uses robust factor exposure matching, and evaluates realized
forward thematic exposure.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.evaluation_v3 import (
    basket_correlation_metric,
    build_forward_theme_benchmarks,
    empirical_placebo_pvalue,
    evaluate_forward_period,
)
from src.matching_v3 import (
    MatchingConfig,
    build_theme_reference_sets,
    fit_factor_loadings,
    score_candidates,
    select_equal_weight_baskets,
)
from src.rppca import fit_rppca

_EPS = 1e-12


def fit_factor_residualizer(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    keep_alpha: bool = False,
    min_obs: int = 52,
) -> Dict[str, object]:
    """Fit a factor residualization model and return trailing residuals.

    The fit uses only the supplied window. ``keep_alpha=True`` retains the
    estimated intercept in the residual series, matching the V2 covariance
    universe treatment.
    """

    factor_cols = [column for column in factors.columns if column != "RF"]
    common = returns.index.intersection(factors.index)
    if len(common) < min_obs:
        raise ValueError(f"Only {len(common)} common observations; need {min_obs}.")

    factor_panel = factors.loc[common, factor_cols].astype(float)
    rf = (
        factors.loc[common, "RF"].astype(float)
        if "RF" in factors.columns
        else pd.Series(0.0, index=common)
    )

    coefficients = pd.DataFrame(
        index=returns.columns,
        columns=["alpha", *factor_cols],
        dtype=float,
    )
    r2 = pd.Series(index=returns.columns, dtype=float, name="r2")
    nobs = pd.Series(index=returns.columns, dtype=float, name="nobs")
    residuals = pd.DataFrame(index=common, columns=returns.columns, dtype=float)

    for ticker in returns.columns:
        y = returns.loc[common, ticker].astype(float) - rf
        valid = y.notna() & factor_panel.notna().all(axis=1)
        n = int(valid.sum())
        nobs.loc[ticker] = n
        if n < min_obs:
            continue

        x = factor_panel.loc[valid].to_numpy(dtype=float)
        yv = y.loc[valid].to_numpy(dtype=float)
        design = np.column_stack([np.ones(n), x])
        coef, *_ = np.linalg.lstsq(design, yv, rcond=None)
        full_fitted = design @ coef
        centered = yv - yv.mean()
        ss_tot = float(np.dot(centered, centered))
        ss_res = float(np.dot(yv - full_fitted, yv - full_fitted))

        coefficients.loc[ticker] = coef
        r2.loc[ticker] = 1.0 - ss_res / ss_tot if ss_tot > _EPS else np.nan

        if keep_alpha:
            # Keep the intercept by subtracting only the factor component.
            residual = yv - x @ coef[1:]
        else:
            residual = yv - full_fitted
        residuals.loc[valid, ticker] = residual

    valid_assets = coefficients.dropna(how="any").index
    model = {
        "coefficients": coefficients.loc[valid_assets],
        "factor_columns": factor_cols,
        "keep_alpha": keep_alpha,
        "r2": r2.loc[valid_assets],
        "nobs": nobs.loc[valid_assets],
    }
    return {"model": model, "residuals": residuals.loc[:, valid_assets]}


def apply_factor_residualizer(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    model: Mapping[str, object],
) -> pd.DataFrame:
    """Apply trailing residualization coefficients to a new period."""

    factor_cols = list(model["factor_columns"])
    coefficients = model["coefficients"]
    keep_alpha = bool(model["keep_alpha"])
    common = returns.index.intersection(factors.index)
    assets = [ticker for ticker in coefficients.index if ticker in returns.columns]
    output = pd.DataFrame(index=common, columns=assets, dtype=float)
    if not assets or len(common) == 0:
        return output

    factor_panel = factors.loc[common, factor_cols].astype(float)
    rf = (
        factors.loc[common, "RF"].astype(float)
        if "RF" in factors.columns
        else pd.Series(0.0, index=common)
    )

    for ticker in assets:
        y = returns.loc[common, ticker].astype(float) - rf
        valid = y.notna() & factor_panel.notna().all(axis=1)
        if not valid.any():
            continue
        x = factor_panel.loc[valid].to_numpy(dtype=float)
        coef = coefficients.loc[ticker].to_numpy(dtype=float)
        if keep_alpha:
            residual = y.loc[valid].to_numpy(dtype=float) - x @ coef[1:]
        else:
            design = np.column_stack([np.ones(int(valid.sum())), x])
            residual = y.loc[valid].to_numpy(dtype=float) - design @ coef
        output.loc[valid, ticker] = residual
    return output


def fit_residual_standardizer(
    residuals: pd.DataFrame,
    winsorize_pct: float = 0.01,
) -> Dict[str, pd.Series]:
    """Fit trailing winsorization limits and volatility scales."""

    lower = residuals.quantile(winsorize_pct)
    upper = residuals.quantile(1.0 - winsorize_pct)
    clipped = residuals.clip(lower=lower, upper=upper, axis=1)
    scale = clipped.std(ddof=1).replace(0.0, np.nan)
    return {"lower": lower, "upper": upper, "scale": scale}


def apply_residual_standardizer(
    residuals: pd.DataFrame,
    standardizer: Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Apply trailing winsorization and volatility parameters."""

    columns = [column for column in residuals.columns if column in standardizer["scale"].index]
    clipped = residuals.loc[:, columns].clip(
        lower=standardizer["lower"].reindex(columns),
        upper=standardizer["upper"].reindex(columns),
        axis=1,
    )
    return clipped.divide(standardizer["scale"].reindex(columns), axis=1)


def _window_index(
    index: pd.DatetimeIndex,
    rebalance_date: pd.Timestamp,
    window: str,
    rolling_window_weeks: int,
) -> pd.DatetimeIndex:
    eligible = index[index <= rebalance_date]
    if window == "expanding":
        return eligible
    if window != "rolling":
        raise ValueError("window must be 'rolling' or 'expanding'.")
    return eligible[-rolling_window_weeks:]


def _resolve_membership(
    source: Optional[Mapping | Callable],
    date: pd.Timestamp,
    fallback: Iterable[str],
) -> list[str]:
    """Resolve optional point in time membership as of a rebalance date."""

    if source is None:
        return list(fallback)
    if callable(source):
        return list(source(date))

    dated = {pd.Timestamp(key): value for key, value in source.items()}
    available_dates = [key for key in dated if key <= date]
    if not available_dates:
        return []
    return list(dated[max(available_dates)])


def _filter_by_coverage(
    returns: pd.DataFrame,
    tickers: Sequence[str],
    min_coverage: float,
) -> list[str]:
    available = [ticker for ticker in tickers if ticker in returns.columns]
    if not available:
        return []
    coverage = returns.loc[:, available].notna().mean()
    return coverage.loc[coverage >= min_coverage].index.tolist()


def _forward_factor_returns(
    standardized_covariance_residuals: pd.DataFrame,
    rppca_result: Mapping[str, object],
) -> pd.DataFrame:
    """Apply frozen RP PCA loadings to the next period covariance residuals."""

    loadings = rppca_result["loadings"]
    columns = loadings.index.tolist()
    aligned = standardized_covariance_residuals.reindex(columns=columns).fillna(0.0)
    values = aligned.to_numpy(dtype=float) @ loadings.to_numpy(dtype=float)
    return pd.DataFrame(values, index=aligned.index, columns=loadings.columns)


def _circular_shift_panel(panel: pd.DataFrame, shift: int) -> pd.DataFrame:
    """Shift the complete ETF panel together to preserve within theme structure."""

    shifted = np.roll(panel.to_numpy(dtype=float), shift=shift, axis=0)
    return pd.DataFrame(shifted, index=panel.index, columns=panel.columns)


def _calibrate_distance_thresholds(
    placebo_scores: Sequence[pd.DataFrame],
    themes: Sequence[str],
    quantile: float,
    min_candidate_r2: float,
) -> Dict[str, float]:
    """Set candidate admission thresholds from the placebo distance distribution."""

    thresholds: Dict[str, float] = {}
    for theme in themes:
        values = []
        for scores in placebo_scores:
            subset = scores.loc[
                (scores["theme"] == theme)
                & (scores["candidate_r2"] >= min_candidate_r2),
                "primary_distance",
            ]
            values.extend(subset.dropna().tolist())
        if values:
            thresholds[theme] = float(np.quantile(values, quantile))
    return thresholds


def run_point_in_time_backtest(
    cov_returns_raw: pd.DataFrame,
    etf_returns_raw: pd.DataFrame,
    target_returns_raw: pd.DataFrame,
    factor_returns: pd.DataFrame,
    etf_config: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    K: int = 15,
    gamma: float = 10.0,
    window: str = "rolling",
    rolling_window_weeks: int = 208,
    top_n: int = 30,
    matching_config: MatchingConfig = MatchingConfig(),
    winsorize_pct: float = 0.01,
    min_train_obs: int = 52,
    min_train_coverage: float = 0.80,
    min_forward_obs: int = 8,
    cov_membership_by_date: Optional[Mapping | Callable] = None,
    target_membership_by_date: Optional[Mapping | Callable] = None,
    n_placebos: int = 50,
    placebo_min_shift_weeks: int = 26,
    placebo_admission_quantile: float = 0.05,
    random_state: int = 42,
    store_candidate_details: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run the revised point in time discovery and forward evaluation process.

    Point in time universe mappings are optional hooks. Until historical Russell
    membership data are supplied, the function falls back to the input columns.
    """

    rng = np.random.default_rng(random_state)
    dates = [pd.Timestamp(date) for date in rebalance_dates]
    dates = sorted(date for date in dates if date <= cov_returns_raw.index.max())

    period_baskets: Dict[pd.Timestamp, Dict[str, dict]] = {}
    period_scores: Dict[pd.Timestamp, pd.DataFrame] = {}
    period_references: Dict[pd.Timestamp, Dict[str, dict]] = {}
    evaluation_frames = []
    detail_frames = []
    threshold_rows = []

    for period_number, rebalance_date in enumerate(dates):
        next_date = dates[period_number + 1] if period_number + 1 < len(dates) else min(
            cov_returns_raw.index.max(),
            etf_returns_raw.index.max(),
            target_returns_raw.index.max(),
            factor_returns.index.max(),
        )
        if next_date <= rebalance_date:
            continue

        if verbose:
            print(
                f"\n{'=' * 72}\n"
                f"Rebalance {period_number + 1}/{len(dates)}: {rebalance_date.date()}\n"
                f"{'=' * 72}"
            )

        train_index = _window_index(
            cov_returns_raw.index,
            rebalance_date,
            window=window,
            rolling_window_weeks=rolling_window_weeks,
        )
        if len(train_index) < min_train_obs:
            continue
        train_start = train_index.min()

        cov_members = _resolve_membership(
            cov_membership_by_date, rebalance_date, cov_returns_raw.columns
        )
        target_members = _resolve_membership(
            target_membership_by_date, rebalance_date, target_returns_raw.columns
        )

        cov_train_raw = cov_returns_raw.loc[train_start:rebalance_date]
        target_train_raw = target_returns_raw.loc[train_start:rebalance_date]
        etf_train_raw = etf_returns_raw.loc[train_start:rebalance_date]
        factor_train = factor_returns.loc[train_start:rebalance_date]

        cov_members = _filter_by_coverage(cov_train_raw, cov_members, min_train_coverage)
        target_members = _filter_by_coverage(target_train_raw, target_members, min_train_coverage)
        cov_train_raw = cov_train_raw.loc[:, cov_members]
        target_train_raw = target_train_raw.loc[:, target_members]

        # CHANGE 1: residualization is fitted inside the rebalance window.
        cov_fit = fit_factor_residualizer(
            cov_train_raw, factor_train, keep_alpha=True, min_obs=min_train_obs
        )
        etf_fit = fit_factor_residualizer(
            etf_train_raw, factor_train, keep_alpha=False, min_obs=min_train_obs
        )
        target_fit = fit_factor_residualizer(
            target_train_raw, factor_train, keep_alpha=False, min_obs=min_train_obs
        )

        # CHANGE 2: winsorization and volatility scaling are also fitted trailing only.
        standardizer = fit_residual_standardizer(
            cov_fit["residuals"], winsorize_pct=winsorize_pct
        )
        cov_train_standardized = apply_residual_standardizer(
            cov_fit["residuals"], standardizer
        ).dropna(axis=1, how="all")
        if cov_train_standardized.shape[1] < K:
            if verbose:
                print("Skipping period: fewer covariance assets than requested factors.")
            continue

        rppca_result = fit_rppca(
            cov_train_standardized,
            K=K,
            gamma=gamma,
            run_oos=False,
            annualise=52.0,
        )
        factor_cov = rppca_result["factors"].cov()

        references = build_theme_reference_sets(
            etf_fit["residuals"],
            rppca_result["factors"],
            etf_config,
            factor_cov=factor_cov.to_numpy(dtype=float),
            top_factors=matching_config.top_factors,
            min_etf_r2=matching_config.min_etf_r2,
            min_obs=min_train_obs,
        )
        if not references:
            continue

        candidate_fit = fit_factor_loadings(
            target_fit["residuals"], rppca_result["factors"], min_obs=min_train_obs
        )

        # CHANGE 3: build time shifted null themes before setting an admission floor.
        unthresholded_config = replace(matching_config, max_relative_distance=None)
        placebo_scores_unthresholded = []
        placebo_reference_sets = []
        max_shift = len(etf_fit["residuals"]) - placebo_min_shift_weeks
        if n_placebos > 0 and max_shift > placebo_min_shift_weeks:
            possible_shifts = np.arange(placebo_min_shift_weeks, max_shift + 1)
            for _ in range(n_placebos):
                shift = int(rng.choice(possible_shifts))
                shifted_etfs = _circular_shift_panel(etf_fit["residuals"], shift)
                placebo_references = build_theme_reference_sets(
                    shifted_etfs,
                    rppca_result["factors"],
                    etf_config,
                    factor_cov=factor_cov.to_numpy(dtype=float),
                    top_factors=matching_config.top_factors,
                    min_etf_r2=matching_config.min_etf_r2,
                    min_obs=min_train_obs,
                )
                placebo_reference_sets.append(placebo_references)
                placebo_scores_unthresholded.append(
                    score_candidates(
                        candidate_fit["betas"],
                        candidate_fit["r2"],
                        placebo_references,
                        factor_cov,
                        config=unthresholded_config,
                    )
                )

        if matching_config.max_relative_distance is None and placebo_scores_unthresholded:
            thresholds = _calibrate_distance_thresholds(
                placebo_scores_unthresholded,
                themes=list(references),
                quantile=placebo_admission_quantile,
                min_candidate_r2=matching_config.min_candidate_r2,
            )
            period_config = replace(matching_config, max_relative_distance=thresholds)
        else:
            thresholds = (
                dict(matching_config.max_relative_distance)
                if isinstance(matching_config.max_relative_distance, Mapping)
                else {
                    theme: matching_config.max_relative_distance for theme in references
                }
            )
            period_config = matching_config

        for theme, threshold in thresholds.items():
            threshold_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "theme": theme,
                    "max_relative_distance": threshold,
                }
            )

        # CHANGE 4: robust multi ETF covariance distance matching.
        scores = score_candidates(
            candidate_fit["betas"],
            candidate_fit["r2"],
            references,
            factor_cov,
            config=period_config,
        )
        baskets = select_equal_weight_baskets(scores, top_n=top_n)

        period_scores[rebalance_date] = scores
        period_baskets[rebalance_date] = baskets
        period_references[rebalance_date] = references

        # CHANGE 5: apply all frozen models to the next period.
        forward_mask_cov = (cov_returns_raw.index > rebalance_date) & (
            cov_returns_raw.index <= next_date
        )
        forward_mask_etf = (etf_returns_raw.index > rebalance_date) & (
            etf_returns_raw.index <= next_date
        )
        forward_mask_target = (target_returns_raw.index > rebalance_date) & (
            target_returns_raw.index <= next_date
        )
        factor_forward = factor_returns.loc[(factor_returns.index > rebalance_date) & (factor_returns.index <= next_date)]

        cov_forward_residuals = apply_factor_residualizer(
            cov_returns_raw.loc[forward_mask_cov], factor_forward, cov_fit["model"]
        )
        cov_forward_standardized = apply_residual_standardizer(
            cov_forward_residuals, standardizer
        )
        forward_factors = _forward_factor_returns(cov_forward_standardized, rppca_result)

        etf_forward_residuals = apply_factor_residualizer(
            etf_returns_raw.loc[forward_mask_etf], factor_forward, etf_fit["model"]
        )
        target_forward_residuals = apply_factor_residualizer(
            target_returns_raw.loc[forward_mask_target], factor_forward, target_fit["model"]
        )
        target_forward_raw = target_returns_raw.loc[forward_mask_target]

        benchmarks = build_forward_theme_benchmarks(
            forward_factors, etf_forward_residuals, references
        )
        evaluation, details = evaluate_forward_period(
            scores,
            baskets,
            target_forward_residuals,
            target_forward_raw,
            benchmarks,
            rebalance_date=rebalance_date,
            min_forward_obs=min_forward_obs,
        )

        # CHANGE 6: compare real basket exposure with time shifted placebo themes.
        placebo_metric_map: Dict[str, list[float]] = {theme: [] for theme in benchmarks}
        for placebo_scores in placebo_scores_unthresholded:
            if placebo_scores.empty:
                continue
            if period_config.max_relative_distance is not None:
                # Reapply the period's calibrated threshold to the placebo candidates.
                placebo_scores = placebo_scores.copy()
                for theme, threshold in thresholds.items():
                    mask = placebo_scores["theme"] == theme
                    placebo_scores.loc[mask, "eligible"] &= (
                        placebo_scores.loc[mask, "primary_distance"] <= threshold
                    )
            placebo_baskets = select_equal_weight_baskets(placebo_scores, top_n=top_n)
            for theme, benchmark_set in benchmarks.items():
                metric = basket_correlation_metric(
                    placebo_baskets.get(theme, {}).get("tickers", []),
                    target_forward_residuals,
                    benchmark_set["synthetic"],
                    min_forward_obs=min_forward_obs,
                )
                placebo_metric_map[theme].append(metric)

        if not evaluation.empty:
            pvalues = []
            placebo_medians = []
            for row in evaluation.itertuples(index=False):
                values = placebo_metric_map.get(row.theme, [])
                pvalues.append(empirical_placebo_pvalue(row.basket_synthetic_corr, values))
                clean_values = [value for value in values if np.isfinite(value)]
                placebo_medians.append(float(np.median(clean_values)) if clean_values else np.nan)
            evaluation["placebo_pvalue"] = pvalues
            evaluation["placebo_median_basket_corr"] = placebo_medians
            evaluation_frames.append(evaluation)
        if store_candidate_details and not details.empty:
            detail_frames.append(details)

        if verbose:
            selected_count = sum(len(item["tickers"]) for item in baskets.values())
            print(
                f"Selected {selected_count} stocks across {len(baskets)} themes; "
                f"forward window has {len(target_forward_raw)} observations."
            )

    evaluation_panel = (
        pd.concat(evaluation_frames, ignore_index=True) if evaluation_frames else pd.DataFrame()
    )
    candidate_forward_panel = (
        pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    )
    threshold_panel = pd.DataFrame(threshold_rows)

    if not evaluation_panel.empty:
        forward_returns = evaluation_panel.pivot(
            index="rebalance_date", columns="theme", values="raw_basket_period_return"
        )
        equity_curves = (1.0 + forward_returns.fillna(0.0)).cumprod()
    else:
        forward_returns = pd.DataFrame()
        equity_curves = pd.DataFrame()

    return {
        "period_baskets": period_baskets,
        "period_scores": period_scores,
        "period_references": period_references,
        "evaluation": evaluation_panel,
        "candidate_forward_exposure": candidate_forward_panel,
        "distance_thresholds": threshold_panel,
        "forward_returns_secondary": forward_returns,
        "equity_curves_secondary": equity_curves,
    }
