"""Point in time ThemeCloner walk forward pipeline.

The pipeline fits all transformations inside each trailing rebalance window,
uses covariance weighted multi ETF matching, and evaluates forward thematic
exposure. Ticker regressions are vectorized. Independent placebo runs use
separate Python processes so Windows can use more than one CPU core.
"""

from __future__ import annotations

from time import perf_counter
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.evaluation_v3 import (
    basket_correlation_metric,
    build_forward_theme_benchmarks,
    empirical_placebo_pvalue,
    evaluate_forward_period,
)
from src.matching_v3 import (
    MatchingConfig,
    apply_rank_placebo_filter,
    build_theme_reference_sets,
    fit_factor_loadings,
    score_candidates,
    select_equal_weight_baskets,
    theme_reference_distance_table,
)
from src.rppca import fit_rppca

_EPS = 1e-12


def fit_factor_residualizer(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    keep_alpha: bool = False,
    min_obs: int = 52,
    n_jobs: int = 1,
) -> Dict[str, object]:
    """Fit trailing factor residualization with vectorized regressions."""

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
    excess_returns = returns.loc[common].subtract(rf, axis=0)

    fitted = fit_factor_loadings(
        excess_returns,
        factor_panel,
        min_obs=min_obs,
        n_jobs=n_jobs,
    )
    # fit_factor_loadings may remove rows with incomplete factor observations.
    common = pd.DatetimeIndex(fitted["common_index"])
    factor_panel = factor_panel.loc[common]
    excess_returns = excess_returns.loc[common]

    coefficients = fitted["coefficients"].dropna(how="any")
    valid_assets = coefficients.index.tolist()
    if not valid_assets:
        return {
            "model": {
                "coefficients": coefficients,
                "factor_columns": factor_cols,
                "keep_alpha": keep_alpha,
                "r2": pd.Series(dtype=float),
                "adjusted_r2": pd.Series(dtype=float),
                "nobs": pd.Series(dtype=float),
            },
            "residuals": pd.DataFrame(index=common),
        }

    x = factor_panel.to_numpy(dtype=float)
    y = excess_returns.loc[:, valid_assets].to_numpy(dtype=float)
    coef = coefficients.loc[valid_assets].to_numpy(dtype=float)
    if keep_alpha:
        prediction = x @ coef[:, 1:].T
    else:
        design = np.column_stack([np.ones(len(common)), x])
        prediction = design @ coef.T
    residual_values = np.where(np.isfinite(y), y - prediction, np.nan)
    residuals = pd.DataFrame(
        residual_values,
        index=common,
        columns=valid_assets,
    )

    model = {
        "coefficients": coefficients,
        "factor_columns": factor_cols,
        "keep_alpha": keep_alpha,
        "r2": fitted["r2"].loc[valid_assets],
        "adjusted_r2": fitted["adjusted_r2"].loc[valid_assets],
        "nobs": fitted["nobs"].loc[valid_assets],
    }
    return {"model": model, "residuals": residuals}


def apply_factor_residualizer(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    model: Mapping[str, object],
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Apply frozen residualization coefficients to a new period."""

    del n_jobs
    factor_cols = list(model["factor_columns"])
    coefficients = model["coefficients"]
    keep_alpha = bool(model["keep_alpha"])
    common = returns.index.intersection(factors.index)
    assets = [ticker for ticker in coefficients.index if ticker in returns.columns]
    if not assets or len(common) == 0:
        return pd.DataFrame(index=common, columns=assets, dtype=float)

    factor_panel = factors.loc[common, factor_cols].astype(float)
    valid_rows = factor_panel.notna().all(axis=1)
    common = common[valid_rows.to_numpy()]
    factor_panel = factor_panel.loc[common]
    if len(common) == 0:
        return pd.DataFrame(index=common, columns=assets, dtype=float)

    rf = (
        factors.loc[common, "RF"].astype(float)
        if "RF" in factors.columns
        else pd.Series(0.0, index=common)
    )
    y_frame = returns.loc[common, assets].subtract(rf, axis=0)
    y = y_frame.to_numpy(dtype=float)
    x = factor_panel.to_numpy(dtype=float)
    coef = coefficients.loc[assets].to_numpy(dtype=float)

    if keep_alpha:
        prediction = x @ coef[:, 1:].T
    else:
        design = np.column_stack([np.ones(len(common)), x])
        prediction = design @ coef.T
    residual = np.where(np.isfinite(y), y - prediction, np.nan)
    return pd.DataFrame(residual, index=common, columns=assets)


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

    columns = [
        column for column in residuals.columns if column in standardizer["scale"].index
    ]
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
    """Apply frozen RP PCA loadings to next period covariance residuals."""

    loadings = rppca_result["loadings"]
    columns = loadings.index.tolist()
    aligned = standardized_covariance_residuals.reindex(columns=columns).fillna(0.0)
    values = aligned.to_numpy(dtype=float) @ loadings.to_numpy(dtype=float)
    return pd.DataFrame(values, index=aligned.index, columns=loadings.columns)


def _circular_shift_panel(panel: pd.DataFrame, shift: int) -> pd.DataFrame:
    """Shift the complete ETF panel together to preserve within theme structure."""

    shifted = np.roll(panel.to_numpy(dtype=float), shift=shift, axis=0)
    return pd.DataFrame(shifted, index=panel.index, columns=panel.columns)


def _run_single_placebo(
    shift: int,
    etf_residuals: pd.DataFrame,
    factors: pd.DataFrame,
    etf_config: pd.DataFrame,
    factor_cov: pd.DataFrame,
    candidate_betas: pd.DataFrame,
    candidate_r2: pd.Series,
    candidate_adjusted_r2: pd.Series,
    matching_config: MatchingConfig,
    min_train_obs: int,
) -> pd.DataFrame:
    """Build and score one time shifted placebo theme."""

    shifted_etfs = _circular_shift_panel(etf_residuals, shift)
    references = build_theme_reference_sets(
        shifted_etfs,
        factors,
        etf_config,
        factor_cov=factor_cov.to_numpy(dtype=float),
        top_factors=matching_config.top_factors,
        min_etf_r2=matching_config.min_etf_r2,
        min_etf_adjusted_r2=matching_config.min_etf_adjusted_r2,
        min_obs=min_train_obs,
        n_jobs=1,
    )
    return score_candidates(
        candidate_betas,
        candidate_r2,
        references,
        factor_cov,
        config=matching_config,
        candidate_adjusted_r2=candidate_adjusted_r2,
    )


def _run_placebo_batch(
    shifts: Sequence[int],
    etf_residuals: pd.DataFrame,
    factors: pd.DataFrame,
    etf_config: pd.DataFrame,
    factor_cov: pd.DataFrame,
    candidate_betas: pd.DataFrame,
    candidate_r2: pd.Series,
    candidate_adjusted_r2: pd.Series,
    matching_config: MatchingConfig,
    min_train_obs: int,
) -> list[pd.DataFrame]:
    """Run a batch in one worker to reduce Windows process launch overhead."""

    return [
        _run_single_placebo(
            shift,
            etf_residuals,
            factors,
            etf_config,
            factor_cov,
            candidate_betas,
            candidate_r2,
            candidate_adjusted_r2,
            matching_config,
            min_train_obs,
        )
        for shift in shifts
    ]


def _parallel_placebos(
    shifts: Sequence[int],
    n_jobs: int,
    *args,
) -> list[pd.DataFrame]:
    """Use process workers for independent placebos."""

    if not shifts:
        return []
    workers = min(max(1, int(n_jobs)), len(shifts))
    if workers == 1:
        return _run_placebo_batch(shifts, *args)

    batches = [batch.tolist() for batch in np.array_split(np.asarray(shifts), workers) if len(batch)]
    nested = Parallel(
        n_jobs=workers,
        backend="loky",
        batch_size=1,
        verbose=0,
    )(
        delayed(_run_placebo_batch)(batch, *args)
        for batch in batches
    )
    return [item for batch in nested for item in batch]


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
    placebo_pvalue_max: float = 0.10,
    placebo_use_fdr: bool = False,
    # Backward compatible alias from the earlier notebook.
    placebo_admission_quantile: Optional[float] = None,
    random_state: int = 42,
    n_jobs: int = 1,
    store_candidate_details: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run point in time discovery and forward exposure evaluation."""

    if placebo_admission_quantile is not None:
        placebo_pvalue_max = float(placebo_admission_quantile)

    rng = np.random.default_rng(random_state)
    dates = sorted(
        pd.Timestamp(date)
        for date in rebalance_dates
        if pd.Timestamp(date) <= cov_returns_raw.index.max()
    )

    period_baskets: Dict[pd.Timestamp, Dict[str, dict]] = {}
    period_scores: Dict[pd.Timestamp, pd.DataFrame] = {}
    period_references: Dict[pd.Timestamp, Dict[str, dict]] = {}
    evaluation_frames = []
    detail_frames = []
    rank_null_frames = []
    reference_distance_frames = []
    overlap_rows = []
    timing_rows = []

    for period_number, rebalance_date in enumerate(dates):
        next_date = (
            dates[period_number + 1]
            if period_number + 1 < len(dates)
            else min(
                cov_returns_raw.index.max(),
                etf_returns_raw.index.max(),
                target_returns_raw.index.max(),
                factor_returns.index.max(),
            )
        )
        if next_date <= rebalance_date:
            continue

        period_start = perf_counter()
        stage_start = period_start
        stage_times: Dict[str, float] = {}

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
            cov_membership_by_date,
            rebalance_date,
            cov_returns_raw.columns,
        )
        target_members = _resolve_membership(
            target_membership_by_date,
            rebalance_date,
            target_returns_raw.columns,
        )

        cov_train_raw = cov_returns_raw.loc[train_start:rebalance_date]
        target_train_raw = target_returns_raw.loc[train_start:rebalance_date]
        etf_train_raw = etf_returns_raw.loc[train_start:rebalance_date]
        factor_train = factor_returns.loc[train_start:rebalance_date]

        cov_members = _filter_by_coverage(
            cov_train_raw,
            cov_members,
            min_train_coverage,
        )
        target_members = _filter_by_coverage(
            target_train_raw,
            target_members,
            min_train_coverage,
        )
        cov_train_raw = cov_train_raw.loc[:, cov_members]
        target_train_raw = target_train_raw.loc[:, target_members]

        cov_fit = fit_factor_residualizer(
            cov_train_raw,
            factor_train,
            keep_alpha=True,
            min_obs=min_train_obs,
            n_jobs=n_jobs,
        )
        etf_fit = fit_factor_residualizer(
            etf_train_raw,
            factor_train,
            keep_alpha=False,
            min_obs=min_train_obs,
            n_jobs=n_jobs,
        )
        target_fit = fit_factor_residualizer(
            target_train_raw,
            factor_train,
            keep_alpha=False,
            min_obs=min_train_obs,
            n_jobs=n_jobs,
        )
        stage_times["residualization"] = perf_counter() - stage_start
        stage_start = perf_counter()

        standardizer = fit_residual_standardizer(
            cov_fit["residuals"],
            winsorize_pct=winsorize_pct,
        )
        cov_train_standardized = apply_residual_standardizer(
            cov_fit["residuals"],
            standardizer,
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
        stage_times["rppca"] = perf_counter() - stage_start
        stage_start = perf_counter()

        references = build_theme_reference_sets(
            etf_fit["residuals"],
            rppca_result["factors"],
            etf_config,
            factor_cov=factor_cov.to_numpy(dtype=float),
            top_factors=matching_config.top_factors,
            min_etf_r2=matching_config.min_etf_r2,
            min_etf_adjusted_r2=matching_config.min_etf_adjusted_r2,
            min_obs=min_train_obs,
            n_jobs=n_jobs,
        )
        if not references:
            continue

        candidate_fit = fit_factor_loadings(
            target_fit["residuals"],
            rppca_result["factors"],
            min_obs=min_train_obs,
            n_jobs=n_jobs,
        )
        base_config = matching_config
        raw_scores = score_candidates(
            candidate_fit["betas"],
            candidate_fit["r2"],
            references,
            factor_cov,
            config=base_config,
            candidate_adjusted_r2=candidate_fit["adjusted_r2"],
        )
        stage_times["reference_and_candidate_fit"] = perf_counter() - stage_start
        stage_start = perf_counter()

        placebo_scores = []
        max_shift = len(etf_fit["residuals"]) - placebo_min_shift_weeks
        if n_placebos > 0 and max_shift > placebo_min_shift_weeks:
            possible_shifts = np.arange(placebo_min_shift_weeks, max_shift + 1)
            shifts = [int(rng.choice(possible_shifts)) for _ in range(n_placebos)]
            placebo_scores = _parallel_placebos(
                shifts,
                n_jobs,
                etf_fit["residuals"],
                rppca_result["factors"],
                etf_config,
                factor_cov,
                candidate_fit["betas"],
                candidate_fit["r2"],
                candidate_fit["adjusted_r2"],
                base_config,
                min_train_obs,
            )

        scores, rank_null = apply_rank_placebo_filter(
            raw_scores,
            placebo_scores,
            alpha=placebo_pvalue_max,
            max_rank=top_n,
            use_fdr=placebo_use_fdr,
        )
        baskets = select_equal_weight_baskets(scores, top_n=top_n)
        stage_times["placebos_and_admission"] = perf_counter() - stage_start
        stage_start = perf_counter()

        period_scores[rebalance_date] = scores
        period_baskets[rebalance_date] = baskets
        period_references[rebalance_date] = references

        if not rank_null.empty:
            rank_null.insert(0, "rebalance_date", rebalance_date)
            rank_null_frames.append(rank_null)

        reference_distances = theme_reference_distance_table(references, factor_cov)
        if not reference_distances.empty:
            reference_distances.insert(0, "rebalance_date", rebalance_date)
            reference_distance_frames.append(reference_distances)

        selected_pairs = [
            (theme, ticker)
            for theme, basket in baskets.items()
            for ticker in basket.get("tickers", [])
        ]
        if selected_pairs:
            selected_frame = pd.DataFrame(selected_pairs, columns=["theme", "ticker"])
            counts = selected_frame.groupby("ticker")["theme"].nunique()
            overlap_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "positions": len(selected_frame),
                    "unique_stocks": selected_frame["ticker"].nunique(),
                    "overlap_positions": len(selected_frame)
                    - selected_frame["ticker"].nunique(),
                    "max_themes_per_stock": int(counts.max()),
                    "mean_themes_per_selected_stock": float(counts.mean()),
                }
            )

        forward_mask_cov = (cov_returns_raw.index > rebalance_date) & (
            cov_returns_raw.index <= next_date
        )
        forward_mask_etf = (etf_returns_raw.index > rebalance_date) & (
            etf_returns_raw.index <= next_date
        )
        forward_mask_target = (target_returns_raw.index > rebalance_date) & (
            target_returns_raw.index <= next_date
        )
        factor_forward = factor_returns.loc[
            (factor_returns.index > rebalance_date)
            & (factor_returns.index <= next_date)
        ]

        cov_forward_residuals = apply_factor_residualizer(
            cov_returns_raw.loc[forward_mask_cov],
            factor_forward,
            cov_fit["model"],
            n_jobs=n_jobs,
        )
        cov_forward_standardized = apply_residual_standardizer(
            cov_forward_residuals,
            standardizer,
        )
        forward_factors = _forward_factor_returns(
            cov_forward_standardized,
            rppca_result,
        )

        etf_forward_residuals = apply_factor_residualizer(
            etf_returns_raw.loc[forward_mask_etf],
            factor_forward,
            etf_fit["model"],
            n_jobs=n_jobs,
        )
        target_forward_residuals = apply_factor_residualizer(
            target_returns_raw.loc[forward_mask_target],
            factor_forward,
            target_fit["model"],
            n_jobs=n_jobs,
        )
        target_forward_raw = target_returns_raw.loc[forward_mask_target]

        benchmarks = build_forward_theme_benchmarks(
            forward_factors,
            etf_forward_residuals,
            references,
            forward_etf_raw=etf_returns_raw.loc[forward_mask_etf],
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
        stage_times["forward_evaluation"] = perf_counter() - stage_start
        stage_start = perf_counter()

        placebo_metric_map: Dict[str, list[float]] = {
            theme: [] for theme in benchmarks
        }
        for placebo_score in placebo_scores:
            if placebo_score.empty:
                continue
            # Placebo scores retain base eligibility. They are the null baskets.
            placebo_baskets = select_equal_weight_baskets(
                placebo_score,
                top_n=top_n,
            )
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
                pvalues.append(
                    empirical_placebo_pvalue(row.basket_synthetic_corr, values)
                )
                clean_values = [value for value in values if np.isfinite(value)]
                placebo_medians.append(
                    float(np.median(clean_values)) if clean_values else np.nan
                )
            evaluation["placebo_pvalue"] = pvalues
            evaluation["placebo_median_basket_corr"] = placebo_medians
            evaluation_frames.append(evaluation)
        if store_candidate_details and not details.empty:
            detail_frames.append(details)

        stage_times["placebo_forward_metrics"] = perf_counter() - stage_start
        stage_times["total"] = perf_counter() - period_start
        timing_rows.append(
            {
                "rebalance_date": rebalance_date,
                **stage_times,
            }
        )

        if verbose:
            selected_count = sum(len(item["tickers"]) for item in baskets.values())
            print(
                f"Selected {selected_count} positions across {len(baskets)} themes; "
                f"forward window has {len(target_forward_raw)} observations."
            )
            print(
                "Timing seconds: "
                + ", ".join(
                    f"{name}={seconds:.1f}"
                    for name, seconds in stage_times.items()
                )
            )

    evaluation_panel = (
        pd.concat(evaluation_frames, ignore_index=True)
        if evaluation_frames
        else pd.DataFrame()
    )
    candidate_forward_panel = (
        pd.concat(detail_frames, ignore_index=True)
        if detail_frames
        else pd.DataFrame()
    )
    rank_null_panel = (
        pd.concat(rank_null_frames, ignore_index=True)
        if rank_null_frames
        else pd.DataFrame()
    )
    reference_distance_panel = (
        pd.concat(reference_distance_frames, ignore_index=True)
        if reference_distance_frames
        else pd.DataFrame()
    )

    if not evaluation_panel.empty:
        forward_returns = evaluation_panel.pivot(
            index="rebalance_date",
            columns="theme",
            values="raw_basket_period_return",
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
        # Kept for compatibility; now contains rank matched placebo thresholds.
        "distance_thresholds": rank_null_panel,
        "placebo_rank_thresholds": rank_null_panel,
        "theme_reference_distances": reference_distance_panel,
        "overlap_diagnostics": pd.DataFrame(overlap_rows),
        "timings": pd.DataFrame(timing_rows),
        "forward_returns_secondary": forward_returns,
        "equity_curves_secondary": equity_curves,
    }
