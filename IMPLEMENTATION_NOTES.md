# ThemeCloner revised walk forward patch

## Files to add

Copy these files into the existing repository:

- `src/matching_v3.py`
- `src/evaluation_v3.py`
- `src/walkforward_v3.py`
- `ThemeCloner_V2_WalkForward_revised.ipynb`

No existing V1 or V2 source module is overwritten. The revised notebook uses the existing data pull, residual factor download, RP PCA solver and rebalance schedule functions.

## Main methodology changes

1. Residualization is fitted separately inside each trailing rebalance window.
2. Winsorization and volatility scaling are also fitted using trailing data only.
3. RP PCA is fitted once on the point in time covariance universe.
4. Each qualifying ETF remains a separate theme reference vector.
5. Candidate matching uses median covariance weighted distance across theme ETFs.
6. Cosine similarity is a secondary directional check.
7. Candidate R squared is an admission filter.
8. The score penalizes the largest single factor mismatch and disagreement across ETFs.
9. Candidate distance thresholds are calibrated from time shifted placebo themes.
10. Baskets contain only qualifying names and are equal weighted.
11. Names are dropped automatically when they fail the next rebalance screen.

## Forward evaluation

The output includes:

- Forward exposure rank correlation
- Top versus bottom score portfolio exposure spread
- Basket correlation and beta to the residualized ETF blend
- Basket correlation and beta to the frozen synthetic theme factor
- Time shifted placebo p values
- Raw basket returns as a secondary diagnostic
- A blinded economic relevance review sheet and control sample

## Historical universe support

`run_point_in_time_backtest` accepts optional `cov_membership_by_date` and `target_membership_by_date` inputs. Each can be a mapping from date to constituent tickers or a callable that returns the membership as of a date. If omitted, the function uses the columns in the supplied return panels.

## Placebo settings

The notebook uses 50 time shifted placebos for a practical first run. Increase this to at least 200 for reported results. More placebos improve threshold and p value precision but increase runtime.

## Validation performed

- All Python modules compile.
- Every notebook code cell compiles.
- The full pipeline was run on synthetic data with two themes, multiple ETFs and known target exposures.
- The repository data were not available in the execution environment, so the revised notebook has not been run against the live cached panels.
