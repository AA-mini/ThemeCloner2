# ThemeCloner data panels

Load any file with `pd.read_parquet(path)`. Index = weekly date (`date`), columns = tickers or factor names, values = simple weekly returns (decimal).

| File | Shape | Dates | Contents |
|---|---|---|---|
| `etf_returns.parquet` | 447 weeks x 42 cols | 2018-01-12 to 2026-07-31 | Weekly returns of every ETF in config/etfs.csv (themes, GICS controls, style factors). |
| `cov_returns.parquet` | 447 weeks x 1832 cols | 2018-01-12 to 2026-07-31 | Weekly returns of the covariance universe (US-listed, >$1bn, S&P 500 + mid/small). Used to estimate the 15 latent factors. |
| `tgt_returns.parquet` | 444 weeks x 811 cols | 2018-01-12 to 2026-07-10 | Weekly returns of the target universe (small caps that get scored against themes). |
| `ff_factors.parquet` | 448 weeks x 7 cols | 2018-01-05 to 2026-07-31 | Weekly Fama-French 5 factors + momentum (Ken French data library), decimal returns. |
| `etf_config.csv` | | | ETF-to-theme mapping (themes, GICS controls, style factors). |

Source: Yahoo Finance via yfinance (prices), Ken French data library (factors).
Caveat: ticker lists are current-listing snapshots, so the stock panels carry survivorship bias. CRSP (via WRDS) is needed for final tables.

Residualized returns and per-rebalance factor loadings are exported separately by the backtest (see files prefixed `wf_`).