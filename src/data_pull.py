# data_pull.py
#
# Pulls return data for three distinct universes, each with a different role:
#
#   1. ETF returns          -- theme definition and OOS validation signal
#                              e.g. BOTZ, ICLN, AINF.L
#                              the ETF IS the theme -- no constituent data needed
#
#   2. Covariance universe  -- broad cross-section used for RP-PCA factor extraction
#                              needs to be rich enough that all themes appear as latent factors
#                              proxy for MSCI ACWI: SP500 + SP400 + SP600 + STOXX600 + Nikkei225
#                              (data access limitation: true ACWI requires a vendor)
#
#   3. Target universe      -- where we hunt for undiscovered thematic exposure
#                              e.g. Russell 2000 small caps, or any regional universe
#                              scored against theme factors extracted from the covariance universe
#
# PAPER NOTE -- data access limitations and proxies used:
#   - MSCI ACWI constituents require a vendor (MSCI, Bloomberg). We proxy with:
#       US:     S&P 500 + S&P 400 midcap + S&P 600 smallcap  (~1500 stocks)
#       Europe: STOXX Europe 600 (600 stocks, via yfinance .DE/.PA/.L suffixes)
#       Japan:  Nikkei 225 (225 stocks, via yfinance .T suffix)
#     This covers ~2300 stocks across 3 major regions, capturing the same
#     large-cap global exposure as ACWI for the purposes of factor extraction.
#   - ETF constituent holdings blocked by providers (iShares, etc.) --
#     we use ETF price series directly as theme labels, not constituent panels.
#   - Russell 2000 full list not available via free API --
#     we proxy with NASDAQ + NYSE ex-SP500, filtered by coverage.
#
# yfinance exchange suffix conventions:
#   US (NYSE/NASDAQ): no suffix    e.g. AAPL
#   London (LSE):     .L           e.g. BP.L
#   Xetra (Germany):  .DE          e.g. SAP.DE
#   Euronext Paris:   .PA          e.g. MC.PA
#   Tokyo:            .T           e.g. 7203.T

import os
import json
import warnings
import requests
import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

# -------------------- config --------------------

DATA_OUT     = os.path.join(os.path.dirname(__file__), "..", "outputs", "data")
os.makedirs(DATA_OUT, exist_ok=True)

START_DATE   = "2018-01-01"
END_DATE     = None          # None = today
FREQ         = "W-FRI"       # weekly Friday close
MIN_COVERAGE = 0.85          # drop tickers with <85% non-null weeks
                             # raised from 0.70 -- zero-fills from low-coverage
                             # stocks create artificial covariance structure

HEADERS      = {"User-Agent": "Mozilla/5.0"}


# -------------------- 1. ETF config --------------------

def load_etf_config(config_path: str) -> pd.DataFrame:
    """
    Reads etfs.csv. Expected columns: ticker, yf_ticker, theme, description.

    ticker     -- Bloomberg-style label for display  (e.g. "AINF LN")
    yf_ticker  -- what we pass to yfinance           (e.g. "AINF.L")

    Multiple ETFs per theme are supported and recommended -- the more ETFs
    per theme, the more robust the theme DNA extraction in theme_dna.py.
    """
    df = pd.read_csv(config_path)
    df.columns = df.columns.str.strip().str.lower()
    df["ticker"] = df["ticker"].str.strip()
    df["theme"]  = df["theme"].str.strip()

    if "yf_ticker" not in df.columns:
        print("  note: no yf_ticker column -- using ticker for yfinance calls")
        df["yf_ticker"] = df["ticker"].str.upper()
    else:
        df["yf_ticker"] = df["yf_ticker"].str.strip()

    # type column: theme / sector / subsector. defaults to 'theme' if absent
    # (sector and subsector rows are CALIBRATION CONTROLS -- they run through
    # the coherence/distinctiveness diagnostic to anchor the coherence scale,
    # but are skipped for candidate basket construction)
    if "type" not in df.columns:
        df["type"] = "theme"
    else:
        df["type"] = df["type"].str.strip().str.lower()

    n_theme = (df["type"] == "theme").sum()
    n_ctrl  = (df["type"] != "theme").sum()
    print(f"loaded {len(df)} ETFs: {n_theme} theme-ETFs, {n_ctrl} control-ETFs "
          f"(sector/subsector calibration anchors)")
    for theme, grp in df.groupby("theme"):
        pairs = [f"{r.ticker} ({r.yf_ticker})" for _, r in grp.iterrows()]
        print(f"  {theme}: {', '.join(pairs)}")
    return df


# -------------------- 2. Generic return puller --------------------

def _pull_returns(tickers: list, label: str, start: str,
                   end=None, freq: str = FREQ) -> pd.DataFrame:
    """
    Downloads weekly log returns for a list of yfinance-format tickers.
    Chunks into batches of 400 to stay within yfinance limits.
    Applies MIN_COVERAGE filter and forward-fills small gaps.

    This is the shared engine used by all three universe pull functions.
    """
    CHUNK  = 400
    chunks = [tickers[i:i+CHUNK] for i in range(0, len(tickers), CHUNK)]
    print(f"\npulling {label}: {len(tickers)} tickers in {len(chunks)} batch(es)")

    frames = []
    for i, chunk in enumerate(chunks):
        print(f"  batch {i+1}/{len(chunks)} ({len(chunk)} tickers)...")
        try:
            raw = yf.download(chunk, start=start, end=end,
                               auto_adjust=True, progress=False)
            if raw.empty:
                continue
            px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            if not isinstance(px, pd.DataFrame):
                px = px.to_frame()
            # single ticker edge case
            if len(chunk) == 1 and px.shape[1] != 1:
                px.columns = chunk
            frames.append(px.resample(freq).last())
        except Exception as e:
            print(f"    batch {i+1} failed: {e}")

    if not frames:
        raise RuntimeError(f"all batches failed for {label}")

    prices  = pd.concat(frames, axis=1)

    # -------------------- log returns --------------------
    returns = np.log(prices / prices.shift(1)).dropna(how="all")

    # -------------------- coverage filter --------------------
    coverage = returns.notna().mean()
    good     = coverage[coverage >= MIN_COVERAGE].index.tolist()
    dropped  = len(tickers) - len(good)
    if dropped:
        print(f"  dropped {dropped} tickers below {MIN_COVERAGE*100:.0f}% coverage")

    returns = returns[good]

    # -------------------- fill small gaps only --------------------
    # forward-fill gaps of 1-2 weeks (holidays, data issues) -- no more
    # do NOT fillna(0): zero returns create artificial covariance structure
    # and cause rank-deficiency in RP-PCA (zero eigenvalues)
    returns = returns.ffill(limit=2)

    # drop any remaining NaNs row-wise (weeks where most stocks missing)
    returns = returns.dropna(how="all")

    print(f"  {label}: {returns.shape[0]} weeks x {returns.shape[1]} stocks")
    print(f"  date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    return returns





# -------------------- 2b. Generic parquet cache wrapper --------------------

def _cached_returns(cache_name: str, pull_fn, refresh: bool = False, **kwargs) -> pd.DataFrame:
    """
    Wraps any of the three _pull_returns-based functions with a parquet cache.
    Set refresh=True to force a fresh pull (e.g. after adding new tickers).
    """
    path = os.path.join(DATA_OUT, f"{cache_name}.parquet")
    if not refresh and os.path.exists(path):
        df = pd.read_parquet(path)
        print(f"  loaded {cache_name} from cache: {df.shape[0]} weeks x {df.shape[1]} cols")
        return df
    df = pull_fn(**kwargs)
    df.to_parquet(path)
    print(f"  cached {cache_name} -> {path}")
    return df






# -------------------- 3. ETF returns --------------------

def pull_etf_returns(etf_config: pd.DataFrame, start: str, end=None) -> pd.DataFrame:
    """
    Pulls weekly returns for the ETFs defined in etfs.csv.
    Uses yf_ticker for the API call, renames columns to Bloomberg ticker.

    Role in pipeline: theme definition + OOS validation benchmark.
    The ETF return IS the theme signal -- no constituent data required.
    """
    yf_tickers  = etf_config["yf_ticker"].tolist()
    bbg_tickers = etf_config["ticker"].tolist()
    yf_to_bbg   = dict(zip(yf_tickers, bbg_tickers))

    returns = _pull_returns(yf_tickers, "ETF panel", start, end)

    # rename yf_ticker columns back to Bloomberg labels
    returns.columns = [yf_to_bbg.get(c, c) for c in returns.columns]

    # warn if any theme lost all its ETFs
    surviving = set(returns.columns)
    for theme, grp in etf_config.groupby("theme"):
        theme_bbg = set(grp["ticker"])
        if not surviving & theme_bbg:
            print(f"  WARNING: theme '{theme}' has no surviving ETFs -- "
                  f"check tickers in etfs.csv")

    print(f"  ETFs kept: {returns.columns.tolist()}")
    return returns


# -------------------- 4. Covariance universe (ACWI proxy) --------------------

# -------------------- hardcoded non-US index constituents --------------------
# Wikipedia and most index providers block automated access.
# These are the top constituents by market cap for each non-US index,
# in yfinance format. Sufficient for factor extraction purposes.
# PAPER NOTE: full Nikkei 225 and STOXX 600 constituent lists require
# a vendor licence (MSCI, Bloomberg, Refinitiv). We use the top ~50-60
# names by market cap as a proxy -- these dominate index returns and
# are sufficient for RP-PCA to identify global latent factors.

_NIKKEI_TOP50 = [
    "7203.T","9984.T","6861.T","6758.T","6098.T","8306.T","9432.T",
    "4063.T","6954.T","7974.T","8035.T","6367.T","9433.T","4519.T",
    "6857.T","8031.T","7267.T","4543.T","6902.T","8316.T","9022.T",
    "4502.T","7751.T","8411.T","6701.T","5401.T","4661.T","6752.T",
    "8058.T","7011.T","3382.T","6503.T","4568.T","8766.T","9020.T",
    "6702.T","7741.T","8604.T","4523.T","9735.T","6301.T","8001.T",
    "6471.T","7733.T","4578.T","5108.T","8002.T","6762.T","7832.T","4507.T",
]

_STOXX_TOP60 = [
    # UK (.L)
    "SHEL.L","AZN.L","HSBA.L","ULVR.L","BP.L","RIO.L","GSK.L","DGE.L",
    "LLOY.L","BARC.L","NG.L","REL.L","VOD.L","TSCO.L",
    # Germany (.DE)
    "SAP.DE","SIE.DE","ALV.DE","MRK.DE","BAYN.DE","BMW.DE","MBG.DE",
    "DTE.DE","BAS.DE","DB1.DE","HEN3.DE","VOW3.DE","EOAN.DE","RWE.DE",
    # France (.PA)
    "MC.PA","OR.PA","SAN.PA","AIR.PA","TTE.PA","BNP.PA","SU.PA",
    "RI.PA","CS.PA","ACA.PA","ENGI.PA","SGO.PA","CAP.PA","VIE.PA",
    # Switzerland (.SW)
    "NESN.SW","NOVN.SW","ROG.SW","ABBN.SW","ZURN.SW","SREN.SW",
    # Netherlands (.AS)
    "ASML.AS","ING.AS","PHIA.AS","UNA.AS","HEIA.AS",
    # Spain (.MC)
    "ITX.MC","SAN.MC","BBVA.MC","IBE.MC",
]

# -------------------- US mid/small cap via NASDAQ+NYSE minus SP500 --------------------
# SP400 and SP600 Wikipedia pages block automated access.
# We approximate US mid+small by taking NASDAQ+NYSE tickers not in SP500.
# The coverage filter in pull_covariance_universe() will keep only active names.
# PAPER NOTE: SP400/SP600 lists require S&P licence; we use exchange listing
# minus SP500 as a free proxy for the mid/small cap segment.

# minimum market cap for the COVARIANCE universe mid/small component.
# higher than the target-universe floor ($50M) because the covariance
# universe defines the factor space -- it must be clean, liquid names
# where co-movement is economically meaningful, not penny-stock noise.
# micro-caps (crypto miners, biotech shells) have huge idiosyncratic
# residual variance that dominates PCA factors and buries real themes.
_COV_MIN_MKTCAP = 1_000_000_000   # $1B floor for covariance universe


def _get_us_mid_small_tickers() -> list:
    """
    Pull NASDAQ+NYSE FULL metadata, remove SP500, apply a $1B market cap floor.

    Uses the full metadata files (with marketCap) rather than bare ticker lists
    so we can exclude micro-caps. The $1B floor removes the penny-stock and
    crypto-miner noise (CETX, EVTV, HIVE, BTCT etc.) that was dominating the
    RP-PCA factors with extreme idiosyncratic variance.

    PAPER NOTE: the covariance universe is intentionally restricted to
    liquid mid/large caps (>$1B) so that extracted factors reflect genuine
    economic co-movement rather than micro-cap idiosyncratic blow-ups.
    The TARGET universe (where we hunt for candidates) keeps the lower
    $50M floor -- we want to find small caps there, we just don't want
    them defining the factor space.
    """
    base = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main"
    try:
        nasdaq_data = requests.get(f"{base}/nasdaq/nasdaq_full_tickers.json",
                                    headers=HEADERS, timeout=30).json()
        nyse_data   = requests.get(f"{base}/nyse/nyse_full_tickers.json",
                                    headers=HEADERS, timeout=30).json()
        all_data    = nasdaq_data + nyse_data
    except Exception as e:
        print(f"  mid/small metadata fetch failed ({e})")
        return []

    try:
        sp500 = set(pd.read_csv(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
            "/main/data/constituents.csv"
        )["Symbol"].str.replace(".", "-").tolist())
    except Exception:
        sp500 = set()

    def is_clean(entry):
        t      = entry.get("symbol", "")
        mktcap = float(entry.get("marketCap") or 0)
        if not t.isalpha() or not (1 <= len(t) <= 5):
            return False
        if len(t) > 2 and t.endswith(("W", "R", "U", "P")):
            return False
        if t in sp500:
            return False
        if mktcap < _COV_MIN_MKTCAP:    # $1B floor
            return False
        return True

    kept = sorted(set(d["symbol"] for d in all_data if is_clean(d)))
    print(f"    mid/small after $1B floor: {len(kept)} names "
          f"(removed micro-cap noise)")
    return kept


def get_acwi_proxy_tickers(use_cache: bool = True,
                            us_only: bool = True) -> dict:
    """
    Builds the covariance universe for RP-PCA factor extraction.

    Two modes:
      us_only=True  (default): SP500 + US mid/small only (~1000 stocks)
                                no foreign tickers, no timezone alignment issues,
                                no FX co-movement contaminating factor structure.
                                THIS IS THE CURRENT RECOMMENDED MODE.

      us_only=False:           includes STOXX 600 top 60 and Nikkei 225 top 50.
                                broader cross-section but introduces:
                                - timing artifacts (London/Tokyo close vs US close)
                                - FX co-movement appearing as latent factors
                                use only if you address these with FX residualization
                                and lag-alignment first.

    PAPER NOTE -- why us_only is the default:
      Foreign-listed stocks close at different times than US (London 11:30 ET,
      Tokyo prior day's 02:00 ET). yfinance returns local-close prices which
      creates spurious lead/lag structure when combined in a single return panel.
      Foreign stocks also share FX exposure (EURUSD, JPYUSD) that FF5 residualization
      cannot strip out -- this FX co-movement appears as latent factors and
      contaminates the thematic signal. Eliminating foreign tickers from the
      covariance universe is the cleanest fix; the thematic ETFs themselves
      can still be global since they serve only as theme labels, not as the
      covariance source.

    use_cache: saves to outputs/data/acwi_proxy_tickers.json -- delete to refresh.
              Cache file is mode-aware (separate cache for us_only=True/False).
    """
    cache_suffix = "us_only" if us_only else "global"
    cache_path   = os.path.join(DATA_OUT, f"acwi_proxy_tickers_{cache_suffix}.json")

    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            tickers = json.load(f)
        total = sum(len(v) for v in tickers.values())
        print(f"loaded covariance universe from cache ({cache_suffix}): "
              f"{total} tickers across {list(tickers.keys())}")
        return tickers

    tickers = {}
    if us_only:
        print("building US-only covariance universe (recommended mode)")
    else:
        print("building global covariance universe -- WARNING: "
              "FX and timing artifacts may contaminate factors")

    # -------------------- US: S&P 500 (large cap) --------------------
    try:
        sp500 = pd.read_csv(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
            "/main/data/constituents.csv"
        )["Symbol"].str.strip().tolist()
        sp500 = [t.replace(".", "-") for t in sp500]
        tickers["us_sp500"] = sp500
        print(f"  US S&P 500:   {len(sp500)} tickers (GitHub)")
    except Exception as e:
        print(f"  SP500 fetch failed: {e}")

    # -------------------- US: mid/small cap proxy (>$1B) --------------------
    # now that the $1B floor removes micro-cap noise, keep all clean names
    # (typically ~1500-2000) -- a larger clean cross-section gives more
    # stable factor estimates than a small subsample
    mid_small = _get_us_mid_small_tickers()
    if mid_small:
        tickers["us_mid_small"] = mid_small
        print(f"  US mid/small: {len(mid_small)} tickers (>$1B, ex-SP500)")

    # -------------------- foreign components (only if requested) --------------------
    # kept in code for future use with proper timing/FX adjustments
    if not us_only:
        tickers["europe_stoxx_top60"] = _STOXX_TOP60
        print(f"  Europe STOXX: {len(_STOXX_TOP60)} tickers "
              f"(top 60 by mkt cap, hardcoded)")
        tickers["japan_nikkei_top50"] = _NIKKEI_TOP50
        print(f"  Japan Nikkei: {len(_NIKKEI_TOP50)} tickers "
              f"(top 50 by mkt cap, hardcoded)")

    total = sum(len(v) for v in tickers.values())
    print(f"\ncovariance universe total: {total} tickers across {list(tickers.keys())}")
    print("  (yfinance coverage filter will reduce to active names only)")

    if use_cache and tickers:
        with open(cache_path, "w") as f:
            json.dump(tickers, f)
        print(f"cached to {cache_path} -- delete to refresh")

    return tickers


def pull_covariance_universe(start: str, end=None,
                              use_cache: bool = True,
                              us_only: bool = True) -> pd.DataFrame:
    """
    Pulls weekly returns for the covariance universe.

    Role in pipeline: the cross-section that RP-PCA runs on.
    Must be broad enough that all themes appear as latent factors.

    us_only: True by default (recommended). Set False to include
             STOXX/Nikkei (introduces FX and timing artifacts).

    Returns a single combined return DataFrame.
    The coverage filter handles any tickers that fail to pull.
    """
    ticker_dict = get_acwi_proxy_tickers(use_cache=use_cache, us_only=us_only)

    # flatten to a single list, deduplicate
    all_tickers = []
    seen        = set()
    for region, tkrs in ticker_dict.items():
        for t in tkrs:
            if t not in seen:
                all_tickers.append(t)
                seen.add(t)

    print(f"\npulling covariance universe: {len(all_tickers)} unique tickers")
    returns = _pull_returns(all_tickers, "covariance universe", start, end)
    return returns


# -------------------- 5. Target universe (Russell 2000 proxy) --------------------

# -------------------- equity filter config --------------------
# industries that identify non-operating companies (funds, SPACs, CEFs)
# we keep REITs and BDCs as they can have genuine thematic exposure
_EXCLUDE_INDUSTRIES = {
    "Finance Companies",        # closed-end funds (QQQX, GOF etc.)
    "Blank Checks",             # SPACs
    "Investment Managers",      # fund managers / ETF sponsors
    "Trusts Except Educational Religious and Charitable",
    "Finance/Investors Services",
}

# name substrings that flag non-operating companies
_EXCLUDE_NAME_KEYWORDS = [
    " ETF", " FUND", " TRUST", "ACQUISITION", " SPAC",
    " WARRANT", "BLANK CHECK", " NOTE ", " DEBENTURE",
]

# minimum market cap -- excludes micro-caps, shells, crypto miners
# $50M is the approximate bottom of the Russell 2000 range
_MIN_MKTCAP = 50_000_000

# maximum market cap -- excludes large-cap names that aren't S&P 500 members
# for reasons unrelated to size (e.g. foreign domicile: VALE, TECK, SCCO).
# matches the covariance universe floor so the two universes stay disjoint by size.
_MAX_MKTCAP_TARGET = 1_000_000_000


def _is_valid_equity(entry: dict, sp500: set) -> bool:
    """
    Returns True if a ticker entry from rreichel3 full metadata
    looks like a genuine small-cap operating company.

    Excludes: ETFs, closed-end funds, SPACs, warrants, preferred shares,
    sub-$50M micro-caps, and S&P 500 large caps.
    """
    sym      = entry.get("symbol", "")
    name     = entry.get("name", "").upper()
    industry = entry.get("industry", "")
    mktcap   = float(entry.get("marketCap") or 0)

    # basic ticker format: letters only, 1-5 chars
    if not sym.isalpha(): return False
    if not (1 <= len(sym) <= 5): return False

    # common non-equity suffixes: W=warrant, R=right, P=preferred, U=unit
    if len(sym) > 2 and sym.endswith(("W", "R", "U", "P")): return False

    # not in S&P 500 (large cap -- goes in covariance universe)
    if sym in sp500: return False

    # explicit market-cap ceiling -- S&P 500 membership alone doesn't catch
    # large foreign-domiciled names (e.g. VALE, TECK, SCCO) that are large-cap
    # but ineligible for the S&P 500 on domicile grounds, not size grounds
    if mktcap >= _MAX_MKTCAP_TARGET: return False

    # exclude by industry classification
    if industry in _EXCLUDE_INDUSTRIES: return False

    # exclude by name keywords
    for kw in _EXCLUDE_NAME_KEYWORDS:
        if kw in name: return False

    # exclude zero / sub-$50M market cap
    if mktcap < _MIN_MKTCAP: return False

    return True


def get_target_universe_tickers(use_cache: bool = True) -> list:
    """
    Builds a US small-cap target universe approximating Russell 2000.

    Uses the rreichel3/US-Stock-Symbols FULL metadata files (not just tickers)
    to filter out ETFs, closed-end funds, SPACs, warrants, preferred shares,
    and sub-$50M micro-caps -- which is why QQQX, GOF, BSTZ etc. no longer
    appear in results.

    PAPER NOTE: True Russell 2000 constituent list requires FTSE Russell licence
    or iShares API (blocked). This proxy uses NASDAQ+NYSE listings with metadata
    filtering to approximate genuine small-cap operating companies. The $50M
    market cap floor and 85%% coverage filter further align with Russell 2000
    methodology (Russell imposes its own liquidity and float screens).

    use_cache: saves to outputs/data/target_universe_tickers.json -- delete to refresh.
    """
    cache_path = os.path.join(DATA_OUT, "target_universe_tickers.json")

    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            tickers = json.load(f)
        print(f"loaded {len(tickers)} target universe tickers from cache")
        return tickers

    print("building target universe (Russell 2000 proxy with metadata filter)...")
    base = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main"

    # -------------------- pull full metadata files --------------------
    # these include sector, industry, marketCap, name -- much richer than
    # the simple ticker lists, allowing proper equity screening
    try:
        nasdaq_data = requests.get(f"{base}/nasdaq/nasdaq_full_tickers.json",
                                    headers=HEADERS, timeout=30).json()
        nyse_data   = requests.get(f"{base}/nyse/nyse_full_tickers.json",
                                    headers=HEADERS, timeout=30).json()
        all_data    = nasdaq_data + nyse_data
        print(f"  fetched metadata for {len(all_data)} listings")
    except Exception as e:
        print(f"  metadata fetch failed ({e}), using fallback")
        return ["AEHR", "ALKT", "AMSF", "BCPC", "CSWI", "DVAX",
                "EVTC", "FCNCA", "GLNG", "HIMS", "IIPR", "JOBY"]

    # -------------------- load S&P 500 for exclusion --------------------
    try:
        sp500 = set(pd.read_csv(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies"
            "/main/data/constituents.csv"
        )["Symbol"].str.replace(".", "-").tolist())
        print(f"  excluding {len(sp500)} S&P 500 names")
    except Exception as e:
        print(f"  SP500 fetch failed ({e})")
        sp500 = set()

    # -------------------- apply equity filter --------------------
    valid   = [d for d in all_data if _is_valid_equity(d, sp500)]
    tickers = sorted(set(d["symbol"] for d in valid))
    print(f"  valid small-cap equities after metadata filter: {len(tickers)}")
    print(f"  (excluded: ETFs, funds, SPACs, warrants, <$50M mktcap, SP500)")

    if use_cache:
        with open(cache_path, "w") as f:
            json.dump(tickers, f)
        print(f"  cached to {cache_path} -- delete to refresh")

    return tickers


def pull_target_universe(start: str, end=None,
                          use_cache: bool = True) -> pd.DataFrame:
    """
    Pulls weekly returns for the target universe (Russell 2000 proxy).

    Role in pipeline: the universe we score against theme factors
    to find undiscovered thematic exposure.
    """
    tickers = get_target_universe_tickers(use_cache=use_cache)
    returns = _pull_returns(tickers, "target universe (Russell proxy)", start, end)
    return returns


# -------------------- 6. Save helper --------------------

def save(df: pd.DataFrame, filename: str):
    path = os.path.join(DATA_OUT, filename)
    df.to_csv(path)
    print(f"saved: {path}  ({df.shape[0]} rows x {df.shape[1]} cols)")


# -------------------- 7. Main --------------------

if __name__ == "__main__":

    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "etfs.csv")
    etf_config  = load_etf_config(config_path)

    etf_returns = pull_etf_returns(etf_config, start=START_DATE)
    save(etf_returns, "etf_returns.csv")

    cov_returns = pull_covariance_universe(start=START_DATE)
    save(cov_returns, "covariance_universe_returns.csv")

    tgt_returns = pull_target_universe(start=START_DATE)
    save(tgt_returns, "target_universe_returns.csv")

    print("\ndata_pull.py done -- next: rppca.py")


# -------------------- helper: split themes from controls --------------------

def split_themes_controls(etf_config: pd.DataFrame):
    """
    Splits the ETF config into discovery themes vs calibration controls.

    Returns (themes_config, controls_config):
      themes_config   -- type == 'theme'; these get full pipeline + baskets
      controls_config -- type in ('sector','subsector'); these run through the
                         coherence/distinctiveness diagnostic ONLY (calibration
                         anchors for the coherence scale), no basket construction.
    """
    themes   = etf_config[etf_config["type"] == "theme"].copy()
    controls = etf_config[etf_config["type"] != "theme"].copy()
    return themes, controls
