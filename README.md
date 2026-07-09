# ThemeCloner

**Find small-cap (or any-universe) analogs of large-cap thematic ETFs using RP-PCA.**

Given a set of thematic ETFs (e.g. three AI infrastructure ETFs), ThemeCloner:
1. Learns the latent factor structure of those ETFs using RP-PCA (Lettau & Pelger 2020)
2. Distils a pure "theme DNA" fingerprint by averaging across ETFs in the same theme — removing construction artifacts specific to any single ETF
3. Projects a target universe (Russell 2000, EuroStoxx, etc.) into that factor space
4. Scores and ranks target stocks by cosine similarity to each theme's DNA

The key insight: stocks that score highly have the **same latent factor exposure** as the theme, even if they have never been labeled thematic. This is the mispricing opportunity.

---

## Repo structure

```
ThemeCloner/
├── config/
│   └── etfs.csv              ← Edit this. One row per ETF, columns: ticker, theme, description
├── src/
│   ├── 01_data_pull.py       ← Pull ETF + target universe returns via yfinance
│   ├── 02_rppca.py           ← Core RP-PCA implementation (Lettau & Pelger 2020)
│   ├── 03_theme_dna.py       ← Ensemble theme DNA distillation across ETFs
│   ├── 04_projection.py      ← Project target universe into theme factor space
│   └── 05_scoring.py         ← Cosine similarity scoring + ranking + robustness check
├── notebooks/
│   └── ThemeCloner.ipynb     ← Master notebook -- interactive version of main.py
├── outputs/                  ← Auto-created, gitignored
├── main.py                   ← Run full pipeline from terminal
└── README.md
```

---

## Quickstart

### 1. Configure your ETFs

Edit `config/etfs.csv`. Multiple ETFs per theme are supported and recommended — the more ETFs per theme, the cleaner the DNA extraction:

```csv
ticker,theme,description
BOTZ,Robotics & AI,Global X Robotics and AI ETF
ROBO,Robotics & AI,ROBO Global Robotics and Automation ETF
AINF,AI Infrastructure,iShares AI Infrastructure ETF
ICLN,Clean Energy,iShares Global Clean Energy ETF
```

### 2. Install dependencies

```bash
pip install numpy pandas yfinance matplotlib seaborn scipy
```

### 3. Run

```bash
# terminal
python main.py

# with custom params
python main.py --universe russell2000 --k 5 --gamma 10 --top_n 30

# or open notebooks/ThemeCloner.ipynb in Jupyter
```

---

## Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `K` | 5 | Number of RP-PCA factors. 5 is the Lettau-Pelger default for broad universes. Use the gamma sweep plot to validate. |
| `gamma` | 10.0 | RP-PCA penalty on cross-sectional mean. gamma=0 is standard PCA. Higher gamma tilts toward higher Sharpe factors. |
| `min_score` | 0.3 | Minimum cosine similarity to include a stock as a candidate. Tune based on how many candidates you want. |
| `top_n` | 30 | Candidates returned per theme. |
| `start_date` | 2018-01-01 | Return history start. 5+ years recommended for stable RP-PCA. |

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/data/etf_returns.csv` | Weekly ETF return panel |
| `outputs/data/universe_returns.csv` | Weekly universe return panel |
| `outputs/data/projections.csv` | Universe stocks × K factor coordinates |
| `outputs/data/scores.csv` | Universe stocks × themes cosine similarity matrix |
| `outputs/ranked_candidates.csv` | Top N candidates per theme with scores |
| `outputs/robustness_check.csv` | Method 1 vs Method 2 agreement table |

---

## Methodology

RP-PCA (Lettau & Pelger, 2020, *Journal of Finance*) extends standard PCA by adding a penalty `γ * μμ'` to the covariance matrix objective, where `μ` is the vector of mean returns. This tilts extracted factors toward directions with high risk premia rather than pure return variance — exactly what matters for thematic investing where the economic driver (not statistical size) is the signal.

The "theme DNA" purification step addresses a known issue with ETF-based research: individual ETFs capture noise from their construction methodology (index rules, liquidity screens, rebalancing schedules) alongside genuine thematic signal. By taking the centroid fingerprint across multiple ETFs tracking the same theme, we extract the shared latent structure and discard ETF-specific artifacts.

---

## Robustness check

The notebook runs both:
- **Method 1 (primary):** RP-PCA on ETF panel → project universe stocks in
- **Method 2 (robustness):** RP-PCA on universe → project ETF DNA in

Stocks that rank in the top-N under **both** methods are high-confidence candidates. Stocks that only appear in Method 1 have the right latent exposure but don't superficially resemble the large-cap basket — these are potentially the most interesting mispricing opportunities.

---

## Reference

Lettau, M. and Pelger, M. (2020). Estimating latent asset-pricing factors. *Journal of Econometrics*, 218(1), 1–31.
