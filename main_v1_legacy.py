# main.py
# Runs the full ThemeCloner pipeline end to end.
# Edit config/etfs.csv to change the themes and ETFs -- that's the only
# file you need to touch for day-to-day use.
#
# Pipeline:
#   1. Pull ETF and universe returns (01_data_pull.py)
#   2. Fit RP-PCA on ETF panel (02_rppca.py)
#   3. Distil theme DNA across ETFs per theme (03_theme_dna.py)
#   4. Project universe stocks into theme factor space (04_projection.py)
#   5. Score and rank candidates (05_scoring.py)
#
# Usage:
#   python main.py
#   python main.py --universe russell2000   (default)
#   python main.py --universe eurostoxx     (swap target universe)
#   python main.py --k 5 --gamma 10        (override RP-PCA params)

import os
import argparse
import pandas as pd

from src.data_pull    import load_etf_config, pull_etf_returns, pull_universe_returns, get_russell2000_tickers
from src.rppca        import fit_rppca
from src.theme_dna    import build_theme_dna
from src.projection   import project_universe, score_universe, save_projections
from src.scoring      import rank_candidates, save_outputs


# -------------------- argument parsing --------------------

def parse_args():
    p = argparse.ArgumentParser(description="ThemeCloner -- find small cap theme analogs")
    p.add_argument("--universe", default="russell2000",
                   help="target universe: 'russell2000' or path to a CSV of tickers")
    p.add_argument("--k",     type=int,   default=5,    help="number of RP-PCA factors")
    p.add_argument("--gamma", type=float, default=10.0, help="RP-PCA gamma parameter")
    p.add_argument("--top_n", type=int,   default=30,   help="candidates to return per theme")
    p.add_argument("--start", default="2018-01-01",     help="data start date")
    return p.parse_args()


# -------------------- main --------------------

def main():
    args = parse_args()

    print("=" * 60)
    print("ThemeCloner")
    print(f"  universe: {args.universe}")
    print(f"  K={args.k}, gamma={args.gamma}, top_n={args.top_n}")
    print("=" * 60)

    config_path = os.path.join("config", "etfs.csv")

    # -------------------- step 1: data --------------------
    print("\n[1/5] pulling data...")
    etf_config = load_etf_config(config_path)
    etf_returns = pull_etf_returns(etf_config["ticker"].tolist(), start=args.start)

    if args.universe == "russell2000":
        universe_tickers = get_russell2000_tickers()
    else:
        universe_tickers = pd.read_csv(args.universe)["ticker"].tolist()

    universe_returns = pull_universe_returns(universe_tickers, start=args.start)

    # -------------------- step 2 & 3: RP-PCA + theme DNA --------------------
    print("\n[2/5] fitting RP-PCA on ETF panel...")
    rppca_result = fit_rppca(etf_returns, K=args.k, gamma=args.gamma)

    print("\n[3/5] distilling theme DNA...")
    dna_result = build_theme_dna(etf_returns, etf_config, K=args.k, gamma=args.gamma)

    # -------------------- step 4: project universe --------------------
    print("\n[4/5] projecting universe into theme factor space...")
    projections = project_universe(universe_returns, etf_returns, rppca_result)
    scores      = score_universe(projections, dna_result["theme_dna"])
    save_projections(projections, scores)

    # -------------------- step 5: rank and save --------------------
    print("\n[5/5] ranking candidates...")
    ranked = rank_candidates(scores, top_n=args.top_n)
    save_outputs(ranked)

    print("\n" + "=" * 60)
    print("done -- results in outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
