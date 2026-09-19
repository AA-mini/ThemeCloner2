# =====================================================================
# CHART CELLS for ThemeCloner_V2_WalkForward_revised.ipynb
# Paste each block as its own new cell, after cell 24 (6b. Period level metrics).
# Charts 1 and 2 run against your CURRENT committed results (no rerun needed).
# Chart 3 requires the src/evaluation_v3.py + src/walkforward_v3.py changes
# AND a fresh backtest run (the new column doesn't exist in the current results).
# =====================================================================


# --------------------------------------------------------------------
# CHART 1 -- Median residual ETF correlation by theme (bar chart)
# Source: evaluation_summary["median_basket_etf_corr"] (already computed)
# --------------------------------------------------------------------
import matplotlib.pyplot as plt

chart1 = evaluation_summary.sort_values("median_basket_etf_corr", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["seagreen" if v >= 0.30 else "goldenrod" if v >= 0.15 else "firebrick"
          for v in chart1["median_basket_etf_corr"]]
ax.barh(chart1["theme"], chart1["median_basket_etf_corr"], color=colors)
ax.axvline(0.30, color="black", linestyle="--", linewidth=1, label="0.30 (moderate)")
ax.axvline(0.15, color="gray", linestyle=":", linewidth=1, label="0.15 (weak)")
ax.set_xlabel("median basket-vs-residual-ETF-blend correlation")
ax.set_title("Chart 1: Basket tracking of the residualized ETF blend, by theme")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("outputs/fig_v3_chart1_median_etf_corr.png", dpi=150)
plt.show()


# --------------------------------------------------------------------
# CHART 2 -- Quarterly tracking correlation through time
# Source: results["evaluation"], column basket_residual_etf_corr, pivoted
#         date x theme (already computed)
# --------------------------------------------------------------------
import matplotlib.pyplot as plt

ev = results["evaluation"]
pivot2 = ev.pivot_table(index="rebalance_date",
                         columns="theme",
                         values="basket_residual_etf_corr")

fig, ax = plt.subplots(figsize=(12, 6))
# highlight the two themes that actually worked; mute the rest to keep it readable
highlight = ["Water", "Cybersecurity"]
for theme in pivot2.columns:
    if theme in highlight:
        ax.plot(pivot2.index, pivot2[theme], linewidth=2.2, marker="o",
                markersize=3, label=theme, zorder=3)
    else:
        ax.plot(pivot2.index, pivot2[theme], linewidth=0.8, color="lightgray",
                alpha=0.7, zorder=1)
ax.axhline(0, color="black", linewidth=0.6)
ax.axhline(0.30, color="green", linestyle="--", alpha=0.4, linewidth=1)
ax.set_ylabel("quarterly basket-vs-residual-ETF-blend correlation")
ax.set_xlabel("rebalance date")
ax.set_title("Chart 2: Tracking correlation through time\n"
             "(Water and Cybersecurity highlighted; all other themes gray)")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig("outputs/fig_v3_chart2_tracking_through_time.png", dpi=150)
plt.show()


# --------------------------------------------------------------------
# CHART 3 -- Cumulative basket return vs raw ETF blend, per theme
# Source: results["evaluation"], columns raw_basket_period_return and
#         raw_etf_blend_period_return.
# REQUIRES: the src/evaluation_v3.py + src/walkforward_v3.py changes AND a
#           fresh backtest run. The guard below prints a clear message if the
#           new column isn't present yet (i.e. you're on old results).
# --------------------------------------------------------------------
import matplotlib.pyplot as plt
import math

ev = results["evaluation"]

if "raw_etf_blend_period_return" not in ev.columns:
    print("Chart 3 needs a fresh backtest run: 'raw_etf_blend_period_return' "
          "is not in results['evaluation'] yet.\n"
          "Apply the src/evaluation_v3.py + src/walkforward_v3.py changes, rerun "
          "the '4. Point in time backtest' cell, then re-run this cell.")
else:
    themes = sorted(ev["theme"].unique())
    ncols = 3
    nrows = math.ceil(len(themes) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows))
    axes = axes.flatten()

    for ax, theme in zip(axes, themes):
        sub = ev[ev["theme"] == theme].sort_values("rebalance_date")
        b = sub["raw_basket_period_return"].fillna(0.0)
        e = sub["raw_etf_blend_period_return"].fillna(0.0)
        basket_curve = (1.0 + b).cumprod().values
        etf_curve = (1.0 + e).cumprod().values
        x = sub["rebalance_date"].values
        ax.plot(x, basket_curve, linewidth=2.0, color="steelblue", label="basket")
        ax.plot(x, etf_curve, linewidth=2.0, linestyle="--", color="firebrick",
                label="raw ETF blend")
        ax.set_title(theme, fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in axes[len(themes):]:
        ax.axis("off")

    fig.suptitle("Chart 3: Cumulative basket return vs. raw ETF blend, by theme "
                 "(compounded across walk-forward periods)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("outputs/fig_v3_chart3_cumulative_return.png", dpi=150)
    plt.show()
