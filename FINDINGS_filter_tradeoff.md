# The Filter Tightness Trade-Off

**Status:** empirically established, August 2026, full ~2,000-stock covariance universe,
quarterly rebalance, 19 periods, 11 themes.

**Why it matters:** this is a general property of screen-then-rank pipelines, not a
quirk of this project. It belongs in the write-up regardless of which framing the
paper ends up taking.

---

## The finding

Candidate admission filters (minimum R-squared against the factor model, minimum
adjusted R-squared, minimum cosine similarity to the theme, minimum fraction of
theme ETFs matched) control a trade-off between two things the paper cares about,
and they move in opposite directions.

**Tight filters** (min_candidate_r2=0.15, min_cosine=0.30, min_etf_match_fraction=0.60):

- Baskets collapsed to 3-9 names against a configured target of 30. Filters were
  removing 70-90% of intended candidates.
- Basket-level tracking was comparatively strong: median basket-vs-residual-ETF
  correlation around 0.25, with individual themes reaching 0.39-0.50.
- Ranking was unmeasurable. Rank IC requires at least 5 names to compute and returned
  NaN in many periods; the top-versus-bottom quintile spread requires at least 10 and
  returned NaN in every single period across the entire project.
- Apparent rank IC values swung violently between quarters (+0.80 to -0.65) because
  each was computed on a handful of stocks.

**Loose filters** (min_candidate_r2=0.05, min_cosine=0.15, min_etf_match_fraction=0.50):

- Every basket filled to exactly 30 names (mean 30.0, standard deviation 0.0, across
  209 theme-periods). No NaNs in any ranking metric.
- Ranking became measurable for the first time, and the top-versus-bottom spread
  produced values at last.
- Basket-level tracking weakened: median correlation fell from roughly 0.25 to
  roughly 0.10, with three themes turning slightly negative.

## The mechanism

Filters admit candidates in descending order of match quality. Tightening them keeps
only the closest matches, which raises the average quality of the basket and therefore
its correlation with the theme. But it also shrinks the basket, and both ranking
metrics need a minimum number of names to say anything at all.

So tightening improves the quantity you can measure well (group tracking) while
destroying your ability to measure the other quantity (ordering within the group).
Loosening does the reverse.

## Why this is a trap for this kind of research

An analyst tuning filters for the best-looking tracking number will naturally tighten
them, and will then be unable to test whether the stock ordering carries information.
The pipeline will appear to work while the central selection claim goes untested.
Worse, the small baskets that result produce unstable ranking estimates that look like
findings: an earlier run on 5-9 stock baskets showed rank IC of +0.34 for Water and
+0.26 for Cybersecurity, which collapsed to +0.16 and +0.09 once baskets filled to 30.
The high values were small-sample noise.

## Practical guidance

Report basket size alongside every ranking metric. A rank IC computed on 6 names is
not comparable to one computed on 30 and should not be presented in the same table
without that context.

Choose filter tightness according to the claim being made. A paper claiming group-level
exposure measurement can justify tight filters. A paper claiming stock selection must
run loose enough that the selection claim is testable, and must accept the weaker
tracking that comes with it.

Never tune filters on the metric you intend to report.

---

# Related: metric sensitivity to cross-section size

**Status:** established the same session, and it is a more severe version of the same
lesson.

The covariance universe was found to be running on approximately 750 stocks after
coverage filtering, due to a stale cached ticker list. Rebuilt properly, it contains
approximately 2,480 tickers.

On the small universe, theme-level ETF agreement (mean pairwise cosine similarity
between the ETFs defining a theme) correlated +0.82 with median rank IC across 11
themes. Every theme above 0.69 agreement had positive rank IC; every theme below 0.53
had negative. This looked like a strong, actionable finding, and a filter was built
to exploit it.

On the properly sized universe the same correlation was -0.52, then -0.10 on a repeat
run. The relationship reversed sign and then vanished. The original +0.82 was an
artifact of unstable factor loadings estimated on too few stocks.

**Lesson for the write-up:** with 11 themes, a correlation of +0.82 is not evidence.
Any cross-theme relationship of this kind must be reproduced on a properly sized
cross-section and across repeat runs before it is treated as real.
