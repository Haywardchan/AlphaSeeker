# Alpha Seeker

A local research dashboard for Berkshire Hathaway Class B (`BRK.B`) and the Vanguard
Total Stock Market ETF (`VTI`). A sidebar toggle switches the complete analysis between
the two instruments. It uses adjusted daily Yahoo Finance bars to estimate:

- nearest volatility-adjusted support and resistance zones;
- probabilities that resistance, support, or neither is touched first within 10 sessions;
- probability distributions for the 10-session path minimum and maximum;
- a next-session intraday spread distribution via conditional quantile regression;
- a proxy market-emotion stack from VIX, SPY, and the selected instrument's technicals;
- local NLP scoring of recent Yahoo Finance headlines;
- fused `BUY`, `HOLD`, or `SELL / REDUCE` guidance combining technical, emotion, and news inputs;
- walk-forward strategy results compared with the selected instrument's buy-and-hold.

The dashboard separates **base technical guidance** from **final fused guidance** so an
emotion/news adjustment cannot silently change a recommendation. Its comparison view can
run BRK.B and VTI with identical settings and contrast guidance, probabilities, risk,
relative performance, and SPY context.

This is research software, not investment advice. Outputs are estimates from historical
daily data, can be wrong, and do not guarantee alpha. Yahoo data is unofficial, intended
for personal/research use, and may be delayed or unavailable.

## Run with Docker

Docker Desktop must be running with Linux containers.

```powershell
docker compose build
docker compose up app
```

Open <http://localhost:8502>. Cached market data is stored in `./data`.

Run quality checks in the same reproducible image:

```powershell
docker compose --profile test build test
docker compose run --rm test
docker compose run --rm test ruff check .
```

## Method

All features and levels for a date use only information available at that close. Swing
highs/lows are detected with SciPy and clustered into zones with kernel density
estimation; ATR barriers are used when a robust zone is unavailable. The classifier is
trained on upper-first, lower-first, and neither outcomes over the following 10 sessions.
Same-day touches of both barriers are excluded because daily bars do not preserve
intraday order.

Candidate classifiers are evaluated on chronological folds separated by the forecast
horizon. Reported probabilities are selected by out-of-sample multiclass log loss and
Brier score. Separate quantile models estimate future path-low and path-high returns.
These are marginal distributions, not simulated joint paths. A dedicated spread model
forecasts the next session's `(High − Low) / Close` distribution from the same causal
feature set.

The advisory signal compares probability-weighted upside and downside after estimated
costs. A fusion layer can adjust that base guidance using proxy emotion (VIX/SPY/stock)
and recent headline sentiment. The signal backtest and Strategy Lab still use the base
technical rule only; news overlay is live-only because Yahoo does not provide archival
headlines for historical replay.

The webpage labels those backtests as technical-only evidence. They should not be read as
historical validation of the final fused recommendation.

### Emotion and news overlays

Emotion is a weighted composite of VIX level, SPY trend/RSI/volume, selected-instrument
RSI/volume, and a TRIN-shaped proxy derived from SPY up/down participation. It is explicitly
approximate—not exchange-calculated TRIN or NYSE breadth.

News uses VADER sentiment plus instrument-specific relevance terms on recent BRK.B or
VTI headlines fetched from Yahoo Finance. Impact is bounded and decays with headline age.

Instrument profiles reduce duplicated SPY exposure and company-news weight for VTI while
using a longer price-level history and stronger company-specific news sensitivity for
BRK.B. These are explicit research assumptions, not independently proven optimal settings.

Fusion applies transparent edge adjustments and stress gates (for example elevated VIX
or strongly negative headline tone) before emitting final guidance.

### Strategy Lab

The Strategy Lab compares four interpretable 0%-100% long/cash candidates: a defensive
probability overlay, a 200-day trend filter, a trend/probability/range-risk hybrid, and a
support-pullback rule. Strategy and benchmark start with the same capital on the same
first actionable next-open date; the earlier implementation incorrectly let buy-and-hold
start before walk-forward strategy predictions existed.

Small parameter grids are selected on the earlier portion of fold-held-out predictions.
The leaderboard is calculated on a later untouched holdout after transaction costs.
A candidate is declared the winner only when it has positive holdout excess return and
maximum drawdown no worse than buy-and-hold. If none qualify, the dashboard reports
buy-and-hold as the winner rather than tuning until apparent alpha appears.

## Important limitations

- A single security supplies a small, non-stationary dataset.
- Emotion indicators are proxies; they do not replicate exchange TRIN or full market breadth.
- News scoring is lexicon-based, not fundamental research, and depends on Yahoo headline coverage.
- Daily OHLC cannot determine the order of multiple intraday events.
- Quantile and first-touch estimates can degrade when market regimes change.
- Backtests remain hypothetical and are sensitive to costs and modeling assumptions.
- Strategy Lab holdout outperformance is historical evidence, not a guarantee of future
  performance; repeated research decisions can still introduce selection bias.

The data layer is isolated so a licensed provider can replace Yahoo before any
commercial or operational use.
