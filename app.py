from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from alpha_seeker.features import build_feature_frame
from alpha_seeker.labels import next_day_spread
from alpha_seeker.pipeline import AnalysisResult, run_analysis

st.set_page_config(page_title="BRK.B Alpha Seeker", page_icon="📈", layout="wide")
st.title("BRK.B Alpha Seeker")
st.caption("Technical research for Berkshire Hathaway Class B · 10-session horizon")


@st.cache_resource(show_spinner="Training walk-forward models…")
def load_analysis(
    lookback_years: int,
    cost_bps: float,
    confidence: float,
    include_emotion: bool,
    include_news: bool,
) -> AnalysisResult:
    return run_analysis(
        lookback_years=lookback_years,
        transaction_cost_bps=cost_bps,
        confidence_margin=confidence,
        include_emotion=include_emotion,
        include_news=include_news,
    )


with st.sidebar:
    st.header("Research controls")
    lookback = st.slider("Training history (years)", 8, 25, 15)
    costs = st.number_input("Round-trip cost (bps)", 0.0, 100.0, 10.0, 5.0)
    confidence = st.slider("Confidence margin", 0.0, 0.25, 0.05, 0.01)
    include_emotion = st.toggle("Include emotion in guidance", value=True)
    include_news = st.toggle("Include news in guidance", value=True)
    if st.button("Refresh data and models", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

try:
    result = load_analysis(lookback, costs, confidence, include_emotion, include_news)
except Exception as exc:  # Streamlit should explain provider/model failures instead of crashing.
    st.error(f"Analysis unavailable: {exc}")
    st.info("Check internet access, then use “Refresh data and models.”")
    st.stop()

price = result.current_price
support = result.support
resistance = result.resistance
prob = result.probabilities
signal = result.signal

st.caption(
    f"Yahoo symbol BRK-B · Last completed bar: {result.as_of:%Y-%m-%d} · "
    f"{result.sample_count:,} model observations"
)

cols = st.columns(5)
cols[0].metric("Last close", f"${price:,.2f}")
cols[1].metric("Lower support", f"${support.price:,.2f}", f"{support.distance_pct:.1%}")
cols[2].metric(
    "Upper resistance", f"${resistance.price:,.2f}", f"+{resistance.distance_pct:.1%}"
)
cols[3].metric("Final guidance", signal.action)
cols[4].metric("Adjusted edge", f"{signal.adjusted_edge:.2%}")

if result.data_is_stale:
    st.warning("The latest cached daily bar appears stale. Guidance may not reflect the market.")

with st.expander("Why this advice?"):
    st.write(
        f"Base technical guidance: **{result.base_signal.action.value}** "
        f"({result.base_signal.edge:+.2%} edge)."
    )
    st.write(
        f"Emotion adjustment: {signal.emotion_adjustment:+.2%} · "
        f"News adjustment: {signal.news_adjustment:+.2%} · "
        f"Confidence margin used: {signal.confidence_margin_used:.1%}."
    )
    if signal.gates_applied:
        st.write("Gates applied: " + "; ".join(signal.gates_applied))
    st.write(result.base_signal.rationale)

st.subheader("10-session first-touch probabilities")
pcols = st.columns(3)
pcols[0].metric("Upper resistance first", f"{prob['upper_first']:.1%}")
pcols[1].metric("Lower support first", f"{prob['lower_first']:.1%}")
pcols[2].metric("Neither level", f"{prob['neither']:.1%}")
st.info(signal.rationale)

tab_names = [
    "Price & levels",
    "10-day min/max distributions",
    "Next-day spread",
    "Emotion & news",
    "Model validation",
    "Signal backtest",
    "Strategy Lab",
]
tab_chart, tab_range, tab_spread, tab_emotion, tab_validation, tab_backtest, tab_strategies = (
    st.tabs(tab_names)
)

with tab_chart:
    bars = result.bars.tail(300)
    indicators = build_feature_frame(result.bars).iloc[-1]
    fig = go.Figure(
        go.Candlestick(
            x=bars.index,
            open=bars["Open"],
            high=bars["High"],
            low=bars["Low"],
            close=bars["Close"],
            name="BRK.B",
        )
    )
    fig.add_hrect(
        y0=support.lower,
        y1=support.upper,
        line_width=0,
        fillcolor="rgba(66, 165, 245, 0.18)",
        annotation_text="Support zone",
    )
    fig.add_hrect(
        y0=resistance.lower,
        y1=resistance.upper,
        line_width=0,
        fillcolor="rgba(255, 167, 38, 0.18)",
        annotation_text="Resistance zone",
    )
    fig.update_layout(height=620, xaxis_rangeslider_visible=False, yaxis_title="Price (USD)")
    st.plotly_chart(fig, width="stretch")
    icols = st.columns(5)
    icols[0].metric("RSI (14)", f"{indicators['rsi_14']:.1f}")
    icols[1].metric("ATR / price", f"{indicators['atr_pct_14']:.2%}")
    icols[2].metric("20-day volatility", f"{indicators['volatility_20']:.2%}")
    icols[3].metric("MACD", f"{indicators['macd']:.2f}")
    icols[4].metric("Bollinger position", f"{indicators['bollinger_position_20']:.2f}")

with tab_range:
    st.caption(
        "These are separate marginal distributions of the lowest low and highest high "
        "during the next 10 sessions—not a simulated joint price path."
    )
    ranges = result.range_summary
    rcols = st.columns(2)
    with rcols[0]:
        st.markdown("#### Path minimum")
        st.metric("Median minimum", f"${ranges.minimum.median:,.2f}")
        st.write(ranges.minimum.frame)
    with rcols[1]:
        st.markdown("#### Path maximum")
        st.metric("Median maximum", f"${ranges.maximum.median:,.2f}")
        st.write(ranges.maximum.frame)

    threshold = st.number_input(
        "Price threshold",
        min_value=float(price * 0.5),
        max_value=float(price * 1.5),
        value=float(price),
        step=1.0,
    )
    c1, c2 = st.columns(2)
    c1.metric(
        f"P(10-day minimum ≤ ${threshold:,.2f})",
        f"{result.range_model.minimum_below(threshold):.1%}",
    )
    c2.metric(
        f"P(10-day maximum ≥ ${threshold:,.2f})",
        f"{result.range_model.maximum_above(threshold):.1%}",
    )

    dist_fig = go.Figure()
    dist_fig.add_trace(
        go.Scatter(
            x=result.range_model.minimum_prices,
            y=result.range_model.minimum_cdf,
            name="Minimum CDF",
        )
    )
    dist_fig.add_trace(
        go.Scatter(
            x=result.range_model.maximum_prices,
            y=result.range_model.maximum_cdf,
            name="Maximum CDF",
        )
    )
    dist_fig.update_layout(xaxis_title="Price (USD)", yaxis_title="Cumulative probability")
    st.plotly_chart(dist_fig, width="stretch")
    st.markdown("#### Range-model validation")
    st.dataframe(
        result.range_model.diagnostics.model_comparison,
        width="stretch",
        hide_index=True,
    )

with tab_spread:
    st.caption(
        "Forecast for the next single session's intraday range "
        "(High − Low) / today's close—not the 10-day path."
    )
    spread_summary = result.spread_summary
    indicators = build_feature_frame(result.bars).iloc[-1]
    historical = next_day_spread(result.bars).dropna().tail(252)
    hist_median = float(historical.median()) if not historical.empty else float("nan")

    scols = st.columns(3)
    scols[0].metric("Median spread (forecast)", f"{spread_summary.median_pct:.2%}")
    scols[1].metric("Median spread (USD)", f"${spread_summary.median_usd:,.2f}")
    scols[2].metric("252-day historical median", f"{hist_median:.2%}")

    st.markdown("#### Predicted intervals")
    st.dataframe(spread_summary.frame, width="stretch", hide_index=True)

    spread_threshold = st.number_input(
        "Spread threshold (%)",
        min_value=0.0,
        max_value=20.0,
        value=float(spread_summary.median_pct * 100),
        step=0.1,
        key="spread_threshold_pct",
    ) / 100
    st.metric(
        f"P(next spread ≥ {spread_threshold:.2%})",
        f"{result.spread_model.spread_at_or_above(spread_threshold):.1%}",
    )

    spread_fig = go.Figure()
    if not historical.empty:
        spread_fig.add_trace(
            go.Histogram(
                x=historical.to_numpy(),
                name="Historical (252 sessions)",
                opacity=0.35,
                nbinsx=30,
                histnorm="probability density",
            )
        )
    spread_fig.add_trace(
        go.Scatter(
            x=result.spread_model.spread_values,
            y=result.spread_model.spread_cdf,
            name="Forecast CDF",
            mode="lines",
            line={"width": 3},
        )
    )
    spread_fig.add_vline(
        x=spread_summary.median_pct,
        line_dash="dash",
        line_color="royalblue",
        annotation_text="Forecast median",
    )
    spread_fig.add_vline(
        x=float(indicators["atr_pct_14"]),
        line_dash="dot",
        line_color="orange",
        annotation_text="ATR / price",
    )
    spread_fig.update_layout(
        xaxis_title="Next-day spread (% of close)",
        yaxis_title="Density / cumulative probability",
        barmode="overlay",
        height=520,
    )
    st.plotly_chart(spread_fig, width="stretch")
    st.markdown("#### Spread-model validation")
    st.dataframe(
        result.spread_model.diagnostics.model_comparison,
        width="stretch",
        hide_index=True,
    )

with tab_emotion:
    emotion = result.emotion
    news = result.news
    st.caption(
        "Emotion uses a proxy stack from VIX, SPY, and BRK.B. TRIN proxy is approximate, "
        "not exchange-calculated TRIN. News scoring is local NLP on recent Yahoo headlines."
    )
    ecols = st.columns(5)
    ecols[0].metric("Emotion regime", emotion.regime.value)
    ecols[1].metric("Composite score", f"{emotion.composite_score:+.2f}")
    ecols[2].metric("VIX z-score", f"{emotion.components.get('vix_zscore', 0.0):.2f}")
    ecols[3].metric("TRIN proxy (approx.)", f"{emotion.trin_proxy:.2f}")
    ecols[4].metric("News impact", f"{news.impact_score:+.2%}")

    history = emotion.history
    emotion_fig = go.Figure()
    emotion_fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["composite_score"],
            name="Emotion score",
            yaxis="y1",
        )
    )
    emotion_fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["vix_close"],
            name="VIX close",
            yaxis="y2",
            line={"dash": "dot"},
        )
    )
    emotion_fig.update_layout(
        yaxis={"title": "Emotion score", "range": [-1.05, 1.05]},
        yaxis2={"title": "VIX", "overlaying": "y", "side": "right"},
        height=420,
    )
    st.plotly_chart(emotion_fig, width="stretch")

    st.markdown("#### Emotion components")
    component_rows = [
        {"component": name, "value": value}
        for name, value in emotion.components.items()
        if not name.endswith("_score")
    ]
    st.dataframe(component_rows, width="stretch", hide_index=True)

    st.markdown("#### Latest headlines")
    ncols = st.columns(3)
    ncols[0].metric("Aggregate sentiment", f"{news.aggregate_sentiment:+.2f}")
    ncols[1].metric("Impact on edge", f"{news.impact_score:+.2%}")
    ncols[2].metric("News confidence", f"{news.confidence:.1%}")
    st.write(news.summary)
    if news.articles:
        article_rows = [
            {
                "headline": article.title,
                "publisher": article.publisher,
                "sentiment": f"{article.sentiment:+.2f}",
                "relevance": f"{article.relevance:.0%}",
                "published": article.published_at.strftime("%Y-%m-%d %H:%M"),
                "link": article.link,
            }
            for article in news.articles
        ]
        st.dataframe(article_rows, width="stretch", hide_index=True)
    else:
        st.warning("No recent relevant BRK.B headlines were returned by Yahoo Finance.")

with tab_validation:
    v = result.validation
    vcols = st.columns(4)
    vcols[0].metric("Selected classifier", v.model_name)
    vcols[1].metric("Walk-forward log loss", f"{v.log_loss:.3f}")
    vcols[2].metric("Multiclass Brier", f"{v.brier_score:.3f}")
    vcols[3].metric("Validation rows", f"{v.validation_rows:,}")
    st.dataframe(v.model_comparison, width="stretch", hide_index=True)
    st.caption(
        "Chronological folds use a forecast-horizon gap. Lower log loss and Brier scores "
        "are better; they measure historical validation, not future certainty."
    )

with tab_backtest:
    metrics = result.backtest.metrics
    bcols = st.columns(6)
    bcols[0].metric("Strategy return", f"{metrics['total_return']:.1%}")
    bcols[1].metric("Buy & hold", f"{metrics['buy_hold_return']:.1%}")
    bcols[2].metric("Excess return", f"{metrics['excess_return']:.1%}")
    bcols[3].metric("Max drawdown", f"{metrics['max_drawdown']:.1%}")
    bcols[4].metric("Trades", f"{metrics['trades']:.0f}")
    bcols[5].metric("Market exposure", f"{metrics['market_exposure']:.1%}")
    equity = result.backtest.equity
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=equity.index, y=equity["Strategy"], name="Strategy"))
    eq_fig.add_trace(go.Scatter(x=equity.index, y=equity["BuyHold"], name="BRK.B buy & hold"))
    eq_fig.update_layout(yaxis_title="Growth of $1", xaxis_title="")
    st.plotly_chart(eq_fig, width="stretch")
    st.caption(
        f"Mean daily return in uptrends: {metrics['uptrend_daily_return']:.3%}; "
        f"in downtrends: {metrics['downtrend_daily_return']:.3%}."
    )

with tab_strategies:
    lab = result.strategy_lab
    st.info(lab.explanation)
    st.caption(
        f"Parameters were selected using OOS data through {lab.development_end:%Y-%m-%d}. "
        f"The leaderboard uses only the untouched holdout beginning "
        f"{lab.holdout_start:%Y-%m-%d}; every position is executed next-open."
    )
    display = lab.leaderboard.copy()
    for column in (
        "total_return",
        "buy_hold_return",
        "excess_return",
        "sharpe",
        "max_drawdown",
        "benchmark_max_drawdown",
        "market_exposure",
        "turnover",
    ):
        display[column] = display[column].map(
            (lambda value: f"{value:.2f}") if column in {"sharpe", "turnover"}
            else (lambda value: f"{value:.1%}")
        )
    st.dataframe(display, width="stretch", hide_index=True)

    lab_fig = go.Figure()
    first_result = next(iter(lab.backtests.values()))
    lab_fig.add_trace(
        go.Scatter(
            x=first_result.equity.index,
            y=first_result.equity["BuyHold"],
            name="Buy & hold",
            line={"width": 3},
        )
    )
    for name, strategy_result in lab.backtests.items():
        lab_fig.add_trace(
            go.Scatter(
                x=strategy_result.equity.index,
                y=strategy_result.equity["Strategy"],
                name=name,
            )
        )
    lab_fig.update_layout(yaxis_title="Growth of $1", xaxis_title="")
    st.plotly_chart(lab_fig, width="stretch")

    exposure_fig = go.Figure()
    for name in lab.exposures:
        exposure_fig.add_trace(
            go.Scatter(
                x=lab.exposures.index,
                y=lab.exposures[name],
                name=name,
                line_shape="hv",
            )
        )
    exposure_fig.update_layout(
        yaxis_title="Target exposure",
        yaxis_range=[-0.05, 1.05],
        xaxis_title="",
    )
    st.plotly_chart(exposure_fig, width="stretch")

st.divider()
st.caption(
    "Research only—not investment advice. Historical probabilities and backtests can fail "
    "under changing regimes. No orders are placed."
)
