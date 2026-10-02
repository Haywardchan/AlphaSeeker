from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alpha_seeker.features import build_feature_frame
from alpha_seeker.labels import next_day_spread
from alpha_seeker.pipeline import AnalysisResult, run_analysis

st.set_page_config(page_title="Alpha Seeker", page_icon="📈", layout="wide")

DISPLAY_SYMBOLS = {"BRK.B": "BRK-B", "VTI": "VTI"}
STRATEGY_NAMES = {
    "defensive_probability_overlay": "Defensive probability overlay",
    "trend_filter": "200-day trend filter",
    "trend_probability_hybrid": "Trend and probability hybrid",
    "support_pullback": "Support pullback",
    "buy_and_hold": "Buy and hold",
}


def style_figure(figure: go.Figure, *, height: int | None = None) -> go.Figure:
    """Apply one consistent, responsive dark chart style."""
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 40, "r": 30, "t": 45, "b": 40},
        hovermode="x unified",
    )
    if height is not None:
        figure.update_layout(height=height)
    return figure


def friendly_error(exc: Exception) -> tuple[str, str]:
    """Map common provider and modelling failures to useful recovery steps."""
    text = str(exc)
    lowered = text.lower()
    if "no data returned" in lowered or "yfinance" in lowered:
        return "Yahoo Finance data is unavailable.", "Refresh shortly or check internet access."
    if "at least" in lowered or "insufficient" in lowered:
        return "There is not enough usable history for this analysis.", text
    if "ohlc" in lowered or "high" in lowered or "low" in lowered:
        return (
            "Yahoo returned an inconsistent market-data row.",
            "Refresh to download a clean copy.",
        )
    return "Analysis could not be completed.", text


def normalized_close(frame: pd.DataFrame, index: pd.Index | None = None) -> pd.Series:
    close = frame["Close"].astype(float)
    if index is not None:
        close = close.reindex(index).ffill().dropna()
    return close / close.iloc[0]


def action_badge(action: str) -> None:
    colors = {
        "BUY": ("#163d2b", "#78dba9"),
        "HOLD": ("#3d3416", "#f1cf72"),
        "SELL_REDUCE": ("#431f24", "#ff9aa5"),
    }
    background, foreground = colors[action]
    label = action.replace("_", " / ")
    st.markdown(
        (
            f"<span style='display:inline-block;padding:0.35rem 0.75rem;"
            f"border-radius:0.35rem;background:{background};color:{foreground};"
            f"font-weight:700'>Final guidance: {label}</span>"
        ),
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Training walk-forward models…")
def load_analysis(
    symbol: str,
    lookback_years: int,
    cost_bps: float,
    confidence: float,
    include_emotion: bool,
    include_news: bool,
) -> AnalysisResult:
    return run_analysis(
        symbol=symbol,
        lookback_years=lookback_years,
        transaction_cost_bps=cost_bps,
        confidence_margin=confidence,
        include_emotion=include_emotion,
        include_news=include_news,
    )


with st.sidebar:
    st.header("Research controls")
    display_symbol = st.radio(
        "Instrument",
        ("BRK.B", "VTI"),
        horizontal=True,
        help="BRK.B is a single company; VTI is a diversified total-market ETF.",
    )
    yahoo_symbol = DISPLAY_SYMBOLS[display_symbol]
    lookback = st.slider(
        "Training history (years)",
        8,
        25,
        15,
        help="Longer histories include more regimes but take longer to train.",
    )
    costs = st.number_input(
        "Round-trip cost (bps)",
        0.0,
        100.0,
        10.0,
        5.0,
        help="Estimated total entry and exit friction; 10 bps equals 0.10%.",
    )
    confidence = st.slider(
        "Confidence margin",
        0.0,
        0.25,
        0.05,
        0.01,
        help="Required upper-versus-lower probability lead for base guidance.",
    )
    include_emotion = st.toggle(
        "Include emotion in guidance",
        value=True,
        help="Adjust the technical edge using VIX, SPY, and instrument indicators.",
    )
    include_news = st.toggle(
        "Include news in guidance",
        value=True,
        help="Apply a bounded, recency-weighted Yahoo headline adjustment.",
    )
    load_comparison = st.toggle(
        "Load BRK.B / VTI comparison",
        value=False,
        help="Runs the other instrument with identical settings and caches the result.",
    )
    if st.button("Refresh data and models", use_container_width=True):
        st.cache_resource.clear()
        st.session_state["refresh_requested"] = True
        st.rerun()

if st.session_state.pop("refresh_requested", False):
    st.toast("Market data and models refreshed.")

instrument_name = (
    "Berkshire Hathaway Class B"
    if display_symbol == "BRK.B"
    else "Vanguard Total Stock Market ETF"
)
st.title(f"{display_symbol} Alpha Seeker")
st.caption(f"Technical research for {instrument_name} · 10-session horizon")

try:
    result = load_analysis(
        yahoo_symbol,
        lookback,
        costs,
        confidence,
        include_emotion,
        include_news,
    )
except Exception as exc:  # Streamlit should explain provider/model failures instead of crashing.
    title, recovery = friendly_error(exc)
    st.error(title)
    st.info(recovery)
    st.stop()

other_result: AnalysisResult | None = None
comparison_error: str | None = None
if load_comparison:
    other_symbol = "VTI" if yahoo_symbol == "BRK-B" else "BRK-B"
    try:
        other_result = load_analysis(
            other_symbol,
            lookback,
            costs,
            confidence,
            include_emotion,
            include_news,
        )
    except Exception as exc:
        comparison_error = str(exc)

price = result.current_price
support = result.support
resistance = result.resistance
prob = result.probabilities
signal = result.signal

st.caption(
    f"Yahoo symbol {yahoo_symbol} · Last completed bar: {result.as_of:%Y-%m-%d} · "
    f"{result.sample_count:,} labelled observations · "
    f"analysis cached {(pd.Timestamp.now() - result.loaded_at).total_seconds() / 60:.0f} min ago"
)

summary_cols = st.columns(3, gap="small")
summary_cols[0].metric("Last close", f"${price:,.2f}")
summary_cols[1].metric(
    "10-session range midpoint",
    f"${(result.range_summary.minimum.median + result.range_summary.maximum.median) / 2:,.2f}",
)
summary_cols[2].metric("Final guidance", signal.final_action.value, f"{signal.adjusted_edge:+.2%}")
action_badge(signal.final_action.value)

if result.data_is_stale:
    st.warning(
        "The latest bar is more than two expected trading sessions old. "
        "Treat guidance as stale until data refreshes."
    )

st.subheader("How the final guidance was formed")
flow = st.columns(4, gap="small")
flow[0].metric(
    "1 · Base technical",
    result.base_signal.action.value,
    f"{result.base_signal.edge:+.2%} edge",
)
flow[1].metric("2 · Emotion", f"{signal.emotion_adjustment:+.2%}")
flow[2].metric("3 · News", f"{signal.news_adjustment:+.2%}")
flow[3].metric("4 · Final", signal.final_action.value, f"{signal.adjusted_edge:+.2%} edge")
if result.base_signal.action != signal.final_action:
    st.info(
        f"Fusion changed {result.base_signal.action.value} to {signal.final_action.value}. "
        "The final result uses a supporting threshold equal to 75% of the base confidence "
        "margin when no safety gate is active."
    )

with st.expander("Full decision explanation", expanded=True):
    st.write(signal.rationale)
    st.write(
        f"Current level method: **{support.source}** · support strength "
        f"{support.strength:.0%} · resistance strength {resistance.strength:.0%}."
    )
    if signal.gates_applied:
        st.write("Gates applied: " + "; ".join(signal.gates_applied))

with st.expander("Advice conditions and historical balance", expanded=True):
    base = result.base_signal
    condition_cols = st.columns(5, gap="small")
    condition_cols[0].metric("Weighted upside", f"{base.weighted_upside:.2%}")
    condition_cols[1].metric("Weighted downside", f"{base.weighted_downside:.2%}")
    condition_cols[2].metric("Upper − lower probability", f"{base.confidence_spread:+.1%}")
    condition_cols[3].metric("Required base margin", f"{confidence:.1%}")
    condition_cols[4].metric("Estimated cost", f"{base.estimated_cost:.2%}")
    st.markdown(
        "- **Base BUY:** probability spread is at least the required margin and "
        "weighted upside minus downside remains positive after cost.\n"
        "- **Base SELL / REDUCE:** the symmetric downside condition clears the same margin "
        "and cost tests.\n"
        "- **Base HOLD:** neither direction clears both tests.\n"
        "- **Final fusion:** a base BUY stays BUY unless the strong-negative-news gate "
        "blocks it. A base HOLD can become BUY or SELL when it clears 75% of the margin "
        "with positive adjusted edge."
    )
    action_counts = (
        result.historical_advice["action"]
        .value_counts()
        .reindex(["BUY", "HOLD", "SELL_REDUCE"], fill_value=0)
    )
    action_balance = pd.DataFrame(
        {
            "action": ["BUY", "HOLD", "SELL / REDUCE"],
            "observations": action_counts.to_numpy(),
            "share": (action_counts / max(1, action_counts.sum())).to_numpy(),
        }
    )
    st.caption("Fold-held-out technical advice distribution")
    st.dataframe(
        action_balance,
        hide_index=True,
        width="stretch",
        column_config={"share": st.column_config.NumberColumn(format="percent")},
    )

st.subheader("10-session first-touch probabilities")
probability_fig = go.Figure(
    go.Bar(
        x=[prob["upper_first"], prob["lower_first"], prob["neither"]],
        y=["Upper resistance", "Lower support", "Neither"],
        orientation="h",
        text=[
            f"{prob['upper_first']:.1%}",
            f"{prob['lower_first']:.1%}",
            f"{prob['neither']:.1%}",
        ],
        textposition="auto",
    )
)
probability_fig.update_layout(
    xaxis={"title": "Probability", "tickformat": ".0%", "range": [0, 1]},
    yaxis={"title": ""},
    showlegend=False,
)
st.plotly_chart(style_figure(probability_fig, height=270), width="stretch")

tab_names = [
    "Overview · Price",
    "Forecasts · 10-day range",
    "Forecasts · Next session",
    "Context · Emotion & news",
    "Validation · Models",
    "Validation · Technical backtest",
    "Validation · Strategy Lab",
    "Compare · BRK.B vs VTI",
]
(
    tab_chart,
    tab_range,
    tab_spread,
    tab_emotion,
    tab_validation,
    tab_backtest,
    tab_strategies,
    tab_compare,
) = st.tabs(tab_names)

with tab_chart:
    chart_controls = st.columns(2)
    chart_sessions = chart_controls[0].select_slider(
        "Price history shown",
        options=[90, 180, 300, 504],
        value=300,
        format_func=lambda value: f"{value} sessions",
    )
    marker_mode = chart_controls[1].radio(
        "Historical advice markers",
        ("Action changes", "Every signal"),
        horizontal=True,
        help="Markers are fold-held-out technical advice, not reconstructed news-fused advice.",
    )
    bars = result.bars.tail(chart_sessions)
    advice = result.historical_advice
    if marker_mode == "Action changes":
        advice = advice.loc[advice["action"].ne(advice["action"].shift())]
    advice = advice.loc[advice.index.intersection(bars.index)]
    indicators = build_feature_frame(result.bars).iloc[-1]
    fig = go.Figure(
        go.Candlestick(
            x=bars.index,
            open=bars["Open"],
            high=bars["High"],
            low=bars["Low"],
            close=bars["Close"],
            name=display_symbol,
        )
    )
    marker_styles = {
        "BUY": {
            "symbol": "triangle-up",
            "color": "#78dba9",
            "position": bars["Low"] * 0.99,
            "label": "BUY",
        },
        "HOLD": {
            "symbol": "circle",
            "color": "#f1cf72",
            "position": bars["Close"],
            "label": "HOLD",
        },
        "SELL_REDUCE": {
            "symbol": "triangle-down",
            "color": "#ff9aa5",
            "position": bars["High"] * 1.01,
            "label": "SELL / REDUCE",
        },
    }
    for action, marker in marker_styles.items():
        action_rows = advice.loc[advice["action"] == action]
        if action_rows.empty:
            continue
        marker_y = marker["position"].reindex(action_rows.index)
        customdata = action_rows[["actual_outcome", "future_return"]].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=action_rows.index,
                y=marker_y,
                mode="markers",
                name=f"OOS {marker['label']}",
                marker={
                    "symbol": marker["symbol"],
                    "color": marker["color"],
                    "size": 11,
                    "line": {"width": 1},
                },
                customdata=customdata,
                hovertemplate=(
                    f"<b>{marker['label']}</b><br>"
                    "Signal date: %{x|%Y-%m-%d}<br>"
                    "Realized first touch: %{customdata[0]}<br>"
                    "10-session return: %{customdata[1]:+.2%}<extra></extra>"
                ),
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
    st.plotly_chart(style_figure(fig, height=620), width="stretch")
    st.caption(
        "Markers are fold-held-out technical advice generated using information available "
        "at that close and executable at the next open. They exclude historical news fusion."
    )
    audit = (
        result.historical_advice.groupby("action", observed=True)
        .agg(
            observations=("action", "size"),
            first_touch_hit_rate=("correct", "mean"),
            average_10_session_return=("future_return", "mean"),
        )
        .reindex(["BUY", "HOLD", "SELL_REDUCE"])
        .dropna(subset=["observations"])
        .reset_index()
    )
    audit["action"] = audit["action"].replace({"SELL_REDUCE": "SELL / REDUCE"})
    st.markdown("#### Historical advice audit")
    st.dataframe(
        audit,
        hide_index=True,
        width="stretch",
        column_config={
            "first_touch_hit_rate": st.column_config.NumberColumn(format="percent"),
            "average_10_session_return": st.column_config.NumberColumn(format="percent"),
        },
    )
    level_cols = st.columns(4, gap="small")
    level_cols[0].metric("Support", f"${support.price:,.2f}", f"{support.distance_pct:.1%}")
    level_cols[1].metric(
        "Resistance", f"${resistance.price:,.2f}", f"+{resistance.distance_pct:.1%}"
    )
    level_cols[2].metric("Level method", support.source.replace("_", " ").title())
    level_cols[3].metric(
        "Level strength",
        f"{(support.strength + resistance.strength) / 2:.0%}",
    )
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
    st.plotly_chart(style_figure(dist_fig, height=460), width="stretch")
    st.markdown("#### Range-model validation")
    st.dataframe(
        result.range_model.diagnostics.model_comparison,
        width="stretch",
        hide_index=True,
    )
    range_diagnostics = result.range_model.diagnostics
    range_validation = st.columns(3)
    range_validation[0].metric(
        "Minimum empirical 80% coverage",
        f"{range_diagnostics.minimum_80_coverage:.1%}",
    )
    range_validation[1].metric(
        "Maximum empirical 80% coverage",
        f"{range_diagnostics.maximum_80_coverage:.1%}",
    )
    range_validation[2].metric("Validation folds", range_diagnostics.folds)

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
    st.plotly_chart(style_figure(spread_fig, height=520), width="stretch")
    st.markdown("#### Spread-model validation")
    st.dataframe(
        result.spread_model.diagnostics.model_comparison,
        width="stretch",
        hide_index=True,
    )
    spread_diagnostics = result.spread_model.diagnostics
    st.caption(
        f"Empirical 80% walk-forward coverage: {spread_diagnostics.coverage_80:.1%} · "
        f"validation {spread_diagnostics.validation_start:%Y-%m-%d} to "
        f"{spread_diagnostics.validation_end:%Y-%m-%d}."
    )

with tab_emotion:
    emotion = result.emotion
    news = result.news
    st.caption(
        f"Emotion uses a proxy stack from VIX, SPY, and {display_symbol}. "
        "TRIN proxy is approximate, "
        "not exchange-calculated TRIN. News scoring is local NLP on recent Yahoo headlines."
    )
    if yahoo_symbol == "VTI":
        st.info(
            "VTI uses an ETF profile: less weight on overlapping SPY components and "
            "company-news impact, with broader macro-market headline relevance."
        )
    else:
        st.info(
            "BRK.B uses a single-stock profile with a longer level history and stronger "
            "company-specific news sensitivity."
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
    st.plotly_chart(style_figure(emotion_fig, height=420), width="stretch")

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
        st.dataframe(
            article_rows,
            width="stretch",
            hide_index=True,
            column_config={
                "link": st.column_config.LinkColumn("Article", display_text="Open")
            },
        )
    else:
        st.warning(
            f"No recent relevant {display_symbol} headlines were returned by Yahoo Finance."
        )

with tab_validation:
    v = result.validation
    vcols = st.columns(4, gap="small")
    vcols[0].metric("Selected classifier", v.model_name)
    vcols[1].metric("Walk-forward log loss", f"{v.log_loss:.3f}")
    vcols[2].metric("Multiclass Brier", f"{v.brier_score:.3f}")
    vcols[3].metric("Walk-forward folds", v.folds)
    detail_cols = st.columns(4, gap="small")
    detail_cols[0].metric("Validation rows", f"{v.validation_rows:,}")
    detail_cols[1].metric("Calibration rows", f"{v.calibration_rows:,}")
    detail_cols[2].metric("Calibration temperature", f"{v.calibration_temperature:.2f}")
    detail_cols[3].metric(
        "Ambiguous labels excluded",
        f"{result.ambiguous_label_rate:.1%}",
        f"{result.ambiguous_label_count} windows",
    )
    st.dataframe(v.model_comparison, width="stretch", hide_index=True)
    st.caption(
        f"Chronological validation ran from {v.validation_start:%Y-%m-%d} to "
        f"{v.validation_end:%Y-%m-%d} with a forecast-horizon gap. Lower log loss and "
        "Brier scores are better; calibration and validation do not guarantee future accuracy."
    )

with tab_backtest:
    st.warning(
        "Technical-only evidence: this historical backtest excludes the live emotion and "
        "news overlays used by Final guidance."
    )
    metrics = result.backtest.metrics
    bcols = st.columns(3, gap="small")
    bcols[0].metric("Technical strategy return", f"{metrics['total_return']:.1%}")
    bcols[1].metric(f"{display_symbol} buy & hold", f"{metrics['buy_hold_return']:.1%}")
    bcols[2].metric("Excess return", f"{metrics['excess_return']:.1%}")
    risk_cols = st.columns(3, gap="small")
    risk_cols[0].metric("Max drawdown", f"{metrics['max_drawdown']:.1%}")
    risk_cols[1].metric("Trades", f"{metrics['trades']:.0f}")
    risk_cols[2].metric("Market exposure", f"{metrics['market_exposure']:.1%}")
    equity = result.backtest.equity
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=equity.index, y=equity["Strategy"], name="Strategy"))
    eq_fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity["BuyHold"],
            name=f"{display_symbol} buy & hold",
        )
    )
    spy_growth = normalized_close(result.benchmark_bars, equity.index)
    eq_fig.add_trace(
        go.Scatter(
            x=spy_growth.index,
            y=spy_growth,
            name="SPY passive benchmark",
            line={"dash": "dot"},
        )
    )
    eq_fig.update_layout(yaxis_title="Growth of $1", xaxis_title="")
    st.plotly_chart(style_figure(eq_fig, height=500), width="stretch")
    st.caption(
        f"Mean daily return in uptrends: {metrics['uptrend_daily_return']:.3%}; "
        f"in downtrends: {metrics['downtrend_daily_return']:.3%}. "
        f"SPY passive return over the displayed test window: {spy_growth.iloc[-1] - 1:.1%}."
    )

with tab_strategies:
    st.warning(
        "Technical-only evidence: Strategy Lab excludes live emotion and headline adjustments."
    )
    lab = result.strategy_lab
    st.info(lab.explanation)
    st.caption(
        f"Parameters were selected using OOS data through {lab.development_end:%Y-%m-%d}. "
        f"The leaderboard uses only the untouched holdout beginning "
        f"{lab.holdout_start:%Y-%m-%d}; every position is executed next-open. "
        f"{len(lab.leaderboard)} strategy families were compared."
    )
    display = lab.leaderboard.copy()
    display["strategy"] = display["strategy"].map(STRATEGY_NAMES).fillna(display["strategy"])
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
        is_winner = name == lab.winner
        lab_fig.add_trace(
            go.Scatter(
                x=strategy_result.equity.index,
                y=strategy_result.equity["Strategy"],
                name=STRATEGY_NAMES.get(name, name),
                line={"width": 4 if is_winner else 1},
                opacity=1.0 if is_winner else 0.45,
            )
        )
    lab_fig.update_layout(yaxis_title="Growth of $1", xaxis_title="")
    st.plotly_chart(style_figure(lab_fig, height=500), width="stretch")

    exposure_fig = go.Figure()
    for name in lab.exposures:
        exposure_fig.add_trace(
            go.Scatter(
                x=lab.exposures.index,
                y=lab.exposures[name],
                name=STRATEGY_NAMES.get(name, name),
                line_shape="hv",
            )
        )
    exposure_fig.update_layout(
        yaxis_title="Target exposure",
        yaxis_range=[-0.05, 1.05],
        xaxis_title="",
    )
    st.plotly_chart(style_figure(exposure_fig, height=420), width="stretch")

with tab_compare:
    st.caption(
        "Same settings and as-of methodology for both instruments. BRK.B is a concentrated "
        "single stock; VTI is a diversified total-market ETF."
    )
    if comparison_error:
        st.error(f"Comparison unavailable: {comparison_error}")
    elif other_result is None:
        st.info("Enable “Load BRK.B / VTI comparison” in the sidebar.")
    else:
        analyses = {
            display_symbol: result,
            "VTI" if display_symbol == "BRK.B" else "BRK.B": other_result,
        }
        comparison_rows = []
        for name, analysis in analyses.items():
            comparison_rows.append(
                {
                    "instrument": name,
                    "base guidance": analysis.base_signal.action.value,
                    "final guidance": analysis.signal.final_action.value,
                    "adjusted edge": analysis.signal.adjusted_edge,
                    "upper first": analysis.probabilities["upper_first"],
                    "lower first": analysis.probabilities["lower_first"],
                    "emotion": analysis.emotion.regime.value,
                    "technical return": analysis.backtest.metrics["total_return"],
                    "buy & hold": analysis.backtest.metrics["buy_hold_return"],
                    "max drawdown": analysis.backtest.metrics["max_drawdown"],
                    "lab winner": STRATEGY_NAMES.get(
                        analysis.strategy_lab.winner,
                        analysis.strategy_lab.winner,
                    ),
                }
            )
        st.dataframe(
            comparison_rows,
            width="stretch",
            hide_index=True,
            column_config={
                "adjusted edge": st.column_config.NumberColumn(format="percent"),
                "upper first": st.column_config.NumberColumn(format="percent"),
                "lower first": st.column_config.NumberColumn(format="percent"),
                "technical return": st.column_config.NumberColumn(format="percent"),
                "buy & hold": st.column_config.NumberColumn(format="percent"),
                "max drawdown": st.column_config.NumberColumn(format="percent"),
            },
        )
        common = result.bars.index.intersection(other_result.bars.index)
        selected_growth = normalized_close(result.bars, common)
        other_growth = normalized_close(other_result.bars, common)
        spy_growth = normalized_close(result.benchmark_bars, common)
        comparison_fig = go.Figure()
        comparison_fig.add_trace(
            go.Scatter(x=common, y=selected_growth, name=display_symbol)
        )
        other_display = "VTI" if display_symbol == "BRK.B" else "BRK.B"
        comparison_fig.add_trace(
            go.Scatter(x=common, y=other_growth, name=other_display)
        )
        comparison_fig.add_trace(
            go.Scatter(
                x=common,
                y=spy_growth,
                name="SPY",
                line={"dash": "dot"},
            )
        )
        comparison_fig.update_layout(
            title="Relative growth over shared history",
            xaxis_title="Date",
            yaxis_title="Growth of $1",
        )
        st.plotly_chart(style_figure(comparison_fig, height=520), width="stretch")
        correlation = result.bars["Close"].pct_change().corr(
            other_result.bars["Close"].pct_change()
        )
        comparison_metrics = st.columns(3)
        comparison_metrics[0].metric("Daily-return correlation", f"{correlation:.2f}")
        comparison_metrics[1].metric(
            f"{display_symbol} shared-history return",
            f"{selected_growth.iloc[-1] - 1:.1%}",
        )
        comparison_metrics[2].metric(
            f"{other_display} shared-history return",
            f"{other_growth.iloc[-1] - 1:.1%}",
        )

st.divider()
st.caption(
    "Research only—not investment advice. Historical probabilities and backtests can fail "
    "under changing regimes. No orders are placed."
)
