import streamlit as st
import plotly.graph_objects as go
from engine import (
    ArticleFetchError,
    HistoricalDataError,
    fetch_country_articles,
    fetch_historical_tension,
    calculate_tension_score,
    generate_forecast_trend,
)
from forecaster import predict_future_events

st.set_page_config(page_title="World Tensions Predictor", layout="wide", page_icon="🌐")

st.title("🌐 AI Geopolitical Tension & Conflict Early-Warning Predictor")


@st.cache_data(ttl=300)
def fetch_cached_country_articles(target: str, num_records: int):
    return fetch_country_articles(target, num_records=num_records)


@st.cache_data(ttl=86400)
def fetch_cached_historical_tension(target: str, days: int):
    return fetch_historical_tension(target, days=days)


# Sidebar setup
st.sidebar.header("Configuration")
mode = st.sidebar.radio("Analysis Mode", ["Single Country", "Bilateral Pair"])

if mode == "Single Country":
    target = st.sidebar.text_input("Target Country", "Taiwan")
else:
    c1 = st.sidebar.text_input("Country A", "India")
    c2 = st.sidebar.text_input("Country B", "Pakistan")
    target = f"{c1} AND {c2}"

sample_size = st.sidebar.slider("Articles to Ingest", 10, 50, 25)
analyze_btn = st.sidebar.button("Run Live Prediction", type="primary")

if analyze_btn:
    with st.spinner(f"Autonomously ingesting news & computing indicators for {target}..."):
        try:
            articles_df = fetch_cached_country_articles(target, sample_size)
        except ArticleFetchError as exc:
            st.error(str(exc))
            st.info("Check your internet connection and try again. The app uses Google News RSS if GDELT is unavailable.")
            st.stop()
        if articles_df.empty:
            if mode == "Bilateral Pair":
                st.warning(
                    "No recent headlines were found that mention both countries. "
                    "There is not enough evidence to calculate a bilateral tension score."
                )
            else:
                st.warning(
                    f"No recent headlines were found for {target}. "
                    "There is not enough evidence to calculate a tension score."
                )
            st.stop()
        metrics = calculate_tension_score(articles_df)

    # Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tension Score", f"{metrics['score']} / 100")
    c2.metric("Threat Tier", metrics['level'])
    c3.metric("Articles Analyzed", metrics['volume'])
    c4.metric("Negative Tone Ratio", f"{metrics['avg_negativity'] * 100:.1f}%")

    st.markdown("---")

    historical_days = 30
    try:
        historical_df = fetch_cached_historical_tension(target, historical_days)
    except HistoricalDataError as exc:
        historical_df = None
        st.warning(
            f"Past {historical_days}-day tone data is currently unavailable. "
            f"The chart will show the forecast only. Details: {exc}"
        )

    # Time-series Plot
    forecast_horizon_days = 15
    forecast_period_days = 3
    _, _, f_dates, f_vals = generate_forecast_trend(
        metrics['score'],
        days_forecast=forecast_horizon_days,
        period_days=forecast_period_days,
    )
    fig = go.Figure()
    if historical_df is not None:
        fig.add_trace(go.Scatter(
            x=historical_df["date"],
            y=historical_df["score"],
            mode="lines",
            name=f"Historical GDELT Tone Indicator (Daily, Past {historical_days} Days)",
            line=dict(color="#00b4d8", width=2),
            hovertemplate="%{x|%b %d, %Y}: %{y:.1f}<extra></extra>",
        ))
    forecast_periods = [
        f"Days {start + 1}-{min(start + forecast_period_days, forecast_horizon_days)}"
        for start in range(0, forecast_horizon_days, forecast_period_days)
    ]
    fig.add_trace(go.Scatter(
        x=f_dates,
        y=f_vals,
        text=forecast_periods,
        mode='lines+markers',
        name='Trend Projection (15D, 3-Day Averages)',
        line=dict(color='#e63946', width=2, dash='dash'),
        hovertemplate='%{text}: %{y:.1f}<extra></extra>',
    ))
    fig.update_layout(title=f"Tension Trend & Predictive Trajectory: {target}", xaxis_title="Date", yaxis_title="Tension Score (0-100)", yaxis_range=[0, 100], template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader(
        f"{forecast_horizon_days}-Day Forecast by "
        f"{forecast_period_days}-Day Period"
    )
    st.table([
        {"Period": period, "Average Predicted Tension": score}
        for period, score in zip(forecast_periods, f_vals)
    ])

    # Narrative Brief & Ingested Wires
    left_col, right_col = st.columns([3, 2])
    with left_col:
        st.subheader("📋 Scenario Forecast: What Will Happen Next")
        titles = articles_df['title'].tolist() if not articles_df.empty else []
        brief = predict_future_events(target, metrics['score'], titles)
        st.markdown(brief)

    with right_col:
        st.subheader("📰 Ingested Wire Articles")
        if not articles_df.empty:
            for _, row in articles_df.iterrows():
                st.markdown(f"- **[{row['title']}]({row['url']})**  \n  *Domain: {row['domain']}*")
        else:
            st.info("No matching articles returned by the current query window.")