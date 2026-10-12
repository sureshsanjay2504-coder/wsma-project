import streamlit as st
import plotly.graph_objects as go
from engine import ArticleFetchError, fetch_country_articles, calculate_tension_score, generate_forecast_trend
from forecaster import predict_future_events

st.set_page_config(page_title="World Tensions Predictor", layout="wide", page_icon="🌐")

st.title("🌐 AI Geopolitical Tension & Conflict Early-Warning Predictor")


@st.cache_data(ttl=300)
def fetch_cached_country_articles(target: str, num_records: int):
    return fetch_country_articles(target, num_records=num_records)


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
            st.warning(
                "No recent headlines were found that mention both countries. "
                "There is not enough evidence to calculate a bilateral tension score."
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

    # Time-series Plot
    h_dates, h_vals, f_dates, f_vals = generate_forecast_trend(metrics['score'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h_dates, y=h_vals, mode='lines+markers', name='Illustrative Trend (Synthetic, 30D)', line=dict(color='#00b4d8', width=2)))
    fig.add_trace(go.Scatter(x=f_dates, y=f_vals, mode='lines+markers', name='Model Projection (Next 14D)', line=dict(color='#e63946', width=2, dash='dash')))
    fig.update_layout(title=f"Tension Trend & Predictive Trajectory: {target}", xaxis_title="Date", yaxis_title="Tension Score (0-100)", yaxis_range=[0, 100], template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

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