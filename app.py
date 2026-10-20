import streamlit as st
import plotly.graph_objects as go
from engine import (
    ArticleFetchError,
    fetch_country_articles,
    calculate_tension_score,
    generate_forecast_trend,
)
from forecaster import predict_future_events

st.set_page_config(page_title="World Tensions Predictor", layout="wide", page_icon="🌐")

st.title("🌐 AI Geopolitical Tension & Conflict Early-Warning Predictor")

research_gaps = [
    {
        "#": 1,
        "Gap": "Warning-response gap: no evidence that forecasts lead to prevention",
        "Evidence in the sources": (
            "De Agostini and Giovanardi: Rhombus and Early Warning Project "
            "warnings were not matched by action; Ullah et al. and "
            "Zakotianskyi stop at prediction."
        ),
        "Research opportunity": (
            "Study how alerts are translated into actions; test response "
            "protocols; track outcomes after warnings."
        ),
    },
    {
        "#": 2,
        "Gap": "Novel onset prediction: models are better at persistence than surprise",
        "Evidence in the sources": (
            "Knack and Balakrishnan via De Agostini and Giovanardi; Arab "
            "Spring; Zakotianskyi's imbalanced data."
        ),
        "Research opportunity": (
            "Evaluate onset forecasting in historically peaceful settings; "
            "report false-alarm and miss costs, not accuracy alone."
        ),
    },
    {
        "#": 3,
        "Gap": "Coverage and language: data-poor regions and non-English sources are under-served",
        "Evidence in the sources": (
            "Zakotianskyi drops poorly covered countries; Good extends "
            "ConflibERT only to Arabic and Spanish; Ullah et al. list data "
            "availability in Africa; Ghosh notes Western-source bias."
        ),
        "Research opportunity": (
            "Build and test multilingual extraction for low-resource "
            "languages; measure forecast performance by data-coverage tier."
        ),
    },
    {
        "#": 4,
        "Gap": "Explainability is demanded but not tested",
        "Evidence in the sources": (
            "Ullah et al. and Ghosh call for transparency; no source evaluates "
            "explanations with users."
        ),
        "Research opportunity": (
            "Run user studies with analysts on explanation formats, trust, "
            "and decision quality."
        ),
    },
    {
        "#": 5,
        "Gap": "No common evaluation standard",
        "Evidence in the sources": (
            "Accuracy claims (80%, 70%) vary in task, horizon, and source; "
            "poster benchmarks differ from those in the brief."
        ),
        "Research opportunity": (
            "Create shared benchmarks, time-split protocols, calibration, and "
            "cost-sensitive metrics; improve reproducibility of reported results."
        ),
    },
    {
        "#": 6,
        "Gap": "Weak empirical base in conceptual papers",
        "Evidence in the sources": (
            "Ullah et al. report no collected data; Ghosh relies on scenarios "
            "and counterfactuals."
        ),
        "Research opportunity": (
            "Execute the proposed case studies, interviews, and cross-regional "
            "comparisons with documented methods."
        ),
    },
    {
        "#": 7,
        "Gap": "Conflict forecasting and sectoral risk modelling are siloed",
        "Evidence in the sources": (
            "Ghosh (energy) and Brandt/Zakotianskyi (conflict) share data "
            "types and aims but do not connect."
        ),
        "Research opportunity": (
            "Link conflict forecasts to downstream outcomes such as energy "
            "supply, trade, or humanitarian need."
        ),
    },
    {
        "#": 8,
        "Gap": "Governance is proposed but not operationalised or tested",
        "Evidence in the sources": (
            "Calls for standards, auditing, and human oversight; Ghosh notes "
            "adversarial manipulation; the G7 brief is Global North-centred."
        ),
        "Research opportunity": (
            "Evaluate governance mechanisms in practice, including safeguards "
            "against misuse and participation of affected and Global South "
            "communities."
        ),
    },
]

with st.expander("Research gaps and opportunities", expanded=False):
    st.dataframe(
        research_gaps,
        use_container_width=True,
        hide_index=True,
    )


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

    # Time-series Plot
    forecast_horizon_days = 15
    forecast_period_days = 3
    h_dates, h_vals, f_dates, f_vals = generate_forecast_trend(
        metrics['score'],
        days_forecast=forecast_horizon_days,
        period_days=forecast_period_days,
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=h_dates,
        y=h_vals,
        mode="lines+markers",
        name="Illustrative Trend (Synthetic, 30D)",
        line=dict(color="#00b4d8", width=2),
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