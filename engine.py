import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import urllib.parse
import xml.etree.ElementTree as ET
import re
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gdeltdoc import GdeltDoc, Filters
from sklearn.ensemble import RandomForestRegressor

# Ensure VADER sentiment lexicon is loaded
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()


class ArticleFetchError(RuntimeError):
    """Raised when neither article source can complete its request."""


def _country_names(target: str) -> list[str]:
    """Parse one country or an ``AND``-separated bilateral target."""
    return [
        country.strip().strip('"')
        for country in re.split(r"\s+AND\s+", target, flags=re.IGNORECASE)
        if country.strip().strip('"')
    ]


def _filter_articles_for_target(
    articles: pd.DataFrame, countries: list[str]
) -> pd.DataFrame:
    """For bilateral searches, retain only headlines naming both countries."""
    if articles.empty or len(countries) < 2:
        return articles

    matching = articles["title"].fillna("").astype(str).map(
        lambda title: all(country.casefold() in title.casefold() for country in countries)
    )
    return articles.loc[matching].reset_index(drop=True)


def fetch_from_rss_fallback(target: str, limit: int = 25) -> pd.DataFrame:
    """Fallback live scraper using Google News RSS if GDELT returns 0 results."""
    try:
        countries = _country_names(target)
        country_query = " ".join(f'"{country}"' for country in countries)
        encoded_query = urllib.parse.quote(
            f"{country_query} (conflict OR military OR sanctions OR tensions)"
        )
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

        resp = requests.get(rss_url, timeout=6)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        records = []
        for item in root.findall('./channel/item')[:limit]:
            title = item.find('title').text if item.find('title') is not None else ""
            link = item.find('link').text if item.find('link') is not None else ""
            source = item.find('source').text if item.find('source') is not None else "NewsWire"
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else str(datetime.now().date())

            records.append({
                'title': title,
                'url': link,
                'seendate': pub_date,
                'domain': source
            })
        return _filter_articles_for_target(pd.DataFrame(records), countries)
    except Exception as exc:
        raise ArticleFetchError(f"Google News RSS fallback failed: {exc}") from exc

def fetch_country_articles(query_target: str, num_records: int = 35) -> pd.DataFrame:
    """Fetch target-related articles, using RSS if GDELT fails or has no matches."""
    gd = GdeltDoc()

    countries = _country_names(query_target)
    country_query = " ".join(f'"{country}"' for country in countries)
    gdelt_keyword = f"{country_query} military tension"

    try:
        filters = Filters(
            keyword=gdelt_keyword,
            timespan="7d",
            num_records=min(num_records, 75)
        )
        df = gd.article_search(filters)

        if df is not None and not df.empty and 'title' in df.columns:
            articles = df[['title', 'url', 'seendate', 'domain']]
            articles = _filter_articles_for_target(articles, countries)
            if not articles.empty:
                return articles.head(num_records)
    except Exception as err:
        print(f"GDELT direct fetch notice: {err}")

    return fetch_from_rss_fallback(" AND ".join(countries), limit=num_records)


def calculate_tension_score(articles_df: pd.DataFrame) -> dict:
    """Calculate a 0-100 tension score from headline sentiment and article volume."""
    if articles_df.empty:
        return {
            "score": 0.0,
            "level": "Insufficient Data",
            "avg_negativity": 0.0,
            "volume": 0,
        }

    neg_scores = [
        sia.polarity_scores(str(title))["neg"]
        for title in articles_df["title"]
    ]
    avg_neg = float(np.mean(neg_scores))
    volume = len(articles_df)
    raw_score = (avg_neg * 180.0) + min(volume * 0.8, 40.0)
    final_score = round(float(np.clip(raw_score, 5.0, 98.0)), 1)

    if final_score < 30:
        level = "Low / Baseline"
    elif final_score < 60:
        level = "Moderate Friction"
    elif final_score < 80:
        level = "High Alert"
    else:
        level = "Critical Escalation"

    return {
        "score": final_score,
        "level": level,
        "avg_negativity": round(avg_neg, 2),
        "volume": volume,
    }


def generate_forecast_trend(
    current_score: float,
    days_history: int = 30,
    days_forecast: int = 14,
):
    """Generate a synthetic historical trend and an ML forecast."""
    dates = [
        datetime.now().date() - timedelta(days=i)
        for i in reversed(range(days_history))
    ]

    np.random.seed(42)
    history = [
        current_score + np.random.normal(0, 3) - (i * 0.25)
        for i in range(days_history)
    ]
    history[-1] = current_score
    history = [float(np.clip(value, 10.0, 95.0)) for value in history]

    X = np.arange(len(history)).reshape(-1, 1)
    model = RandomForestRegressor(n_estimators=40, random_state=42)
    model.fit(X, np.array(history))

    future_X = np.arange(
        len(history), len(history) + days_forecast
    ).reshape(-1, 1)
    future_preds = model.predict(future_X)
    future_dates = [
        datetime.now().date() + timedelta(days=i)
        for i in range(1, days_forecast + 1)
    ]

    return (
        dates,
        history,
        future_dates,
        [round(float(value), 1) for value in future_preds],
    )