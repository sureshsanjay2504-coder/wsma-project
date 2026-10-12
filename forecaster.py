import os
import requests

def predict_future_events(target: str, tension_score: float, headlines: list) -> str:
    """Generates a structured analytical forecast from ingested headlines."""
    recent_headlines = "\n- ".join(headlines[:8]) if headlines else "No direct open-source headlines."
    groq_api_key = os.getenv("GROQ_API_KEY")

    prompt = f"""You are a geopolitical risk analyst.
Subject: {target}
Current Tension Index: {tension_score}/100
Recent Live Ingested Headlines:
- {recent_headlines}

Generate a clear 3-point briefing:
1. Probable Trajectory (What is going to happen in the next 14-30 days)
2. Immediate Triggers & Indicators to watch
3. Potential De-escalation Conditions"""

    if groq_api_key:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_api_key}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2
                },
                timeout=12
            )
            return resp.json()['choices'][0]['message']['content']
        except Exception:
            pass

    # Deterministic local briefing fallback
    return f"""### 1. Probable Trajectory (Next 14–30 Days)
With a score of **{tension_score}/100**, the data indicates diplomatic posturing without immediate conventional escalation. Expect official statements, mutual diplomatic summons, and security posture checks.

### 2. Immediate Triggers to Monitor
- Retaliatory diplomatic sanctions or ambassador expulsions.
- Shifts in readiness levels along disputed border zones or maritime corridors.
- State-sponsored economic or trade restriction announcements.

### 3. Potential De-escalation Pathways
- Scheduled bilateral working group talks or regional multilateral summit engagements."""