# ============================================================
# AI TRAVEL ANALYST — STREAMLIT VERSION
# Converted from ai_travel_analyst_v2.py
#
# Required companion module:
#   ai_travel_analyst.py
#
# That module must provide:
#   clean_df
#   gradio_analyzer(...)
#   predict_flight_price(...)
#
# Required optional result files:
#   ai_travel_model_comparison.csv
#   ai_travel_feature_importance.csv
# ============================================================

import os
import re
import html
import itertools
import pandas as pd
import streamlit as st

# Import the existing ML/data logic from your project.
# If your existing file has a different name, change this import.
from ai_travel_analyst import clean_df, gradio_analyzer, predict_flight_price


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Travel Analyst",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ------------------------------------------------------------
# STYLING
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

    :root {
        --bg:#0A0E17; --panel:#12161F; --panel-2:#171C27; --line:#232938;
        --amber:#F5B942; --amber-dim:rgba(245,185,66,.18);
        --green:#3DDC97; --green-dim:rgba(61,220,151,.12);
        --coral:#FF6D6D; --coral-dim:rgba(255,109,109,.12);
        --text:#E9ECF3; --muted:#8890A3;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1180px;
        padding: 28px 32px 60px 32px;
    }

    .hero { text-align:center; padding:36px 20px 30px 20px; }
    .hero-eyebrow {
        font-family:'JetBrains Mono',monospace; font-size:12px;
        letter-spacing:3px; color:var(--amber); text-transform:uppercase;
        margin-bottom:14px;
    }
    .hero-title {
        font-family:'Space Grotesk',sans-serif; font-size:46px;
        font-weight:700; color:var(--text); margin-bottom:10px;
    }
    .hero-title .accent { color:var(--amber); }
    .hero-subtitle { font-size:17px; color:var(--muted); margin-bottom:22px; }
    .hero-flow {
        display:inline-flex; align-items:center; gap:14px;
        font-family:'JetBrains Mono',monospace; font-size:13px;
        font-weight:500; color:var(--text); background:var(--panel);
        border:1px solid var(--line); padding:10px 22px; border-radius:999px;
    }
    .hero-flow .sep { color:var(--amber); }

    .section-title {
        text-align:left; padding:16px 20px; margin:30px 0 16px 0;
        border-left:3px solid var(--amber);
        background:linear-gradient(90deg,var(--amber-dim),transparent 60%);
        border-radius:0 10px 10px 0;
    }
    .section-title h2 {
        margin:0; font-family:'Space Grotesk',sans-serif;
        font-size:19px; font-weight:700; color:var(--text);
    }
    .section-title p { margin:5px 0 0 0; color:var(--muted); font-size:13px; }

    .ai-dashboard, .scenario-dashboard, .optimizer-dashboard, .perf-dashboard {
        width:100%; box-sizing:border-box; font-family:'Inter',sans-serif;
        color:var(--text); margin-top:14px;
    }
    .ai-result-header, .scenario-header, .optimizer-header, .perf-header {
        padding:24px; border:1px solid var(--line); border-radius:16px;
        margin-bottom:16px; background:linear-gradient(135deg,var(--panel),var(--panel-2));
    }
    .ai-eyebrow {
        font-family:'JetBrains Mono',monospace; font-size:11px;
        letter-spacing:2px; color:var(--amber); text-transform:uppercase;
    }
    .ai-route, .scenario-route, .optimizer-route {
        margin-top:10px; font-family:'Space Grotesk',sans-serif;
        font-size:27px; font-weight:700;
    }
    .ai-meta, .scenario-subtitle { margin-top:7px; color:var(--muted); font-size:13.5px; }

    .ai-top-row {
        display:grid; grid-template-columns:150px 1fr; gap:20px;
        align-items:center; padding:22px; border:1px solid var(--line);
        border-radius:16px; background:var(--panel); margin-bottom:14px;
    }
    .gauge-wrap { display:flex; flex-direction:column; align-items:center; }
    .gauge-label {
        margin-top:6px; font-family:'JetBrains Mono',monospace;
        font-size:10.5px; letter-spacing:1.5px; color:var(--muted);
        text-transform:uppercase;
    }
    .ai-headline-price {
        font-family:'JetBrains Mono',monospace; font-size:13px; color:var(--muted);
    }
    .ai-headline-price .value {
        display:block; font-size:34px; font-weight:700;
        color:var(--text); margin-top:4px;
    }
    .ai-headline-price .delta {
        margin-top:8px; display:inline-block; font-size:13px;
        padding:4px 10px; border-radius:999px;
    }
    .delta-good { background:var(--green-dim); color:var(--green); }
    .delta-bad { background:var(--coral-dim); color:var(--coral); }
    .delta-flat { background:var(--amber-dim); color:var(--amber); }

    .ai-card-grid, .scenario-comparison {
        display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px;
    }
    .ai-card {
        padding:18px; border:1px solid var(--line); border-radius:14px;
        background:var(--panel);
    }
    .ai-card-label {
        font-family:'JetBrains Mono',monospace; font-size:10.5px;
        font-weight:700; color:var(--muted); letter-spacing:1px;
        text-transform:uppercase;
    }
    .ai-card-value {
        margin-top:9px; font-family:'JetBrains Mono',monospace;
        font-size:21px; font-weight:700;
    }
    .ai-card-sub { margin-top:6px; font-size:11.5px; color:var(--muted); }

    .ai-decision {
        margin-top:14px; padding:22px; border-radius:16px;
        border:1px solid var(--line);
    }
    .decision-good { background:var(--green-dim); border-color:rgba(61,220,151,.35); }
    .decision-neutral { background:var(--amber-dim); border-color:rgba(245,185,66,.35); }
    .decision-warning { background:var(--coral-dim); border-color:rgba(255,109,109,.35); }
    .decision-label {
        font-family:'JetBrains Mono',monospace; font-size:11px;
        font-weight:700; color:var(--muted);
    }
    .decision-main {
        margin-top:8px; font-family:'Space Grotesk',sans-serif;
        font-size:22px; font-weight:700;
    }
    .decision-reason { margin-top:6px; color:var(--muted); font-size:14px; }
    .ai-model {
        display:flex; justify-content:space-between; margin-top:12px;
        padding:12px 18px; border-radius:12px; background:var(--panel);
        border:1px solid var(--line); color:var(--muted);
        font-family:'JetBrains Mono',monospace; font-size:11px;
    }
    .ai-model strong { color:var(--text); font-size:12.5px; }

    .drivers {
        margin-top:14px; padding:22px; border:1px solid var(--line);
        border-radius:16px; background:var(--panel);
    }
    .drivers-label {
        font-family:'JetBrains Mono',monospace; font-size:11px;
        font-weight:700; color:var(--muted); letter-spacing:1px;
        margin-bottom:12px;
    }
    .driver-row {
        display:grid; grid-template-columns:140px 1fr 40px;
        align-items:center; gap:10px; margin-bottom:8px;
    }
    .driver-name { font-size:12.5px; color:var(--text); }
    .driver-bar-bg { height:8px; background:var(--line); border-radius:4px; overflow:hidden; }
    .driver-bar-fill {
        height:100%; background:linear-gradient(90deg,var(--amber),var(--green));
    }
    .driver-pct {
        font-family:'JetBrains Mono',monospace; font-size:11px;
        color:var(--muted); text-align:right;
    }

    .scenario-grid {
        display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
        gap:12px; margin-bottom:12px;
    }
    .scenario-card {
        padding:20px; border:1px solid var(--line);
        border-radius:14px; background:var(--panel);
    }
    .scenario-highlight { border-color:var(--amber); background:var(--amber-dim); }
    .scenario-label {
        font-family:'JetBrains Mono',monospace; font-size:11px;
        font-weight:700; color:var(--muted);
    }
    .scenario-main { margin-top:10px; font-size:14px; font-weight:600; }
    .scenario-price {
        margin-top:10px; font-family:'JetBrains Mono',monospace;
        font-size:25px; font-weight:700;
    }
    .scenario-comparison > div {
        padding:15px; border:1px solid var(--line);
        border-radius:12px; background:var(--panel);
    }
    .scenario-comparison span {
        display:block; font-family:'JetBrains Mono',monospace;
        font-size:10.5px; color:var(--muted); font-weight:700;
    }
    .scenario-comparison strong {
        display:block; margin-top:7px; font-family:'JetBrains Mono',monospace;
        font-size:18px;
    }
    .scenario-verdict, .optimizer-result {
        margin-top:12px; padding:20px; border-radius:14px;
        border:1px solid rgba(61,220,151,.35); background:var(--green-dim);
    }
    .scenario-verdict-label, .optimizer-result-label {
        font-family:'JetBrains Mono',monospace; font-size:11px;
        font-weight:700; color:var(--muted);
    }
    .scenario-verdict-main, .optimizer-result-main {
        margin-top:7px; font-family:'Space Grotesk',sans-serif;
        font-size:19px; font-weight:700;
    }
    .scenario-verdict-text { margin-top:6px; color:var(--muted); font-size:13.5px; }

    .optimizer-grid {
        display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
        gap:12px; margin-top:14px;
    }
    .optimizer-card {
        padding:18px; border:1px solid var(--line);
        border-radius:14px; background:var(--panel); text-align:center;
    }
    .optimizer-card-label {
        font-family:'JetBrains Mono',monospace; font-size:10.5px;
        color:var(--muted); letter-spacing:1px;
    }
    .optimizer-card-value {
        margin-top:8px; font-family:'Space Grotesk',sans-serif;
        font-size:22px; font-weight:700; color:var(--amber);
    }

    .perf-table { width:100%; border-collapse:collapse; margin-top:10px; }
    .perf-table th, .perf-table td {
        padding:10px 14px; text-align:left; border-bottom:1px solid var(--line);
        font-family:'JetBrains Mono',monospace; font-size:13px;
    }
    .perf-table th {
        color:var(--muted); font-size:11px; letter-spacing:1px; text-transform:uppercase;
    }
    .perf-best { color:var(--green); font-weight:700; }

    .ai-empty {
        padding:34px; border:1px dashed var(--line); border-radius:16px;
        background:var(--panel); text-align:center;
    }
    .ai-empty-title {
        font-family:'Space Grotesk',sans-serif; font-size:18px; font-weight:700;
        margin-top:8px;
    }
    .ai-empty-text { color:var(--muted); margin-top:6px; font-size:13.5px; }

    .footer {
        text-align:center; padding:28px; margin-top:30px; color:var(--muted);
        font-size:12px; font-family:'JetBrains Mono',monospace;
    }

    @media (max-width:700px) {
        .ai-card-grid,.scenario-comparison,.scenario-grid,.optimizer-grid {
            grid-template-columns:1fr;
        }
        .ai-top-row { grid-template-columns:1fr; }
        .hero-title { font-size:32px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# DATA-DRIVEN OPTIONS
# ------------------------------------------------------------
airlines = sorted(clean_df["Airline"].dropna().unique().tolist())
sources = sorted(clean_df["Source"].dropna().unique().tolist())
destinations = sorted(clean_df["Destination"].dropna().unique().tolist())
travel_classes = sorted(clean_df["Travel_Class"].dropna().unique().tolist())
seasons = sorted(clean_df["Season"].dropna().unique().tolist())
weekdays = sorted(clean_df["Weekday"].dropna().unique().tolist())
aircraft_types = sorted(clean_df["Aircraft_Type"].dropna().unique().tolist())
booking_channels = sorted(clean_df["Booking_Channel"].dropna().unique().tolist())


def first_or(options, preferred, fallback_index=0):
    if preferred in options:
        return preferred
    return options[fallback_index] if options else None


# ------------------------------------------------------------
# DISPLAY HELPERS
# ------------------------------------------------------------
def render_gauge_svg(score_0_100, size=120):
    try:
        score = max(0.0, min(100.0, float(score_0_100)))
    except (TypeError, ValueError):
        score = 0.0

    radius = 50
    circumference = 2 * 3.14159265 * radius
    offset = circumference * (1 - score / 100)
    color = "#3DDC97" if score >= 66 else ("#F5B942" if score >= 40 else "#FF6D6D")

    return f"""<svg width="{size}" height="{size}" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="{radius}" fill="none" stroke="#232938" stroke-width="10"/>
        <circle cx="60" cy="60" r="{radius}" fill="none" stroke="{color}" stroke-width="10"
            stroke-linecap="round" stroke-dasharray="{circumference:.2f}"
            stroke-dashoffset="{offset:.2f}" transform="rotate(-90 60 60)"/>
        <text x="60" y="56" text-anchor="middle" font-family="JetBrains Mono, monospace"
            font-size="26" font-weight="700" fill="#E9ECF3">{score:.0f}</text>
        <text x="60" y="74" text-anchor="middle" font-family="JetBrains Mono, monospace"
            font-size="11" fill="#8890A3">/ 100</text>
    </svg>"""


EMPTY_STATE_HTML = """<div class="ai-empty"><div style="font-size:26px;">🛰️</div>
<div class="ai-empty-title">Awaiting Flight Data</div>
<div class="ai-empty-text">Enter your flight details above and click <b>Analyze Flight</b>.</div></div>"""


def render_price_drivers_html():
    path = "ai_travel_feature_importance.csv"
    if not os.path.exists(path):
        return ""

    df = pd.read_csv(path).head(7)
    if df.empty or "Importance" not in df.columns or "Feature" not in df.columns:
        return ""

    max_imp = df["Importance"].max()
    if not max_imp:
        return ""

    rows = ""
    for _, r in df.iterrows():
        pct = r["Importance"] / max_imp * 100
        rows += f"""<div class="driver-row">
            <div class="driver-name">{html.escape(str(r['Feature']))}</div>
            <div class="driver-bar-bg"><div class="driver-bar-fill" style="width:{pct:.1f}%"></div></div>
            <div class="driver-pct">{pct:.0f}%</div>
        </div>"""

    return f"""<div class="drivers"><div class="drivers-label">
        🧠 PRICE DRIVERS — WHY THIS PREDICTION
    </div>{rows}</div>"""


def format_ai_output_html(raw_output):
    if raw_output is None:
        return EMPTY_STATE_HTML

    text = str(raw_output)

    def extract(pattern, default="—"):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else default

    predicted_price = extract(r"Predicted Price\s*:\s*([^\n]+)")
    historical_median = extract(r"Historical Median\s*:\s*([^\n]+)")
    difference = extract(r"Difference\s*:\s*([₹\-0-9,.\s]+)")
    difference_percent = extract(r"Difference \(%\)\s*:\s*([^\n]+)")
    deal_score = extract(r"Deal Score\s*:\s*([0-9.]+)")
    price_percentile = extract(r"Price Percentile\s*:\s*([0-9.]+)")
    comparable_flights = extract(r"Comparable Flights\s*:\s*([0-9]+)")
    model_name = extract(r"Model\s*:\s*([^\n]+)")
    airline = extract(r"AIRLINE\s*\n\s*([^\n]+)")
    travel_class = extract(r"TRAVEL CLASS\s*\n\s*([^\n]+)")
    days_before = extract(r"DAYS BEFORE DEPARTURE\s*\n\s*([^\n]+)")

    route_match = re.search(r"ROUTE\s*\n\s*(.+?)\s*→\s*(.+)", text, re.IGNORECASE)
    route = (
        f"{html.escape(route_match.group(1).strip())} → "
        f"{html.escape(route_match.group(2).strip())}"
        if route_match else "Flight Analysis"
    )

    recommendation, recommendation_reason = "—", ""
    match = re.search(
        r"(BOOK NOW|CONSIDER BOOKING|COMPARE OPTIONS|CONSIDER ALTERNATIVES)"
        r"\s*[–-]?\s*([^\n]*)",
        text,
        re.IGNORECASE,
    )
    if match:
        recommendation = match.group(1).upper()
        recommendation_reason = match.group(2).strip()

    delta_class = "delta-flat"
    if difference.strip().startswith("-"):
        delta_class = "delta-good"
    elif re.search(r"[1-9]", difference):
        delta_class = "delta-bad"

    gauge_svg = render_gauge_svg(deal_score if deal_score != "—" else 0)
    drivers_html = render_price_drivers_html()

    values = [
        predicted_price, historical_median, difference, difference_percent,
        deal_score, price_percentile, comparable_flights, model_name,
        airline, travel_class, days_before, recommendation, recommendation_reason,
    ]
    (
        predicted_price, historical_median, difference, difference_percent,
        deal_score, price_percentile, comparable_flights, model_name,
        airline, travel_class, days_before, recommendation, recommendation_reason,
    ) = [html.escape(str(v)) for v in values]

    if "BOOK NOW" in recommendation or "CONSIDER BOOKING" in recommendation:
        decision_class, decision_icon = "decision-good", "🟢"
    elif "COMPARE" in recommendation:
        decision_class, decision_icon = "decision-neutral", "🟡"
    elif "ALTERNATIVE" in recommendation:
        decision_class, decision_icon = "decision-warning", "🔴"
    else:
        decision_class, decision_icon = "decision-neutral", "🔵"

    return f"""<div class="ai-dashboard">
        <div class="ai-result-header">
            <div class="ai-eyebrow">🛰️ Flight Intelligence Report</div>
            <div class="ai-route">{route}</div>
            <div class="ai-meta">{airline} &nbsp;•&nbsp; {travel_class}
                &nbsp;•&nbsp; {days_before} days before departure</div>
        </div>
        <div class="ai-top-row">
            <div class="gauge-wrap">{gauge_svg}<div class="gauge-label">Deal Score</div></div>
            <div class="ai-headline-price">PREDICTED FARE
                <span class="value">{predicted_price}</span>
                <span class="delta {delta_class}">{difference}
                ({difference_percent}) vs median</span>
            </div>
        </div>
        <div class="ai-card-grid">
            <div class="ai-card"><div class="ai-card-label">Historical Median</div>
                <div class="ai-card-value">{historical_median}</div>
                <div class="ai-card-sub">Comparable historical flights</div></div>
            <div class="ai-card"><div class="ai-card-label">Price Percentile</div>
                <div class="ai-card-value">{price_percentile}</div>
                <div class="ai-card-sub">Position vs. historical fares</div></div>
            <div class="ai-card"><div class="ai-card-label">Comparable Flights</div>
                <div class="ai-card-value">{comparable_flights}</div>
                <div class="ai-card-sub">Matches used in this analysis</div></div>
        </div>
        {drivers_html}
        <div class="ai-decision {decision_class}">
            <div class="decision-label">💡 AI DECISION</div>
            <div class="decision-main">{decision_icon} {recommendation}</div>
            <div class="decision-reason">{recommendation_reason}</div>
        </div>
        <div class="ai-model"><span>🧠 MODEL</span><strong>{model_name}</strong></div>
    </div>"""


def render_model_performance_html():
    path = "ai_travel_model_comparison.csv"

    if not os.path.exists(path):
        return """<div class="ai-empty"><div style="font-size:22px;">📊</div>
        <div class="ai-empty-title">Model Comparison Not Found</div>
        <div class="ai-empty-text">Run phase1_model_evaluation.py first to generate
        ai_travel_model_comparison.csv.</div></div>"""

    df = pd.read_csv(path)
    if "R2" not in df.columns or "MAE" not in df.columns or "RMSE" not in df.columns:
        return """<div class="ai-empty"><div class="ai-empty-title">
        Invalid model comparison file</div></div>"""

    df = df.sort_values("R2", ascending=False).reset_index(drop=True)
    best_name = df.iloc[0]["Model"]
    rows = ""

    for _, r in df.iterrows():
        is_best = r["Model"] == best_name
        cls = "perf-best" if is_best else ""
        star = " 🏆" if is_best else ""
        rows += (
            f"<tr><td class='{cls}'>{html.escape(str(r['Model']))}{star}</td>"
            f"<td>₹{r['MAE']:,.2f}</td><td>{r['RMSE']:,.2f}</td>"
            f"<td>{r['R2']:.3f}</td></tr>"
        )

    return f"""<div class="perf-dashboard"><div class="perf-header">
        <div class="ai-eyebrow">📊 Model Performance</div>
        <div class="ai-meta" style="margin-top:8px;">{html.escape(best_name)}
        was selected as the final model because it achieved the best validation R²
        and lowest MAE among the evaluated models.</div>
        <table class="perf-table"><thead><tr><th>Model</th><th>MAE</th>
        <th>RMSE</th><th>R²</th></tr></thead><tbody>{rows}</tbody></table>
    </div></div>"""


# ------------------------------------------------------------
# BUSINESS LOGIC
# ------------------------------------------------------------
def analyze_flight(
    airline, source, destination, departure_date, travel_class, days,
    departure_hour, arrival_hour, duration, stops, distance, season,
    weekday, aircraft, booking_channel, passengers,
):
    raw_result = gradio_analyzer(
        airline, source, destination, departure_date, travel_class, days,
        departure_hour, arrival_hour, duration, stops, distance, season,
        weekday, aircraft, booking_channel, passengers,
    )
    return format_ai_output_html(raw_result)


def simulate_scenario(
    scenario_stops, scenario_class, scenario_days,
    airline, source, destination, departure_date, current_class, current_days,
    departure_hour, arrival_hour, duration, current_stops, distance, season,
    weekday, aircraft, booking_channel, passengers,
):
    scenario_price = predict_flight_price(
        airline=airline, source=source, destination=destination,
        departure_date=departure_date, departure_hour=int(departure_hour),
        arrival_hour=int(arrival_hour), duration_minutes=float(duration),
        total_stops=int(scenario_stops), distance_km=float(distance),
        travel_class=scenario_class, days_before_departure=int(scenario_days),
        season=season, weekday=weekday, aircraft_type=aircraft,
        booking_channel=booking_channel, passenger_count=int(passengers),
    )

    current_price = predict_flight_price(
        airline=airline, source=source, destination=destination,
        departure_date=departure_date, departure_hour=int(departure_hour),
        arrival_hour=int(arrival_hour), duration_minutes=float(duration),
        total_stops=int(current_stops), distance_km=float(distance),
        travel_class=current_class, days_before_departure=int(current_days),
        season=season, weekday=weekday, aircraft_type=aircraft,
        booking_channel=booking_channel, passenger_count=int(passengers),
    )

    difference = scenario_price - current_price
    saving = current_price - scenario_price
    pct_change = (difference / current_price * 100) if current_price else 0

    if saving > 0:
        verdict = "🟢 THIS SCENARIO MAY SAVE MONEY"
        verdict_text = f"Potential saving of ₹{saving:,.2f} ({abs(pct_change):.2f}%)."
    elif saving < 0:
        verdict = "🔴 THIS SCENARIO COSTS MORE"
        verdict_text = (
            f"This scenario is ₹{abs(saving):,.2f} more expensive "
            f"({abs(pct_change):.2f}% higher)."
        )
    else:
        verdict = "🟡 NO PRICE CHANGE"
        verdict_text = "The predicted price is unchanged."

    return f"""<div class="scenario-dashboard">
        <div class="scenario-header"><div class="ai-eyebrow">🔮 What-If Result</div>
            <div class="scenario-route">{html.escape(str(source))} →
            {html.escape(str(destination))}</div>
            <div class="scenario-subtitle">Compare your current booking with
            the scenario you selected.</div></div>
        <div class="scenario-grid">
            <div class="scenario-card"><div class="scenario-label">CURRENT BOOKING</div>
                <div class="scenario-main">{int(current_days)} days ·
                {html.escape(str(current_class))} · {int(current_stops)} stop(s)</div>
                <div class="scenario-price">₹{current_price:,.2f}</div></div>
            <div class="scenario-card scenario-highlight">
                <div class="scenario-label">WHAT-IF SCENARIO</div>
                <div class="scenario-main">{int(scenario_days)} days ·
                {html.escape(str(scenario_class))} · {int(scenario_stops)} stop(s)</div>
                <div class="scenario-price">₹{scenario_price:,.2f}</div></div>
        </div>
        <div class="scenario-comparison">
            <div><span>PRICE DIFFERENCE</span><strong>₹{difference:,.2f}</strong></div>
            <div><span>POTENTIAL SAVING</span><strong>₹{max(saving,0):,.2f}</strong></div>
            <div><span>PRICE CHANGE</span><strong>{pct_change:+.2f}%</strong></div>
        </div>
        <div class="scenario-verdict"><div class="scenario-verdict-label">🧠 AI VERDICT</div>
            <div class="scenario-verdict-main">{verdict}</div>
            <div class="scenario-verdict-text">{verdict_text}</div></div>
    </div>"""


def find_optimal_strategy(
    airline, source, destination, departure_date, current_class, current_days,
    departure_hour, arrival_hour, duration, current_stops, distance, season,
    weekday, aircraft, booking_channel, passengers,
):
    stop_options = [0, 1, 2]
    class_options = ["Economy", "Premium Economy", "Business", "First"]
    day_options = [7, 14, 30, 45, 60, 90]

    current_price = predict_flight_price(
        airline=airline, source=source, destination=destination,
        departure_date=departure_date, departure_hour=int(departure_hour),
        arrival_hour=int(arrival_hour), duration_minutes=float(duration),
        total_stops=int(current_stops), distance_km=float(distance),
        travel_class=current_class, days_before_departure=int(current_days),
        season=season, weekday=weekday, aircraft_type=aircraft,
        booking_channel=booking_channel, passenger_count=int(passengers),
    )

    best = None

    for stops, cls, days in itertools.product(
        stop_options, class_options, day_options
    ):
        price = predict_flight_price(
            airline=airline, source=source, destination=destination,
            departure_date=departure_date, departure_hour=int(departure_hour),
            arrival_hour=int(arrival_hour), duration_minutes=float(duration),
            total_stops=stops, distance_km=float(distance),
            travel_class=cls, days_before_departure=days,
            season=season, weekday=weekday, aircraft_type=aircraft,
            booking_channel=booking_channel, passenger_count=int(passengers),
        )

        if best is None or price < best["price"]:
            best = {"stops": stops, "class": cls, "days": days, "price": price}

    saving = current_price - best["price"]
    pct = (saving / current_price * 100) if current_price else 0

    return f"""<div class="optimizer-dashboard">
        <div class="optimizer-header"><div class="ai-eyebrow">🏆 Optimal Booking Strategy</div>
            <div class="optimizer-route">{html.escape(str(source))} →
            {html.escape(str(destination))}</div>
            <div class="ai-meta">Searched {len(stop_options)*len(class_options)*len(day_options)}
            combinations of stops, class and booking window</div></div>
        <div class="optimizer-grid">
            <div class="optimizer-card"><div class="optimizer-card-label">STOPS</div>
                <div class="optimizer-card-value">{best['stops']}</div></div>
            <div class="optimizer-card"><div class="optimizer-card-label">CLASS</div>
                <div class="optimizer-card-value">{html.escape(best['class'])}</div></div>
            <div class="optimizer-card"><div class="optimizer-card-label">BOOK BY</div>
                <div class="optimizer-card-value">{best['days']}d before</div></div>
        </div>
        <div class="optimizer-result">
            <div class="optimizer-result-label">💰 ESTIMATED PRICE</div>
            <div class="optimizer-result-main">₹{best['price']:,.2f}</div>
            <div class="scenario-verdict-text">Potential saving of
            ₹{max(saving,0):,.2f} ({pct:.1f}%) vs your current selection
            (₹{current_price:,.2f}).</div>
        </div>
    </div>"""


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.markdown(
    """<div class="hero">
        <div class="hero-eyebrow">FLIGHT INTELLIGENCE SYSTEM</div>
        <div class="hero-title">✈️ AI TRAVEL <span class="accent">ANALYST</span></div>
        <div class="hero-subtitle">AI Travel Decision Support System</div>
        <div class="hero-flow">PREDICT <span class="sep">→</span> EXPLAIN
        <span class="sep">→</span> COMPARE <span class="sep">→</span> SIMULATE
        <span class="sep">→</span> OPTIMIZE</div>
    </div>""",
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="section-title"><h2>📊 Model Performance</h2>
    <p>How the final model was chosen.</p></div>""",
    unsafe_allow_html=True,
)
st.markdown(render_model_performance_html(), unsafe_allow_html=True)

st.markdown(
    """<div class="section-title"><h2>✈️ Flight Information</h2>
    <p>Tell the AI about the flight you are considering.</p></div>""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    airline_input = st.selectbox("Airline", airlines, index=(
        airlines.index("qatar airways") if "qatar airways" in airlines else 0
    ))
with c2:
    source_input = st.selectbox("Source", sources, index=(
        sources.index("Delhi") if "Delhi" in sources else 0
    ))
with c3:
    destination_input = st.selectbox("Destination", destinations, index=(
        destinations.index("London") if "London" in destinations else 0
    ))

c1, c2, c3 = st.columns(3)
with c1:
    travel_class_input = st.selectbox("Travel Class", travel_classes, index=(
        travel_classes.index("Economy") if "Economy" in travel_classes else 0
    ))
with c2:
    departure_date_input = st.text_input("Departure Date", value="2026-12-15")
with c3:
    days_input = st.slider("Days Before Departure", 1, 180, 30, 1)

st.markdown(
    """<div class="section-title"><h2>🕐 Flight Details</h2>
    <p>Configure timing, duration, stops and passenger information.</p></div>""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    departure_hour_input = st.slider("Departure Hour", 0, 23, 10, 1)
with c2:
    arrival_hour_input = st.slider("Arrival Hour", 0, 23, 18, 1)
with c3:
    duration_input = st.number_input("Duration (minutes)", min_value=1.0, value=540.0)

c1, c2, c3 = st.columns(3)
with c1:
    stops_input = st.slider("Total Stops", 0, 2, 1, 1)
with c2:
    distance_input = st.number_input("Distance (km)", min_value=1.0, value=6700.0)
with c3:
    passengers_input = st.slider("Passengers", 1, 6, 1, 1)

st.markdown(
    """<div class="section-title"><h2>🌍 Travel Conditions</h2>
    <p>Additional factors used by the machine-learning model.</p></div>""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    season_input = st.selectbox("Season", seasons, index=(
        seasons.index("Winter") if "Winter" in seasons else 0
    ))
with c2:
    weekday_input = st.selectbox("Weekday", weekdays, index=(
        weekdays.index("Tuesday") if "Tuesday" in weekdays else 0
    ))
with c3:
    aircraft_input = st.selectbox("Aircraft Type", aircraft_types, index=(
        aircraft_types.index("Boeing 777") if "Boeing 777" in aircraft_types else 0
    ))
with c4:
    booking_input = st.selectbox("Booking Channel", booking_channels, index=(
        booking_channels.index("Website") if "Website" in booking_channels else 0
    ))

if st.button("🔮 ANALYZE FLIGHT", type="primary", use_container_width=True):
    try:
        with st.spinner("Analyzing flight..."):
            result = analyze_flight(
                airline_input, source_input, destination_input, departure_date_input,
                travel_class_input, days_input, departure_hour_input,
                arrival_hour_input, duration_input, stops_input, distance_input,
                season_input, weekday_input, aircraft_input, booking_input,
                passengers_input,
            )
        st.markdown(result, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
else:
    st.markdown(EMPTY_STATE_HTML, unsafe_allow_html=True)


st.markdown(
    """<div class="section-title"><h2>🧠 AI What-If Travel Simulator</h2>
    <p>Defaults below are set differently from your current booking above,
    so you can see a real contrast immediately — then tweak either side.</p></div>""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    scenario_stops = st.slider("Number of Stops", 0, 2, 0, 1, key="scenario_stops")
with c2:
    scenario_class = st.selectbox(
        "Scenario Travel Class",
        ["Economy", "Premium Economy", "Business", "First"],
        index=0,
        key="scenario_class",
    )
with c3:
    scenario_days = st.slider(
        "Scenario Days Before Departure", 1, 180, 60, 1,
        key="scenario_days",
    )

if st.button("🔮 SIMULATE SCENARIO", type="primary", use_container_width=True):
    try:
        with st.spinner("Simulating scenario..."):
            result = simulate_scenario(
                scenario_stops, scenario_class, scenario_days,
                airline_input, source_input, destination_input,
                departure_date_input, travel_class_input, days_input,
                departure_hour_input, arrival_hour_input, duration_input,
                stops_input, distance_input, season_input, weekday_input,
                aircraft_input, booking_input, passengers_input,
            )
        st.markdown(result, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Scenario simulation failed: {e}")


st.markdown(
    """<div class="section-title"><h2>🏆 AI Booking Strategy Optimizer</h2>
    <p>Don't guess scenarios — let the AI search all of them and find the
    cheapest realistic strategy for this route.</p></div>""",
    unsafe_allow_html=True,
)

if st.button("🏆 FIND OPTIMAL STRATEGY", type="primary", use_container_width=True):
    try:
        with st.spinner("Searching booking strategies..."):
            result = find_optimal_strategy(
                airline_input, source_input, destination_input,
                departure_date_input, travel_class_input, days_input,
                departure_hour_input, arrival_hour_input, duration_input,
                stops_input, distance_input, season_input, weekday_input,
                aircraft_input, booking_input, passengers_input,
            )
        st.markdown(result, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Optimizer failed: {e}")


st.markdown(
    """<div class="footer">AI TRAVEL ANALYST · Predict → Explain →
    Compare → Simulate → Optimize</div>""",
    unsafe_allow_html=True,
)
