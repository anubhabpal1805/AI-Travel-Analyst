# ============================================================
# AI TRAVEL ANALYST — v2
# Builds on the previously-fixed dashboard. Adds:
#   - Model Performance section (reads ai_travel_model_comparison.csv,
#     produced by phase1_model_evaluation.py)
#   - "Price Drivers" explainability bars in the result card
#     (reads ai_travel_feature_importance.csv if present)
#   - What-If defaults now differ from the main form by default,
#     so the demo shows a real contrast on first load
#   - NEW: AI Booking Strategy Optimizer — searches stops x class
#     x days-before-departure and returns the cheapest combo
#
# Still assumes clean_df, gradio_analyzer(...), and
# predict_flight_price(...) exist earlier in your notebook.
# ============================================================

import gradio as gr
import re
import html
import os
import itertools
import pandas as pd


airlines = sorted(clean_df["Airline"].dropna().unique().tolist())
sources = sorted(clean_df["Source"].dropna().unique().tolist())
destinations = sorted(clean_df["Destination"].dropna().unique().tolist())
travel_classes = sorted(clean_df["Travel_Class"].dropna().unique().tolist())
seasons = sorted(clean_df["Season"].dropna().unique().tolist())
weekdays = sorted(clean_df["Weekday"].dropna().unique().tolist())
aircraft_types = sorted(clean_df["Aircraft_Type"].dropna().unique().tolist())
booking_channels = sorted(clean_df["Booking_Channel"].dropna().unique().tolist())


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');
:root {
    --bg:#0A0E17; --panel:#12161F; --panel-2:#171C27; --line:#232938;
    --amber:#F5B942; --amber-dim:rgba(245,185,66,.18);
    --green:#3DDC97; --green-dim:rgba(61,220,151,.12);
    --coral:#FF6D6D; --coral-dim:rgba(255,109,109,.12);
    --text:#E9ECF3; --muted:#8890A3;
}
body, .gradio-container { background: var(--bg) !important; }
.gradio-container { max-width:1180px !important; margin:auto !important; padding:28px 32px 60px 32px !important; font-family:'Inter',sans-serif !important; }
.hero { text-align:center; padding:36px 20px 30px 20px; }
.hero-eyebrow { font-family:'JetBrains Mono',monospace; font-size:12px; letter-spacing:3px; color:var(--amber); text-transform:uppercase; margin-bottom:14px; }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:46px; font-weight:700; color:var(--text); margin-bottom:10px; }
.hero-title .accent { color:var(--amber); }
.hero-subtitle { font-size:17px; color:var(--muted); margin-bottom:22px; }
.hero-flow { display:inline-flex; align-items:center; gap:14px; font-family:'JetBrains Mono',monospace; font-size:13px; font-weight:500; color:var(--text); background:var(--panel); border:1px solid var(--line); padding:10px 22px; border-radius:999px; }
.hero-flow .sep { color:var(--amber); }
.section-title { text-align:left; padding:16px 20px; margin:30px 0 16px 0; border-left:3px solid var(--amber); background:linear-gradient(90deg,var(--amber-dim),transparent 60%); border-radius:0 10px 10px 0; }
.section-title h2 { margin:0; font-family:'Space Grotesk',sans-serif; font-size:19px; font-weight:700; color:var(--text); }
.section-title p { margin:5px 0 0 0; color:var(--muted); font-size:13px; }
label span { color:var(--muted) !important; font-size:12px !important; font-weight:600 !important; }
input, select, textarea { font-family:'JetBrains Mono',monospace !important; background:var(--panel) !important; border:1px solid var(--line) !important; color:var(--text) !important; border-radius:10px !important; }
input:focus, select:focus { border-color:var(--amber) !important; box-shadow:0 0 0 3px var(--amber-dim) !important; }
input[type="range"] { accent-color: var(--amber) !important; }
button.primary { background:linear-gradient(135deg,var(--amber),#E0A22F) !important; color:#14110A !important; border:none !important; box-shadow:0 8px 24px rgba(245,185,66,.25) !important; font-family:'Space Grotesk',sans-serif !important; font-weight:700 !important; border-radius:12px !important; }
button.primary:hover { transform:translateY(-2px); }
.ai-dashboard, .scenario-dashboard, .optimizer-dashboard, .perf-dashboard { width:100%; box-sizing:border-box; font-family:'Inter',sans-serif; color:var(--text); margin-top:14px; }
.ai-result-header, .scenario-header, .optimizer-header, .perf-header { padding:24px; border:1px solid var(--line); border-radius:16px; margin-bottom:16px; background:linear-gradient(135deg,var(--panel),var(--panel-2)); }
.ai-eyebrow { font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:2px; color:var(--amber); text-transform:uppercase; }
.ai-route, .scenario-route, .optimizer-route { margin-top:10px; font-family:'Space Grotesk',sans-serif; font-size:27px; font-weight:700; }
.ai-meta, .scenario-subtitle { margin-top:7px; color:var(--muted); font-size:13.5px; }
.ai-top-row { display:grid; grid-template-columns:150px 1fr; gap:20px; align-items:center; padding:22px; border:1px solid var(--line); border-radius:16px; background:var(--panel); margin-bottom:14px; }
.gauge-wrap { display:flex; flex-direction:column; align-items:center; }
.gauge-label { margin-top:6px; font-family:'JetBrains Mono',monospace; font-size:10.5px; letter-spacing:1.5px; color:var(--muted); text-transform:uppercase; }
.ai-headline-price { font-family:'JetBrains Mono',monospace; font-size:13px; color:var(--muted); }
.ai-headline-price .value { display:block; font-size:34px; font-weight:700; color:var(--text); margin-top:4px; }
.ai-headline-price .delta { margin-top:8px; display:inline-block; font-size:13px; padding:4px 10px; border-radius:999px; }
.delta-good { background:var(--green-dim); color:var(--green); }
.delta-bad { background:var(--coral-dim); color:var(--coral); }
.delta-flat { background:var(--amber-dim); color:var(--amber); }
.ai-card-grid, .scenario-comparison { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }
.ai-card { padding:18px; border:1px solid var(--line); border-radius:14px; background:var(--panel); }
.ai-card-label { font-family:'JetBrains Mono',monospace; font-size:10.5px; font-weight:700; color:var(--muted); letter-spacing:1px; text-transform:uppercase; }
.ai-card-value { margin-top:9px; font-family:'JetBrains Mono',monospace; font-size:21px; font-weight:700; }
.ai-card-sub { margin-top:6px; font-size:11.5px; color:var(--muted); }
.ai-decision { margin-top:14px; padding:22px; border-radius:16px; border:1px solid var(--line); }
.decision-good { background:var(--green-dim); border-color:rgba(61,220,151,.35); }
.decision-neutral { background:var(--amber-dim); border-color:rgba(245,185,66,.35); }
.decision-warning { background:var(--coral-dim); border-color:rgba(255,109,109,.35); }
.decision-label { font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700; color:var(--muted); }
.decision-main { margin-top:8px; font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:700; }
.decision-reason { margin-top:6px; color:var(--muted); font-size:14px; }
.ai-model { display:flex; justify-content:space-between; margin-top:12px; padding:12px 18px; border-radius:12px; background:var(--panel); border:1px solid var(--line); color:var(--muted); font-family:'JetBrains Mono',monospace; font-size:11px; }
.ai-model strong { color:var(--text); font-size:12.5px; }
.drivers { margin-top:14px; padding:22px; border:1px solid var(--line); border-radius:16px; background:var(--panel); }
.drivers-label { font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700; color:var(--muted); letter-spacing:1px; margin-bottom:12px; }
.driver-row { display:grid; grid-template-columns:140px 1fr 40px; align-items:center; gap:10px; margin-bottom:8px; }
.driver-name { font-size:12.5px; color:var(--text); }
.driver-bar-bg { height:8px; background:var(--line); border-radius:4px; overflow:hidden; }
.driver-bar-fill { height:100%; background:linear-gradient(90deg,var(--amber),var(--green)); }
.driver-pct { font-family:'JetBrains Mono',monospace; font-size:11px; color:var(--muted); text-align:right; }
.scenario-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin-bottom:12px; }
.scenario-card { padding:20px; border:1px solid var(--line); border-radius:14px; background:var(--panel); }
.scenario-highlight { border-color:var(--amber); background:var(--amber-dim); }
.scenario-label { font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700; color:var(--muted); }
.scenario-main { margin-top:10px; font-size:14px; font-weight:600; }
.scenario-price { margin-top:10px; font-family:'JetBrains Mono',monospace; font-size:25px; font-weight:700; }
.scenario-comparison > div { padding:15px; border:1px solid var(--line); border-radius:12px; background:var(--panel); }
.scenario-comparison span { display:block; font-family:'JetBrains Mono',monospace; font-size:10.5px; color:var(--muted); font-weight:700; }
.scenario-comparison strong { display:block; margin-top:7px; font-family:'JetBrains Mono',monospace; font-size:18px; }
.scenario-verdict, .optimizer-result { margin-top:12px; padding:20px; border-radius:14px; border:1px solid rgba(61,220,151,.35); background:var(--green-dim); }
.scenario-verdict-label, .optimizer-result-label { font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700; color:var(--muted); }
.scenario-verdict-main, .optimizer-result-main { margin-top:7px; font-family:'Space Grotesk',sans-serif; font-size:19px; font-weight:700; }
.scenario-verdict-text { margin-top:6px; color:var(--muted); font-size:13.5px; }
.optimizer-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-top:14px; }
.optimizer-card { padding:18px; border:1px solid var(--line); border-radius:14px; background:var(--panel); text-align:center; }
.optimizer-card-label { font-family:'JetBrains Mono',monospace; font-size:10.5px; color:var(--muted); letter-spacing:1px; }
.optimizer-card-value { margin-top:8px; font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:700; color:var(--amber); }
.perf-table { width:100%; border-collapse:collapse; margin-top:10px; }
.perf-table th, .perf-table td { padding:10px 14px; text-align:left; border-bottom:1px solid var(--line); font-family:'JetBrains Mono',monospace; font-size:13px; }
.perf-table th { color:var(--muted); font-size:11px; letter-spacing:1px; text-transform:uppercase; }
.perf-best { color:var(--green); font-weight:700; }
.ai-empty { padding:34px; border:1px dashed var(--line); border-radius:16px; background:var(--panel); text-align:center; }
.ai-empty-title { font-family:'Space Grotesk',sans-serif; font-size:18px; font-weight:700; margin-top:8px; }
.ai-empty-text { color:var(--muted); margin-top:6px; font-size:13.5px; }
.footer { text-align:center; padding:28px; margin-top:30px; color:var(--muted); font-size:12px; font-family:'JetBrains Mono',monospace; }
@media (max-width:700px) { .ai-card-grid,.scenario-comparison,.scenario-grid,.optimizer-grid { grid-template-columns:1fr; } .ai-top-row { grid-template-columns:1fr; } .hero-title { font-size:32px; } }
"""


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
        <circle cx="60" cy="60" r="{radius}" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round"
            stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}" transform="rotate(-90 60 60)"/>
        <text x="60" y="56" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="26" font-weight="700" fill="#E9ECF3">{score:.0f}</text>
        <text x="60" y="74" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="11" fill="#8890A3">/ 100</text>
    </svg>"""


EMPTY_STATE_HTML = """<div class="ai-empty"><div style="font-size:26px;">🛰️</div>
<div class="ai-empty-title">Awaiting Flight Data</div>
<div class="ai-empty-text">Enter your flight details above and click <b>Analyze Flight</b>.</div></div>"""


def render_price_drivers_html():
    """Reads ai_travel_feature_importance.csv produced by phase1_model_evaluation.py.
    Returns '' if the file doesn't exist yet, so nothing breaks."""
    path = "ai_travel_feature_importance.csv"
    if not os.path.exists(path):
        return ""
    df = pd.read_csv(path).head(7)
    if df.empty:
        return ""
    max_imp = df["Importance"].max()
    rows = ""
    for _, r in df.iterrows():
        pct = r["Importance"] / max_imp * 100
        rows += f"""<div class="driver-row">
            <div class="driver-name">{html.escape(str(r['Feature']))}</div>
            <div class="driver-bar-bg"><div class="driver-bar-fill" style="width:{pct:.1f}%"></div></div>
            <div class="driver-pct">{pct:.0f}%</div>
        </div>"""
    return f"""<div class="drivers"><div class="drivers-label">🧠 PRICE DRIVERS — WHY THIS PREDICTION</div>{rows}</div>"""


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
    route = (f"{html.escape(route_match.group(1).strip())} → {html.escape(route_match.group(2).strip())}"
              if route_match else "Flight Analysis")

    recommendation, recommendation_reason = "—", ""
    match = re.search(r"(BOOK NOW|CONSIDER BOOKING|COMPARE OPTIONS|CONSIDER ALTERNATIVES)\s*[–-]?\s*([^\n]*)", text, re.IGNORECASE)
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

    values = [predicted_price, historical_median, difference, difference_percent, deal_score,
              price_percentile, comparable_flights, model_name, airline, travel_class,
              days_before, recommendation, recommendation_reason]
    (predicted_price, historical_median, difference, difference_percent, deal_score, price_percentile,
     comparable_flights, model_name, airline, travel_class, days_before, recommendation,
     recommendation_reason) = [html.escape(str(v)) for v in values]

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
            <div class="ai-meta">{airline} &nbsp;•&nbsp; {travel_class} &nbsp;•&nbsp; {days_before} days before departure</div>
        </div>
        <div class="ai-top-row">
            <div class="gauge-wrap">{gauge_svg}<div class="gauge-label">Deal Score</div></div>
            <div class="ai-headline-price">PREDICTED FARE<span class="value">{predicted_price}</span>
                <span class="delta {delta_class}">{difference} ({difference_percent}) vs median</span></div>
        </div>
        <div class="ai-card-grid">
            <div class="ai-card"><div class="ai-card-label">Historical Median</div><div class="ai-card-value">{historical_median}</div><div class="ai-card-sub">Comparable historical flights</div></div>
            <div class="ai-card"><div class="ai-card-label">Price Percentile</div><div class="ai-card-value">{price_percentile}</div><div class="ai-card-sub">Position vs. historical fares</div></div>
            <div class="ai-card"><div class="ai-card-label">Comparable Flights</div><div class="ai-card-value">{comparable_flights}</div><div class="ai-card-sub">Matches used in this analysis</div></div>
        </div>
        {drivers_html}
        <div class="ai-decision {decision_class}">
            <div class="decision-label">💡 AI DECISION</div>
            <div class="decision-main">{decision_icon} {recommendation}</div>
            <div class="decision-reason">{recommendation_reason}</div>
        </div>
        <div class="ai-model"><span>🧠 MODEL</span><strong>{model_name}</strong></div>
    </div>"""


def gradio_analyzer_visual(*args):
    raw_result = gradio_analyzer(*args)
    return format_ai_output_html(raw_result)


def simulate_scenario(scenario_stops, scenario_class, scenario_days, airline, source, destination,
                       departure_date, current_class, current_days, departure_hour, arrival_hour,
                       duration, current_stops, distance, season, weekday, aircraft, booking_channel, passengers):
    scenario_price = predict_flight_price(
        airline=airline, source=source, destination=destination, departure_date=departure_date,
        departure_hour=int(departure_hour), arrival_hour=int(arrival_hour), duration_minutes=float(duration),
        total_stops=int(scenario_stops), distance_km=float(distance), travel_class=scenario_class,
        days_before_departure=int(scenario_days), season=season, weekday=weekday, aircraft_type=aircraft,
        booking_channel=booking_channel, passenger_count=int(passengers))
    current_price = predict_flight_price(
        airline=airline, source=source, destination=destination, departure_date=departure_date,
        departure_hour=int(departure_hour), arrival_hour=int(arrival_hour), duration_minutes=float(duration),
        total_stops=int(current_stops), distance_km=float(distance), travel_class=current_class,
        days_before_departure=int(current_days), season=season, weekday=weekday, aircraft_type=aircraft,
        booking_channel=booking_channel, passenger_count=int(passengers))

    difference = scenario_price - current_price
    saving = current_price - scenario_price
    pct_change = (difference / current_price * 100) if current_price else 0
    if saving > 0:
        verdict, verdict_text = "🟢 THIS SCENARIO MAY SAVE MONEY", f"Potential saving of ₹{saving:,.2f} ({abs(pct_change):.2f}%)."
    elif saving < 0:
        verdict, verdict_text = "🔴 THIS SCENARIO COSTS MORE", f"This scenario is ₹{abs(saving):,.2f} more expensive ({abs(pct_change):.2f}% higher)."
    else:
        verdict, verdict_text = "🟡 NO PRICE CHANGE", "The predicted price is unchanged."

    return f"""<div class="scenario-dashboard">
        <div class="scenario-header"><div class="ai-eyebrow">🔮 What-If Result</div>
            <div class="scenario-route">{html.escape(str(source))} → {html.escape(str(destination))}</div>
            <div class="scenario-subtitle">Compare your current booking with the scenario you selected.</div></div>
        <div class="scenario-grid">
            <div class="scenario-card"><div class="scenario-label">CURRENT BOOKING</div>
                <div class="scenario-main">{int(current_days)} days · {html.escape(str(current_class))} · {int(current_stops)} stop(s)</div>
                <div class="scenario-price">₹{current_price:,.2f}</div></div>
            <div class="scenario-card scenario-highlight"><div class="scenario-label">WHAT-IF SCENARIO</div>
                <div class="scenario-main">{int(scenario_days)} days · {html.escape(str(scenario_class))} · {int(scenario_stops)} stop(s)</div>
                <div class="scenario-price">₹{scenario_price:,.2f}</div></div>
        </div>
        <div class="scenario-comparison">
            <div><span>PRICE DIFFERENCE</span><strong>₹{difference:,.2f}</strong></div>
            <div><span>POTENTIAL SAVING</span><strong>₹{max(saving,0):,.2f}</strong></div>
            <div><span>PRICE CHANGE</span><strong>{pct_change:+.2f}%</strong></div>
        </div>
        <div class="scenario-verdict"><div class="scenario-verdict-label">🧠 AI VERDICT</div>
            <div class="scenario-verdict-main">{verdict}</div><div class="scenario-verdict-text">{verdict_text}</div></div>
    </div>"""


# ------------------------------------------------------------
# PHASE 2 — AI BOOKING STRATEGY OPTIMIZER (the "killer feature")
# Grid-searches stops x class x days-before-departure and finds
# the cheapest realistic combination, holding route/airline/etc fixed.
# ------------------------------------------------------------

def find_optimal_strategy(airline, source, destination, departure_date, current_class, current_days,
                           departure_hour, arrival_hour, duration, current_stops, distance, season,
                           weekday, aircraft, booking_channel, passengers):
    stop_options = [0, 1, 2]
    class_options = ["Economy", "Premium Economy", "Business", "First"]
    day_options = [7, 14, 30, 45, 60, 90]

    current_price = predict_flight_price(
        airline=airline, source=source, destination=destination, departure_date=departure_date,
        departure_hour=int(departure_hour), arrival_hour=int(arrival_hour), duration_minutes=float(duration),
        total_stops=int(current_stops), distance_km=float(distance), travel_class=current_class,
        days_before_departure=int(current_days), season=season, weekday=weekday, aircraft_type=aircraft,
        booking_channel=booking_channel, passenger_count=int(passengers))

    best = None
    for stops, cls, days in itertools.product(stop_options, class_options, day_options):
        price = predict_flight_price(
            airline=airline, source=source, destination=destination, departure_date=departure_date,
            departure_hour=int(departure_hour), arrival_hour=int(arrival_hour), duration_minutes=float(duration),
            total_stops=stops, distance_km=float(distance), travel_class=cls,
            days_before_departure=days, season=season, weekday=weekday, aircraft_type=aircraft,
            booking_channel=booking_channel, passenger_count=int(passengers))
        if best is None or price < best["price"]:
            best = {"stops": stops, "class": cls, "days": days, "price": price}

    saving = current_price - best["price"]
    pct = (saving / current_price * 100) if current_price else 0

    return f"""<div class="optimizer-dashboard">
        <div class="optimizer-header"><div class="ai-eyebrow">🏆 Optimal Booking Strategy</div>
            <div class="optimizer-route">{html.escape(str(source))} → {html.escape(str(destination))}</div>
            <div class="ai-meta">Searched {len(stop_options)*len(class_options)*len(day_options)} combinations of stops, class and booking window</div></div>
        <div class="optimizer-grid">
            <div class="optimizer-card"><div class="optimizer-card-label">STOPS</div><div class="optimizer-card-value">{best['stops']}</div></div>
            <div class="optimizer-card"><div class="optimizer-card-label">CLASS</div><div class="optimizer-card-value">{html.escape(best['class'])}</div></div>
            <div class="optimizer-card"><div class="optimizer-card-label">BOOK BY</div><div class="optimizer-card-value">{best['days']}d before</div></div>
        </div>
        <div class="optimizer-result">
            <div class="optimizer-result-label">💰 ESTIMATED PRICE</div>
            <div class="optimizer-result-main">₹{best['price']:,.2f}</div>
            <div class="scenario-verdict-text">Potential saving of ₹{max(saving,0):,.2f} ({pct:.1f}%) vs your current selection (₹{current_price:,.2f}).</div>
        </div>
    </div>"""


# ------------------------------------------------------------
# MODEL PERFORMANCE SECTION (reads phase1_model_evaluation.py output)
# ------------------------------------------------------------

def render_model_performance_html():
    path = "ai_travel_model_comparison.csv"
    if not os.path.exists(path):
        return """<div class="ai-empty"><div style="font-size:22px;">📊</div>
        <div class="ai-empty-title">Model Comparison Not Found</div>
        <div class="ai-empty-text">Run phase1_model_evaluation.py first to generate ai_travel_model_comparison.csv.</div></div>"""
    df = pd.read_csv(path).sort_values("R2", ascending=False).reset_index(drop=True)
    best_name = df.iloc[0]["Model"]
    rows = ""
    for _, r in df.iterrows():
        is_best = r["Model"] == best_name
        cls = "perf-best" if is_best else ""
        star = " 🏆" if is_best else ""
        rows += f"<tr><td class='{cls}'>{html.escape(str(r['Model']))}{star}</td><td>₹{r['MAE']:,.2f}</td><td>{r['RMSE']:,.2f}</td><td>{r['R2']:.3f}</td></tr>"
    return f"""<div class="perf-dashboard"><div class="perf-header">
        <div class="ai-eyebrow">📊 Model Performance</div>
        <div class="ai-meta" style="margin-top:8px;">{html.escape(best_name)} was selected as the final model because it achieved the best validation R² and lowest MAE among the evaluated models.</div>
        <table class="perf-table"><thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead><tbody>{rows}</tbody></table>
    </div></div>"""


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(title="AI Travel Analyst", css=custom_css) as demo:

    gr.HTML("""<div class="hero"><div class="hero-eyebrow">FLIGHT INTELLIGENCE SYSTEM</div>
        <div class="hero-title">✈️ AI TRAVEL <span class="accent">ANALYST</span></div>
        <div class="hero-subtitle">AI Travel Decision Support System</div>
        <div class="hero-flow">PREDICT <span class="sep">→</span> EXPLAIN <span class="sep">→</span> COMPARE <span class="sep">→</span> SIMULATE <span class="sep">→</span> OPTIMIZE</div></div>""")

    gr.HTML("""<div class="section-title"><h2>📊 Model Performance</h2><p>How the final model was chosen.</p></div>""")
    gr.HTML(render_model_performance_html())

    gr.Markdown("### Enter your flight details\nThe AI will estimate the flight price, explain the prediction, compare it with historical patterns, and recommend the best booking strategy.")

    gr.HTML("""<div class="section-title"><h2>✈️ Flight Information</h2><p>Tell the AI about the flight you are considering.</p></div>""")
    with gr.Row():
        airline_input = gr.Dropdown(choices=airlines, label="Airline", value="qatar airways" if "qatar airways" in airlines else airlines[0], interactive=True)
        source_input = gr.Dropdown(choices=sources, label="Source", value="Delhi" if "Delhi" in sources else sources[0], interactive=True)
        destination_input = gr.Dropdown(choices=destinations, label="Destination", value="London" if "London" in destinations else destinations[0], interactive=True)
    with gr.Row():
        travel_class_input = gr.Dropdown(choices=travel_classes, label="Travel Class", value="Economy" if "Economy" in travel_classes else travel_classes[0], interactive=True)
        departure_date_input = gr.Textbox(label="Departure Date", value="2026-12-15", interactive=True)
        days_input = gr.Slider(minimum=1, maximum=180, value=30, step=1, label="Days Before Departure", interactive=True)

    gr.HTML("""<div class="section-title"><h2>🕐 Flight Details</h2><p>Configure timing, duration, stops and passenger information.</p></div>""")
    with gr.Row():
        departure_hour_input = gr.Slider(minimum=0, maximum=23, value=10, step=1, label="Departure Hour", interactive=True)
        arrival_hour_input = gr.Slider(minimum=0, maximum=23, value=18, step=1, label="Arrival Hour", interactive=True)
        duration_input = gr.Number(value=540, label="Duration (minutes)", interactive=True)
    with gr.Row():
        stops_input = gr.Slider(minimum=0, maximum=2, value=1, step=1, label="Total Stops", interactive=True)
        distance_input = gr.Number(value=6700, label="Distance (km)", interactive=True)
        passengers_input = gr.Slider(minimum=1, maximum=6, value=1, step=1, label="Passengers", interactive=True)

    gr.HTML("""<div class="section-title"><h2>🌍 Travel Conditions</h2><p>Additional factors used by the machine-learning model.</p></div>""")
    with gr.Row():
        season_input = gr.Dropdown(choices=seasons, label="Season", value="Winter" if "Winter" in seasons else seasons[0], interactive=True)
        weekday_input = gr.Dropdown(choices=weekdays, label="Weekday", value="Tuesday" if "Tuesday" in weekdays else weekdays[0], interactive=True)
        aircraft_input = gr.Dropdown(choices=aircraft_types, label="Aircraft Type", value="Boeing 777" if "Boeing 777" in aircraft_types else aircraft_types[0], interactive=True)
        booking_input = gr.Dropdown(choices=booking_channels, label="Booking Channel", value="Website" if "Website" in booking_channels else booking_channels[0], interactive=True)

    analyze_button = gr.Button("🔮 ANALYZE FLIGHT", variant="primary")
    output_box = gr.HTML(value=EMPTY_STATE_HTML, label="AI Travel Intelligence")

    gr.HTML("""<div class="section-title"><h2>🧠 AI What-If Travel Simulator</h2><p>Defaults below are set differently from your current booking above, so you can see a real contrast immediately — then tweak either side.</p></div>""")
    with gr.Row():
        # NOTE: defaults intentionally differ from the main form (30 days / Economy / 1 stop)
        # so the demo shows an actual price change on first load, per Phase 1.4.
        scenario_stops = gr.Slider(minimum=0, maximum=2, step=1, value=0, label="Number of Stops", interactive=True)
        scenario_class = gr.Dropdown(choices=["Economy", "Premium Economy", "Business", "First"], value="Economy", label="Travel Class", interactive=True)
        scenario_days = gr.Slider(minimum=1, maximum=180, step=1, value=60, label="Days Before Departure", interactive=True)
    simulate_button = gr.Button("🔮 SIMULATE SCENARIO", variant="primary")
    scenario_output = gr.HTML()

    gr.HTML("""<div class="section-title"><h2>🏆 AI Booking Strategy Optimizer</h2><p>Don't guess scenarios — let the AI search all of them and find the cheapest realistic strategy for this route.</p></div>""")
    optimize_button = gr.Button("🏆 FIND OPTIMAL STRATEGY", variant="primary")
    optimizer_output = gr.HTML()

    gr.HTML("""<div class="footer">AI TRAVEL ANALYST · Predict → Explain → Compare → Simulate → Optimize</div>""")

    analyze_button.click(
        fn=gradio_analyzer_visual,
        inputs=[airline_input, source_input, destination_input, departure_date_input, travel_class_input,
                days_input, departure_hour_input, arrival_hour_input, duration_input, stops_input,
                distance_input, season_input, weekday_input, aircraft_input, booking_input, passengers_input],
        outputs=output_box
    )

    simulate_button.click(
        fn=simulate_scenario,
        inputs=[scenario_stops, scenario_class, scenario_days, airline_input, source_input, destination_input,
                departure_date_input, travel_class_input, days_input, departure_hour_input, arrival_hour_input,
                duration_input, stops_input, distance_input, season_input, weekday_input, aircraft_input,
                booking_input, passengers_input],
        outputs=scenario_output
    )

    optimize_button.click(
        fn=find_optimal_strategy,
        inputs=[airline_input, source_input, destination_input, departure_date_input, travel_class_input,
                days_input, departure_hour_input, arrival_hour_input, duration_input, stops_input,
                distance_input, season_input, weekday_input, aircraft_input, booking_input, passengers_input],
        outputs=optimizer_output
    )

print("Dashboard v2 created successfully!")
demo.launch()
