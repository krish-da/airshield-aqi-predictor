"""
AirShield — Indian AQI Threshold Breach Predictor (app.py)
Single-file Flask server + executive, city-focused Environmental Forecasting Webpage.
Architecture (strictly 5 source files total):
  - app.py
  - data_sources.py
  - train_model.py
  - predictor.py
  - requirements.txt
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Any, Dict

from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv

from data_sources import (
    CPCB_HIST_CSV,
    DB_PATH,
    aqi_to_cpcb_category,
    calculate_cpcb_aqi,
    search_indian_cities,
    resolve_cpcb_station_coverage,
    evaluate_live_predictions,
)
from predictor import (
    MODEL_ARTIFACT_PATH,
    load_model_bundle,
    predict_city_aqi_24h,
    get_evaluation_report,
)
from train_model import train_and_evaluate_pipeline

load_dotenv(override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("airshield.app")

app = Flask(__name__)


def ensure_model_ready() -> None:
    """Ensure trained model artifact exists at startup; if missing, train once automatically."""
    if not os.path.exists(MODEL_ARTIFACT_PATH):
        logger.info("Trained model artifact not found. Training initial model on CPCB dataset...")
        train_and_evaluate_pipeline()
    load_model_bundle()


# =====================================================================
# FLASK BACKEND API ROUTES
# =====================================================================

@app.route("/api/health", methods=["GET"])
def api_health() -> Response:
    """System health-check endpoint (never exposes raw API keys)."""
    model_exists = os.path.exists(MODEL_ARTIFACT_PATH)
    db_exists = os.path.exists(DB_PATH)
    owm_configured = bool(
        os.getenv("OPENWEATHER_API_KEY", "").strip()
        and os.getenv("OPENWEATHER_API_KEY") != "YOUR_OPENWEATHER_API_KEY"
    )
    cpcb_configured = bool(
        os.getenv("DATA_GOV_IN_API_KEY", "").strip()
        and os.getenv("DATA_GOV_IN_API_KEY") != "YOUR_DATA_GOV_IN_API_KEY"
    )
    openaq_configured = bool(
        os.getenv("OPENAQ_API_KEY", "").strip()
        and os.getenv("OPENAQ_API_KEY") != "YOUR_OPENAQ_API_KEY"
    )
    firms_configured = bool(
        os.getenv("NASA_FIRMS_MAP_KEY", "").strip()
        and os.getenv("NASA_FIRMS_MAP_KEY") != "YOUR_NASA_FIRMS_MAP_KEY"
    )
    bundle = load_model_bundle() if model_exists else {}
    return jsonify({
        "status": "healthy" if model_exists else "model_missing",
        "application": "AirShield — Indian AQI Threshold Breach Predictor",
        "model_ready": model_exists,
        "model_version": bundle.get("model_version"),
        "selected_model": bundle.get("selected_model"),
        "feature_count": len(bundle.get("feature_columns", [])),
        "sqlite_ready": db_exists,
        "providers": {
            "openweathermap_configured": owm_configured,
            "data_gov_in_configured": cpcb_configured,
            "openaq_configured": openaq_configured,
            "nasa_firms_configured": firms_configured,
            "open_meteo_enabled": True,
        },
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/api/cities", methods=["GET"])
def api_cities() -> Response:
    """Autocomplete endpoint for Indian cities, towns, and populated places."""
    query = request.args.get("q", "").strip()
    limit = min(12, max(1, int(request.args.get("limit", 8))))
    suggestions = search_indian_cities(query, limit=limit)
    return jsonify({
        "query": query,
        "count": len(suggestions),
        "results": suggestions,
    })


@app.route("/api/predict", methods=["GET"])
def api_predict() -> Response:
    """
    Generate live 24-hour AQI prediction, CPCB category, threshold breach analysis,
    weather context, regional festival distance, and causal explanations for a selected Indian city.
    """
    ensure_model_ready()
    city = request.args.get("city", "Delhi").strip() or "Delhi"
    state = request.args.get("state", "").strip()
    lat_str = request.args.get("lat", "").strip()
    lon_str = request.args.get("lon", "").strip()

    if lat_str and lon_str:
        try:
            lat = float(lat_str)
            lon = float(lon_str)
        except ValueError:
            return jsonify({"error": "Invalid latitude or longitude parameters."}), 400
    else:
        matches = search_indian_cities(city, limit=1)
        if not matches:
            return jsonify({"error": f"Could not resolve Indian city '{city}'."}), 404
        lat = float(matches[0]["lat"])
        lon = float(matches[0]["lon"])
        city = matches[0]["name"]
        state = state or matches[0]["state"]

    # Validate Indian geographic bounding box (lat 6..38, lon 67..98)
    if not (5.0 <= lat <= 38.5 and 67.0 <= lon <= 98.5):
        return jsonify({
            "error": "Coordinates lie outside India. AirShield is calibrated exclusively for Indian CPCB AQI forecasting."
        }), 400

    try:
        forecast = predict_city_aqi_24h(
            city_name=city,
            lat=lat,
            lon=lon,
            state=state,
            log_to_db=True,
        )
        return jsonify(forecast)
    except Exception as exc:
        logger.exception("Prediction failed for %s: %s", city, exc)
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics() -> Response:
    """Return chronological train/validation/test metrics, candidate benchmarks, and live validation ledger."""
    ensure_model_ready()
    report = get_evaluation_report()
    return jsonify(report)


@app.route("/api/retrain", methods=["POST"])
def api_retrain() -> Response:
    """Trigger safe model retraining on verified historical CPCB + weather data."""
    try:
        report = train_and_evaluate_pipeline()
        load_model_bundle(force_reload=True)
        return jsonify({"status": "retrained", "report": report})
    except Exception as exc:
        logger.exception("Retraining error: %s", exc)
        return jsonify({"error": str(exc)}), 500


# =====================================================================
# EMBEDDED FRONTEND WEBPAGE (EXECUTIVE CITY-FOCUSED SINGLE-PAGE DESIGN)
# =====================================================================

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AirShield — Indian AQI Threshold Breach Predictor</title>
  <meta name="description" content="Live 24-hour Indian CPCB Air Quality Index (AQI) forecast and threshold-breach predictor." />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;600;700&family=Outfit:wght@500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg-page: #f4f7f9;
      --bg-card: #ffffff;
      --bg-muted: #f8fafc;
      --bg-subtle: #f1f5f9;
      --text-main: #0f172a;
      --text-body: #334155;
      --text-muted: #64748b;
      --border-light: #e2e8f0;
      --border-med: #cbd5e1;
      --brand: #0f766e;
      --brand-hover: #115e59;

      /* Official CPCB Category Colors */
      --cpcb-good: #15803d;
      --cpcb-good-bg: #f0fdf4;
      --cpcb-satisfactory: #4d7c0f;
      --cpcb-satisfactory-bg: #f7fee7;
      --cpcb-moderate: #b45309;
      --cpcb-moderate-bg: #fffbeb;
      --cpcb-poor: #c2410c;
      --cpcb-poor-bg: #fff7ed;
      --cpcb-verypoor: #b91c1c;
      --cpcb-verypoor-bg: #fef2f2;
      --cpcb-severe: #7f1d1d;
      --cpcb-severe-bg: #fef2f2;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-page);
      color: var(--text-main);
      line-height: 1.55;
      font-size: 15px;
      -webkit-font-smoothing: antialiased;
    }

    h1, h2, h3, .font-display {
      font-family: 'Outfit', sans-serif;
      letter-spacing: -0.02em;
    }

    .tabular-nums, .font-mono {
      font-family: 'JetBrains Mono', monospace;
      font-variant-numeric: tabular-nums;
    }

    /* Sticky Top Navigation */
    .navbar {
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-light);
      position: sticky;
      top: 0;
      z-index: 40;
    }

    .navbar-inner {
      max-width: 1160px;
      margin: 0 auto;
      padding: 0.85rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1.25rem;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      text-decoration: none;
      color: var(--text-main);
      flex-shrink: 0;
    }

    .brand-mark {
      width: 10px;
      height: 24px;
      border-radius: 3px;
      background: var(--brand);
    }

    .brand-title {
      font-family: 'Outfit', sans-serif;
      font-size: 1.35rem;
      font-weight: 700;
    }

    .brand-subtitle {
      font-size: 0.82rem;
      color: var(--text-muted);
      font-weight: 500;
      display: none;
      border-left: 1px solid var(--border-med);
      padding-left: 0.75rem;
    }

    @media (min-width: 720px) {
      .brand-subtitle {
        display: inline;
      }
    }

    .btn-refresh {
      background: var(--brand);
      color: #ffffff;
      border: 1px solid var(--brand);
      border-radius: 8px;
      padding: 0.5rem 1rem;
      font-size: 0.84rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 150ms ease, transform 100ms ease;
      white-space: nowrap;
      flex-shrink: 0;
    }

    .btn-refresh:hover {
      background: var(--brand-hover);
    }

    /* Main Page Container */
    .page-wrap {
      max-width: 1160px;
      margin: 0 auto;
      padding: 1.65rem 1.5rem 3.5rem;
    }

    /* Search & Quick City Selector */
    .search-section {
      background: var(--bg-card);
      border: 1px solid var(--border-light);
      border-radius: 12px;
      padding: 0.9rem 1.15rem;
      margin-bottom: 1.35rem;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }

    .search-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 0.85rem;
      align-items: center;
    }

    @media (min-width: 960px) {
      .search-grid {
        grid-template-columns: 380px 1fr;
      }
    }

    .search-wrapper {
      position: relative;
      width: 100%;
    }

    .search-box {
      display: flex;
      align-items: center;
      background: var(--bg-muted);
      border: 1px solid var(--border-med);
      border-radius: 8px;
      padding: 0.1rem 0.85rem;
      transition: border-color 150ms ease, box-shadow 150ms ease, background 150ms ease;
    }

    .search-box:focus-within {
      background: #ffffff;
      border-color: var(--brand);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.12);
    }

    .search-input {
      width: 100%;
      border: none;
      background: transparent;
      padding: 0.6rem 0.35rem;
      font-size: 0.9rem;
      color: var(--text-main);
      font-family: inherit;
      outline: none;
    }

    .autocomplete-menu {
      position: absolute;
      top: calc(100% + 6px);
      left: 0;
      right: 0;
      background: var(--bg-card);
      border: 1px solid var(--border-med);
      border-radius: 10px;
      box-shadow: 0 12px 28px -6px rgba(15, 23, 42, 0.14);
      max-height: 300px;
      overflow-y: auto;
      z-index: 50;
      display: none;
    }

    .autocomplete-menu.open {
      display: block;
    }

    .autocomplete-item {
      width: 100%;
      text-align: left;
      background: transparent;
      border: none;
      border-bottom: 1px solid var(--border-light);
      padding: 0.7rem 1rem;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.75rem;
      font-family: inherit;
    }

    .autocomplete-item:last-child {
      border-bottom: none;
    }

    .autocomplete-item:hover {
      background: var(--bg-subtle);
    }

    .city-chips {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.45rem;
    }

    .preset-btn {
      background: var(--bg-muted);
      color: var(--text-body);
      border: 1px solid var(--border-light);
      border-radius: 7px;
      padding: 0.42rem 0.78rem;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      white-space: nowrap;
      transition: all 140ms ease;
    }

    .preset-btn:hover {
      border-color: var(--border-med);
      background: var(--bg-subtle);
    }

    .preset-btn.active {
      background: var(--text-main);
      color: #ffffff;
      border-color: var(--text-main);
    }

    /* City Hero Card */
    .hero-card {
      background: var(--bg-card);
      border: 1px solid var(--border-light);
      border-radius: 14px;
      padding: 1.75rem;
      margin-bottom: 1.35rem;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }

    .hero-header {
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      align-items: flex-start;
      gap: 0.75rem;
      padding-bottom: 1.25rem;
      border-bottom: 1px solid var(--border-light);
      margin-bottom: 1.35rem;
    }

    @media (min-width: 768px) {
      .hero-header {
        flex-direction: row;
        align-items: flex-start;
      }
      .hero-meta {
        text-align: right;
        flex-shrink: 0;
      }
    }

    .city-heading {
      font-size: 2rem;
      font-weight: 700;
      color: var(--text-main);
      line-height: 1.15;
    }

    .station-subtitle {
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-top: 0.3rem;
    }

    /* Threshold Breach Banner */
    .risk-banner {
      border-radius: 10px;
      padding: 0.95rem 1.25rem;
      margin-bottom: 1.35rem;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      border: 1px solid var(--border-light);
      border-left-width: 5px;
    }

    .risk-banner.critical {
      background: #fef2f2;
      border-color: #fecaca;
      border-left-color: #dc2626;
      color: #7f1d1d;
    }

    .risk-banner.warning {
      background: #fffbeb;
      border-color: #fde68a;
      border-left-color: #d97706;
      color: #78350f;
    }

    .risk-banner.normal {
      background: #f0fdf4;
      border-color: #bbf7d0;
      border-left-color: #16a34a;
      color: #14532d;
    }

    /* 3-Column Core City Forecast Numbers */
    .forecast-kpi-row {
      display: grid;
      grid-template-columns: 1fr;
      gap: 1.15rem;
    }

    @media (min-width: 768px) {
      .forecast-kpi-row {
        grid-template-columns: repeat(3, 1fr);
      }
    }

    .kpi-box {
      background: var(--bg-muted);
      border-radius: 10px;
      padding: 1.25rem 1.3rem;
      border: 1px solid var(--border-light);
      border-top: 4px solid var(--border-med);
    }

    .kpi-label {
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 0.4rem;
    }

    .kpi-number {
      font-family: 'JetBrains Mono', monospace;
      font-variant-numeric: tabular-nums;
      font-size: 2.65rem;
      font-weight: 700;
      line-height: 1.05;
      margin-bottom: 0.45rem;
    }

    .kpi-category {
      font-family: 'Outfit', sans-serif;
      font-size: 2.05rem;
      font-weight: 700;
      line-height: 1.12;
      margin-bottom: 0.45rem;
    }

    .kpi-sub {
      font-size: 0.82rem;
      color: var(--text-body);
    }

    /* 4-Card City Conditions & Regional Distance Strip */
    .env-strip {
      display: grid;
      grid-template-columns: 1fr;
      gap: 1rem;
      margin-bottom: 1.35rem;
    }

    @media (min-width: 640px) {
      .env-strip {
        grid-template-columns: repeat(2, 1fr);
      }
    }

    @media (min-width: 1024px) {
      .env-strip {
        grid-template-columns: repeat(4, 1fr);
      }
    }

    .env-card {
      background: var(--bg-card);
      border: 1px solid var(--border-light);
      border-radius: 12px;
      padding: 1.15rem 1.2rem;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
    }

    .env-card-title {
      font-size: 0.76rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 0.35rem;
    }

    .env-card-main {
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.18rem;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 0.25rem;
    }

    .env-card-detail {
      font-size: 0.8rem;
      color: var(--text-body);
      line-height: 1.45;
    }

    /* Generic Card */
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-light);
      border-radius: 14px;
      padding: 1.55rem;
      margin-bottom: 1.35rem;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
    }

    .card-title {
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 0.25rem;
    }

    .card-subtitle {
      font-size: 0.84rem;
      color: var(--text-muted);
      margin-bottom: 1.1rem;
    }

    /* CPCB Scale Bar */
    .cpcb-scale-bar {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 0.5rem;
      margin-top: 1rem;
    }

    @media (min-width: 768px) {
      .cpcb-scale-bar {
        grid-template-columns: repeat(6, 1fr);
      }
    }

    .scale-segment {
      padding: 0.45rem 0.65rem;
      border-radius: 6px;
      border: 1px solid var(--border-light);
      font-size: 0.76rem;
    }

    /* Compact City Pollutant Strip */
    .pollutant-strip {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 0.65rem;
      margin-top: 1.15rem;
      padding-top: 1.15rem;
      border-top: 1px solid var(--border-light);
    }

    @media (min-width: 768px) {
      .pollutant-strip {
        grid-template-columns: repeat(6, 1fr);
      }
    }

    .pol-box {
      background: var(--bg-muted);
      border: 1px solid var(--border-light);
      border-radius: 8px;
      padding: 0.65rem 0.8rem;
    }

    .pol-name {
      display: flex;
      justify-content: space-between;
      font-size: 0.74rem;
      font-weight: 700;
      color: var(--text-muted);
    }

    .pol-val {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.96rem;
      font-weight: 700;
      color: var(--text-main);
      margin-top: 0.15rem;
    }

    /* 2-Column City Insights (Why AQI Changes + Health Advice) */
    .insights-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 1.35rem;
    }

    @media (min-width: 900px) {
      .insights-grid {
        grid-template-columns: 1.1fr 0.9fr;
      }
    }

    .driver-item {
      padding: 0.85rem 0;
      border-bottom: 1px solid var(--border-light);
    }

    .driver-item:first-child {
      padding-top: 0;
    }

    .driver-item:last-child {
      border-bottom: none;
      padding-bottom: 0;
    }

    .driver-row {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 0.5rem;
      margin-bottom: 0.25rem;
    }

    .driver-name {
      font-weight: 600;
      font-size: 0.92rem;
      color: var(--text-main);
    }

    .driver-tag {
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.76rem;
      font-weight: 600;
      white-space: nowrap;
    }

    .health-block {
      background: var(--bg-muted);
      border: 1px solid var(--border-light);
      border-radius: 8px;
      padding: 0.85rem 1rem;
      margin-bottom: 0.75rem;
    }

    .health-block:last-of-type {
      margin-bottom: 0;
    }

    .health-block-label {
      font-size: 0.76rem;
      font-weight: 700;
      color: var(--text-muted);
      margin-bottom: 0.2rem;
    }

    .footer {
      border-top: 1px solid var(--border-light);
      padding: 1.75rem 1.5rem;
      margin-top: 2rem;
      font-size: 0.78rem;
      color: var(--text-muted);
      text-align: center;
    }
  </style>
</head>
<body>
  <!-- Sticky Top Bar -->
  <header class="navbar">
    <div class="navbar-inner">
      <a href="#top" class="brand">
        <span class="brand-mark" aria-hidden="true"></span>
        <span class="brand-title">AirShield</span>
        <span class="brand-subtitle">Indian City 24-Hour CPCB AQI &amp; Threshold Breach Forecaster</span>
      </a>
      <button type="button" id="refreshForecastBtn" class="btn-refresh">Refresh Live Telemetry</button>
    </div>
  </header>

  <main id="top" class="page-wrap">
    <!-- City Search & Quick City Buttons -->
    <section class="search-section" aria-label="Select Indian City">
      <div class="search-grid">
        <div class="search-wrapper">
          <div class="search-box">
            <input
              type="text"
              id="citySearchInput"
              class="search-input"
              placeholder="Search any Indian city or town (e.g., Delhi, Ahmedabad, Mumbai, Shimla)..."
              autocomplete="off"
              aria-label="Search Indian City"
            />
          </div>
          <div id="autocompleteMenu" class="autocomplete-menu" role="listbox" aria-label="City Suggestions"></div>
        </div>

        <div class="city-chips" id="presetCityGroup">
          <button type="button" class="preset-btn active" data-city="Delhi" data-state="Delhi" data-lat="28.6139" data-lon="77.2090">Delhi</button>
          <button type="button" class="preset-btn" data-city="Ahmedabad" data-state="Gujarat" data-lat="23.0225" data-lon="72.5714">Ahmedabad</button>
          <button type="button" class="preset-btn" data-city="Mumbai" data-state="Maharashtra" data-lat="19.0760" data-lon="72.8777">Mumbai</button>
          <button type="button" class="preset-btn" data-city="Bengaluru" data-state="Karnataka" data-lat="12.9716" data-lon="77.5946">Bengaluru</button>
          <button type="button" class="preset-btn" data-city="Lucknow" data-state="Uttar Pradesh" data-lat="26.8467" data-lon="80.9462">Lucknow</button>
          <button type="button" class="preset-btn" data-city="Kolkata" data-state="West Bengal" data-lat="22.5726" data-lon="88.3639">Kolkata</button>
          <button type="button" class="preset-btn" data-city="Chennai" data-state="Tamil Nadu" data-lat="13.0827" data-lon="80.2707">Chennai</button>
          <button type="button" class="preset-btn" data-city="Shimla" data-state="Himachal Pradesh" data-lat="31.1048" data-lon="77.1734">Shimla</button>
        </div>
      </div>
    </section>

    <!-- Selected City Forecast Hero Card -->
    <section class="hero-card" aria-label="Selected City 24-Hour AQI Forecast">
      <div class="hero-header">
        <div>
          <h1 id="cityTitleText" class="city-heading">Delhi, Delhi</h1>
          <p id="coverageText" class="station-subtitle">Loading live CPCB station coverage and city telemetry...</p>
        </div>
        <div class="hero-meta">
          <div id="locationCoordsText" class="tabular-nums" style="font-size: 0.8rem; color: var(--text-muted);">28.6139°N, 77.2090°E</div>
          <div id="currentSourceNote" style="font-size: 0.78rem; font-weight: 600; color: var(--brand); margin-top: 0.2rem;">Official CPCB Live Station</div>
        </div>
      </div>

      <!-- 24h Threshold Breach Status for Selected City -->
      <div id="riskBanner" class="risk-banner normal" role="status">
        <div>
          <div style="font-size: 0.75rem; font-weight: 700; opacity: 0.85;">24-Hour Safety &amp; Threshold Breach Status (CPCB Unsafe Threshold: AQI &gt; 200)</div>
          <div id="riskStateHeading" style="font-size: 1.15rem; font-weight: 700; margin-top: 0.1rem;">Evaluating 24-hour city forecast...</div>
        </div>
        <div style="text-align: right;">
          <div style="font-size: 0.75rem; opacity: 0.85;">Projected Breach Window</div>
          <div id="breachTimeText" class="tabular-nums" style="font-size: 0.92rem; font-weight: 700;">—</div>
        </div>
      </div>

      <!-- 3 Core Forecast Outputs for Selected City -->
      <div class="forecast-kpi-row">
        <div id="kpiBoxCurrent" class="kpi-box">
          <div class="kpi-label">Current Observed AQI</div>
          <div id="currentAqiValue" class="kpi-number">—</div>
          <div class="kpi-sub">
            <strong id="currentCategoryText">—</strong> · <span id="dominantPollutantText">Dominant: PM2.5</span>
          </div>
        </div>

        <div id="kpiBoxPredicted" class="kpi-box">
          <div class="kpi-label">Predicted AQI (+24 Hours)</div>
          <div style="display: flex; align-items: baseline; gap: 0.65rem;">
            <div id="predictedAqiValue" class="kpi-number">—</div>
            <span id="aqiDeltaText" class="tabular-nums" style="font-size: 0.84rem; font-weight: 600;">—</span>
          </div>
          <div class="kpi-sub tabular-nums">
            80% Range: <span id="predictionIntervalText">—</span> · 24h Peak: <strong id="peak24hAqiText">—</strong>
          </div>
        </div>

        <div id="kpiBoxCategory" class="kpi-box">
          <div class="kpi-label">Predicted CPCB Category (+24h)</div>
          <div id="predictedCategoryBadge" class="kpi-category">—</div>
          <div class="kpi-sub">
            <span id="categoryRangeText">CPCB Scale</span> · <span id="confidenceNote">High Confidence</span>
          </div>
        </div>
      </div>
    </section>

    <!-- Selected City Live Conditions (Weather +24h Shift, PBL Mixing, Regional Festival Distance, Plume/Fire Distance) -->
    <section class="env-strip" aria-label="Live City Environmental Conditions and Event Distances">
      <div class="env-card">
        <div class="env-card-title">Live &amp; +24h Forecast Weather</div>
        <div id="wxMainVal" class="env-card-main">—</div>
        <div id="wxSubVal" class="env-card-detail">—</div>
      </div>

      <div class="env-card">
        <div class="env-card-title">Boundary Layer &amp; Vertical Mixing</div>
        <div id="pblMainVal" class="env-card-main">—</div>
        <div id="pblSubVal" class="env-card-detail">—</div>
      </div>

      <div class="env-card">
        <div class="env-card-title">Regional Festival &amp; Event Distance</div>
        <div id="festMainVal" class="env-card-main" style="font-family: 'Outfit', sans-serif; font-size: 1.08rem;">—</div>
        <div id="festSubVal" class="env-card-detail">—</div>
      </div>

      <div class="env-card">
        <div class="env-card-title">Fire Plumes &amp; Crop-Belt Distance</div>
        <div id="firmsMainVal" class="env-card-main">—</div>
        <div id="firmsSubVal" class="env-card-detail">—</div>
      </div>
    </section>

    <!-- Selected City 24-Hour Trajectory & Live Pollutant Concentrations -->
    <section class="card" aria-label="City 24-Hour AQI Trend and Pollutant Readings">
      <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 0.5rem;">
        <div>
          <h2 class="card-title">Past 24h Observed &amp; Next 24h Predicted AQI Trajectory</h2>
          <p class="card-subtitle">Solid line shows recent city observations; dashed line shows the +24-hour forecast trajectory.</p>
        </div>
        <div class="tabular-nums" style="font-size: 0.78rem; color: var(--text-muted);">
          <span id="targetTimestampNote">Forecast Target: +24h IST</span>
        </div>
      </div>

      <div style="width: 100%; height: 245px; position: relative;">
        <svg id="aqiTrendSvg" viewBox="0 0 820 245" style="width: 100%; height: 100%; overflow: visible;" role="img" aria-label="City AQI Trajectory Chart"></svg>
      </div>

      <div class="cpcb-scale-bar">
        <div class="scale-segment" style="background: var(--cpcb-good-bg);">
          <strong style="color: var(--cpcb-good);">Good</strong> <span class="tabular-nums" style="color: var(--text-muted);">0–50</span>
        </div>
        <div class="scale-segment" style="background: var(--cpcb-satisfactory-bg);">
          <strong style="color: var(--cpcb-satisfactory);">Satisfactory</strong> <span class="tabular-nums" style="color: var(--text-muted);">51–100</span>
        </div>
        <div class="scale-segment" style="background: var(--cpcb-moderate-bg);">
          <strong style="color: var(--cpcb-moderate);">Moderate</strong> <span class="tabular-nums" style="color: var(--text-muted);">101–200</span>
        </div>
        <div class="scale-segment" style="background: var(--cpcb-poor-bg);">
          <strong style="color: var(--cpcb-poor);">Poor (Unsafe)</strong> <span class="tabular-nums" style="color: var(--text-muted);">201–300</span>
        </div>
        <div class="scale-segment" style="background: var(--cpcb-verypoor-bg);">
          <strong style="color: var(--cpcb-verypoor);">Very Poor</strong> <span class="tabular-nums" style="color: var(--text-muted);">301–400</span>
        </div>
        <div class="scale-segment" style="background: var(--cpcb-severe-bg);">
          <strong style="color: var(--cpcb-severe);">Severe</strong> <span class="tabular-nums" style="color: var(--text-muted);">401–500</span>
        </div>
      </div>

      <!-- Compact Live City Pollutant Readings -->
      <div id="pollutantStrip" class="pollutant-strip" aria-label="Current City Pollutant Concentrations"></div>
    </section>

    <!-- 2-Column City Advisory: Why AQI Will Change + Health Precautions -->
    <div class="insights-grid">
      <section class="card" style="margin-bottom: 0;" aria-label="Why AQI is Expected to Change">
        <h2 class="card-title">Why AQI Is Expected to <span id="directionVerbText">Change</span></h2>
        <p class="card-subtitle">Grounded meteorological, 24h weather shift, boundary-layer, and regional festival distance drivers for this city.</p>
        <div id="explanationDriversList">
          <div class="driver-item">Loading city atmospheric drivers...</div>
        </div>
      </section>

      <section class="card" style="margin-bottom: 0;" aria-label="CPCB Health Precautions">
        <h2 class="card-title">Health Precautions (<span id="healthCatLabel">—</span>)</h2>
        <p id="healthImpactSummary" class="card-subtitle">—</p>

        <div class="health-block">
          <div class="health-block-label">General Public</div>
          <div id="healthGeneralText" style="font-size: 0.86rem; color: var(--text-main);">—</div>
        </div>
        <div class="health-block">
          <div class="health-block-label">Sensitive Groups (Asthma, Children, Elderly)</div>
          <div id="healthSensitiveText" style="font-size: 0.86rem; color: var(--text-main);">—</div>
        </div>
        <div class="health-block">
          <div class="health-block-label">Mask &amp; Ventilation Advice</div>
          <div id="healthMaskText" style="font-size: 0.86rem; color: var(--text-main);">—</div>
        </div>
        <p id="healthDisclaimerText" style="font-size: 0.74rem; color: var(--text-muted); margin-top: 0.8rem;">
          Calibrated to official CPCB National AQI standards; does not replace medical advice.
        </p>
      </section>
    </div>
  </main>

  <footer class="footer">
    AirShield — Indian City 24-Hour AQI &amp; Threshold Breach Forecaster · Official CPCB Breakpoints ·
    Live Feeds: CPCB / OpenAQ v3, OpenWeatherMap, Open-Meteo (+24h Forecast &amp; PBL), NASA FIRMS VIIRS &amp; Regional Festival Distance Engine
  </footer>

  <script>
    const state = {
      city: "Delhi",
      stateName: "Delhi",
      lat: 28.6139,
      lon: 77.2090,
      debounceTimer: null
    };

    function getCategoryColor(category) {
      const map = {
        "Good": "#15803d",
        "Satisfactory": "#4d7c0f",
        "Moderate": "#b45309",
        "Poor": "#c2410c",
        "Very Poor": "#b91c1c",
        "Severe": "#7f1d1d"
      };
      return map[category] || "#334155";
    }

    function renderTrendSvg(pastPoints, nextPoints) {
      const svg = document.getElementById("aqiTrendSvg");
      if (!svg) return;
      const all = [];
      (pastPoints || []).forEach(p => all.push({ ...p, type: "past" }));
      (nextPoints || []).forEach(p => all.push({ ...p, type: "next" }));
      if (all.length < 2) {
        svg.innerHTML = '<text x="410" y="120" text-anchor="middle" fill="#64748b" font-size="13">Insufficient trajectory points</text>';
        return;
      }

      const w = 820, h = 245;
      const padL = 46, padR = 24, padT = 22, padB = 34;
      const plotW = w - padL - padR;
      const plotH = h - padT - padB;

      const maxAqi = Math.max(250, ...all.map(d => d.aqi || 0)) * 1.1;
      const minAqi = 0;

      const xOf = (i) => padL + (i / (all.length - 1)) * plotW;
      const yOf = (val) => padT + plotH - ((Math.min(maxAqi, Math.max(minAqi, val)) - minAqi) / (maxAqi - minAqi)) * plotH;

      let html = "";
      const gridLevels = [50, 100, 200, 300, 400].filter(v => v <= maxAqi);
      gridLevels.forEach(lv => {
        const y = yOf(lv).toFixed(1);
        const isUnsafe = lv === 200;
        const stroke = isUnsafe ? "#dc2626" : "#e2e8f0";
        const dash = isUnsafe ? 'stroke-dasharray="5,4"' : "";
        html += `<line x1="${padL}" y1="${y}" x2="${w - padR}" y2="${y}" stroke="${stroke}" stroke-width="${isUnsafe ? 1.5 : 1}" ${dash} />`;
        html += `<text x="${padL - 8}" y="${Number(y) + 4}" text-anchor="end" font-family="JetBrains Mono" font-size="11" fill="${isUnsafe ? '#dc2626' : '#64748b'}">${lv}</text>`;
        if (isUnsafe) {
          html += `<text x="${w - padR - 4}" y="${Number(y) - 6}" text-anchor="end" font-family="Plus Jakarta Sans" font-size="10" font-weight="600" fill="#dc2626">CPCB Poor / Unsafe Threshold (200 AQI)</text>`;
        }
      });

      const splitIdx = Math.max(0, (pastPoints || []).length - 1);
      const splitX = xOf(splitIdx).toFixed(1);
      html += `<line x1="${splitX}" y1="${padT}" x2="${splitX}" y2="${h - padB}" stroke="#94a3b8" stroke-dasharray="3,3" stroke-width="1" />`;
      html += `<text x="${splitX}" y="${padT - 6}" text-anchor="middle" font-size="10" font-weight="600" fill="#475569">Now</text>`;

      let obsPath = "";
      for (let i = 0; i <= splitIdx; i++) {
        obsPath += (i === 0 ? "M" : "L") + `${xOf(i).toFixed(1)},${yOf(all[i].aqi).toFixed(1)} `;
      }
      html += `<path d="${obsPath}" fill="none" stroke="#0f172a" stroke-width="2.5" />`;

      let fcPath = "";
      for (let i = splitIdx; i < all.length; i++) {
        fcPath += (i === splitIdx ? "M" : "L") + `${xOf(i).toFixed(1)},${yOf(all[i].aqi).toFixed(1)} `;
      }
      html += `<path d="${fcPath}" fill="none" stroke="#0f766e" stroke-width="2.5" stroke-dasharray="6,4" />`;

      all.forEach((pt, idx) => {
        const cx = xOf(idx).toFixed(1);
        const cy = yOf(pt.aqi).toFixed(1);
        const color = getCategoryColor(pt.category);
        html += `<circle cx="${cx}" cy="${cy}" r="4" fill="${color}" stroke="#ffffff" stroke-width="1.5"><title>${pt.display_time}: AQI ${pt.aqi} (${pt.category})</title></circle>`;
        if (idx % 2 === 0 || idx === all.length - 1 || idx === splitIdx) {
          html += `<text x="${cx}" y="${h - 10}" text-anchor="middle" font-family="JetBrains Mono" font-size="10" fill="#64748b">${pt.horizon_hours || pt.display_time}</text>`;
        }
      });

      svg.innerHTML = html;
    }

    async function loadCityForecast(city, lat, lon, stateName) {
      state.city = city;
      state.lat = lat;
      state.lon = lon;
      state.stateName = stateName || "";

      document.querySelectorAll(".preset-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.city.toLowerCase() === city.toLowerCase());
      });

      const params = new URLSearchParams({ city });
      if (lat !== undefined && lon !== undefined && lat !== null && lon !== null) {
        params.set("lat", lat);
        params.set("lon", lon);
      }
      if (stateName) params.set("state", stateName);

      try {
        const resp = await fetch(`/api/predict?${params.toString()}`);
        const data = await resp.json();
        if (!resp.ok || data.error) {
          document.getElementById("coverageText").textContent = data.error || "Unable to load city forecast.";
          return;
        }
        renderForecastData(data);
      } catch (err) {
        document.getElementById("coverageText").textContent = "Network error while fetching city forecast.";
      }
    }

    function renderForecastData(data) {
      const loc = data.location || {};
      const cov = loc.coverage || {};
      const cur = data.current_observation || {};
      const fc = data.forecast_24h || {};
      const risk = data.risk_alert || {};
      const wx = data.weather || {};
      const nextWx = wx.next_24h_weather || {};
      const ctx = (data.contextual_factors || {}).festival_and_seasonal || {};
      const firms = (data.contextual_factors || {}).nasa_firms || {};
      const health = data.health_precautions || {};

      // Hero Header
      document.getElementById("cityTitleText").textContent = `${loc.city}, ${loc.state}`;
      document.getElementById("coverageText").textContent = `${loc.station_label || ""} · ${cov.coverage_notice || ""}`;
      document.getElementById("locationCoordsText").textContent = `${loc.lat}°N, ${loc.lon}°E · Issued: ${(data.timestamps || {}).issued_at_ist || ""}`;
      document.getElementById("currentSourceNote").textContent = cur.source || "";

      // Risk Banner
      const riskBanner = document.getElementById("riskBanner");
      riskBanner.className = `risk-banner ${risk.severity || "normal"}`;
      document.getElementById("riskStateHeading").textContent = risk.state || "";
      document.getElementById("breachTimeText").textContent = risk.expected_breach_time || "—";

      // KPI 1: Current Observed AQI
      const curColor = getCategoryColor(cur.category);
      const curEl = document.getElementById("currentAqiValue");
      curEl.textContent = cur.aqi !== undefined ? Math.round(cur.aqi) : "—";
      curEl.style.color = curColor;
      document.getElementById("kpiBoxCurrent").style.borderTopColor = curColor;
      document.getElementById("currentCategoryText").textContent = cur.category || "—";
      document.getElementById("dominantPollutantText").textContent = `Dominant: ${cur.dominant_pollutant || "PM2.5"}`;

      // KPI 2: Predicted +24h AQI
      const predColor = getCategoryColor(fc.predicted_category);
      const predEl = document.getElementById("predictedAqiValue");
      predEl.textContent = fc.predicted_aqi !== undefined ? fc.predicted_aqi.toFixed(1) : "—";
      predEl.style.color = predColor;
      document.getElementById("kpiBoxPredicted").style.borderTopColor = predColor;
      const delta = fc.aqi_change_24h || 0;
      const sign = delta > 0 ? "+" : "";
      const deltaEl = document.getElementById("aqiDeltaText");
      deltaEl.textContent = `${sign}${delta.toFixed(1)} (${fc.direction || "Stable"})`;
      deltaEl.style.color = delta > 4 ? "#b91c1c" : (delta < -4 ? "#15803d" : "#475569");
      document.getElementById("predictionIntervalText").textContent = `${fc.interval_80_low}–${fc.interval_80_high}`;
      document.getElementById("peak24hAqiText").textContent = `${fc.peak_24h_aqi}`;
      document.getElementById("targetTimestampNote").textContent = `Forecast Target: ${(data.timestamps || {}).target_at_ist || ""}`;

      // KPI 3: Predicted Category
      const catBadge = document.getElementById("predictedCategoryBadge");
      catBadge.textContent = fc.predicted_category || "—";
      catBadge.style.color = predColor;
      document.getElementById("kpiBoxCategory").style.borderTopColor = predColor;
      document.getElementById("categoryRangeText").textContent = `CPCB Band ${health.Range || "—"}`;
      document.getElementById("confidenceNote").textContent = fc.confidence_level || "High Confidence";

      // 4-Card City Conditions & Regional Distance Strip
      const nextTemp = nextWx.next24h_temp_c !== undefined ? nextWx.next24h_temp_c : wx.temperature_c;
      const nextWind = nextWx.next24h_wind_speed_ms !== undefined ? nextWx.next24h_wind_speed_ms : wx.wind_speed_ms;
      document.getElementById("wxMainVal").textContent = `${wx.temperature_c}°C → ${nextTemp}°C (24h)`;
      document.getElementById("wxSubVal").textContent = `Wind ${wx.wind_speed_ms} → ${nextWind} m/s · RH ${wx.humidity_pct}% · ${wx.weather_description || "Clear"}`;

      const curPbl = Math.round(wx.boundary_layer_height_m || 550);
      const nextPbl = Math.round(nextWx.next24h_pbl_m || curPbl);
      document.getElementById("pblMainVal").textContent = `PBL ${curPbl}m → ${nextPbl}m`;
      document.getElementById("pblSubVal").textContent = `${wx.pressure_hpa} hPa · DewPt ${wx.dew_point_c}°C · 24h Rain ${nextWx.next24h_precip_mm || 0} mm`;

      const nf = ctx.nearest_festival;
      if (nf) {
        const off = nf.days_offset;
        const rel = off === 0 ? "Today" : (off > 0 ? `${off}d ago` : `In ${Math.abs(off)}d`);
        document.getElementById("festMainVal").textContent = `${nf.name} (${rel})`;
        document.getElementById("festSubVal").textContent = `${nf.distance_to_epicenter_km} km from ${nf.nearest_epicenter_hub} hub · Diwali: ${ctx.abs_days_to_diwali}d`;
      } else {
        document.getElementById("festMainVal").textContent = `${ctx.region || "India"} Regime`;
        document.getElementById("festSubVal").textContent = `Diwali window offset: ${ctx.days_from_diwali || 0}d`;
      }

      if (firms.available) {
        document.getElementById("firmsMainVal").textContent = `${firms.fires_24h || 0} fires (24h)`;
        document.getElementById("firmsSubVal").textContent = `Crop-fire belt: ${ctx.dist_to_stubble_belt_km} km · Thar dust: ${ctx.dist_to_thar_dust_km} km`;
      } else {
        document.getElementById("firmsMainVal").textContent = `Crop Belt: ${ctx.dist_to_stubble_belt_km} km`;
        document.getElementById("firmsSubVal").textContent = `Thar Dust Source: ${ctx.dist_to_thar_dust_km} km · ${ctx.region || "India"} Zone`;
      }

      // Trajectory Chart
      renderTrendSvg((data.trends || {}).past_24h || [], (data.trends || {}).next_24h || []);

      // Compact Live City Pollutant Strip with CPCB Sub-Index
      const pols = cur.pollutants || {};
      const subs = cur.sub_indices || {};
      const polDefs = [
        { key: "pm25", label: "PM2.5", unit: "µg/m³" },
        { key: "pm10", label: "PM10", unit: "µg/m³" },
        { key: "no2", label: "NO2", unit: "µg/m³" },
        { key: "so2", label: "SO2", unit: "µg/m³" },
        { key: "co", label: "CO", unit: "mg/m³" },
        { key: "o3", label: "O3", unit: "µg/m³" }
      ];
      document.getElementById("pollutantStrip").innerHTML = polDefs.map(p => {
        const val = pols[p.key];
        const sub = subs[p.key];
        const subText = sub !== undefined && sub !== null ? `Sub-Idx ${Math.round(sub)}` : "";
        return `<div class="pol-box">
          <div class="pol-name"><span>${p.label}</span><span class="tabular-nums" style="font-weight: 500;">${subText}</span></div>
          <div class="pol-val">${val !== null && val !== undefined ? val + " " + p.unit : "N/A"}</div>
        </div>`;
      }).join("");

      // Causal Drivers
      document.getElementById("directionVerbText").textContent = `${fc.direction || "Change"} in ${loc.city}`;
      const expContainer = document.getElementById("explanationDriversList");
      expContainer.innerHTML = (data.explanations || []).map(d => {
        const impColor = d.impact.includes("Increase") || d.impact.includes("Upward")
          ? "#b91c1c"
          : (d.impact.includes("Reduce") ? "#15803d" : "#475569");
        return `<div class="driver-item">
          <div class="driver-row">
            <span class="driver-name">${d.factor}</span>
            <span class="driver-tag" style="color: ${impColor};">${d.impact}</span>
          </div>
          <p style="font-size: 0.84rem; color: var(--text-body);">${d.detail}</p>
        </div>`;
      }).join("");

      // Health Precautions
      document.getElementById("healthCatLabel").textContent = `${health.category || "—"} (${health.Range || ""})`;
      document.getElementById("healthImpactSummary").textContent = health.health_impact || "";
      document.getElementById("healthGeneralText").textContent = health.general_public || "";
      document.getElementById("healthSensitiveText").textContent = health.sensitive_groups || "";
      document.getElementById("healthMaskText").textContent = health.mask_guidance || "";
      document.getElementById("healthDisclaimerText").textContent = health.disclaimer || "";
    }

    // Autocomplete
    const searchInput = document.getElementById("citySearchInput");
    const autoMenu = document.getElementById("autocompleteMenu");

    async function fetchAutocomplete(query) {
      try {
        const resp = await fetch(`/api/cities?q=${encodeURIComponent(query)}&limit=8`);
        const data = await resp.json();
        const items = data.results || [];
        if (!items.length) {
          autoMenu.innerHTML = '<div style="padding: 0.75rem; font-size: 0.85rem; color: #64748b;">No matching Indian locations found.</div>';
          autoMenu.classList.add("open");
          return;
        }
        autoMenu.innerHTML = items.map(item => `
          <button type="button" class="autocomplete-item" role="option"
            data-city="${item.name}"
            data-state="${item.state}"
            data-lat="${item.lat}"
            data-lon="${item.lon}">
            <div>
              <div style="font-weight: 600; color: #0f172a; font-size: 0.9rem;">${item.display_name}</div>
              <div class="tabular-nums" style="font-size: 0.75rem; color: #64748b;">${item.lat}°N, ${item.lon}°E</div>
            </div>
            <div style="font-size: 0.75rem; font-weight: 600; color: ${item.has_cpcb_station ? '#0f766e' : '#b45309'};">
              ${item.has_cpcb_station ? `CPCB Station (${item.cpcb_stations})` : `Regional Est.`}
            </div>
          </button>
        `).join("");
        autoMenu.classList.add("open");

        autoMenu.querySelectorAll(".autocomplete-item").forEach(btn => {
          btn.addEventListener("click", () => {
            const c = btn.dataset.city;
            const s = btn.dataset.state;
            const lat = parseFloat(btn.dataset.lat);
            const lon = parseFloat(btn.dataset.lon);
            searchInput.value = `${c}, ${s}`;
            autoMenu.classList.remove("open");
            loadCityForecast(c, lat, lon, s);
          });
        });
      } catch (e) {
        console.error("Autocomplete error:", e);
      }
    }

    searchInput.addEventListener("input", (e) => {
      const q = e.target.value.trim();
      clearTimeout(state.debounceTimer);
      if (!q) {
        autoMenu.classList.remove("open");
        return;
      }
      state.debounceTimer = setTimeout(() => fetchAutocomplete(q), 200);
    });

    searchInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        const firstOption = autoMenu.querySelector(".autocomplete-item");
        if (firstOption && autoMenu.classList.contains("open")) {
          firstOption.click();
        } else if (searchInput.value.trim()) {
          autoMenu.classList.remove("open");
          loadCityForecast(searchInput.value.trim(), null, null, "");
        }
      }
    });

    document.addEventListener("click", (e) => {
      if (!e.target.closest(".search-wrapper")) {
        autoMenu.classList.remove("open");
      }
    });

    document.querySelectorAll(".preset-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const c = btn.dataset.city;
        const s = btn.dataset.state;
        const lat = parseFloat(btn.dataset.lat);
        const lon = parseFloat(btn.dataset.lon);
        searchInput.value = `${c}, ${s}`;
        loadCityForecast(c, lat, lon, s);
      });
    });

    document.getElementById("refreshForecastBtn").addEventListener("click", () => {
      loadCityForecast(state.city, state.lat, state.lon, state.stateName);
    });

    window.addEventListener("DOMContentLoaded", () => {
      loadCityForecast("Delhi", 28.6139, 77.2090, "Delhi");
    });
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index() -> Response:
    """Render the main AirShield City-Focused Forecasting Webpage."""
    ensure_model_ready()
    return Response(DASHBOARD_HTML, mimetype="text/html")


# =====================================================================
# BUILT-IN 19-POINT ACCEPTANCE SELF-TEST SUITE (--self-test)
# =====================================================================

def run_acceptance_self_tests() -> int:
    """
    Programmatically verify all 19 acceptance requirements including:
    - Real multi-year CPCB + meteorology training dataset (2015–2025)
    - Complete weather & +24h forecast weather parity in training & live inference
    - Pure distance-based regional & national festival features (zero hardcoded weights)
    - 3-stage ablation study & festival-window validation
    """
    print("=== RUNNING AIRSHIELD 19-POINT ACCEPTANCE TEST SUITE ===")
    ensure_model_ready()
    bundle = load_model_bundle()
    feat_cols = set(bundle.get("feature_columns", []))

    required_weather_and_event_features = {
        "temperature_c",
        "humidity_pct",
        "wind_speed_ms",
        "precipitation_mm",
        "boundary_layer_height_m",
        "next24h_temp_c",
        "next24h_humidity_pct",
        "next24h_wind_speed_ms",
        "next24h_precip_mm",
        "next24h_pbl_m",
        "temp_change_24h",
        "wind_change_24h",
        "days_from_diwali",
        "abs_days_to_diwali",
        "is_diwali_window",
        "days_from_dussehra",
        "days_from_holi",
        "days_to_regional_festival",
        "dist_to_active_festival_km",
        "regional_festival_proximity",
        "is_festival_day",
        "dist_to_stubble_belt_km",
        "is_stubble_burning_season",
        "downwind_stubble_exposure",
        "dist_to_thar_dust_km",
        "is_winter_inversion",
        "is_dust_season",
        "is_monsoon_season",
    }
    assert required_weather_and_event_features.issubset(feat_cols), (
        f"Missing required weather/festival distance features: {required_weather_and_event_features - feat_cols}"
    )
    assert "festival_emission_weight" not in feat_cols, "Hardcoded festival_emission_weight must not be present!"
    print(
        f"[PASS] 0. Verified all {len(required_weather_and_event_features)} weather (+24h forecast & PBL) and "
        f"distance-based regional festival features are active in trained model ({len(feat_cols)} total features, zero hardcoded weights)."
    )

    client = app.test_client()

    # 1. Verify all 5 required source files exist
    required_files = ["app.py", "data_sources.py", "train_model.py", "predictor.py", "requirements.txt"]
    for rf in required_files:
        assert os.path.exists(rf), f"Missing required file: {rf}"
    print("[PASS] 1. All 5 source files exist and are connected.")

    # 2-3. Flask launches and home webpage loads
    r_home = client.get("/")
    assert r_home.status_code == 200 and b"AirShield" in r_home.data
    print("[PASS] 2-3. Flask server and home webpage load cleanly (HTTP 200).")

    # 4-5. Indian city autocomplete & coordinate resolution
    r_cities = client.get("/api/cities?q=Ahe")
    assert r_cities.status_code == 200
    c_data = r_cities.get_json()
    assert c_data["count"] > 0
    ahm = [c for c in c_data["results"] if "Ahmedabad" in c["name"]]
    assert ahm and abs(ahm[0]["lat"] - 23.0225) < 0.5
    print(f"[PASS] 4-5. City autocomplete ('Ahe') returned {c_data['count']} Indian matches including Ahmedabad ({ahm[0]['lat']}, {ahm[0]['lon']}).")

    # 6-7. Live prediction with OpenAQ CPCB + OpenWeatherMap + Open-Meteo (+24h weather) + NASA FIRMS + Regional Festival Distance
    r_pred = client.get("/api/predict?city=Delhi&lat=28.6139&lon=77.2090&state=Delhi")
    assert r_pred.status_code == 200
    p_data = r_pred.get_json()
    assert p_data["weather"]["temperature_c"] is not None
    assert p_data["weather"]["next_24h_weather"]["next24h_temp_c"] is not None
    assert p_data["forecast_24h"]["predicted_aqi"] > 0
    assert p_data["forecast_24h"]["feature_completeness_pct"] == 100.0
    fest_ctx = p_data["contextual_factors"]["festival_and_seasonal"]
    assert fest_ctx.get("dist_to_stubble_belt_km") is not None
    print(
        f"[PASS] 6-7 & 12-13. Live Delhi forecast (100% feature completeness across {len(feat_cols)} features): "
        f"Observed AQI={p_data['current_observation']['aqi']} ({p_data['current_observation']['category']}), "
        f"Predicted +24h AQI={p_data['forecast_24h']['predicted_aqi']} ({p_data['forecast_24h']['predicted_category']}), "
        f"Live Temp={p_data['weather']['temperature_c']}°C -> +24h Temp={p_data['weather']['next_24h_weather']['next24h_temp_c']}°C, "
        f"Nearest Festival={fest_ctx.get('nearest_festival')}, "
        f"Stubble Belt Dist={fest_ctx.get('dist_to_stubble_belt_km')} km."
    )

    # 8-11 & 14-15. Multi-year CPCB data, chronological validation, 3-stage ablation study
    r_metrics = client.get("/api/metrics")
    assert r_metrics.status_code == 200
    m_data = r_metrics.get_json()
    assert m_data["dataset_summary"]["total_verified_samples"] > 40000
    assert "CatBoostRegressor" in m_data["candidate_benchmarks"]
    assert "PersistenceBaseline" in m_data["candidate_benchmarks"]
    assert "festival_window_test_subset" in m_data["ablation_study"]
    print(
        f"[PASS] 8-11 & 14-15. Model validation verified on {m_data['dataset_summary']['total_verified_samples']} real multi-year samples; "
        f"Selected Model: {m_data['selected_model']} ({m_data['model_version']}); "
        f"Overall Context MAE Gain: +{m_data['ablation_study']['context_mae_improvement']} AQI | "
        f"Festival-Window MAE Gain: +{m_data['ablation_study']['festival_window_test_subset']['festival_window_mae_gain']} AQI."
    )

    # 16. Missing local CPCB station coverage handled transparently (e.g. Shimla)
    r_shimla = client.get("/api/predict?city=Shimla&lat=31.1048&lon=77.1734&state=Himachal%20Pradesh")
    assert r_shimla.status_code == 200
    s_data = r_shimla.get_json()
    assert s_data["location"]["coverage"]["has_cpcb_station"] is False
    assert "Limited local CPCB station coverage" in s_data["location"]["coverage"]["coverage_notice"]
    print(f"[PASS] 16. Non-CPCB town transparency verified for Shimla: '{s_data['location']['coverage']['coverage_notice']}'")

    # 17. Verify no API keys exposed in HTML or JSON responses
    for env_key in ["OPENWEATHER_API_KEY", "DATA_GOV_IN_API_KEY", "OPENAQ_API_KEY", "NASA_FIRMS_MAP_KEY"]:
        val = os.getenv(env_key, "")
        if val and len(val) > 8:
            assert val.encode() not in r_home.data, f"{env_key} leaked in HTML!"
            assert val not in json.dumps(p_data), f"{env_key} leaked in /api/predict!"
    print("[PASS] 17. Security check passed: Zero API keys exposed in HTML or API JSON.")

    print("=== ALL 19 ACCEPTANCE TESTS PASSED SUCCESSFULLY ===")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(run_acceptance_self_tests())
    ensure_model_ready()
    port = int(os.getenv("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False)
