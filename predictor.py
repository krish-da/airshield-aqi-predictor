"""
AirShield — Live 24-Hour AQI Inference, Threshold-Breach & Explanation Engine (predictor.py)
Responsibilities:
1. Load persisted trained ML model, imputer, and evaluation report from models/
2. Orchestrate live data collection for any selected Indian city (CPCB, OpenWeatherMap, Open-Meteo, NASA FIRMS, Nager.Date)
3. Construct the exact 76-feature vector matching train_model.py (including current + next-24h weather,
   boundary layer height, and pure distance-based regional/national festival & plume features)
4. Predict next-24h numeric Indian CPCB AQI and derive official CPCB AQI category
5. Compute 24-hour trajectory, threshold-breach risk alert, and expected breach time
6. Generate grounded environmental explanations and CPCB health recommendations
7. Persist live forecasts to SQLite for delayed live validation
"""

import os
import json
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import joblib
import numpy as np
import pandas as pd

from data_sources import (
    IST,
    MODELS_DIR,
    aqi_to_cpcb_category,
    calculate_cpcb_aqi,
    calculate_cpcb_subindex,
    normalize_co_to_mg_m3,
    resolve_cpcb_station_coverage,
    fetch_openweather_live,
    fetch_openmeteo_live_and_history,
    fetch_cpcb_live_station_data,
    fetch_nasa_firms_fires,
    get_festival_and_seasonal_context,
    log_live_prediction,
    evaluate_live_predictions,
)

logger = logging.getLogger("airshield.predictor")

MODEL_ARTIFACT_PATH = os.path.join(MODELS_DIR, "airshield_model.joblib")
EVAL_REPORT_PATH = os.path.join(MODELS_DIR, "evaluation_report.json")

_MODEL_CACHE: Optional[Dict[str, Any]] = None


def load_model_bundle(force_reload: bool = False) -> Dict[str, Any]:
    """Load persisted model bundle once into memory. Never retrains on user requests."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None and not force_reload:
        return _MODEL_CACHE

    if not os.path.exists(MODEL_ARTIFACT_PATH):
        raise FileNotFoundError(
            f"Trained model artifact not found at {MODEL_ARTIFACT_PATH}. Run 'python train_model.py' first."
        )

    _MODEL_CACHE = joblib.load(MODEL_ARTIFACT_PATH)
    logger.info(
        "Loaded persisted model %s (%s)",
        _MODEL_CACHE.get("selected_model"),
        _MODEL_CACHE.get("model_version"),
    )
    return _MODEL_CACHE


def get_evaluation_report() -> Dict[str, Any]:
    """Load the saved chronological validation report and combine with live validation ledger."""
    if os.path.exists(EVAL_REPORT_PATH):
        with open(EVAL_REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
    else:
        bundle = load_model_bundle()
        report = bundle.get("evaluation_report", {})

    report["live_validation"] = evaluate_live_predictions()
    return report


def get_health_precautions(category: str, predicted_aqi: float) -> Dict[str, Any]:
    """
    Return official CPCB-aligned health impacts, precautionary guidance, and sensitive group advice.
    """
    cat_map = {
        "Good": {
            "Range": "0–50",
            "color": "#16a34a",
            "health_impact": "Minimal impact. Air quality is satisfactory and poses little or no risk.",
            "general_public": "Ideal conditions for outdoor exercise, commuting, and natural indoor ventilation.",
            "sensitive_groups": "No restrictions for children, elderly, or individuals with asthma.",
            "mask_guidance": "No respirator required.",
        },
        "Satisfactory": {
            "Range": "51–100",
            "color": "#65a30d",
            "health_impact": "Minor breathing discomfort to sensitive people during prolonged outdoor exertion.",
            "general_public": "Normal outdoor activities can continue safely.",
            "sensitive_groups": "Individuals with severe respiratory sensitivity should monitor symptoms during heavy exertion.",
            "mask_guidance": "Optional for highly sensitive individuals near heavy traffic corridors.",
        },
        "Moderate": {
            "Range": "101–200",
            "color": "#d97706",
            "health_impact": "Breathing discomfort to people with lung disease (asthma, COPD), children, and older adults.",
            "general_public": "Limit prolonged outdoor exertion near congested roadways or industrial zones.",
            "sensitive_groups": "Keep rescue inhalers accessible; reduce vigorous outdoor exercise during late evening and early morning hours.",
            "mask_guidance": "N95/FFP2 mask recommended for sensitive commuters in heavy traffic.",
        },
        "Poor": {
            "Range": "201–300",
            "color": "#ea580c",
            "health_impact": "Breathing discomfort to most people on prolonged exposure; pronounced effects on heart and lung patients.",
            "general_public": "Avoid prolonged outdoor exercise (jogging, cycling) especially between 10 PM and 9 AM when inversion traps pollutants.",
            "sensitive_groups": "Remain indoors where possible; use HEPA air filtration if available.",
            "mask_guidance": "Well-fitted N95 / KN95 mask strongly advised outdoors.",
        },
        "Very Poor": {
            "Range": "301–400",
            "color": "#dc2626",
            "health_impact": "Respiratory illness on prolonged exposure. Marked aggravation in people with lung and heart diseases.",
            "general_public": "Eliminate outdoor physical exertion. Keep windows closed during peak morning/night smog hours.",
            "sensitive_groups": "Strictly avoid outdoor exposure; children, pregnant women, and elderly should stay in filtered indoor environments.",
            "mask_guidance": "N95 / N99 respirator required for any outdoor travel.",
        },
        "Severe": {
            "Range": "401–500",
            "color": "#991b1b",
            "health_impact": "Affects healthy people and seriously impacts those with existing respiratory or cardiovascular diseases.",
            "general_public": "Health emergency conditions: avoid all outdoor activity, work from home if feasible, and seal indoor air leaks.",
            "sensitive_groups": "Remain strictly indoors with continuous HEPA filtration; seek medical care immediately if experiencing shortness of breath.",
            "mask_guidance": "Certified N95 / N99 respirator mandatory if stepping outside.",
        },
    }
    info = cat_map.get(category, cat_map["Moderate"]).copy()
    info["category"] = category
    info["predicted_aqi"] = round(predicted_aqi, 1)
    info["disclaimer"] = (
        "AirShield is an educational and research forecasting tool. Predictions do not replace official "
        "Central Pollution Control Board (CPCB) monitoring bulletins or professional medical advice."
    )
    return info


def build_explanation_drivers(
    current_aqi: float,
    predicted_aqi: float,
    pollutants: Dict[str, Optional[float]],
    weather: Dict[str, Any],
    context: Dict[str, Any],
    firms: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Generate transparent, grounded causal explanations supported strictly by observed
    meteorological, 24h forecast weather, pollutant, boundary-layer, and regional festival distance inputs.
    """
    drivers: List[Dict[str, str]] = []
    delta = predicted_aqi - current_aqi

    # 1. Baseline pollution momentum
    pm25 = pollutants.get("pm25")
    pm10 = pollutants.get("pm10")
    if current_aqi >= 201:
        drivers.append({
            "factor": "Elevated Baseline Aerosol Load",
            "impact": "Upward Pressure",
            "detail": (
                f"Current AQI is already elevated at {round(current_aqi)} "
                f"(PM2.5: {round(pm25, 1) if pm25 else 'N/A'} µg/m³, PM10: {round(pm10, 1) if pm10 else 'N/A'} µg/m³), "
                "creating strong 24-hour atmospheric persistence."
            ),
        })
    elif current_aqi <= 85:
        drivers.append({
            "factor": "Clean Baseline Air Mass",
            "impact": "Stabilizing",
            "detail": (
                f"Baseline AQI is relatively low ({round(current_aqi)}), limiting immediate accumulation "
                "unless sharp stagnation or external plume transport occurs."
            ),
        })

    # 2. Current & Next-24h Wind Speed & Horizontal Dispersion
    wind_ms = weather.get("wind_speed_ms", 2.5)
    next_wx = weather.get("next_24h_weather", {})
    next_wind_ms = next_wx.get("next24h_wind_speed_ms", wind_ms)
    if wind_ms is not None:
        if wind_ms < 2.2 or (next_wind_ms is not None and next_wind_ms < 2.2):
            drivers.append({
                "factor": "Low Wind Speed & Weak 24h Horizontal Dispersion",
                "impact": "Increases AQI",
                "detail": (
                    f"Current surface wind is {wind_ms:.1f} m/s with a 24h forecast average of {next_wind_ms:.1f} m/s. "
                    "Winds below 2.2 m/s inhibit horizontal ventilation and trap vehicular and urban emissions."
                ),
            })
        elif wind_ms >= 3.8 or (next_wind_ms is not None and next_wind_ms >= 3.8):
            drivers.append({
                "factor": "Favorable Surface Wind Ventilation",
                "impact": "Reduces AQI",
                "detail": (
                    f"Surface wind speed ({wind_ms:.1f} m/s now, {next_wind_ms:.1f} m/s 24h forecast) promotes "
                    "horizontal advection and dilution of near-surface particulate matter."
                ),
            })

    # 3. Planetary Boundary Layer (PBL) height & vertical mixing
    pbl_m = weather.get("boundary_layer_height_m")
    next_pbl_m = next_wx.get("next24h_pbl_m", pbl_m)
    if pbl_m is not None:
        if pbl_m < 450 or (next_pbl_m is not None and next_pbl_m < 450):
            drivers.append({
                "factor": "Shallow Planetary Boundary Layer (PBL)",
                "impact": "Increases AQI",
                "detail": (
                    f"Mixing height is ~{round(pbl_m)} m (24h forecast avg ~{round(next_pbl_m or pbl_m)} m). "
                    "A compressed boundary layer restricts vertical turbulent dilution near breathing height."
                ),
            })
        elif pbl_m > 1050:
            drivers.append({
                "factor": "Deep Convective Boundary Layer",
                "impact": "Reduces AQI",
                "detail": (
                    f"Boundary layer height reaches ~{round(pbl_m)} m, enabling deep vertical mixing "
                    "of surface emissions into the upper troposphere."
                ),
            })

    # 4. Precipitation & 24h Forecast Wet Deposition
    precip_mm = weather.get("precipitation_mm", 0.0) or 0.0
    next_precip_mm = next_wx.get("next24h_precip_mm", 0.0) or 0.0
    if precip_mm >= 1.0 or next_precip_mm >= 1.5:
        drivers.append({
            "factor": "Precipitation Scavenging (Wet Deposition)",
            "impact": "Reduces AQI",
            "detail": (
                f"Observed/forecast rainfall ({precip_mm:.1f} mm current, {next_precip_mm:.1f} mm over next 24h) "
                "scavenges coarse PM10 and soluble aerosols via wet deposition."
            ),
        })

    # 5. Regional & National Festival / Celebration Distance Context
    nf = context.get("nearest_festival")
    if nf and abs(nf.get("days_offset", 99)) <= 7 and nf.get("distance_to_epicenter_km", 9999) <= 550:
        d_off = nf["days_offset"]
        when_str = "active today" if d_off == 0 else (f"{d_off} day(s) ago" if d_off > 0 else f"in {abs(d_off)} day(s)")
        drivers.append({
            "factor": f"Regional/National Event Proximity: {nf['name']}",
            "impact": "Increases AQI" if abs(d_off) <= 3 else "Regional Context",
            "detail": (
                f"City is {nf['distance_to_epicenter_km']} km from the {nf['nearest_epicenter_hub']} celebration epicenter "
                f"({when_str}, spatial-temporal proximity={nf['spatial_temporal_proximity']:.2f})."
            ),
        })

    # 6. Crop-residue burning belt distance & NASA FIRMS
    if context.get("is_stubble_burning_season"):
        drivers.append({
            "factor": "Post-Monsoon Crop-Residue Plume Corridor",
            "impact": "Increases AQI",
            "detail": (
                f"Located {context.get('dist_to_stubble_belt_km')} km from the Punjab–Haryana paddy stubble belt "
                f"(downwind transport exposure index: {context.get('downwind_stubble_exposure', 0):.2f})."
            ),
        })
    if firms.get("available") and (firms.get("fires_24h") or 0) > 10:
        drivers.append({
            "factor": "Active Satellite Thermal Anomalies (NASA FIRMS)",
            "impact": "Increases AQI",
            "detail": (
                f"{firms['fires_24h']} VIIRS fire detections observed within 220 km in the past 24 hours "
                f"({firms['fires_72h']} in 72h, avg FRP {firms.get('mean_frp_mw', 0)} MW), contributing regional biomass smoke."
            ),
        })

    if len(drivers) < 2:
        temp_c = weather.get("temperature_c", 25.0)
        rh_pct = weather.get("humidity_pct", 55.0)
        direction_word = "modest increase" if delta > 3 else ("gradual improvement" if delta < -3 else "near-steady persistence")
        drivers.append({
            "factor": "Synoptic Meteorological Balance",
            "impact": "Neutral / Moderate",
            "detail": (
                f"Temperature ({temp_c:.1f} °C, 24h Δ {next_wx.get('temp_change_24h', 0.0):+.1f} °C), "
                f"humidity ({round(rh_pct)}%), and wind ({wind_ms:.1f} m/s) indicate {direction_word} over the next 24 hours."
            ),
        })

    return drivers


def predict_city_aqi_24h(
    city_name: str,
    lat: float,
    lon: float,
    state: str = "",
    log_to_db: bool = True,
) -> Dict[str, Any]:
    """
    Execute the complete AirShield live inference pipeline for an Indian city,
    constructing all 76 zero-leakage pollutant, weather (+24h forecast & PBL),
    and distance-based regional festival features matching train_model.py.
    """
    bundle = load_model_bundle()
    model = bundle["model"]
    imputer = bundle["imputer"]
    feature_columns: List[str] = bundle["feature_columns"]
    model_version: str = bundle["model_version"]
    selected_model_name: str = bundle["selected_model"]
    quantiles: Dict[str, float] = bundle.get("uncertainty_quantiles", {"q80_abs_error": 30.0, "q90_abs_error": 45.0})

    now_ist = datetime.now(IST)
    target_ist = now_ist + timedelta(hours=24)

    coverage = resolve_cpcb_station_coverage(lat, lon, city_name)
    resolved_state = state or coverage.get("nearest_cpcb_state", "India")

    cpcb_live = fetch_cpcb_live_station_data(city_name, lat, lon)
    owm_live = fetch_openweather_live(lat, lon)
    om_live = fetch_openmeteo_live_and_history(lat, lon)
    firms_live = fetch_nasa_firms_fires(lat, lon)

    curr_temp = float(owm_live.get("temperature_c") if owm_live.get("available") else om_live.get("temperature_c", 26.0))
    curr_rh = float(owm_live.get("humidity_pct") if owm_live.get("available") else om_live.get("humidity_pct", 55.0))
    curr_wind = float(owm_live.get("wind_speed_ms") if owm_live.get("available") else om_live.get("wind_speed_ms", 2.2))
    curr_pbl = float(om_live.get("boundary_layer_height_m", 550.0))

    om_next24 = om_live.get("next_24h_weather", {})
    next24h_temp_c = float(om_next24.get("next24h_temp_c", curr_temp))
    next24h_humidity_pct = float(om_next24.get("next24h_humidity_pct", curr_rh))
    next24h_wind_speed_ms = float(om_next24.get("next24h_wind_speed_ms", curr_wind))
    next24h_precip_mm = float(om_next24.get("next24h_precip_mm", 0.0))
    next24h_pbl_m = float(om_next24.get("next24h_pbl_m", curr_pbl))
    temp_change_24h = round(next24h_temp_c - curr_temp, 2)
    wind_change_24h = round(next24h_wind_speed_ms - curr_wind, 2)

    weather_merged = {
        "temperature_c": curr_temp,
        "feels_like_c": owm_live.get("feels_like_c") if owm_live.get("available") else curr_temp,
        "humidity_pct": curr_rh,
        "dew_point_c": om_live.get("dew_point_c") or owm_live.get("dew_point_c", 15.0),
        "pressure_hpa": owm_live.get("pressure_hpa") if owm_live.get("available") else om_live.get("pressure_hpa", 1010.0),
        "wind_speed_ms": curr_wind,
        "wind_direction_deg": owm_live.get("wind_direction_deg") if owm_live.get("available") else om_live.get("wind_direction_deg", 270.0),
        "wind_gust_ms": owm_live.get("wind_gust_ms") if owm_live.get("available") else curr_wind * 1.35,
        "cloud_cover_pct": owm_live.get("cloud_cover_pct") if owm_live.get("available") else om_live.get("cloud_cover_pct", 20.0),
        "precipitation_mm": owm_live.get("precipitation_mm") if owm_live.get("available") else om_live.get("precipitation_mm", 0.0),
        "weather_condition": owm_live.get("weather_condition", "Clear"),
        "weather_description": owm_live.get("weather_description", "Clear Sky"),
        "boundary_layer_height_m": curr_pbl,
        "solar_radiation_wm2": om_live.get("solar_radiation_wm2", 180.0),
        "uv_index": om_live.get("uv_index", 4.0),
        "aerosol_optical_depth": om_live.get("aerosol_optical_depth", 0.35),
        "next_24h_weather": {
            "next24h_temp_c": next24h_temp_c,
            "next24h_humidity_pct": next24h_humidity_pct,
            "next24h_wind_speed_ms": next24h_wind_speed_ms,
            "next24h_precip_mm": next24h_precip_mm,
            "next24h_pbl_m": next24h_pbl_m,
            "temp_change_24h": temp_change_24h,
            "wind_change_24h": wind_change_24h,
        },
        "weather_source": owm_live.get("source", "Open-Meteo Weather Fallback"),
        "weather_timestamp": owm_live.get("observed_timestamp") or now_ist.isoformat(),
    }

    om_pols = om_live.get("auxiliary_pollutants", {}) if om_live.get("available") else {}
    owm_pols = owm_live.get("owm_pollutants", {}) if owm_live.get("available") else {}

    if cpcb_live.get("available") and cpcb_live.get("stations"):
        st0 = cpcb_live["stations"][0]
        raw_pols = dict(st0.get("pollutants", {}))
        for k in ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"]:
            if raw_pols.get(k) is None:
                raw_pols[k] = om_pols.get(k) if om_pols.get(k) is not None else owm_pols.get(k)
        pollution_source_label = cpcb_live.get("source") or "Official CPCB Live Station"
        station_label = st0.get("station", f"{city_name} CPCB CAAQMS")
    else:
        raw_pols = {}
        for k in ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"]:
            val_om = om_pols.get(k)
            val_owm = owm_pols.get(k)
            raw_pols[k] = val_om if val_om is not None else val_owm
        pollution_source_label = (
            "Official CPCB Formula applied to Live Open-Meteo & OpenWeatherMap Pollutant Measurements"
        )
        station_label = (
            f"{city_name} ({coverage['station_count']} CPCB CAAQMS in network)"
            if coverage["has_cpcb_station"]
            else f"Regional Grid ({city_name} — nearest CPCB station: {coverage['nearest_cpcb_city']}, {coverage['distance_to_station_km']} km)"
        )

    cpcb_calc = calculate_cpcb_aqi(raw_pols, require_three_pollutants=True)
    if cpcb_calc["aqi"] is None:
        cpcb_calc = calculate_cpcb_aqi(raw_pols, require_three_pollutants=False)

    if cpcb_live.get("available") and cpcb_live.get("stations") and cpcb_live["stations"][0].get("official_aqi"):
        current_aqi = float(cpcb_live["stations"][0]["official_aqi"])
    else:
        current_aqi = float(cpcb_calc["aqi"] or 110.0)
    current_category = aqi_to_cpcb_category(current_aqi)

    hourly_hist = om_live.get("hourly_history", [])
    hist_aqis = [float(h["aqi"]) for h in hourly_hist if h.get("aqi") is not None]
    if not hist_aqis:
        hist_aqis = [current_aqi]

    def get_lag_hours_ago(hours_back: int) -> float:
        idx = max(0, len(hist_aqis) - 1 - hours_back)
        return float(hist_aqis[idx])

    lag_1h = get_lag_hours_ago(1)
    lag_3h = get_lag_hours_ago(3)
    lag_6h = get_lag_hours_ago(6)
    lag_12h = get_lag_hours_ago(12)
    lag_24h = get_lag_hours_ago(24)
    lag_48h = get_lag_hours_ago(48)

    recent_window = hist_aqis[-24:] if len(hist_aqis) >= 24 else hist_aqis
    roll_mean_3 = float(np.mean(recent_window))
    roll_max_3 = float(np.max(recent_window))
    roll_min_3 = float(np.min(recent_window))
    roll_std_3 = float(np.std(recent_window)) if len(recent_window) > 1 else 4.0
    roll_mean_7 = float(np.mean(hist_aqis))

    daily_equiv_series = pd.Series([lag_48h, lag_24h, current_aqi], dtype=float)
    aqi_ewm_3 = float(daily_equiv_series.ewm(span=3, min_periods=1).mean().iloc[-1])
    aqi_ewm_7 = float(daily_equiv_series.ewm(span=7, min_periods=1).mean().iloc[-1])
    aqi_ewm_14 = float(daily_equiv_series.ewm(span=14, min_periods=1).mean().iloc[-1])

    # Contextual regional festival & seasonal distance features evaluated at the +24h forecast target date
    context = get_festival_and_seasonal_context(
        target_date=target_ist,
        lat=lat,
        lon=lon,
        state=resolved_state,
        temp_c=weather_merged["temperature_c"],
        wind_speed_ms=weather_merged["wind_speed_ms"],
        wind_dir_deg=weather_merged["wind_direction_deg"],
        rh_pct=weather_merged["humidity_pct"],
        pbl_m=weather_merged["boundary_layer_height_m"],
        precip_mm=weather_merged["precipitation_mm"],
    )

    city_priors = bundle.get("city_priors", {})
    global_prior = float(city_priors.get("global_median", 108.0))
    city_medians = city_priors.get("city_medians", {})
    city_month_medians = city_priors.get("city_month_medians", {})
    prior_city_key = city_name if city_name in city_medians else coverage.get("nearest_cpcb_city", city_name)
    city_train_median_aqi = float(city_medians.get(prior_city_key, global_prior))
    city_month_train_median_aqi = float(
        city_month_medians.get(f"{prior_city_key}:{int(context['month'])}", city_train_median_aqi)
    )
    aqi_anomaly_vs_city_month = current_aqi - city_month_train_median_aqi

    wind_rad = math.radians(float(weather_merged["wind_direction_deg"] or 270.0))
    doy = now_ist.timetuple().tm_yday
    pm25_val = float(raw_pols.get("pm25") if raw_pols.get("pm25") is not None else 42.0)
    pm10_val = float(raw_pols.get("pm10") if raw_pols.get("pm10") is not None else 85.0)
    sub_pm25 = calculate_cpcb_subindex("pm25", pm25_val) or (current_aqi * 0.9)
    sub_pm10 = calculate_cpcb_subindex("pm10", pm10_val) or (current_aqi * 0.85)

    hist_pm25 = [float(h["pm25"]) for h in hourly_hist if h.get("pm25") is not None]
    hist_pm10 = [float(h["pm10"]) for h in hourly_hist if h.get("pm10") is not None]
    pm25_lag_1 = float(hist_pm25[-24]) if len(hist_pm25) >= 24 else pm25_val
    pm25_roll_3 = float(np.mean(hist_pm25[-24:])) if hist_pm25 else pm25_val
    pm10_lag_1 = float(hist_pm10[-24]) if len(hist_pm10) >= 24 else pm10_val
    pm10_roll_3 = float(np.mean(hist_pm10[-24:])) if hist_pm10 else pm10_val

    # Build exact 76-feature dictionary matching FEATURE_COLUMNS in train_model.py
    raw_feature_dict: Dict[str, Any] = {
        "current_aqi": current_aqi,
        "aqi_lag_1": lag_24h,
        "aqi_lag_2": lag_48h,
        "aqi_lag_3": lag_48h,
        "aqi_lag_7": roll_mean_7,
        "aqi_delta_24h": current_aqi - lag_24h,
        "aqi_delta_48h": current_aqi - lag_48h,
        "aqi_ewm_3": aqi_ewm_3,
        "aqi_ewm_7": aqi_ewm_7,
        "aqi_ewm_14": aqi_ewm_14,
        "aqi_roll_mean_3": roll_mean_3,
        "aqi_roll_max_3": roll_max_3,
        "aqi_roll_min_3": roll_min_3,
        "aqi_roll_std_3": roll_std_3,
        "aqi_roll_mean_7": roll_mean_7,
        "aqi_ratio_to_roll7": current_aqi / max(roll_mean_7, 10.0),
        "city_train_median_aqi": city_train_median_aqi,
        "city_month_train_median_aqi": city_month_train_median_aqi,
        "aqi_anomaly_vs_city_month": aqi_anomaly_vs_city_month,
        "pm25": raw_pols.get("pm25"),
        "pm25_lag_1": pm25_lag_1,
        "pm25_roll_3": pm25_roll_3,
        "sub_pm25": sub_pm25,
        "pm10": raw_pols.get("pm10"),
        "pm10_lag_1": pm10_lag_1,
        "pm10_roll_3": pm10_roll_3,
        "sub_pm10": sub_pm10,
        "no2": raw_pols.get("no2"),
        "so2": raw_pols.get("so2"),
        "co": normalize_co_to_mg_m3(raw_pols.get("co")),
        "o3": raw_pols.get("o3"),
        "nh3": raw_pols.get("nh3"),
        "pm25_to_pm10_ratio": round(min(1.0, max(0.05, float(pm25_val) / max(float(pm10_val), 1.0))), 3),
        "temperature_c": weather_merged["temperature_c"],
        "humidity_pct": weather_merged["humidity_pct"],
        "dew_point_c": weather_merged["dew_point_c"],
        "pressure_hpa": weather_merged["pressure_hpa"],
        "cloud_cover_pct": weather_merged["cloud_cover_pct"],
        "wind_speed_ms": weather_merged["wind_speed_ms"],
        "wind_gust_ms": weather_merged["wind_gust_ms"],
        "wind_dir_sin": math.sin(wind_rad),
        "wind_dir_cos": math.cos(wind_rad),
        "precipitation_mm": weather_merged["precipitation_mm"],
        "solar_radiation_wm2": weather_merged["solar_radiation_wm2"],
        "boundary_layer_height_m": weather_merged["boundary_layer_height_m"],
        "ventilation_proxy": round(float(weather_merged["wind_speed_ms"] or 2.0) * (float(weather_merged["boundary_layer_height_m"] or 550.0) / 100.0), 2),
        "next24h_temp_c": next24h_temp_c,
        "next24h_humidity_pct": next24h_humidity_pct,
        "next24h_wind_speed_ms": next24h_wind_speed_ms,
        "next24h_precip_mm": next24h_precip_mm,
        "next24h_pbl_m": next24h_pbl_m,
        "temp_change_24h": temp_change_24h,
        "wind_change_24h": wind_change_24h,
        "latitude": lat,
        "longitude": lon,
        "month": context["month"],
        "doy_sin": math.sin(2.0 * math.pi * doy / 365.25),
        "doy_cos": math.cos(2.0 * math.pi * doy / 365.25),
        "day_of_week": context["day_of_week"],
        "is_weekend": context["is_weekend"],
        "days_from_diwali": context["days_from_diwali"],
        "abs_days_to_diwali": context["abs_days_to_diwali"],
        "is_diwali_window": context["is_diwali_window"],
        "days_from_dussehra": context["days_from_dussehra"],
        "days_from_holi": context["days_from_holi"],
        "days_to_regional_festival": context["days_to_regional_festival"],
        "dist_to_active_festival_km": context["dist_to_active_festival_km"],
        "regional_festival_proximity": context["regional_festival_proximity"],
        "is_festival_day": context["is_festival_day"],
        "dist_to_stubble_belt_km": context["dist_to_stubble_belt_km"],
        "is_stubble_burning_season": context["is_stubble_burning_season"],
        "downwind_stubble_exposure": context["downwind_stubble_exposure"],
        "dist_to_thar_dust_km": context["dist_to_thar_dust_km"],
        "is_winter_inversion": context["is_winter_inversion"],
        "is_dust_season": context["is_dust_season"],
        "is_monsoon_season": context["is_monsoon_season"],
    }

    present_features = sum(1 for k in feature_columns if raw_feature_dict.get(k) is not None)
    feature_completeness_pct = round((present_features / len(feature_columns)) * 100.0, 1)

    row_df = pd.DataFrame(
        [[np.nan if raw_feature_dict.get(col) is None else float(raw_feature_dict[col]) for col in feature_columns]],
        columns=feature_columns,
    )
    X_live = imputer.transform(row_df)

    raw_pred = float(model.predict(X_live)[0])
    predicted_aqi = round(float(np.clip(raw_pred, 12.0, 500.0)), 1)
    predicted_category = aqi_to_cpcb_category(predicted_aqi)

    q80 = quantiles.get("q80_abs_error", 30.0)
    if not coverage["has_cpcb_station"]:
        q80 = round(q80 * 1.2, 1)
        confidence_level = "Moderate (Regional Estimate — No Local CPCB Station)"
    elif feature_completeness_pct >= 90.0:
        confidence_level = "High (Complete Multi-Sensor Feature Coverage)"
    else:
        confidence_level = "Moderate (Partial Imputation Applied)"

    pred_low_80 = round(max(10.0, predicted_aqi - q80), 1)
    pred_high_80 = round(min(500.0, predicted_aqi + q80), 1)

    forecast_trajectory = []
    breach_threshold = 200.0
    expected_breach_time = None

    for step_h in range(3, 27, 3):
        frac = step_h / 24.0
        step_dt = now_ist + timedelta(hours=step_h)
        hr = step_dt.hour
        diurnal_mod = 0.0
        if step_h < 24:
            if hr in (22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8):
                diurnal_mod = 0.06 * current_aqi
            elif hr in (13, 14, 15, 16, 17):
                diurnal_mod = -0.05 * current_aqi
        step_aqi = round(
            max(10.0, min(500.0, (1.0 - frac) * current_aqi + frac * predicted_aqi + diurnal_mod * (1.0 - abs(2 * frac - 1)))),
            1,
        )
        if step_h == 24:
            step_aqi = predicted_aqi
        step_cat = aqi_to_cpcb_category(step_aqi)
        if step_aqi > breach_threshold and expected_breach_time is None:
            expected_breach_time = step_dt.strftime("%d %b, %I:%M %p IST")
        forecast_trajectory.append({
            "horizon_hours": f"+{step_h}h",
            "timestamp": step_dt.strftime("%Y-%m-%d %H:%M"),
            "display_time": step_dt.strftime("%I:%M %p"),
            "aqi": step_aqi,
            "category": step_cat,
        })

    peak_24h_aqi = max([pt["aqi"] for pt in forecast_trajectory] + [predicted_aqi])
    if current_aqi > 200.0:
        if predicted_aqi > current_aqi + 5:
            risk_state = "Currently Unsafe — Further Deterioration Expected"
            risk_severity = "critical"
        elif predicted_aqi <= 200.0:
            risk_state = "Currently Unsafe — Gradual Dispersion Expected Within 24h"
            risk_severity = "warning"
        else:
            risk_state = "Currently Unsafe — Persistent Poor/Severe Air Quality"
            risk_severity = "critical"
    elif peak_24h_aqi > 200.0:
        risk_state = "Threshold Breach Alert — Expected to Become Unsafe (>200 AQI)"
        risk_severity = "warning"
    else:
        risk_state = "Expected to Remain Within Safer / Moderate Threshold (≤200 AQI)"
        risk_severity = "normal"

    past_24h_points = []
    hist_slice = hourly_hist[-25::3] if len(hourly_hist) >= 9 else hourly_hist
    for item in hist_slice:
        ts_raw = item.get("timestamp", "")
        try:
            dt_p = datetime.fromisoformat(ts_raw)
            disp = dt_p.strftime("%I:%M %p")
        except ValueError:
            disp = ts_raw[-5:]
        past_24h_points.append({
            "timestamp": ts_raw,
            "display_time": disp,
            "aqi": round(float(item.get("aqi") or current_aqi), 1),
            "category": aqi_to_cpcb_category(item.get("aqi") or current_aqi),
        })

    explanations = build_explanation_drivers(
        current_aqi=current_aqi,
        predicted_aqi=predicted_aqi,
        pollutants=raw_pols,
        weather=weather_merged,
        context=context,
        firms=firms_live,
    )
    health_advice = get_health_precautions(predicted_category, predicted_aqi)

    prediction_id = None
    if log_to_db:
        prediction_id = log_live_prediction(
            city=city_name,
            state=resolved_state,
            station=station_label,
            lat=lat,
            lon=lon,
            current_aqi=current_aqi,
            predicted_aqi=predicted_aqi,
            predicted_category=predicted_category,
            model_version=model_version,
            data_source=pollution_source_label,
            features={
                k: raw_feature_dict.get(k)
                for k in [
                    "current_aqi", "aqi_lag_1", "pm25", "pm10",
                    "temperature_c", "wind_speed_ms", "boundary_layer_height_m",
                    "next24h_temp_c", "next24h_wind_speed_ms",
                    "dist_to_active_festival_km", "regional_festival_proximity",
                    "dist_to_stubble_belt_km",
                ]
            },
        )

    return {
        "prediction_id": prediction_id,
        "location": {
            "city": city_name,
            "state": resolved_state,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "timezone": "Asia/Kolkata (IST, UTC+05:30)",
            "station_label": station_label,
            "coverage": coverage,
        },
        "timestamps": {
            "issued_at_ist": now_ist.strftime("%d %b %Y, %I:%M %p IST"),
            "target_at_ist": target_ist.strftime("%d %b %Y, %I:%M %p IST"),
            "weather_observed_at": weather_merged["weather_timestamp"],
        },
        "current_observation": {
            "aqi": current_aqi,
            "category": current_category,
            "dominant_pollutant": cpcb_calc.get("dominant_pollutant") or "PM2.5",
            "sub_indices": cpcb_calc.get("sub_indices", {}),
            "pollutants": {
                "pm25": round(float(raw_pols["pm25"]), 1) if raw_pols.get("pm25") is not None else None,
                "pm10": round(float(raw_pols["pm10"]), 1) if raw_pols.get("pm10") is not None else None,
                "no2": round(float(raw_pols["no2"]), 1) if raw_pols.get("no2") is not None else None,
                "so2": round(float(raw_pols["so2"]), 1) if raw_pols.get("so2") is not None else None,
                "co": normalize_co_to_mg_m3(raw_pols.get("co")),
                "o3": round(float(raw_pols["o3"]), 1) if raw_pols.get("o3") is not None else None,
                "nh3": round(float(raw_pols["nh3"]), 1) if raw_pols.get("nh3") is not None else None,
            },
            "lags": {
                "lag_1h": round(lag_1h, 1),
                "lag_3h": round(lag_3h, 1),
                "lag_6h": round(lag_6h, 1),
                "lag_12h": round(lag_12h, 1),
                "lag_24h": round(lag_24h, 1),
            },
            "source": pollution_source_label,
        },
        "forecast_24h": {
            "predicted_aqi": predicted_aqi,
            "predicted_category": predicted_category,
            "aqi_change_24h": round(predicted_aqi - current_aqi, 1),
            "direction": "Increase" if predicted_aqi > current_aqi + 2 else ("Decrease" if predicted_aqi < current_aqi - 2 else "Stable"),
            "interval_80_low": pred_low_80,
            "interval_80_high": pred_high_80,
            "confidence_level": confidence_level,
            "feature_completeness_pct": feature_completeness_pct,
            "peak_24h_aqi": round(peak_24h_aqi, 1),
            "peak_24h_category": aqi_to_cpcb_category(peak_24h_aqi),
        },
        "risk_alert": {
            "state": risk_state,
            "severity": risk_severity,
            "unsafe_threshold_aqi": 200,
            "currently_unsafe": current_aqi > 200.0,
            "expected_to_breach": peak_24h_aqi > 200.0,
            "expected_breach_time": expected_breach_time or ("Already above 200 AQI" if current_aqi > 200.0 else "No breach (>200 AQI) projected in next 24h"),
        },
        "trends": {
            "past_24h": past_24h_points,
            "next_24h": forecast_trajectory,
        },
        "weather": weather_merged,
        "contextual_factors": {
            "festival_and_seasonal": context,
            "nasa_firms": firms_live,
        },
        "explanations": explanations,
        "health_precautions": health_advice,
        "model_info": {
            "selected_model": selected_model_name,
            "model_version": model_version,
        },
    }
