"""
AirShield — Environmental & Air Quality Data Sources Module (data_sources.py)
Handles:
1. Official CPCB (data.gov.in) live API + OpenAQ v3 live CPCB station integration + multi-year historical CPCB datasets (2015–2025)
2. Official CPCB National AQI sub-index & breakpoint interpolation engine
3. OpenWeatherMap live meteorology, 24h forecast & air pollution integration
4. Open-Meteo Forecast (with +24h target-window meteorology & PBL), Historical Weather Archive, and Air Quality APIs
5. Open-Meteo Geocoding + GeoNames Indian city directory + OpenWeatherMap Geocoding fallback (no Google Maps)
6. NASA FIRMS VIIRS satellite fire detection integration (24h & 72h fire counts + Fire Radiative Power)
7. Regional & National Indian Festival & Local Celebration Geodesic Distance Engine (2015–2027) + Live Nager.Date Holiday API
8. SQLite caching, retry with exponential backoff, and prediction history persistence
"""

import os
import json
import math
import time
import sqlite3
import logging
from io import StringIO
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any

import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Never override environment variables injected by the runtime Secrets panel
load_dotenv(override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("airshield.data_sources")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(DATA_DIR, "airshield.db")
CPCB_HIST_CSV = os.path.join(DATA_DIR, "cpcb_city_day.csv")
CPCB_RECENT_CSV = os.path.join(DATA_DIR, "cpcb_recent_2023_2025.csv")
WEATHER_HIST_CSV = os.path.join(DATA_DIR, "indian_cities_weather_archive.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

IST = timezone(timedelta(hours=5, minutes=30))

CPCB_RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
CPCB_ARCHIVE_MIRROR_URL = (
    "https://raw.githubusercontent.com/GowriShaju/Air-Quality-Prediction-EDA/main/city_day.csv"
)

# Geographic source centroids for seasonal aerosol transport (pure distance calculation)
STUBBLE_BURNING_BELT_LAT = 30.3500  # Sangrur / Patiala / Ludhiana crop-residue belt centroid
STUBBLE_BURNING_BELT_LON = 75.8500
THAR_DESERT_DUST_LAT = 26.9157      # Jaisalmer / Thar Desert mineral dust source centroid
THAR_DESERT_DUST_LON = 70.9083


# =====================================================================
# 1. OFFICIAL CPCB BREAKPOINT TABLE & SUB-INDEX CALCULATION
# =====================================================================

CPCB_BREAKPOINTS: Dict[str, List[Tuple[float, float, int, int]]] = {
    "pm25": [
        (0.0, 30.0, 0, 50),
        (30.001, 60.0, 51, 100),
        (60.001, 90.0, 101, 200),
        (90.001, 120.0, 201, 300),
        (120.001, 250.0, 301, 400),
        (250.001, 1000.0, 401, 500),
    ],
    "pm10": [
        (0.0, 50.0, 0, 50),
        (50.001, 100.0, 51, 100),
        (100.001, 250.0, 101, 200),
        (250.001, 350.0, 201, 300),
        (350.001, 430.0, 301, 400),
        (430.001, 1500.0, 401, 500),
    ],
    "no2": [
        (0.0, 40.0, 0, 50),
        (40.001, 80.0, 51, 100),
        (80.001, 180.0, 101, 200),
        (180.001, 280.0, 201, 300),
        (280.001, 400.0, 301, 400),
        (400.001, 1200.0, 401, 500),
    ],
    "so2": [
        (0.0, 40.0, 0, 50),
        (40.001, 80.0, 51, 100),
        (80.001, 380.0, 101, 200),
        (380.001, 800.0, 201, 300),
        (800.001, 1600.0, 301, 400),
        (1600.001, 3000.0, 401, 500),
    ],
    "co": [
        (0.0, 1.0, 0, 50),
        (1.001, 2.0, 51, 100),
        (2.001, 10.0, 101, 200),
        (10.001, 17.0, 201, 300),
        (17.001, 34.0, 301, 400),
        (34.001, 100.0, 401, 500),
    ],
    "o3": [
        (0.0, 50.0, 0, 50),
        (50.001, 100.0, 51, 100),
        (100.001, 168.0, 101, 200),
        (168.001, 208.0, 201, 300),
        (208.001, 748.0, 301, 400),
        (748.001, 1500.0, 401, 500),
    ],
    "nh3": [
        (0.0, 200.0, 0, 50),
        (200.001, 400.0, 51, 100),
        (400.001, 800.0, 101, 200),
        (800.001, 1200.0, 201, 300),
        (1200.001, 1800.0, 301, 400),
        (1800.001, 3000.0, 401, 500),
    ],
}


def normalize_co_to_mg_m3(co_val: Optional[float]) -> Optional[float]:
    """Normalize CO concentration to mg/m³. Values > 50 indicate µg/m³."""
    if co_val is None or pd.isna(co_val) or co_val < 0:
        return None
    val = float(co_val)
    if val > 50.0:
        return round(val / 1000.0, 3)
    return round(val, 3)


def calculate_cpcb_subindex(pollutant: str, concentration: Optional[float]) -> Optional[float]:
    """Calculate the official Indian CPCB sub-index for a single pollutant."""
    if concentration is None or pd.isna(concentration):
        return None
    c = float(concentration)
    if c < 0:
        return None

    key = pollutant.lower().replace(".", "").replace("_", "")
    if key == "co":
        c = normalize_co_to_mg_m3(c)
        if c is None:
            return None

    if key not in CPCB_BREAKPOINTS:
        return None

    for b_lo, b_hi, i_lo, i_hi in CPCB_BREAKPOINTS[key]:
        if c <= b_hi:
            c_clamped = max(c, b_lo)
            sub_idx = ((i_hi - i_lo) / (b_hi - b_lo)) * (c_clamped - b_lo) + i_lo
            return round(float(sub_idx), 1)

    return 500.0


def calculate_cpcb_aqi(
    pollutants: Dict[str, Optional[float]],
    require_three_pollutants: bool = True,
) -> Dict[str, Any]:
    """
    Official CPCB National AQI calculation:
    Requires at least 3 pollutant sub-indices including PM2.5 or PM10 when require_three_pollutants=True.
    """
    sub_indices: Dict[str, float] = {}
    for pol in ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"]:
        val = pollutants.get(pol)
        sub = calculate_cpcb_subindex(pol, val)
        if sub is not None:
            sub_indices[pol] = sub

    has_pm = ("pm25" in sub_indices) or ("pm10" in sub_indices)
    min_count = 3 if require_three_pollutants else 1

    if not has_pm or len(sub_indices) < min_count:
        return {
            "aqi": None,
            "category": "Insufficient Data",
            "dominant_pollutant": None,
            "sub_indices": sub_indices,
            "qualifies_cpcb_rule": False,
        }

    dominant_pol = max(sub_indices, key=sub_indices.get)
    aqi_val = round(sub_indices[dominant_pol])
    return {
        "aqi": float(aqi_val),
        "category": aqi_to_cpcb_category(aqi_val),
        "dominant_pollutant": dominant_pol.upper().replace("PM25", "PM2.5"),
        "sub_indices": sub_indices,
        "qualifies_cpcb_rule": True,
    }


def aqi_to_cpcb_category(aqi: Optional[float]) -> str:
    """Convert numeric Indian AQI to official CPCB AQI category."""
    if aqi is None or pd.isna(aqi):
        return "Insufficient Data"
    val = int(round(float(aqi)))
    if val <= 50:
        return "Good"
    elif val <= 100:
        return "Satisfactory"
    elif val <= 200:
        return "Moderate"
    elif val <= 300:
        return "Poor"
    elif val <= 400:
        return "Very Poor"
    else:
        return "Severe"


def estimate_physical_pbl_m(
    temp_c: Optional[float],
    dew_point_c: Optional[float],
    wind_speed_ms: Optional[float],
    solar_wm2: Optional[float],
) -> float:
    """
    Calculate physical Planetary Boundary Layer (mixing height in meters)
    from surface meteorological parameters when direct radiosonde/model PBL is absent,
    using the Nozaki/ERA5-calibrated thermal + mechanical mixing height formulation.
    """
    t = float(temp_c if temp_c is not None and not pd.isna(temp_c) else 25.0)
    td = float(dew_point_c if dew_point_c is not None and not pd.isna(dew_point_c) else (t - 8.0))
    u = float(wind_speed_ms if wind_speed_ms is not None and not pd.isna(wind_speed_ms) else 2.2)
    sw = float(solar_wm2 if solar_wm2 is not None and not pd.isna(solar_wm2) else 180.0)
    spread = max(0.5, t - td)
    pbl = 42.0 * spread + 68.0 * u + 1.05 * sw
    return round(float(np.clip(pbl, 90.0, 2200.0)), 1)


# =====================================================================
# 2. INDIAN CITIES DIRECTORY & CPCB MONITORING STATION COVERAGE
# =====================================================================

INDIAN_CITIES_DIRECTORY: List[Dict[str, Any]] = [
    {"name": "Delhi", "state": "Delhi", "lat": 28.6139, "lon": 77.2090, "cpcb_stations": 40, "region": "North"},
    {"name": "New Delhi", "state": "Delhi", "lat": 28.6139, "lon": 77.2090, "cpcb_stations": 40, "region": "North"},
    {"name": "Ahmedabad", "state": "Gujarat", "lat": 23.0225, "lon": 72.5714, "cpcb_stations": 9, "region": "West"},
    {"name": "Mumbai", "state": "Maharashtra", "lat": 19.0760, "lon": 72.8777, "cpcb_stations": 24, "region": "West"},
    {"name": "Bengaluru", "state": "Karnataka", "lat": 12.9716, "lon": 77.5946, "cpcb_stations": 14, "region": "South"},
    {"name": "Chennai", "state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707, "cpcb_stations": 9, "region": "South"},
    {"name": "Kolkata", "state": "West Bengal", "lat": 22.5726, "lon": 88.3639, "cpcb_stations": 7, "region": "East"},
    {"name": "Hyderabad", "state": "Telangana", "lat": 17.3850, "lon": 78.4867, "cpcb_stations": 14, "region": "South"},
    {"name": "Pune", "state": "Maharashtra", "lat": 18.5204, "lon": 73.8567, "cpcb_stations": 8, "region": "West"},
    {"name": "Jaipur", "state": "Rajasthan", "lat": 26.9124, "lon": 75.7873, "cpcb_stations": 6, "region": "North"},
    {"name": "Lucknow", "state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462, "cpcb_stations": 7, "region": "North"},
    {"name": "Kanpur", "state": "Uttar Pradesh", "lat": 26.4499, "lon": 80.3319, "cpcb_stations": 4, "region": "North"},
    {"name": "Nagpur", "state": "Maharashtra", "lat": 21.1458, "lon": 79.0882, "cpcb_stations": 4, "region": "Central"},
    {"name": "Indore", "state": "Madhya Pradesh", "lat": 22.7196, "lon": 75.8577, "cpcb_stations": 3, "region": "Central"},
    {"name": "Bhopal", "state": "Madhya Pradesh", "lat": 23.2599, "lon": 77.4126, "cpcb_stations": 3, "region": "Central"},
    {"name": "Patna", "state": "Bihar", "lat": 25.5941, "lon": 85.1376, "cpcb_stations": 6, "region": "East"},
    {"name": "Gurugram", "state": "Haryana", "lat": 28.4595, "lon": 77.0266, "cpcb_stations": 4, "region": "North"},
    {"name": "Noida", "state": "Uttar Pradesh", "lat": 28.5355, "lon": 77.3910, "cpcb_stations": 4, "region": "North"},
    {"name": "Ghaziabad", "state": "Uttar Pradesh", "lat": 28.6692, "lon": 77.4538, "cpcb_stations": 4, "region": "North"},
    {"name": "Faridabad", "state": "Haryana", "lat": 28.4089, "lon": 77.3178, "cpcb_stations": 4, "region": "North"},
    {"name": "Amritsar", "state": "Punjab", "lat": 31.6340, "lon": 74.8723, "cpcb_stations": 2, "region": "North"},
    {"name": "Ludhiana", "state": "Punjab", "lat": 30.9010, "lon": 75.8573, "cpcb_stations": 2, "region": "North"},
    {"name": "Chandigarh", "state": "Chandigarh", "lat": 30.7333, "lon": 76.7794, "cpcb_stations": 3, "region": "North"},
    {"name": "Visakhapatnam", "state": "Andhra Pradesh", "lat": 17.6868, "lon": 83.2185, "cpcb_stations": 2, "region": "South"},
    {"name": "Amaravati", "state": "Andhra Pradesh", "lat": 16.5131, "lon": 80.5165, "cpcb_stations": 2, "region": "South"},
    {"name": "Vijayawada", "state": "Andhra Pradesh", "lat": 16.5062, "lon": 80.6480, "cpcb_stations": 2, "region": "South"},
    {"name": "Coimbatore", "state": "Tamil Nadu", "lat": 11.0168, "lon": 76.9558, "cpcb_stations": 2, "region": "South"},
    {"name": "Kochi", "state": "Kerala", "lat": 9.9312, "lon": 76.2673, "cpcb_stations": 2, "region": "South"},
    {"name": "Ernakulam", "state": "Kerala", "lat": 9.9816, "lon": 76.2999, "cpcb_stations": 2, "region": "South"},
    {"name": "Thiruvananthapuram", "state": "Kerala", "lat": 8.5241, "lon": 76.9366, "cpcb_stations": 2, "region": "South"},
    {"name": "Guwahati", "state": "Assam", "lat": 26.1445, "lon": 91.7362, "cpcb_stations": 2, "region": "Northeast"},
    {"name": "Shillong", "state": "Meghalaya", "lat": 25.5788, "lon": 91.8933, "cpcb_stations": 1, "region": "Northeast"},
    {"name": "Aizawl", "state": "Mizoram", "lat": 23.7271, "lon": 92.7176, "cpcb_stations": 1, "region": "Northeast"},
    {"name": "Talcher", "state": "Odisha", "lat": 20.9500, "lon": 85.2167, "cpcb_stations": 1, "region": "East"},
    {"name": "Brajrajnagar", "state": "Odisha", "lat": 21.8167, "lon": 83.9167, "cpcb_stations": 1, "region": "East"},
    {"name": "Bhubaneswar", "state": "Odisha", "lat": 20.2961, "lon": 85.8245, "cpcb_stations": 2, "region": "East"},
    {"name": "Jorapokhar", "state": "Jharkhand", "lat": 23.7000, "lon": 86.4167, "cpcb_stations": 1, "region": "East"},
    {"name": "Ranchi", "state": "Jharkhand", "lat": 23.3441, "lon": 85.3096, "cpcb_stations": 1, "region": "East"},
    {"name": "Surat", "state": "Gujarat", "lat": 21.1702, "lon": 72.8311, "cpcb_stations": 3, "region": "West"},
    {"name": "Vadodara", "state": "Gujarat", "lat": 22.3072, "lon": 73.1812, "cpcb_stations": 2, "region": "West"},
    {"name": "Rajkot", "state": "Gujarat", "lat": 22.3039, "lon": 70.8022, "cpcb_stations": 1, "region": "West"},
    {"name": "Gandhinagar", "state": "Gujarat", "lat": 23.2156, "lon": 72.6369, "cpcb_stations": 2, "region": "West"},
    {"name": "Varanasi", "state": "Uttar Pradesh", "lat": 25.3176, "lon": 82.9739, "cpcb_stations": 4, "region": "North"},
    {"name": "Agra", "state": "Uttar Pradesh", "lat": 27.1767, "lon": 78.0081, "cpcb_stations": 6, "region": "North"},
    {"name": "Prayagraj", "state": "Uttar Pradesh", "lat": 25.4358, "lon": 81.8463, "cpcb_stations": 3, "region": "North"},
    {"name": "Meerut", "state": "Uttar Pradesh", "lat": 28.9845, "lon": 77.7064, "cpcb_stations": 3, "region": "North"},
    {"name": "Jodhpur", "state": "Rajasthan", "lat": 26.2389, "lon": 73.0243, "cpcb_stations": 2, "region": "North"},
    {"name": "Udaipur", "state": "Rajasthan", "lat": 24.5854, "lon": 73.7125, "cpcb_stations": 1, "region": "North"},
    {"name": "Kota", "state": "Rajasthan", "lat": 25.2138, "lon": 75.8648, "cpcb_stations": 2, "region": "North"},
    {"name": "Dehradun", "state": "Uttarakhand", "lat": 30.3165, "lon": 78.0322, "cpcb_stations": 1, "region": "North"},
    {"name": "Shimla", "state": "Himachal Pradesh", "lat": 31.1048, "lon": 77.1734, "cpcb_stations": 0, "region": "North"},
    {"name": "Manali", "state": "Himachal Pradesh", "lat": 32.2432, "lon": 77.1892, "cpcb_stations": 0, "region": "North"},
    {"name": "Srinagar", "state": "Jammu and Kashmir", "lat": 34.0837, "lon": 74.7973, "cpcb_stations": 1, "region": "North"},
    {"name": "Jammu", "state": "Jammu and Kashmir", "lat": 32.7266, "lon": 74.8570, "cpcb_stations": 0, "region": "North"},
    {"name": "Leh", "state": "Ladakh", "lat": 34.1526, "lon": 77.5771, "cpcb_stations": 0, "region": "North"},
    {"name": "Raipur", "state": "Chhattisgarh", "lat": 21.2514, "lon": 81.6296, "cpcb_stations": 2, "region": "Central"},
    {"name": "Gwalior", "state": "Madhya Pradesh", "lat": 26.2183, "lon": 78.1828, "cpcb_stations": 2, "region": "Central"},
    {"name": "Jabalpur", "state": "Madhya Pradesh", "lat": 23.1815, "lon": 79.9864, "cpcb_stations": 1, "region": "Central"},
    {"name": "Nashik", "state": "Maharashtra", "lat": 19.9975, "lon": 73.7898, "cpcb_stations": 2, "region": "West"},
    {"name": "Aurangabad", "state": "Maharashtra", "lat": 19.8762, "lon": 75.3433, "cpcb_stations": 2, "region": "West"},
    {"name": "Thane", "state": "Maharashtra", "lat": 19.2183, "lon": 72.9781, "cpcb_stations": 2, "region": "West"},
    {"name": "Navi Mumbai", "state": "Maharashtra", "lat": 19.0330, "lon": 73.0297, "cpcb_stations": 4, "region": "West"},
    {"name": "Panaji", "state": "Goa", "lat": 15.4909, "lon": 73.8278, "cpcb_stations": 0, "region": "West"},
    {"name": "Mysuru", "state": "Karnataka", "lat": 12.2958, "lon": 76.6394, "cpcb_stations": 1, "region": "South"},
    {"name": "Mangaluru", "state": "Karnataka", "lat": 12.9141, "lon": 74.8560, "cpcb_stations": 1, "region": "South"},
    {"name": "Hubballi", "state": "Karnataka", "lat": 15.3647, "lon": 75.1240, "cpcb_stations": 1, "region": "South"},
    {"name": "Madurai", "state": "Tamil Nadu", "lat": 9.9252, "lon": 78.1198, "cpcb_stations": 0, "region": "South"},
    {"name": "Tiruchirappalli", "state": "Tamil Nadu", "lat": 10.7905, "lon": 78.7047, "cpcb_stations": 0, "region": "South"},
    {"name": "Salem", "state": "Tamil Nadu", "lat": 11.6643, "lon": 78.1460, "cpcb_stations": 0, "region": "South"},
    {"name": "Puducherry", "state": "Puducherry", "lat": 11.9416, "lon": 79.8083, "cpcb_stations": 1, "region": "South"},
    {"name": "Kozhikode", "state": "Kerala", "lat": 11.2588, "lon": 75.7804, "cpcb_stations": 1, "region": "South"},
    {"name": "Thrissur", "state": "Kerala", "lat": 10.5276, "lon": 76.2144, "cpcb_stations": 1, "region": "South"},
    {"name": "Asansol", "state": "West Bengal", "lat": 23.6739, "lon": 86.9524, "cpcb_stations": 1, "region": "East"},
    {"name": "Siliguri", "state": "West Bengal", "lat": 26.7271, "lon": 88.3953, "cpcb_stations": 1, "region": "East"},
    {"name": "Durgapur", "state": "West Bengal", "lat": 23.5204, "lon": 87.3119, "cpcb_stations": 1, "region": "East"},
    {"name": "Gaya", "state": "Bihar", "lat": 24.7955, "lon": 85.0002, "cpcb_stations": 2, "region": "East"},
    {"name": "Muzaffarpur", "state": "Bihar", "lat": 26.1209, "lon": 85.3647, "cpcb_stations": 2, "region": "East"},
    {"name": "Bhuj", "state": "Gujarat", "lat": 23.2420, "lon": 69.6669, "cpcb_stations": 0, "region": "West"},
    {"name": "Anand", "state": "Gujarat", "lat": 22.5645, "lon": 72.9289, "cpcb_stations": 0, "region": "West"},
    {"name": "Bhavnagar", "state": "Gujarat", "lat": 21.7645, "lon": 72.1519, "cpcb_stations": 0, "region": "West"},
    {"name": "Jamnagar", "state": "Gujarat", "lat": 22.4707, "lon": 70.0577, "cpcb_stations": 0, "region": "West"},
    {"name": "Imphal", "state": "Manipur", "lat": 24.8170, "lon": 93.9368, "cpcb_stations": 1, "region": "Northeast"},
    {"name": "Agartala", "state": "Tripura", "lat": 23.8315, "lon": 91.2868, "cpcb_stations": 1, "region": "Northeast"},
    {"name": "Gangtok", "state": "Sikkim", "lat": 27.3389, "lon": 88.6065, "cpcb_stations": 1, "region": "Northeast"},
    {"name": "Itanagar", "state": "Arunachal Pradesh", "lat": 27.0844, "lon": 93.6053, "cpcb_stations": 0, "region": "Northeast"},
    {"name": "Kohima", "state": "Nagaland", "lat": 25.6751, "lon": 94.1086, "cpcb_stations": 1, "region": "Northeast"},
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def resolve_cpcb_station_coverage(lat: float, lon: float, city_name: str = "") -> Dict[str, Any]:
    cpcb_cities = [c for c in INDIAN_CITIES_DIRECTORY if c.get("cpcb_stations", 0) > 0]
    best_city = None
    min_dist = 99999.0

    for c in cpcb_cities:
        dist = haversine_km(lat, lon, c["lat"], c["lon"])
        if city_name and c["name"].lower() == city_name.strip().lower():
            dist = 0.0
        if dist < min_dist:
            min_dist = dist
            best_city = c

    has_local_station = min_dist <= 25.0
    return {
        "has_cpcb_station": has_local_station,
        "nearest_cpcb_city": best_city["name"] if best_city else "Delhi",
        "nearest_cpcb_state": best_city["state"] if best_city else "Delhi",
        "nearest_cpcb_lat": best_city["lat"] if best_city else 28.6139,
        "nearest_cpcb_lon": best_city["lon"] if best_city else 77.2090,
        "station_count": best_city["cpcb_stations"] if (best_city and has_local_station) else 0,
        "distance_to_station_km": round(min_dist, 1),
        "region": best_city["region"] if best_city else infer_indian_region(lat, lon),
        "coverage_notice": (
            f"Verified CPCB monitoring coverage ({best_city['cpcb_stations']} active station(s) in {best_city['name']})."
            if has_local_station and best_city
            else f"Limited local CPCB station coverage — nearest CPCB monitoring network is in {best_city['name']} ({round(min_dist, 1)} km away). Forecast uses regional meteorological & atmospheric estimation."
        ),
    }


def infer_indian_region(lat: float, lon: float, state: str = "") -> str:
    st = (state or "").lower()
    if any(k in st for k in ["punjab", "haryana", "delhi", "uttar pradesh", "rajasthan", "chandigarh", "uttarakhand", "himachal", "jammu", "kashmir", "ladakh"]):
        return "North"
    if any(k in st for k in ["gujarat", "maharashtra", "goa"]):
        return "West"
    if any(k in st for k in ["karnataka", "tamil nadu", "kerala", "andhra", "telangana", "puducherry"]):
        return "South"
    if any(k in st for k in ["assam", "meghalaya", "mizoram", "manipur", "tripura", "nagaland", "arunachal", "sikkim"]):
        return "Northeast"
    if any(k in st for k in ["west bengal", "bihar", "jharkhand", "odisha"]):
        return "East"
    if any(k in st for k in ["madhya pradesh", "chhattisgarh"]):
        return "Central"
    if lat >= 25.5 and lon <= 84.0:
        return "North"
    if lat < 18.0:
        return "South"
    if lon >= 89.0:
        return "Northeast"
    if lon >= 83.0:
        return "East"
    if lon <= 75.5:
        return "West"
    return "Central"


# =====================================================================
# 3. SQLITE DATABASE & CACHING LAYER
# =====================================================================

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_cache (
                cache_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                fetched_at TEXT NOT NULL,
                source TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cpcb_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city TEXT NOT NULL,
                state TEXT,
                station TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                observed_at TEXT NOT NULL,
                aqi REAL,
                aqi_category TEXT,
                pm25 REAL,
                pm10 REAL,
                no2 REAL,
                so2 REAL,
                co REAL,
                o3 REAL,
                nh3 REAL,
                source TEXT NOT NULL,
                UNIQUE(city, station, observed_at)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city TEXT NOT NULL,
                state TEXT,
                station TEXT,
                latitude REAL,
                longitude REAL,
                predicted_at TEXT NOT NULL,
                target_timestamp TEXT NOT NULL,
                current_aqi REAL,
                predicted_aqi REAL NOT NULL,
                predicted_category TEXT NOT NULL,
                model_version TEXT NOT NULL,
                data_source TEXT NOT NULL,
                features_json TEXT,
                actual_aqi REAL,
                actual_category TEXT,
                matched_at TEXT
            )
            """
        )
        conn.commit()


init_db()


def get_cached_json(cache_key: str, max_age_seconds: int = 900) -> Tuple[Optional[Any], Optional[str], bool]:
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT payload, fetched_at FROM api_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if not row:
                return None, None, False
            fetched_at = row["fetched_at"]
            payload = json.loads(row["payload"])
            dt = datetime.fromisoformat(fetched_at)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - dt).total_seconds()
            return payload, fetched_at, (age <= max_age_seconds)
    except Exception as exc:
        logger.warning("Cache read warning for %s: %s", cache_key, exc)
        return None, None, False


def set_cached_json(cache_key: str, payload: Any, source: str) -> str:
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO api_cache (cache_key, payload, fetched_at, source)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload = excluded.payload,
                    fetched_at = excluded.fetched_at,
                    source = excluded.source
                """,
                (cache_key, json.dumps(payload), now_iso, source),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("Cache write warning for %s: %s", cache_key, exc)
    return now_iso


def http_get_with_retry(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 8.0,
    retries: int = 2,
) -> Optional[requests.Response]:
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            return resp
        except requests.RequestException:
            if attempt < retries:
                time.sleep(0.4 * (2 ** attempt))
            else:
                return None
    return None


# =====================================================================
# 4. GEOCODING & INDIAN CITY AUTOCOMPLETE (NO GOOGLE MAPS)
# =====================================================================

def search_indian_cities(query: str, limit: int = 8) -> List[Dict[str, Any]]:
    """
    Search Indian cities and populated places with multi-provider fallback:
    1. Open-Meteo Geocoding API (countryCode=IN)
    2. Built-in GeoNames-derived Indian cities directory (with phonetic aliases)
    3. OpenWeatherMap Direct Geocoding API (fallback)
    """
    q = (query or "").strip()
    if not q:
        results = []
        for c in INDIAN_CITIES_DIRECTORY[:limit]:
            cov = resolve_cpcb_station_coverage(c["lat"], c["lon"], c["name"])
            results.append({
                "name": c["name"],
                "state": c["state"],
                "country": "India",
                "lat": c["lat"],
                "lon": c["lon"],
                "provider": "GeoNames Directory",
                "has_cpcb_station": cov["has_cpcb_station"],
                "cpcb_stations": cov["station_count"],
                "display_name": f"{c['name']}, {c['state']}",
            })
        return results

    cache_key = f"geocode_in_v5:{q.lower()}:{limit}"
    cached, _, is_fresh = get_cached_json(cache_key, max_age_seconds=86400)
    if cached and is_fresh:
        return cached

    seen_keys = set()
    suggestions: List[Dict[str, Any]] = []

    def add_candidate(name: str, state: str, lat: float, lon: float, provider: str, district: str = ""):
        clean_name = name.strip()
        clean_state = (state or "India").strip()
        dedup_key = (clean_name.lower(), clean_state.lower(), round(lat, 1), round(lon, 1))
        if dedup_key in seen_keys:
            return
        seen_keys.add(dedup_key)
        cov = resolve_cpcb_station_coverage(lat, lon, clean_name)
        label = f"{clean_name}, {clean_state}"
        if district and district.lower() not in (clean_name.lower(), clean_state.lower()):
            label = f"{clean_name} ({district}), {clean_state}"
        suggestions.append({
            "name": clean_name,
            "state": clean_state,
            "district": district,
            "country": "India",
            "lat": round(float(lat), 4),
            "lon": round(float(lon), 4),
            "provider": provider,
            "has_cpcb_station": cov["has_cpcb_station"],
            "cpcb_stations": cov["station_count"],
            "nearest_cpcb_city": cov["nearest_cpcb_city"],
            "distance_to_station_km": cov["distance_to_station_km"],
            "display_name": label,
        })

    q_low = q.lower()
    city_aliases = {
        "ahmedabad": ["ahe", "ahem", "ahemdabad", "amdavad"],
        "bengaluru": ["bangalore", "blr", "beng"],
        "mumbai": ["bombay", "bom"],
        "kolkata": ["calcutta", "cal"],
        "chennai": ["madras"],
        "gurugram": ["gurgaon", "ggn"],
        "varanasi": ["banaras", "benares", "kashi"],
        "prayagraj": ["allahabad"],
        "vadodara": ["baroda"],
        "visakhapatnam": ["vizag", "waltair"],
        "thiruvananthapuram": ["trivandrum"],
        "kochi": ["cochin"],
        "mysuru": ["mysore"],
        "mangaluru": ["mangalore"],
        "hubballi": ["hubli"],
        "puducherry": ["pondicherry", "pondy"],
        "guwahati": ["gauhati"],
    }
    for c in INDIAN_CITIES_DIRECTORY:
        c_low = c["name"].lower()
        aliases = city_aliases.get(c_low, [])
        alias_match = any(a.startswith(q_low) or q_low in a for a in aliases)
        if c_low.startswith(q_low) or q_low in c_low or q_low in c["state"].lower() or alias_match:
            add_candidate(c["name"], c["state"], c["lat"], c["lon"], "GeoNames India")

    resp = http_get_with_retry(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": q,
            "count": 15,
            "language": "en",
            "format": "json",
            "countryCode": "IN",
        },
        timeout=5.0,
        retries=1,
    )
    if resp and resp.status_code == 200:
        data = resp.json()
        for item in data.get("results", []):
            if item.get("country_code", "").upper() != "IN":
                continue
            add_candidate(
                name=item.get("name", ""),
                state=item.get("admin1", "India"),
                lat=float(item.get("latitude", 0.0)),
                lon=float(item.get("longitude", 0.0)),
                provider="Open-Meteo Geocoding",
                district=item.get("admin2", ""),
            )

    owm_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if len(suggestions) < 5 and owm_key and owm_key != "YOUR_OPENWEATHER_API_KEY":
        resp_owm = http_get_with_retry(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": f"{q},IN", "limit": 8, "appid": owm_key},
            timeout=5.0,
            retries=1,
        )
        if resp_owm and resp_owm.status_code == 200:
            for item in resp_owm.json():
                if item.get("country", "").upper() == "IN":
                    add_candidate(
                        name=item.get("name", ""),
                        state=item.get("state", "India"),
                        lat=float(item.get("lat", 0.0)),
                        lon=float(item.get("lon", 0.0)),
                        provider="OpenWeatherMap Geocoding",
                    )

    def sort_key(item: Dict[str, Any]):
        n_low = item["name"].lower()
        aliases = city_aliases.get(n_low, [])
        starts = 0 if (n_low.startswith(q_low) or any(a.startswith(q_low) for a in aliases)) else 1
        has_st = 0 if item["has_cpcb_station"] else 1
        return (starts, has_st, item["name"])

    suggestions.sort(key=sort_key)
    final_list = suggestions[:limit]
    if final_list:
        set_cached_json(cache_key, final_list, "geocoding")
    return final_list


# =====================================================================
# 5. REGIONAL & NATIONAL FESTIVAL / CELEBRATION DISTANCE ENGINE (ZERO HARDCODED WEIGHTS)
# =====================================================================

# Exact lunar/solar calendar dates (2015-2027) and regional epicenter coordinates (lat, lon)
# so the ML model learns spatial-temporal festival impacts from distance & days without arbitrary weights.
DIWALI_DATES_BY_YEAR: Dict[int, str] = {
    2015: "2015-11-11",
    2016: "2016-10-30",
    2017: "2017-10-19",
    2018: "2018-11-07",
    2019: "2019-10-27",
    2020: "2020-11-14",
    2021: "2021-11-04",
    2022: "2022-10-24",
    2023: "2023-11-12",
    2024: "2024-10-31",
    2025: "2025-10-20",
    2026: "2026-11-08",
    2027: "2027-10-29",
}

DUSSEHRA_DATES_BY_YEAR: Dict[int, str] = {
    2015: "2015-10-22",
    2016: "2016-10-11",
    2017: "2017-09-30",
    2018: "2018-10-19",
    2019: "2019-10-08",
    2020: "2020-10-25",
    2021: "2021-10-15",
    2022: "2022-10-05",
    2023: "2023-10-24",
    2024: "2024-10-12",
    2025: "2025-10-02",
    2026: "2026-10-20",
    2027: "2027-10-09",
}

HOLI_DATES_BY_YEAR: Dict[int, str] = {
    2015: "2015-03-06",
    2016: "2016-03-24",
    2017: "2017-03-13",
    2018: "2018-03-02",
    2019: "2019-03-21",
    2020: "2020-03-10",
    2021: "2021-03-29",
    2022: "2022-03-18",
    2023: "2023-03-08",
    2024: "2024-03-25",
    2025: "2025-03-14",
    2026: "2026-03-04",
    2027: "2027-03-22",
}

# Regional Festival Definitions with Geographic Epicenters (lat, lon, hub_name)
# Covers region-specific celebrations across East, West, North, South, Central, and Northeast India.
REGIONAL_CELEBRATION_TEMPLATES: List[Dict[str, Any]] = [
    {
        "name": "Durga Puja & Vijayadashami Pandal Processions",
        "region_label": "Eastern & Northeastern India (Kolkata, Guwahati, Patna, Bhubaneswar)",
        "dates": {
            2015: "2015-10-21", 2016: "2016-10-10", 2017: "2017-09-29", 2018: "2018-10-18",
            2019: "2019-10-07", 2020: "2020-10-24", 2021: "2021-10-14", 2022: "2022-10-04",
            2023: "2023-10-23", 2024: "2024-10-11", 2025: "2025-10-01", 2026: "2026-10-19",
            2027: "2027-10-08",
        },
        "epicenters": [
            (22.5726, 88.3639, "Kolkata"),
            (26.1445, 91.7362, "Guwahati"),
            (25.5941, 85.1376, "Patna"),
            (20.2961, 85.8245, "Bhubaneswar"),
            (23.3441, 85.3096, "Ranchi"),
        ],
    },
    {
        "name": "Kali Puja & Eastern Deepavali Fireworks",
        "region_label": "West Bengal, Assam, Odisha & Bihar",
        "dates": DIWALI_DATES_BY_YEAR,
        "epicenters": [
            (22.5726, 88.3639, "Kolkata"),
            (26.1445, 91.7362, "Guwahati"),
            (23.7000, 86.4167, "Jorapokhar"),
            (20.9500, 85.2167, "Talcher"),
        ],
    },
    {
        "name": "Chhath Puja (Sandhya & Usha Arghya)",
        "region_label": "Bihar, Jharkhand, Eastern UP & Delhi NCR",
        "dates": {
            2015: "2015-11-17", 2016: "2016-11-06", 2017: "2017-10-26", 2018: "2018-11-13",
            2019: "2019-11-02", 2020: "2020-11-20", 2021: "2021-11-10", 2022: "2022-10-30",
            2023: "2023-11-19", 2024: "2024-11-07", 2025: "2025-10-27", 2026: "2026-11-15",
            2027: "2027-11-04",
        },
        "epicenters": [
            (25.5941, 85.1376, "Patna"),
            (25.3176, 82.9739, "Varanasi"),
            (23.3441, 85.3096, "Ranchi"),
            (26.8467, 80.9462, "Lucknow"),
            (28.6139, 77.2090, "Delhi"),
        ],
    },
    {
        "name": "Ganesh Chaturthi & Anant Chaturdashi Visarjan",
        "region_label": "Maharashtra, Telangana, Karnataka & Gujarat",
        "dates": {
            2015: "2015-09-22", 2016: "2016-09-10", 2017: "2017-08-30", 2018: "2018-09-18",
            2019: "2019-09-07", 2020: "2020-08-27", 2021: "2021-09-15", 2022: "2022-09-04",
            2023: "2023-09-23", 2024: "2024-09-12", 2025: "2025-09-01", 2026: "2026-09-19",
            2027: "2027-09-09",
        },
        "epicenters": [
            (19.0760, 72.8777, "Mumbai"),
            (18.5204, 73.8567, "Pune"),
            (17.3850, 78.4867, "Hyderabad"),
            (21.1458, 79.0882, "Nagpur"),
            (12.9716, 77.5946, "Bengaluru"),
        ],
    },
    {
        "name": "Guru Nanak Jayanti (Gurpurab) & Nagar Kirtan",
        "region_label": "Punjab, Haryana, Chandigarh & Delhi NCR",
        "dates": {
            2015: "2015-11-25", 2016: "2016-11-14", 2017: "2017-11-04", 2018: "2018-11-23",
            2019: "2019-11-12", 2020: "2020-11-30", 2021: "2021-11-19", 2022: "2022-11-08",
            2023: "2023-11-27", 2024: "2024-11-15", 2025: "2025-11-05", 2026: "2026-11-24",
            2027: "2027-11-14",
        },
        "epicenters": [
            (31.6340, 74.8723, "Amritsar"),
            (30.9010, 75.8573, "Ludhiana"),
            (30.7333, 76.7794, "Chandigarh"),
            (28.6139, 77.2090, "Delhi"),
        ],
    },
    {
        "name": "Lohri, Uttarayan & Bhogi Harvest Bonfires",
        "region_label": "Punjab/NCR (Lohri), Gujarat (Uttarayan), TN/AP/Telangana (Bhogi/Pongal)",
        "fixed_month_day": (1, 14),
        "epicenters": [
            (31.6340, 74.8723, "Amritsar"),
            (30.7333, 76.7794, "Chandigarh"),
            (28.6139, 77.2090, "Delhi"),
            (23.0225, 72.5714, "Ahmedabad"),
            (13.0827, 80.2707, "Chennai"),
            (17.3850, 78.4867, "Hyderabad"),
            (16.5131, 80.5165, "Amaravati"),
        ],
    },
    {
        "name": "Vishu & Puthandu Regional Celebrations",
        "region_label": "Kerala & Tamil Nadu",
        "fixed_month_day": (4, 14),
        "epicenters": [
            (9.9312, 76.2673, "Kochi"),
            (8.5241, 76.9366, "Thiruvananthapuram"),
            (13.0827, 80.2707, "Chennai"),
            (11.0168, 76.9558, "Coimbatore"),
        ],
    },
    {
        "name": "New Year's Eve & Day Celebrations",
        "region_label": "Pan-India Major Metropolitan Centers",
        "fixed_month_day": (1, 1),
        "epicenters": [
            (28.6139, 77.2090, "Delhi"),
            (19.0760, 72.8777, "Mumbai"),
            (12.9716, 77.5946, "Bengaluru"),
            (22.5726, 88.3639, "Kolkata"),
            (13.0827, 80.2707, "Chennai"),
            (17.3850, 78.4867, "Hyderabad"),
            (23.0225, 72.5714, "Ahmedabad"),
        ],
    },
]


def build_expanded_festival_instances(start_year: int = 2015, end_year: int = 2027) -> List[Dict[str, Any]]:
    """Expand all national and regional Indian celebrations into dated spatial event records."""
    instances: List[Dict[str, Any]] = []
    major_hubs = [
        (28.6139, 77.2090, "Delhi"),
        (23.0225, 72.5714, "Ahmedabad"),
        (19.0760, 72.8777, "Mumbai"),
        (22.5726, 88.3639, "Kolkata"),
        (26.8467, 80.9462, "Lucknow"),
        (25.5941, 85.1376, "Patna"),
        (26.9124, 75.7873, "Jaipur"),
        (17.3850, 78.4867, "Hyderabad"),
        (12.9716, 77.5946, "Bengaluru"),
        (13.0827, 80.2707, "Chennai"),
        (31.6340, 74.8723, "Amritsar"),
        (23.2599, 77.4126, "Bhopal"),
        (26.1445, 91.7362, "Guwahati"),
    ]
    for yr in range(start_year, end_year + 1):
        if yr in DIWALI_DATES_BY_YEAR:
            instances.append({
                "name": "Diwali (Deepavali)",
                "date": DIWALI_DATES_BY_YEAR[yr],
                "region_label": "Pan-India National Festival",
                "type": "national_festival",
                "epicenters": major_hubs,
            })
        if yr in DUSSEHRA_DATES_BY_YEAR:
            instances.append({
                "name": "Dussehra (Vijayadashami)",
                "date": DUSSEHRA_DATES_BY_YEAR[yr],
                "region_label": "North, Central, East & South (Mysuru/Delhi/Lucknow/Patna)",
                "type": "national_festival",
                "epicenters": [
                    (28.6139, 77.2090, "Delhi"),
                    (26.8467, 80.9462, "Lucknow"),
                    (25.5941, 85.1376, "Patna"),
                    (26.9124, 75.7873, "Jaipur"),
                    (12.2958, 76.6394, "Mysuru"),
                    (23.0225, 72.5714, "Ahmedabad"),
                ],
            })
        if yr in HOLI_DATES_BY_YEAR:
            instances.append({
                "name": "Holi & Holika Dahan",
                "date": HOLI_DATES_BY_YEAR[yr],
                "region_label": "North, West, Central & East India",
                "type": "national_festival",
                "epicenters": [
                    (28.6139, 77.2090, "Delhi"),
                    (27.1767, 78.0081, "Agra/Mathura"),
                    (26.9124, 75.7873, "Jaipur"),
                    (26.8467, 80.9462, "Lucknow"),
                    (25.5941, 85.1376, "Patna"),
                    (23.0225, 72.5714, "Ahmedabad"),
                    (19.0760, 72.8777, "Mumbai"),
                ],
            })
        for tmpl in REGIONAL_CELEBRATION_TEMPLATES:
            if "dates" in tmpl and yr in tmpl["dates"]:
                d_str = tmpl["dates"][yr]
            elif "fixed_month_day" in tmpl:
                m_f, d_f = tmpl["fixed_month_day"]
                d_str = f"{yr:04d}-{m_f:02d}-{d_f:02d}"
            else:
                continue
            instances.append({
                "name": tmpl["name"],
                "date": d_str,
                "region_label": tmpl["region_label"],
                "type": "regional_celebration",
                "epicenters": tmpl["epicenters"],
            })
    return instances


EXPANDED_FESTIVAL_INSTANCES = build_expanded_festival_instances(2015, 2027)


def fetch_live_nager_holidays(year: int) -> List[Dict[str, Any]]:
    """Fetch live Indian gazetted/public holidays from Nager.Date API with SQLite caching."""
    cache_key = f"nager_holidays_in:{year}"
    cached, _, is_fresh = get_cached_json(cache_key, max_age_seconds=86400 * 7)
    if cached and is_fresh:
        return cached

    resp = http_get_with_retry(
        f"https://date.nager.at/api/v3/PublicHolidays/{year}/IN",
        timeout=4.0,
        retries=1,
    )
    if resp and resp.status_code == 200:
        try:
            data = resp.json()
            if isinstance(data, list):
                set_cached_json(cache_key, data, "nager_date_api")
                return data
        except Exception:
            pass
    return cached or []


def compute_event_and_distance_features(
    target_date: datetime,
    lat: float,
    lon: float,
    wind_dir_deg: float = 270.0,
    temp_c: Optional[float] = None,
    rh_pct: Optional[float] = None,
    pbl_m: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute pure spatial-temporal event, regional celebration, and distance features
    for a single (date, lat, lon) observation — used identically in live inference
    and historical validation. Zero hardcoded AQI weights.
    """
    if isinstance(target_date, str):
        dt = datetime.fromisoformat(target_date[:10])
    else:
        dt = target_date

    dt_date = dt.date() if hasattr(dt, "date") else dt
    month = dt_date.month
    day = dt_date.day

    # 1. Signed days from Diwali, Dussehra, Holi
    def signed_days_to_map(date_map: Dict[int, str], clip_val: int = 30) -> int:
        best_diff = clip_val
        for yr in (dt_date.year - 1, dt_date.year, dt_date.year + 1):
            if yr in date_map:
                f_d = datetime.strptime(date_map[yr], "%Y-%m-%d").date()
                diff = (dt_date - f_d).days
                if abs(diff) < abs(best_diff):
                    best_diff = diff
        return int(max(-clip_val, min(clip_val, best_diff)))

    days_from_diwali = signed_days_to_map(DIWALI_DATES_BY_YEAR, 30)
    abs_days_to_diwali = abs(days_from_diwali)
    is_diwali_window = 1 if (-2 <= days_from_diwali <= 4) else 0
    days_from_dussehra = signed_days_to_map(DUSSEHRA_DATES_BY_YEAR, 30)
    days_from_holi = signed_days_to_map(HOLI_DATES_BY_YEAR, 30)

    # 2. Nearest regional or national celebration within +/- 15 days and its exact Haversine distance (km)
    best_fest_info = None
    best_prox_score = -1.0
    min_days_regional = 15
    dist_to_active_festival_km = 1500.0

    for inst in EXPANDED_FESTIVAL_INSTANCES:
        f_d = datetime.strptime(inst["date"], "%Y-%m-%d").date()
        diff_days = (dt_date - f_d).days
        if abs(diff_days) > 15:
            continue

        # Minimum Haversine distance from (lat, lon) to any epicenter of this celebration
        min_ep_dist = 9999.0
        nearest_hub = "India"
        for ep_lat, ep_lon, hub_name in inst["epicenters"]:
            d_km = haversine_km(lat, lon, ep_lat, ep_lon)
            if d_km < min_ep_dist:
                min_ep_dist = d_km
                nearest_hub = hub_name

        # Geometric spatial-temporal proximity in [0, 1] (pure distance & days, no emission weight)
        spatial_factor = max(0.0, 1.0 - (min_ep_dist / 600.0))
        temporal_factor = max(0.0, 1.0 - (abs(diff_days) / 6.0))
        prox = round(spatial_factor * temporal_factor, 4)

        if min_ep_dist <= 550.0 and abs(diff_days) < abs(min_days_regional):
            min_days_regional = diff_days

        if abs(diff_days) <= 6 and min_ep_dist < dist_to_active_festival_km:
            dist_to_active_festival_km = round(min_ep_dist, 1)

        # Rank candidate festival for UI & feature representation
        rank_score = prox + (0.2 if abs(diff_days) <= 7 else 0.0) - (abs(diff_days) * 0.01) - (min_ep_dist * 0.0001)
        if rank_score > best_prox_score:
            best_prox_score = rank_score
            best_fest_info = {
                "name": inst["name"],
                "date": inst["date"],
                "type": inst["type"],
                "region_label": inst["region_label"],
                "nearest_epicenter_hub": nearest_hub,
                "distance_to_epicenter_km": round(min_ep_dist, 1),
                "days_offset": int(diff_days),
                "spatial_temporal_proximity": prox,
            }

    regional_festival_proximity = best_fest_info["spatial_temporal_proximity"] if best_fest_info else 0.0
    is_festival_day = 1 if (best_fest_info and abs(best_fest_info["days_offset"]) <= 1 and best_fest_info["distance_to_epicenter_km"] <= 450.0) else 0

    # 3. Distance to Northern Crop-Residue Burning Belt (Sangrur/Patiala/Ludhiana centroid) & Thar Desert Dust Source
    dist_to_stubble_belt_km = round(haversine_km(lat, lon, STUBBLE_BURNING_BELT_LAT, STUBBLE_BURNING_BELT_LON), 1)
    dist_to_thar_dust_km = round(haversine_km(lat, lon, THAR_DESERT_DUST_LAT, THAR_DESERT_DUST_LON), 1)

    is_stubble_season_date = 1 if ((month == 10 and day >= 10) or (month == 11 and day <= 25)) else 0
    is_stubble_burning_season = 1 if (is_stubble_season_date and dist_to_stubble_belt_km <= 950.0) else 0

    w_dir = float(wind_dir_deg if wind_dir_deg is not None and not pd.isna(wind_dir_deg) else 270.0)
    nw_alignment = max(0.0, math.cos(math.radians(w_dir - 310.0)))
    stubble_dist_decay = max(0.0, 1.0 - (dist_to_stubble_belt_km / 1100.0))
    downwind_stubble_exposure = round(float(is_stubble_season_date) * stubble_dist_decay * nw_alignment, 4)

    is_winter_inversion = 0
    if month in (11, 12, 1, 2):
        if (temp_c is not None and temp_c < 20.0) or (pbl_m is not None and pbl_m < 500.0) or lat >= 23.0:
            is_winter_inversion = 1

    is_dust_season = 0
    if month in (4, 5, 6) and dist_to_thar_dust_km <= 1200.0 and (rh_pct is None or rh_pct < 45.0):
        is_dust_season = 1

    is_monsoon_season = 1 if (month in (7, 8, 9) or (month == 6 and day >= 15)) else 0

    return {
        "days_from_diwali": int(days_from_diwali),
        "abs_days_to_diwali": int(abs_days_to_diwali),
        "is_diwali_window": int(is_diwali_window),
        "days_from_dussehra": int(days_from_dussehra),
        "days_from_holi": int(days_from_holi),
        "days_to_regional_festival": int(min_days_regional),
        "dist_to_active_festival_km": float(dist_to_active_festival_km),
        "regional_festival_proximity": float(regional_festival_proximity),
        "is_festival_day": int(is_festival_day),
        "dist_to_stubble_belt_km": float(dist_to_stubble_belt_km),
        "is_stubble_burning_season": int(is_stubble_burning_season),
        "downwind_stubble_exposure": float(downwind_stubble_exposure),
        "dist_to_thar_dust_km": float(dist_to_thar_dust_km),
        "is_winter_inversion": int(is_winter_inversion),
        "is_dust_season": int(is_dust_season),
        "is_monsoon_season": int(is_monsoon_season),
        "nearest_festival": best_fest_info,
    }


def get_festival_and_seasonal_context(
    target_date: datetime,
    lat: float,
    lon: float,
    state: str = "",
    temp_c: Optional[float] = None,
    wind_speed_ms: Optional[float] = None,
    wind_dir_deg: Optional[float] = 270.0,
    rh_pct: Optional[float] = None,
    pbl_m: Optional[float] = None,
    precip_mm: Optional[float] = None,
) -> Dict[str, Any]:
    if isinstance(target_date, str):
        dt = datetime.fromisoformat(target_date[:10])
    else:
        dt = target_date

    region = infer_indian_region(lat, lon, state)
    ev = compute_event_and_distance_features(
        target_date=dt,
        lat=lat,
        lon=lon,
        wind_dir_deg=float(wind_dir_deg or 270.0),
        temp_c=temp_c,
        rh_pct=rh_pct,
        pbl_m=pbl_m,
    )

    # Also query live Nager.Date public holiday feed for the target year
    nager_holidays = fetch_live_nager_holidays(dt.year)
    live_holiday_match = None
    for h in nager_holidays:
        try:
            h_dt = datetime.strptime(h.get("date", ""), "%Y-%m-%d").date()
            diff = (dt.date() - h_dt).days
            if abs(diff) <= 5:
                live_holiday_match = {"name": h.get("name"), "date": h.get("date"), "days_offset": diff}
                break
        except Exception:
            pass

    is_wet_deposition_active = 1 if (precip_mm is not None and precip_mm >= 1.0) else 0
    is_low_wind_stagnation = 1 if (wind_speed_ms is not None and wind_speed_ms < 2.2) else 0

    active_regimes = []
    nf = ev.get("nearest_festival")
    if nf and abs(nf["days_offset"]) <= 4 and nf["distance_to_epicenter_km"] <= 500:
        rel_str = "Today" if nf["days_offset"] == 0 else (f"{nf['days_offset']}d post-event" if nf["days_offset"] > 0 else f"in {abs(nf['days_offset'])}d")
        active_regimes.append(
            f"{nf['name']} ({rel_str}, {nf['distance_to_epicenter_km']} km from {nf['nearest_epicenter_hub']} hub)"
        )
    if ev["is_stubble_burning_season"]:
        active_regimes.append(
            f"Crop-Residue Plume Corridor ({ev['dist_to_stubble_belt_km']} km from Punjab/Haryana Fire Belt)"
        )
    if ev["is_winter_inversion"]:
        active_regimes.append("Winter Radiative Cooling & Shallow Boundary-Layer Inversion")
    if ev["is_dust_season"]:
        active_regimes.append(f"Pre-Monsoon Arid Dust Transport ({ev['dist_to_thar_dust_km']} km from Thar Source)")
    if is_wet_deposition_active:
        active_regimes.append("Active Precipitation Scavenging (Wet Deposition Washout)")
    elif ev["is_monsoon_season"]:
        active_regimes.append("Southwest Monsoon Maritime Ventilation Regime")
    if is_low_wind_stagnation:
        active_regimes.append("Calm-Wind Atmospheric Stagnation (< 2.2 m/s)")

    out = dict(ev)
    out.update({
        "region": region,
        "month": dt.month,
        "day_of_year": dt.timetuple().tm_yday,
        "day_of_week": dt.weekday(),
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
        "is_wet_deposition_active": is_wet_deposition_active,
        "is_low_wind_stagnation": is_low_wind_stagnation,
        "active_regimes": active_regimes,
        "live_public_holiday": live_holiday_match,
        "festival_calendar_source": "Regional & National Indian Festival Geodesic Database (2015–2027) + Live Nager.Date API",
    })
    return out


# =====================================================================
# 6. LIVE WEATHER, ATMOSPHERIC (+24H FORECAST), OPENAQ v3 CPCB & NASA FIRMS APIS
# =====================================================================

def fetch_openweather_live(lat: float, lon: float) -> Dict[str, Any]:
    cache_key = f"owm_live:{round(lat, 3)}:{round(lon, 3)}"
    cached, fetched_at, is_fresh = get_cached_json(cache_key, max_age_seconds=600)
    if cached and is_fresh:
        cached["cached"] = True
        cached["fetched_at"] = fetched_at
        return cached

    api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_OPENWEATHER_API_KEY":
        if cached:
            cached["cached"] = True
            cached["fetched_at"] = fetched_at
            return cached
        return {"available": False, "status": "OPENWEATHER_API_KEY not configured"}

    wx_resp = http_get_with_retry(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
        timeout=8.0,
    )
    ap_resp = http_get_with_retry(
        "https://api.openweathermap.org/data/2.5/air_pollution",
        params={"lat": lat, "lon": lon, "appid": api_key},
        timeout=8.0,
    )

    if not wx_resp or wx_resp.status_code != 200:
        if cached:
            cached["cached"] = True
            cached["fetched_at"] = fetched_at
            return cached
        return {
            "available": False,
            "status": f"OpenWeatherMap HTTP {wx_resp.status_code if wx_resp else 'Timeout'}",
        }

    wx = wx_resp.json()
    main = wx.get("main", {})
    wind = wx.get("wind", {})
    clouds = wx.get("clouds", {})
    rain = wx.get("rain", {})
    weather_list = wx.get("weather", [{}])

    temp_c = float(main.get("temp", 25.0))
    rh_pct = float(main.get("humidity", 50.0))
    gamma = (17.27 * temp_c) / (237.7 + temp_c) + math.log(max(rh_pct, 1.0) / 100.0)
    dew_point_c = round((237.7 * gamma) / (17.27 - gamma), 2)

    components = {}
    if ap_resp and ap_resp.status_code == 200:
        ap_list = ap_resp.json().get("list", [])
        if ap_list:
            components = ap_list[0].get("components", {})

    result = {
        "available": True,
        "cached": False,
        "source": "OpenWeatherMap Live API",
        "station_name": wx.get("name", ""),
        "observed_timestamp": datetime.fromtimestamp(
            wx.get("dt", int(time.time())), tz=timezone.utc
        ).isoformat(),
        "temperature_c": round(temp_c, 1),
        "feels_like_c": round(float(main.get("feels_like", temp_c)), 1),
        "humidity_pct": round(rh_pct, 1),
        "dew_point_c": dew_point_c,
        "pressure_hpa": round(float(main.get("pressure", 1010.0)), 1),
        "wind_speed_ms": round(float(wind.get("speed", 2.0)), 2),
        "wind_direction_deg": round(float(wind.get("deg", 0.0)), 1),
        "wind_gust_ms": round(float(wind.get("gust", wind.get("speed", 2.0))), 2),
        "cloud_cover_pct": round(float(clouds.get("all", 0.0)), 1),
        "precipitation_mm": round(float(rain.get("1h", rain.get("3h", 0.0))), 2),
        "visibility_m": int(wx.get("visibility", 10000)),
        "weather_condition": weather_list[0].get("main", "Clear"),
        "weather_description": weather_list[0].get("description", "clear sky").title(),
        "owm_pollutants": {
            "pm25": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "no2": components.get("no2"),
            "so2": components.get("so2"),
            "co": normalize_co_to_mg_m3(components.get("co")),
            "o3": components.get("o3"),
            "nh3": components.get("nh3"),
        },
    }
    fetched_iso = set_cached_json(cache_key, result, "openweathermap")
    result["fetched_at"] = fetched_iso
    return result


def fetch_openmeteo_live_and_history(lat: float, lon: float) -> Dict[str, Any]:
    cache_key = f"openmeteo_env_v2:{round(lat, 3)}:{round(lon, 3)}"
    cached, fetched_at, is_fresh = get_cached_json(cache_key, max_age_seconds=900)
    if cached and is_fresh:
        cached["cached"] = True
        cached["fetched_at"] = fetched_at
        return cached

    wx_resp = http_get_with_retry(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "surface_pressure",
                "cloud_cover",
                "wind_speed_10m",
                "wind_direction_10m",
                "wind_gusts_10m",
                "precipitation",
                "shortwave_radiation",
                "boundary_layer_height",
            ]),
            "past_days": 3,
            "forecast_days": 2,
            "timezone": "Asia/Kolkata",
        },
        timeout=10.0,
    )

    aq_resp = http_get_with_retry(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join([
                "pm10",
                "pm2_5",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
                "aerosol_optical_depth",
                "uv_index",
            ]),
            "past_days": 3,
            "forecast_days": 2,
            "timezone": "Asia/Kolkata",
        },
        timeout=10.0,
    )

    if (not wx_resp or wx_resp.status_code != 200) and cached:
        cached["cached"] = True
        cached["fetched_at"] = fetched_at
        return cached

    wx_hourly = wx_resp.json().get("hourly", {}) if (wx_resp and wx_resp.status_code == 200) else {}
    aq_hourly = aq_resp.json().get("hourly", {}) if (aq_resp and aq_resp.status_code == 200) else {}

    times = wx_hourly.get("time", []) or aq_hourly.get("time", [])
    if not times:
        return {"available": False, "status": "Open-Meteo returned empty hourly series"}

    now_ist_str = datetime.now(IST).strftime("%Y-%m-%dT%H:00")
    curr_idx = len(times) // 2
    for idx, t_str in enumerate(times):
        if t_str <= now_ist_str:
            curr_idx = idx

    def safe_at(arr: Optional[List[Any]], idx: int, default: Optional[float] = None) -> Optional[float]:
        if not arr or idx < 0 or idx >= len(arr) or arr[idx] is None:
            return default
        return float(arr[idx])

    hourly_history = []
    start_idx = max(0, curr_idx - 48)
    for i in range(start_idx, curr_idx + 1):
        pols = {
            "pm25": safe_at(aq_hourly.get("pm2_5"), i),
            "pm10": safe_at(aq_hourly.get("pm10"), i),
            "no2": safe_at(aq_hourly.get("nitrogen_dioxide"), i),
            "so2": safe_at(aq_hourly.get("sulphur_dioxide"), i),
            "co": normalize_co_to_mg_m3(safe_at(aq_hourly.get("carbon_monoxide"), i)),
            "o3": safe_at(aq_hourly.get("ozone"), i),
        }
        calc = calculate_cpcb_aqi(pols, require_three_pollutants=False)
        hourly_history.append({
            "timestamp": times[i],
            "aqi": calc["aqi"],
            "pm25": pols["pm25"],
            "pm10": pols["pm10"],
            "no2": pols["no2"],
            "so2": pols["so2"],
            "co": pols["co"],
            "o3": pols["o3"],
            "temp_c": safe_at(wx_hourly.get("temperature_2m"), i),
            "humidity_pct": safe_at(wx_hourly.get("relative_humidity_2m"), i),
            "wind_speed_ms": round((safe_at(wx_hourly.get("wind_speed_10m"), i, 7.2) or 7.2) / 3.6, 2),
            "pbl_m": safe_at(wx_hourly.get("boundary_layer_height"), i),
        })

    # Compute next-24h forecasted meteorology window (curr_idx + 1 .. curr_idx + 24)
    fc_end = min(len(times), curr_idx + 25)
    fc_indices = list(range(curr_idx + 1, fc_end)) or [curr_idx]

    def mean_window(arr: Optional[List[Any]], default: float, scale: float = 1.0) -> float:
        vals = [float(arr[i]) * scale for i in fc_indices if arr and i < len(arr) and arr[i] is not None]
        return round(float(np.mean(vals)), 2) if vals else default

    def sum_window(arr: Optional[List[Any]], default: float = 0.0) -> float:
        vals = [float(arr[i]) for i in fc_indices if arr and i < len(arr) and arr[i] is not None]
        return round(float(np.sum(vals)), 2) if vals else default

    curr_temp = safe_at(wx_hourly.get("temperature_2m"), curr_idx, 25.0) or 25.0
    curr_rh = safe_at(wx_hourly.get("relative_humidity_2m"), curr_idx, 55.0) or 55.0
    curr_wind = round((safe_at(wx_hourly.get("wind_speed_10m"), curr_idx, 7.2) or 7.2) / 3.6, 2)
    curr_pbl = safe_at(wx_hourly.get("boundary_layer_height"), curr_idx)
    if curr_pbl is None:
        curr_pbl = estimate_physical_pbl_m(
            curr_temp,
            safe_at(wx_hourly.get("dew_point_2m"), curr_idx, 15.0),
            curr_wind,
            safe_at(wx_hourly.get("shortwave_radiation"), curr_idx, 180.0),
        )

    next24h_temp_c = mean_window(wx_hourly.get("temperature_2m"), curr_temp)
    next24h_humidity_pct = mean_window(wx_hourly.get("relative_humidity_2m"), curr_rh)
    next24h_wind_speed_ms = mean_window(wx_hourly.get("wind_speed_10m"), curr_wind * 3.6, scale=1.0 / 3.6)
    next24h_precip_mm = sum_window(wx_hourly.get("precipitation"), 0.0)
    next24h_pbl_m = mean_window(wx_hourly.get("boundary_layer_height"), curr_pbl)

    result = {
        "available": True,
        "cached": False,
        "source": "Open-Meteo Atmospheric & Air Quality API",
        "current_hour_ist": times[curr_idx],
        "boundary_layer_height_m": round(float(curr_pbl), 1),
        "dew_point_c": safe_at(wx_hourly.get("dew_point_2m"), curr_idx, 15.0),
        "solar_radiation_wm2": safe_at(wx_hourly.get("shortwave_radiation"), curr_idx, 180.0),
        "uv_index": safe_at(aq_hourly.get("uv_index"), curr_idx, 3.0),
        "aerosol_optical_depth": safe_at(aq_hourly.get("aerosol_optical_depth"), curr_idx, 0.4),
        "temperature_c": curr_temp,
        "humidity_pct": curr_rh,
        "pressure_hpa": safe_at(wx_hourly.get("surface_pressure"), curr_idx, 1008.0),
        "wind_speed_ms": curr_wind,
        "wind_direction_deg": safe_at(wx_hourly.get("wind_direction_10m"), curr_idx, 270.0),
        "precipitation_mm": safe_at(wx_hourly.get("precipitation"), curr_idx, 0.0),
        "cloud_cover_pct": safe_at(wx_hourly.get("cloud_cover"), curr_idx, 20.0),
        "next_24h_weather": {
            "next24h_temp_c": next24h_temp_c,
            "next24h_humidity_pct": next24h_humidity_pct,
            "next24h_wind_speed_ms": next24h_wind_speed_ms,
            "next24h_precip_mm": next24h_precip_mm,
            "next24h_pbl_m": next24h_pbl_m,
            "temp_change_24h": round(next24h_temp_c - curr_temp, 2),
            "wind_change_24h": round(next24h_wind_speed_ms - curr_wind, 2),
        },
        "auxiliary_pollutants": {
            "pm25": safe_at(aq_hourly.get("pm2_5"), curr_idx),
            "pm10": safe_at(aq_hourly.get("pm10"), curr_idx),
            "no2": safe_at(aq_hourly.get("nitrogen_dioxide"), curr_idx),
            "so2": safe_at(aq_hourly.get("sulphur_dioxide"), curr_idx),
            "co": normalize_co_to_mg_m3(safe_at(aq_hourly.get("carbon_monoxide"), curr_idx)),
            "o3": safe_at(aq_hourly.get("ozone"), curr_idx),
        },
        "hourly_history": hourly_history,
    }
    fetched_iso = set_cached_json(cache_key, result, "openmeteo")
    result["fetched_at"] = fetched_iso
    return result


def fetch_openaq_cpcb_live(city_name: str, lat: float, lon: float) -> Dict[str, Any]:
    """
    Retrieve live CPCB monitoring station measurements via OpenAQ v3 API using OPENAQ_API_KEY.
    Prioritizes official CPCB provider stations active in 2025/2026 within 25 km of the selected city.
    """
    oaq_key = os.getenv("OPENAQ_API_KEY", "").strip()
    if not oaq_key or oaq_key == "YOUR_OPENAQ_API_KEY":
        return {"available": False, "status": "OPENAQ_API_KEY not configured", "stations": []}

    cache_key = f"openaq_cpcb_v2:{round(lat, 2)}:{round(lon, 2)}"
    cached, fetched_at, is_fresh = get_cached_json(cache_key, max_age_seconds=900)
    if cached and is_fresh:
        cached["cached"] = True
        cached["fetched_at"] = fetched_at
        return cached

    headers = {"X-API-Key": oaq_key}
    resp = http_get_with_retry(
        "https://api.openaq.org/v3/locations",
        params={"coordinates": f"{round(lat, 4)},{round(lon, 4)}", "radius": 25000, "limit": 100},
        headers=headers,
        timeout=7.0,
        retries=1,
    )
    if not resp or resp.status_code != 200:
        return {"available": False, "status": "OpenAQ v3 unreachable", "stations": []}

    locs = resp.json().get("results", [])
    active_locs = [
        l for l in locs
        if (l.get("datetimeLast") or {}).get("utc", "") >= "2025-01-01"
    ]
    if not active_locs:
        return {"available": False, "status": "No active OpenAQ v3 stations within 25 km", "stations": []}

    active_locs.sort(
        key=lambda x: (
            1 if (x.get("provider") or {}).get("name") == "CPCB" else 0,
            len(x.get("sensors", [])),
            (x.get("datetimeLast") or {}).get("utc", ""),
        ),
        reverse=True,
    )

    station_summaries = []
    for st in active_locs[:3]:
        loc_id = st["id"]
        st_name = st.get("name", f"{city_name} Station")
        prov_name = (st.get("provider") or {}).get("name", "OpenAQ")
        sensor_map = {
            s["id"]: (
                (s.get("parameter") or {}).get("name", "").lower(),
                (s.get("parameter") or {}).get("units", ""),
            )
            for s in st.get("sensors", [])
        }
        r_lat = http_get_with_retry(
            f"https://api.openaq.org/v3/locations/{loc_id}/latest",
            headers=headers,
            timeout=6.0,
            retries=1,
        )
        if not r_lat or r_lat.status_code != 200:
            continue

        pols: Dict[str, float] = {}
        last_upd = (st.get("datetimeLast") or {}).get("local") or datetime.now(IST).isoformat()
        for m in r_lat.json().get("results", []):
            dt_utc = (m.get("datetime") or {}).get("utc", "")
            if dt_utc < "2025-01-01":
                continue
            sid = m.get("sensorsId")
            val = m.get("value")
            if sid not in sensor_map or val is None or float(val) < 0:
                continue
            p_name, _ = sensor_map[sid]
            v_f = float(val)
            if p_name in ("pm25", "pm10", "no2", "so2", "o3", "nh3"):
                pols[p_name] = round(v_f, 1)
            elif p_name == "co":
                pols["co"] = normalize_co_to_mg_m3(v_f) or v_f

        calc = calculate_cpcb_aqi(pols, require_three_pollutants=False)
        if calc["aqi"] is not None:
            station_summaries.append({
                "city": city_name,
                "state": "",
                "station": f"{st_name} ({prov_name})",
                "provider": prov_name,
                "latitude": float((st.get("coordinates") or {}).get("latitude", lat)),
                "longitude": float((st.get("coordinates") or {}).get("longitude", lon)),
                "last_update": last_upd,
                "pollutants": pols,
                "aqi": calc["aqi"],
                "category": calc["category"],
                "dominant_pollutant": calc["dominant_pollutant"],
            })

    if not station_summaries:
        return {"available": False, "status": "OpenAQ v3 stations had insufficient live pollutant readings", "stations": []}

    with get_db_connection() as conn:
        for st_info in station_summaries:
            conn.execute(
                """
                INSERT OR IGNORE INTO cpcb_observations
                (city, state, station, latitude, longitude, observed_at, aqi, aqi_category,
                 pm25, pm10, no2, so2, co, o3, nh3, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    st_info["city"],
                    st_info["state"],
                    st_info["station"],
                    st_info["latitude"],
                    st_info["longitude"],
                    st_info["last_update"],
                    st_info["aqi"],
                    st_info["category"],
                    st_info["pollutants"].get("pm25"),
                    st_info["pollutants"].get("pm10"),
                    st_info["pollutants"].get("no2"),
                    st_info["pollutants"].get("so2"),
                    st_info["pollutants"].get("co"),
                    st_info["pollutants"].get("o3"),
                    st_info["pollutants"].get("nh3"),
                    f"OpenAQ v3 Live ({st_info['provider']})",
                ),
            )
        conn.commit()

    best = station_summaries[0]
    res = {
        "available": True,
        "cached": False,
        "source": f"Live Station via OpenAQ v3 ({best['station']})",
        "city_aqi": best["aqi"],
        "city_category": best["category"],
        "stations": station_summaries,
        "last_update": best["last_update"],
    }
    fetched_iso = set_cached_json(cache_key, res, "openaq_v3")
    res["fetched_at"] = fetched_iso
    return res


def fetch_cpcb_live_station_data(city_name: str, lat: float, lon: float) -> Dict[str, Any]:
    """
    Retrieve live CPCB station observations:
    1. Try official data.gov.in CPCB API (Resource ID: 3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69).
    2. If data.gov.in is unreachable from the cloud network, automatically fall back to
       OpenAQ v3 Live CPCB stations (`OPENAQ_API_KEY`).
    """
    api_key = os.getenv("DATA_GOV_IN_API_KEY", "").strip()
    cache_key = f"cpcb_live_v2:{city_name.lower().strip()}"
    cached, fetched_at, is_fresh = get_cached_json(cache_key, max_age_seconds=900)
    if cached and is_fresh:
        cached["cached"] = True
        cached["fetched_at"] = fetched_at
        return cached

    records_collected = []
    if api_key and api_key != "YOUR_DATA_GOV_IN_API_KEY":
        resp = http_get_with_retry(
            f"https://api.data.gov.in/resource/{CPCB_RESOURCE_ID}",
            params={
                "api-key": api_key,
                "format": "json",
                "filters[city]": city_name,
                "offset": 0,
                "limit": 100,
            },
            timeout=3.5,
            retries=0,
        )
        if resp and resp.status_code == 200:
            records_collected = resp.json().get("records", [])

    if records_collected:
        stations: Dict[str, Dict[str, Any]] = {}
        for row in records_collected:
            st_name = row.get("station", "CPCB Station")
            if st_name not in stations:
                stations[st_name] = {
                    "city": row.get("city", city_name),
                    "state": row.get("state", ""),
                    "station": st_name,
                    "latitude": float(row.get("latitude") or lat),
                    "longitude": float(row.get("longitude") or lon),
                    "last_update": row.get("last_update", datetime.now(IST).isoformat()),
                    "pollutants": {},
                }
            pol_id = (row.get("pollutant_id") or "").lower().replace(".", "")
            pol_avg = row.get("pollutant_avg")
            if pol_id and pol_avg not in (None, "NA", ""):
                try:
                    stations[st_name]["pollutants"][pol_id] = float(pol_avg)
                except ValueError:
                    pass

        station_summaries = []
        with get_db_connection() as conn:
            for st_name, st_info in stations.items():
                calc = calculate_cpcb_aqi(st_info["pollutants"], require_three_pollutants=True)
                if calc["aqi"] is not None:
                    st_info["aqi"] = calc["aqi"]
                    st_info["category"] = calc["category"]
                    st_info["dominant_pollutant"] = calc["dominant_pollutant"]
                    station_summaries.append(st_info)
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO cpcb_observations
                        (city, state, station, latitude, longitude, observed_at, aqi, aqi_category,
                         pm25, pm10, no2, so2, co, o3, nh3, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            st_info["city"],
                            st_info["state"],
                            st_info["station"],
                            st_info["latitude"],
                            st_info["longitude"],
                            st_info["last_update"],
                            st_info["aqi"],
                            st_info["category"],
                            st_info["pollutants"].get("pm25"),
                            st_info["pollutants"].get("pm10"),
                            st_info["pollutants"].get("no2"),
                            st_info["pollutants"].get("so2"),
                            st_info["pollutants"].get("co"),
                            st_info["pollutants"].get("o3"),
                            st_info["pollutants"].get("nh3"),
                            "CPCB data.gov.in Live API",
                        ),
                    )
            conn.commit()

        if station_summaries:
            avg_aqi = round(float(np.mean([s["aqi"] for s in station_summaries])), 1)
            result = {
                "available": True,
                "cached": False,
                "source": "Official CPCB (data.gov.in) Real-Time API",
                "city_aqi": avg_aqi,
                "city_category": aqi_to_cpcb_category(avg_aqi),
                "stations": station_summaries,
                "last_update": station_summaries[0]["last_update"],
            }
            fetched_iso = set_cached_json(cache_key, result, "cpcb_live")
            result["fetched_at"] = fetched_iso
            return result

    oaq_res = fetch_openaq_cpcb_live(city_name, lat, lon)
    if oaq_res.get("available"):
        set_cached_json(cache_key, oaq_res, "openaq_cpcb_live")
        return oaq_res

    return {
        "available": False,
        "status": "Direct CPCB station feed unavailable for this location; using CPCB breakpoint formula on live atmospheric telemetry.",
        "stations": [],
    }


def fetch_nasa_firms_fires(lat: float, lon: float) -> Dict[str, Any]:
    """
    Query NASA FIRMS VIIRS_SNPP_NRT satellite fire detections within ~220 km bounding box
    over the last 72 hours using NASA_FIRMS_MAP_KEY.
    """
    map_key = os.getenv("NASA_FIRMS_MAP_KEY", "").strip()
    if not map_key or map_key == "YOUR_NASA_FIRMS_MAP_KEY":
        return {
            "available": False,
            "fires_24h": None,
            "fires_72h": None,
            "mean_frp_mw": None,
            "status": "NASA_FIRMS_MAP_KEY not configured",
        }

    cache_key = f"firms_v2:{round(lat, 1)}:{round(lon, 1)}"
    cached, _, is_fresh = get_cached_json(cache_key, max_age_seconds=3600)
    if cached and is_fresh:
        return cached

    bbox = f"{round(lon - 2.0, 2)},{round(lat - 2.0, 2)},{round(lon + 2.0, 2)},{round(lat + 2.0, 2)}"
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/VIIRS_SNPP_NRT/{bbox}/3"
    resp = http_get_with_retry(url, timeout=10.0, retries=1)
    if not resp or resp.status_code != 200 or "latitude" not in resp.text:
        return {
            "available": False,
            "fires_24h": None,
            "fires_72h": None,
            "mean_frp_mw": None,
            "status": "NASA FIRMS API unavailable",
        }

    try:
        df_f = pd.read_csv(StringIO(resp.text))
        fires_72h = int(len(df_f))
        if fires_72h == 0:
            fires_24h = 0
            mean_frp = 0.0
        else:
            max_date = str(df_f["acq_date"].max())
            fires_24h = int((df_f["acq_date"].astype(str) == max_date).sum())
            mean_frp = round(float(df_f["frp"].mean()), 1) if "frp" in df_f.columns else 0.0
    except Exception:
        lines = [ln for ln in resp.text.strip().splitlines() if ln.strip()]
        fires_72h = max(0, len(lines) - 1)
        fires_24h = fires_72h // 3
        mean_frp = 0.0

    res = {
        "available": True,
        "fires_24h": fires_24h,
        "fires_72h": fires_72h,
        "mean_frp_mw": mean_frp,
        "status": f"{fires_24h} active fire detections in 24h ({fires_72h} in 72h, avg FRP {mean_frp} MW) within 220 km",
    }
    set_cached_json(cache_key, res, "nasa_firms")
    return res


# =====================================================================
# 7. MULTI-YEAR REAL CPCB & METEOROLOGICAL DATASET COLLECTION (2015–2025)
# =====================================================================

def _fetch_openmeteo_weather_batch(
    cities_batch: List[Tuple[str, float, float]],
    start_date: str,
    end_date: str,
) -> List[pd.DataFrame]:
    """Fetch daily historical meteorology for a batch of Indian cities using Open-Meteo Archive."""
    if not cities_batch:
        return []
    lats = ",".join(str(c[1]) for c in cities_batch)
    lons = ",".join(str(c[2]) for c in cities_batch)
    resp = http_get_with_retry(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lats,
            "longitude": lons,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join([
                "temperature_2m_mean",
                "relative_humidity_2m_mean",
                "dew_point_2m_mean",
                "surface_pressure_mean",
                "cloud_cover_mean",
                "wind_speed_10m_mean",
                "wind_direction_10m_dominant",
                "wind_gusts_10m_max",
                "precipitation_sum",
                "shortwave_radiation_sum",
            ]),
            "timezone": "Asia/Kolkata",
        },
        timeout=25.0,
        retries=2,
    )
    if not resp or resp.status_code != 200:
        return []
    payload = resp.json()
    items = payload if isinstance(payload, list) else [payload]
    frames = []
    for (city_name, _, _), obj in zip(cities_batch, items):
        daily = obj.get("daily", {})
        if not daily.get("time"):
            continue
        frames.append(pd.DataFrame({
            "City": city_name,
            "Date": daily["time"],
            "temperature_c": daily.get("temperature_2m_mean"),
            "humidity_pct": daily.get("relative_humidity_2m_mean"),
            "dew_point_c": daily.get("dew_point_2m_mean"),
            "pressure_hpa": daily.get("surface_pressure_mean"),
            "cloud_cover_pct": daily.get("cloud_cover_mean"),
            "wind_speed_ms": [
                round(float(w) / 3.6, 2) if w is not None else np.nan
                for w in daily.get("wind_speed_10m_mean", [])
            ],
            "wind_direction_deg": daily.get("wind_direction_10m_dominant"),
            "wind_gust_ms": [
                round(float(w) / 3.6, 2) if w is not None else np.nan
                for w in daily.get("wind_gusts_10m_max", [])
            ],
            "precipitation_mm": daily.get("precipitation_sum"),
            "solar_radiation_wm2": [
                round(float(s) * 11.57, 1) if s is not None else np.nan
                for s in daily.get("shortwave_radiation_sum", [])
            ],
        }))
    return frames


def _fetch_recent_cpcb_standardized_aq(cities_list: List[Tuple[str, float, float]]) -> pd.DataFrame:
    """
    Fetch 2023-01-01 to 2025-12-31 hourly air pollutant observations across all 26 Indian CPCB cities,
    compute daily 24-hour mean concentrations and official CPCB 3-pollutant AQI, and cache to CSV.
    """
    df_existing = pd.DataFrame()
    existing_cities = set()
    if os.path.exists(CPCB_RECENT_CSV):
        try:
            df_existing = pd.read_csv(CPCB_RECENT_CSV)
            existing_cities = set(df_existing["City"].dropna().unique())
        except Exception:
            pass

    missing_cities = [c for c in cities_list if c[0] not in existing_cities]
    if not missing_cities and not df_existing.empty:
        return df_existing

    logger.info("Fetching 2023–2025 air quality observations for %d remaining CPCB cities...", len(missing_cities))
    all_daily_rows = [df_existing] if not df_existing.empty else []
    batch_size = 3
    for b_start in range(0, len(missing_cities), batch_size):
        batch = missing_cities[b_start : b_start + batch_size]
        lats = ",".join(str(c[1]) for c in batch)
        lons = ",".join(str(c[2]) for c in batch)
        resp = http_get_with_retry(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lats,
                "longitude": lons,
                "start_date": "2023-01-01",
                "end_date": "2025-12-31",
                "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
                "timezone": "Asia/Kolkata",
            },
            timeout=30.0,
            retries=3,
        )
        time.sleep(0.35)
        if not resp or resp.status_code != 200:
            continue
        payload = resp.json()
        items = payload if isinstance(payload, list) else [payload]
        for (city_name, _, _), obj in zip(batch, items):
            hourly = obj.get("hourly", {})
            times = hourly.get("time", [])
            if not times:
                continue
            df_h = pd.DataFrame({
                "Date": [t[:10] for t in times],
                "PM2.5": pd.to_numeric(hourly.get("pm2_5"), errors="coerce"),
                "PM10": pd.to_numeric(hourly.get("pm10"), errors="coerce"),
                "NO2": pd.to_numeric(hourly.get("nitrogen_dioxide"), errors="coerce"),
                "SO2": pd.to_numeric(hourly.get("sulphur_dioxide"), errors="coerce"),
                "CO": pd.to_numeric(hourly.get("carbon_monoxide"), errors="coerce") / 1000.0,
                "O3": pd.to_numeric(hourly.get("ozone"), errors="coerce"),
            })
            df_d = df_h.groupby("Date", as_index=False).mean()
            df_d["City"] = city_name
            df_d["NH3"] = np.nan

            aqis = []
            for _, r in df_d.iterrows():
                calc = calculate_cpcb_aqi(
                    {
                        "pm25": r["PM2.5"],
                        "pm10": r["PM10"],
                        "no2": r["NO2"],
                        "so2": r["SO2"],
                        "co": r["CO"],
                        "o3": r["O3"],
                    },
                    require_three_pollutants=True,
                )
                aqis.append(calc["aqi"])
            df_d["AQI"] = aqis
            df_d = df_d.dropna(subset=["AQI"])
            all_daily_rows.append(df_d[["City", "Date", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3", "AQI"]])

    if all_daily_rows:
        df_recent = pd.concat(all_daily_rows, ignore_index=True).drop_duplicates(subset=["City", "Date"])
        df_recent.to_csv(CPCB_RECENT_CSV, index=False)
        logger.info("Saved %d recent (2023–2025) CPCB-standardized city-day records across %d cities to %s", len(df_recent), df_recent["City"].nunique(), CPCB_RECENT_CSV)
        return df_recent
    return pd.DataFrame()


def ensure_historical_training_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download and cache:
    1. Official CPCB historical observations (2015–2020) in cpcb_city_day.csv
    2. Recent multi-year CPCB-standardized daily observations (2023–2025) in cpcb_recent_2023_2025.csv
    3. Complete matching daily historical meteorology (2015–2020 and 2023–2025) from Open-Meteo Archive
       in indian_cities_weather_archive.csv across all 26 CPCB cities (~52,000+ total samples).
    """
    if not os.path.exists(CPCB_HIST_CSV):
        logger.info("Downloading official CPCB multi-city historical dataset...")
        resp = http_get_with_retry(CPCB_ARCHIVE_MIRROR_URL, timeout=25.0, retries=2)
        if not resp or resp.status_code != 200:
            raise RuntimeError("Could not download historical CPCB dataset.")
        with open(CPCB_HIST_CSV, "wb") as f:
            f.write(resp.content)

    cpcb_df = pd.read_csv(CPCB_HIST_CSV)
    all_cpcb_cities = sorted(cpcb_df["City"].dropna().unique().tolist())
    city_coords = {c["name"]: (c["lat"], c["lon"]) for c in INDIAN_CITIES_DIRECTORY}
    valid_cities = [(c, city_coords[c][0], city_coords[c][1]) for c in all_cpcb_cities if c in city_coords]

    # 1. Fetch 2023-2025 recent multi-year CPCB-standardized dataset
    recent_df = _fetch_recent_cpcb_standardized_aq(valid_cities)
    if not recent_df.empty:
        combined_cpcb = pd.concat([cpcb_df, recent_df], ignore_index=True)
        combined_cpcb = combined_cpcb.drop_duplicates(subset=["City", "Date"], keep="first")
    else:
        combined_cpcb = cpcb_df

    # 2. Ensure historical weather archive covers 2015-2016, 2017-2020, and 2023-2025 for all 26 cities
    wx_df = pd.read_csv(WEATHER_HIST_CSV) if os.path.exists(WEATHER_HIST_CSV) else pd.DataFrame()
    wx_frames = [wx_df] if not wx_df.empty else []
    updated_wx = False

    cities_with_2015 = set(wx_df[wx_df["Date"] <= "2015-06-01"]["City"].unique()) if not wx_df.empty else set()
    cities_with_2025 = set(wx_df[wx_df["Date"] >= "2025-06-01"]["City"].unique()) if not wx_df.empty else set()

    missing_2015 = [c for c in valid_cities if c[0] not in cities_with_2015]
    missing_2025 = [c for c in valid_cities if c[0] not in cities_with_2025]

    batch_size = 4
    if missing_2015:
        logger.info("Fetching 2015–2016 historical weather archive for %d CPCB cities...", len(missing_2015))
        for i in range(0, len(missing_2015), batch_size):
            batch_frames = _fetch_openmeteo_weather_batch(missing_2015[i : i + batch_size], "2015-01-01", "2016-12-31")
            time.sleep(0.3)
            if batch_frames:
                wx_frames.extend(batch_frames)
                updated_wx = True

    if missing_2025:
        logger.info("Fetching 2023–2025 historical weather archive for %d CPCB cities...", len(missing_2025))
        for i in range(0, len(missing_2025), batch_size):
            batch_frames = _fetch_openmeteo_weather_batch(missing_2025[i : i + batch_size], "2023-01-01", "2025-12-31")
            time.sleep(0.3)
            if batch_frames:
                wx_frames.extend(batch_frames)
                updated_wx = True

    if updated_wx and wx_frames:
        wx_df = pd.concat(wx_frames, ignore_index=True)
        wx_df = wx_df.drop_duplicates(subset=["City", "Date"], keep="last").sort_values(["City", "Date"])
        wx_df.to_csv(WEATHER_HIST_CSV, index=False)
        logger.info("Updated %s with %d total city-day weather records across %d cities (2015–2025)", WEATHER_HIST_CSV, len(wx_df), wx_df["City"].nunique())

    return combined_cpcb, wx_df


# =====================================================================
# 8. LIVE PREDICTION LOGGING & DELAYED ACTUAL CPCB VALIDATION
# =====================================================================

def log_live_prediction(
    city: str,
    state: str,
    station: str,
    lat: float,
    lon: float,
    current_aqi: Optional[float],
    predicted_aqi: float,
    predicted_category: str,
    model_version: str,
    data_source: str,
    features: Dict[str, Any],
) -> int:
    now_utc = datetime.now(timezone.utc)
    target_utc = now_utc + timedelta(hours=24)
    with get_db_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions_log
            (city, state, station, latitude, longitude, predicted_at, target_timestamp,
             current_aqi, predicted_aqi, predicted_category, model_version, data_source, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                city,
                state,
                station,
                lat,
                lon,
                now_utc.isoformat(),
                target_utc.isoformat(),
                current_aqi,
                round(float(predicted_aqi), 1),
                predicted_category,
                model_version,
                data_source,
                json.dumps(features),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def evaluate_live_predictions() -> Dict[str, Any]:
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        unmatched = conn.execute(
            """
            SELECT id, city, target_timestamp
            FROM predictions_log
            WHERE actual_aqi IS NULL AND target_timestamp <= ?
            """,
            (now_iso,),
        ).fetchall()

        for row in unmatched:
            obs = conn.execute(
                """
                SELECT aqi, aqi_category, observed_at
                FROM cpcb_observations
                WHERE LOWER(city) = LOWER(?)
                  AND ABS(julianday(observed_at) - julianday(?)) <= 0.15
                  AND aqi IS NOT NULL
                ORDER BY ABS(julianday(observed_at) - julianday(?)) ASC
                LIMIT 1
                """,
                (row["city"], row["target_timestamp"], row["target_timestamp"]),
            ).fetchone()
            if obs:
                conn.execute(
                    """
                    UPDATE predictions_log
                    SET actual_aqi = ?, actual_category = ?, matched_at = ?
                    WHERE id = ?
                    """,
                    (obs["aqi"], obs["aqi_category"], now_iso, row["id"]),
                )
        conn.commit()

        total_logged = conn.execute("SELECT COUNT(*) AS cnt FROM predictions_log").fetchone()["cnt"]
        matched_rows = conn.execute(
            """
            SELECT city, predicted_at, target_timestamp, predicted_aqi, predicted_category,
                   actual_aqi, actual_category, model_version, data_source
            FROM predictions_log
            WHERE actual_aqi IS NOT NULL
            ORDER BY target_timestamp DESC
            """
        ).fetchall()
        recent_logged = conn.execute(
            """
            SELECT id, city, state, station, predicted_at, target_timestamp,
                   current_aqi, predicted_aqi, predicted_category, model_version,
                   data_source, actual_aqi, actual_category
            FROM predictions_log
            ORDER BY id DESC
            LIMIT 10
            """
        ).fetchall()

    recent_list = [dict(r) for r in recent_logged]
    if not matched_rows:
        return {
            "status": "Live validation pending — awaiting actual CPCB observations.",
            "matched_count": 0,
            "total_predictions_logged": total_logged,
            "metrics": None,
            "recent_predictions": recent_list,
        }

    y_true = np.array([float(r["actual_aqi"]) for r in matched_rows])
    y_pred = np.array([float(r["predicted_aqi"]) for r in matched_rows])
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    cat_true = [r["actual_category"] for r in matched_rows]
    cat_pred = [r["predicted_category"] for r in matched_rows]
    acc = float(np.mean([1.0 if ct == cp else 0.0 for ct, cp in zip(cat_true, cat_pred)]))

    return {
        "status": f"Verified against {len(matched_rows)} actual CPCB observations.",
        "matched_count": len(matched_rows),
        "total_predictions_logged": total_logged,
        "evaluation_period": {
            "from": matched_rows[-1]["target_timestamp"],
            "to": matched_rows[0]["target_timestamp"],
        },
        "metrics": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "category_accuracy": round(acc * 100.0, 2),
        },
        "recent_predictions": recent_list,
    }
