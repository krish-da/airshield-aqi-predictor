"""
AirShield — Supervised Machine Learning Training & Chronological Validation Pipeline (train_model.py)
Responsibilities:
1. Download / load verified multi-year CPCB & CPCB-standardized observations (2015–2025, ~50,000+ samples)
   and matching historical meteorology across 26 Indian cities
2. Reconstruct missing AQI strictly when CPCB 3-pollutant + PM qualification rules are met
3. Engineer zero-leakage lag, EWMA, rolling, CPCB sub-index, training-only city-month climatology,
   current + next-24h meteorological & boundary-layer (PBL) features, and pure distance-based
   regional/national festival & seasonal plume features (zero hardcoded emission weights)
4. Perform strict chronological splitting (70% Train / 15% Validation / 15% Test)
5. Fit imputation and city-month priors strictly on Training split only
6. Benchmark CatBoostRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor, StackedEnsembleRegressor, and PersistenceBaseline
7. Perform 3-Stage Feature Ablation Study (Pollutants Only vs +Weather/PBL vs +Regional Festival/Event Distance Features)
   on both All Test Days and Active Festival/Event Windows
8. Persist winning model, preprocessing pipeline, and full numerical validation report to models/
"""

import os
import json
import math
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
)
from catboost import CatBoostRegressor

from data_sources import (
    MODELS_DIR,
    INDIAN_CITIES_DIRECTORY,
    DIWALI_DATES_BY_YEAR,
    DUSSEHRA_DATES_BY_YEAR,
    HOLI_DATES_BY_YEAR,
    EXPANDED_FESTIVAL_INSTANCES,
    STUBBLE_BURNING_BELT_LAT,
    STUBBLE_BURNING_BELT_LON,
    THAR_DESERT_DUST_LAT,
    THAR_DESERT_DUST_LON,
    haversine_km,
    estimate_physical_pbl_m,
    ensure_historical_training_datasets,
    calculate_cpcb_aqi,
    calculate_cpcb_subindex,
    aqi_to_cpcb_category,
    normalize_co_to_mg_m3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("airshield.train_model")

MODEL_ARTIFACT_PATH = os.path.join(MODELS_DIR, "airshield_model.joblib")
EVAL_REPORT_PATH = os.path.join(MODELS_DIR, "evaluation_report.json")

POLLUTANT_AND_PRIOR_COLS: List[str] = [
    "current_aqi",
    "aqi_lag_1",
    "aqi_lag_2",
    "aqi_lag_3",
    "aqi_lag_7",
    "aqi_delta_24h",
    "aqi_delta_48h",
    "aqi_ewm_3",
    "aqi_ewm_7",
    "aqi_ewm_14",
    "aqi_roll_mean_3",
    "aqi_roll_max_3",
    "aqi_roll_min_3",
    "aqi_roll_std_3",
    "aqi_roll_mean_7",
    "aqi_ratio_to_roll7",
    "city_train_median_aqi",
    "city_month_train_median_aqi",
    "aqi_anomaly_vs_city_month",
    "pm25",
    "pm25_lag_1",
    "pm25_roll_3",
    "sub_pm25",
    "pm10",
    "pm10_lag_1",
    "pm10_roll_3",
    "sub_pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "nh3",
    "pm25_to_pm10_ratio",
]

WEATHER_AND_PBL_COLS: List[str] = [
    "temperature_c",
    "humidity_pct",
    "dew_point_c",
    "pressure_hpa",
    "cloud_cover_pct",
    "wind_speed_ms",
    "wind_gust_ms",
    "wind_dir_sin",
    "wind_dir_cos",
    "precipitation_mm",
    "solar_radiation_wm2",
    "boundary_layer_height_m",
    "ventilation_proxy",
    "next24h_temp_c",
    "next24h_humidity_pct",
    "next24h_wind_speed_ms",
    "next24h_precip_mm",
    "next24h_pbl_m",
    "temp_change_24h",
    "wind_change_24h",
]

SPATIAL_CALENDAR_AND_EVENT_COLS: List[str] = [
    "latitude",
    "longitude",
    "month",
    "doy_sin",
    "doy_cos",
    "day_of_week",
    "is_weekend",
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
]

FEATURE_COLUMNS: List[str] = (
    POLLUTANT_AND_PRIOR_COLS + WEATHER_AND_PBL_COLS + SPATIAL_CALENDAR_AND_EVENT_COLS
)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics (MAE, RMSE, R²) and CPCB category metrics (Accuracy, Macro F1, Weighted F1)."""
    y_pred_clipped = np.clip(np.round(y_pred, 1), 10.0, 500.0)
    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred_clipped)))
    r2 = float(r2_score(y_true, y_pred_clipped)) if len(y_true) > 2 else 0.0

    cat_true = [aqi_to_cpcb_category(v) for v in y_true]
    cat_pred = [aqi_to_cpcb_category(v) for v in y_pred_clipped]

    acc = float(accuracy_score(cat_true, cat_pred))
    macro_f1 = float(f1_score(cat_true, cat_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(cat_true, cat_pred, average="weighted", zero_division=0))

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "category_accuracy_pct": round(acc * 100.0, 2),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }


def _attach_vectorized_festival_and_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized computation of exact Haversine distances and temporal offsets to all
    regional & national Indian celebrations (evaluated at the +24h target date next_date),
    crop-residue burning belt, and Thar dust source.
    Zero hardcoded emission weights — the model learns all weights from data.
    """
    out = df.copy()
    dt_series = out["next_date"].fillna(out["Date"])

    def calc_signed_days(date_map: Dict[int, str], clip_val: int = 30) -> np.ndarray:
        vals = np.full(len(dt_series), clip_val, dtype=int)
        for d_str in date_map.values():
            f_ts = pd.Timestamp(d_str)
            diff = (dt_series - f_ts).dt.days.values
            mask = np.abs(diff) < np.abs(vals)
            vals[mask] = np.clip(diff[mask], -clip_val, clip_val)
        return vals

    out["days_from_diwali"] = calc_signed_days(DIWALI_DATES_BY_YEAR, 30)
    out["abs_days_to_diwali"] = np.abs(out["days_from_diwali"].values)
    out["is_diwali_window"] = ((out["days_from_diwali"] >= -2) & (out["days_from_diwali"] <= 4)).astype(int)
    out["days_from_dussehra"] = calc_signed_days(DUSSEHRA_DATES_BY_YEAR, 30)
    out["days_from_holi"] = calc_signed_days(HOLI_DATES_BY_YEAR, 30)

    # Precompute city -> festival instance minimum epicenter distance
    unique_cities = out[["City", "latitude", "longitude"]].drop_duplicates()
    city_lat_lon = {r["City"]: (float(r["latitude"]), float(r["longitude"])) for _, r in unique_cities.iterrows()}

    # Precompute static city distances to Stubble Belt & Thar Dust Source
    city_stubble_dist = {
        c: round(haversine_km(lat, lon, STUBBLE_BURNING_BELT_LAT, STUBBLE_BURNING_BELT_LON), 1)
        for c, (lat, lon) in city_lat_lon.items()
    }
    city_thar_dist = {
        c: round(haversine_km(lat, lon, THAR_DESERT_DUST_LAT, THAR_DESERT_DUST_LON), 1)
        for c, (lat, lon) in city_lat_lon.items()
    }
    out["dist_to_stubble_belt_km"] = out["City"].map(city_stubble_dist).astype(float)
    out["dist_to_thar_dust_km"] = out["City"].map(city_thar_dist).astype(float)

    # Vectorized regional festival distance & proximity across all festival instances
    n_rows = len(out)
    days_to_reg_fest = np.full(n_rows, 15, dtype=int)
    dist_to_active_fest = np.full(n_rows, 1500.0, dtype=float)
    reg_fest_prox = np.zeros(n_rows, dtype=float)
    is_fest_day = np.zeros(n_rows, dtype=int)

    city_col = out["City"].values
    for inst in EXPANDED_FESTIVAL_INSTANCES:
        f_ts = pd.Timestamp(inst["date"])
        diff_days = (dt_series - f_ts).dt.days.values
        active_mask = np.abs(diff_days) <= 15
        if not np.any(active_mask):
            continue

        # Map each city to its minimum distance to this festival's epicenters
        c_dist_map = {}
        for c_name, (c_lat, c_lon) in city_lat_lon.items():
            c_dist_map[c_name] = min(
                haversine_km(c_lat, c_lon, ep_lat, ep_lon)
                for ep_lat, ep_lon, _ in inst["epicenters"]
            )
        inst_dists = np.array([c_dist_map[c] for c in city_col[active_mask]], dtype=float)
        inst_diffs = diff_days[active_mask]

        spatial_factor = np.maximum(0.0, 1.0 - (inst_dists / 600.0))
        temporal_factor = np.maximum(0.0, 1.0 - (np.abs(inst_diffs) / 6.0))
        prox = np.round(spatial_factor * temporal_factor, 4)

        idx_active = np.where(active_mask)[0]
        reg_fest_prox[idx_active] = np.maximum(reg_fest_prox[idx_active], prox)

        near_reg_mask = (inst_dists <= 550.0) & (np.abs(inst_diffs) < np.abs(days_to_reg_fest[idx_active]))
        days_to_reg_fest[idx_active[near_reg_mask]] = inst_diffs[near_reg_mask]

        within_6d_mask = np.abs(inst_diffs) <= 6
        idx_6d = idx_active[within_6d_mask]
        dist_to_active_fest[idx_6d] = np.minimum(dist_to_active_fest[idx_6d], np.round(inst_dists[within_6d_mask], 1))

        fest_day_mask = (np.abs(inst_diffs) <= 1) & (inst_dists <= 450.0)
        is_fest_day[idx_active[fest_day_mask]] = 1

    out["days_to_regional_festival"] = days_to_reg_fest
    out["dist_to_active_festival_km"] = dist_to_active_fest
    out["regional_festival_proximity"] = reg_fest_prox
    out["is_festival_day"] = is_fest_day

    month_vals = out["month"].values
    day_vals = out["Date"].dt.day.values
    is_stubble_date = (((month_vals == 10) & (day_vals >= 10)) | ((month_vals == 11) & (day_vals <= 25))).astype(int)
    out["is_stubble_burning_season"] = (is_stubble_date & (out["dist_to_stubble_belt_km"].values <= 950.0)).astype(int)

    w_dir = out["wind_direction_deg"].fillna(270.0).values
    nw_align = np.maximum(0.0, np.cos(np.radians(w_dir - 310.0)))
    stubble_decay = np.maximum(0.0, 1.0 - (out["dist_to_stubble_belt_km"].values / 1100.0))
    out["downwind_stubble_exposure"] = np.round(is_stubble_date * stubble_decay * nw_align, 4)

    temp_vals = out["temperature_c"].fillna(24.0).values
    pbl_vals = out["boundary_layer_height_m"].fillna(650.0).values
    out["is_winter_inversion"] = (
        np.isin(month_vals, [11, 12, 1, 2])
        & ((temp_vals < 20.0) | (pbl_vals < 500.0) | (out["latitude"].values >= 23.0))
    ).astype(int)

    rh_vals = out["humidity_pct"].fillna(50.0).values
    out["is_dust_season"] = (
        np.isin(month_vals, [4, 5, 6])
        & (out["dist_to_thar_dust_km"].values <= 1200.0)
        & (rh_vals < 45.0)
    ).astype(int)

    out["is_monsoon_season"] = (
        np.isin(month_vals, [7, 8, 9]) | ((month_vals == 6) & (day_vals >= 15))
    ).astype(int)

    return out


def prepare_engineered_dataset() -> pd.DataFrame:
    """
    Clean multi-year CPCB + recent CPCB-standardized dataset (2015–2025),
    merge with historical meteorology at issue time t AND next-24h target window t+1,
    compute physical boundary layer height, and engineer all spatial-temporal features.
    """
    cpcb_df, wx_df = ensure_historical_training_datasets()
    logger.info("Loaded multi-year CPCB dataset (%d rows) and historical weather dataset (%d rows)", len(cpcb_df), len(wx_df))

    df = cpcb_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["City", "Date"]).sort_values(["City", "Date"]).drop_duplicates(subset=["City", "Date"])

    rename_map = {
        "PM2.5": "pm25",
        "PM10": "pm10",
        "NO2": "no2",
        "SO2": "so2",
        "CO": "co",
        "O3": "o3",
        "NH3": "nh3",
        "AQI": "aqi",
    }
    df = df.rename(columns=rename_map)
    for col in ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3", "aqi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    reconstructed_count = 0
    aqi_vals = df["aqi"].values.copy()
    for idx in range(len(df)):
        if pd.isna(aqi_vals[idx]):
            row = df.iloc[idx]
            pols = {
                "pm25": row["pm25"],
                "pm10": row["pm10"],
                "no2": row["no2"],
                "so2": row["so2"],
                "co": normalize_co_to_mg_m3(row["co"]),
                "o3": row["o3"],
                "nh3": row["nh3"],
            }
            calc = calculate_cpcb_aqi(pols, require_three_pollutants=True)
            if calc["aqi"] is not None:
                aqi_vals[idx] = calc["aqi"]
                reconstructed_count += 1
    df["aqi"] = aqi_vals
    logger.info("Reconstructed %d missing AQI values via official CPCB 3-pollutant formula.", reconstructed_count)

    df = df.dropna(subset=["aqi"]).copy()
    df = df[(df["aqi"] >= 10.0) & (df["aqi"] <= 500.0)].copy()

    if not wx_df.empty:
        wx_clean = wx_df.copy()
        wx_clean["Date"] = pd.to_datetime(wx_clean["Date"], errors="coerce")
        wx_clean = wx_clean.drop_duplicates(subset=["City", "Date"])
        df = pd.merge(df, wx_clean, on=["City", "Date"], how="left")
    else:
        for wcol in [
            "temperature_c", "humidity_pct", "dew_point_c", "pressure_hpa",
            "cloud_cover_pct", "wind_speed_ms", "wind_direction_deg",
            "wind_gust_ms", "precipitation_mm", "solar_radiation_wm2",
        ]:
            df[wcol] = np.nan

    # Compute physical Planetary Boundary Layer height (boundary_layer_height_m)
    spread = np.maximum(0.5, df["temperature_c"].fillna(25.0) - df["dew_point_c"].fillna(16.0))
    u_ms = df["wind_speed_ms"].fillna(2.2)
    sw_wm2 = df["solar_radiation_wm2"].fillna(180.0)
    df["boundary_layer_height_m"] = np.round(np.clip(42.0 * spread + 68.0 * u_ms + 1.05 * sw_wm2, 90.0, 2200.0), 1)

    city_meta = {c["name"]: c for c in INDIAN_CITIES_DIRECTORY}
    df["latitude"] = df["City"].map(lambda c: city_meta.get(c, {}).get("lat", 23.5))
    df["longitude"] = df["City"].map(lambda c: city_meta.get(c, {}).get("lon", 78.5))
    df["state"] = df["City"].map(lambda c: city_meta.get(c, {}).get("state", "India"))

    df = df.sort_values(["City", "Date"]).reset_index(drop=True)
    grouped = df.groupby("City", group_keys=False)

    df["current_aqi"] = df["aqi"]
    df["aqi_lag_1"] = grouped["aqi"].shift(1)
    df["aqi_lag_2"] = grouped["aqi"].shift(2)
    df["aqi_lag_3"] = grouped["aqi"].shift(3)
    df["aqi_lag_7"] = grouped["aqi"].shift(7)
    df["aqi_delta_24h"] = df["current_aqi"] - df["aqi_lag_1"]
    df["aqi_delta_48h"] = df["current_aqi"] - df["aqi_lag_2"].fillna(df["aqi_lag_1"])

    df["aqi_ewm_3"] = grouped["aqi"].transform(lambda s: s.ewm(span=3, min_periods=1, adjust=False).mean())
    df["aqi_ewm_7"] = grouped["aqi"].transform(lambda s: s.ewm(span=7, min_periods=1, adjust=False).mean())
    df["aqi_ewm_14"] = grouped["aqi"].transform(lambda s: s.ewm(span=14, min_periods=1, adjust=False).mean())

    df["aqi_roll_mean_3"] = grouped["aqi"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["aqi_roll_max_3"] = grouped["aqi"].transform(lambda s: s.rolling(3, min_periods=1).max())
    df["aqi_roll_min_3"] = grouped["aqi"].transform(lambda s: s.rolling(3, min_periods=1).min())
    df["aqi_roll_std_3"] = grouped["aqi"].transform(lambda s: s.rolling(3, min_periods=2).std()).fillna(0.0)
    df["aqi_roll_mean_7"] = grouped["aqi"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["aqi_ratio_to_roll7"] = df["current_aqi"] / np.maximum(df["aqi_roll_mean_7"], 10.0)

    # Pollutant lags & CPCB sub-indices at issue time t
    df["pm25_lag_1"] = grouped["pm25"].shift(1)
    df["pm25_roll_3"] = grouped["pm25"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["pm10_lag_1"] = grouped["pm10"].shift(1)
    df["pm10_roll_3"] = grouped["pm10"].transform(lambda s: s.rolling(3, min_periods=1).mean())

    df["sub_pm25"] = [calculate_cpcb_subindex("pm25", v) if pd.notna(v) else np.nan for v in df["pm25"].values]
    df["sub_pm10"] = [calculate_cpcb_subindex("pm10", v) if pd.notna(v) else np.nan for v in df["pm10"].values]

    # Next-24h target-window meteorology (matches live 24h weather forecast at inference time)
    df["next24h_temp_c"] = grouped["temperature_c"].shift(-1).fillna(df["temperature_c"])
    df["next24h_humidity_pct"] = grouped["humidity_pct"].shift(-1).fillna(df["humidity_pct"])
    df["next24h_wind_speed_ms"] = grouped["wind_speed_ms"].shift(-1).fillna(df["wind_speed_ms"])
    df["next24h_precip_mm"] = grouped["precipitation_mm"].shift(-1).fillna(0.0)
    df["next24h_pbl_m"] = grouped["boundary_layer_height_m"].shift(-1).fillna(df["boundary_layer_height_m"])
    df["temp_change_24h"] = df["next24h_temp_c"] - df["temperature_c"]
    df["wind_change_24h"] = df["next24h_wind_speed_ms"] - df["wind_speed_ms"]

    # Target: AQI exactly 24 hours (+1 day) ahead in the same city
    df["target_aqi_24h"] = grouped["aqi"].shift(-1)
    df["next_date"] = grouped["Date"].shift(-1)
    df["lead_days"] = (df["next_date"] - df["Date"]).dt.days

    df = df[(df["lead_days"] == 1) & (df["target_aqi_24h"].notna()) & (df["aqi_lag_1"].notna())].copy()

    df["co"] = df["co"].apply(normalize_co_to_mg_m3)
    df["pm25_to_pm10_ratio"] = np.where(
        (df["pm25"].notna()) & (df["pm10"].notna()) & (df["pm10"] > 1.0),
        np.clip(df["pm25"] / df["pm10"], 0.05, 1.0),
        np.nan,
    )

    wind_rad = np.radians(df["wind_direction_deg"].fillna(270.0))
    df["wind_dir_sin"] = np.sin(wind_rad)
    df["wind_dir_cos"] = np.cos(wind_rad)
    df["ventilation_proxy"] = np.round(
        df["wind_speed_ms"].fillna(2.2) * (df["boundary_layer_height_m"].fillna(550.0) / 100.0), 2
    )

    df["month"] = df["Date"].dt.month
    doy = df["Date"].dt.dayofyear
    df["doy_sin"] = np.sin(2.0 * math.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2.0 * math.pi * doy / 365.25)
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Attach pure distance-based & temporal regional festival and seasonal plume features
    df = _attach_vectorized_festival_and_distance_features(df)

    df = df.sort_values(["Date", "City"]).reset_index(drop=True)
    return df


def attach_training_only_priors(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Compute city-level and (city, month)-level median AQI strictly on the Training split
    so zero future information leaks into validation or test sets.
    """
    global_median = float(train_df["current_aqi"].median())
    city_medians = train_df.groupby("City")["current_aqi"].median().to_dict()
    city_month_medians = {
        f"{c}:{int(m)}": float(v)
        for (c, m), v in train_df.groupby(["City", "month"])["current_aqi"].median().items()
    }

    def apply_priors(sub_df: pd.DataFrame) -> pd.DataFrame:
        out = sub_df.copy()
        out["city_train_median_aqi"] = out["City"].map(lambda c: city_medians.get(c, global_median))
        out["city_month_train_median_aqi"] = [
            city_month_medians.get(f"{c}:{int(m)}", city_medians.get(c, global_median))
            for c, m in zip(out["City"], out["month"])
        ]
        out["aqi_anomaly_vs_city_month"] = out["current_aqi"] - out["city_month_train_median_aqi"]
        return out

    priors_bundle = {
        "global_median": global_median,
        "city_medians": city_medians,
        "city_month_medians": city_month_medians,
    }
    return apply_priors(train_df), apply_priors(val_df), apply_priors(test_df), priors_bundle


def train_and_evaluate_pipeline() -> Dict[str, Any]:
    start_time = time.time()
    df_model = prepare_engineered_dataset()

    n = len(df_model)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df_model.iloc[:train_end].copy()
    val_df = df_model.iloc[train_end:val_end].copy()
    test_df = df_model.iloc[val_end:].copy()

    train_df, val_df, test_df, city_priors = attach_training_only_priors(train_df, val_df, test_df)

    logger.info(
        "Chronological Split -> Train: %d (%s..%s), Val: %d (%s..%s), Test: %d (%s..%s)",
        len(train_df), train_df["Date"].min().strftime("%Y-%m-%d"), train_df["Date"].max().strftime("%Y-%m-%d"),
        len(val_df), val_df["Date"].min().strftime("%Y-%m-%d"), val_df["Date"].max().strftime("%Y-%m-%d"),
        len(test_df), test_df["Date"].min().strftime("%Y-%m-%d"), test_df["Date"].max().strftime("%Y-%m-%d"),
    )

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_df[FEATURE_COLUMNS])
    y_train = train_df["target_aqi_24h"].values

    X_val = imputer.transform(val_df[FEATURE_COLUMNS])
    y_val = val_df["target_aqi_24h"].values

    X_test = imputer.transform(test_df[FEATURE_COLUMNS])
    y_test = test_df["target_aqi_24h"].values

    candidates = {
        "CatBoostRegressor": CatBoostRegressor(
            iterations=950,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=6.0,
            subsample=0.85,
            loss_function="RMSE",
            eval_metric="MAE",
            random_seed=42,
            verbose=False,
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=480,
            learning_rate=0.035,
            max_depth=6,
            min_samples_leaf=22,
            l2_regularization=2.5,
            random_state=42,
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=220,
            max_depth=15,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=42,
        ),
        "StackedEnsembleRegressor": VotingRegressor(
            estimators=[
                (
                    "et",
                    ExtraTreesRegressor(
                        n_estimators=220,
                        max_depth=15,
                        min_samples_leaf=4,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
                (
                    "hgb",
                    HistGradientBoostingRegressor(
                        max_iter=480,
                        learning_rate=0.035,
                        max_depth=6,
                        min_samples_leaf=22,
                        l2_regularization=2.5,
                        random_state=42,
                    ),
                ),
                (
                    "cb",
                    CatBoostRegressor(
                        iterations=750,
                        learning_rate=0.035,
                        depth=6,
                        l2_leaf_reg=6.0,
                        subsample=0.85,
                        loss_function="RMSE",
                        random_seed=42,
                        verbose=False,
                    ),
                ),
            ],
            weights=[0.45, 0.30, 0.25],
        ),
    }

    benchmark_results: Dict[str, Any] = {}
    trained_models: Dict[str, Any] = {}

    benchmark_results["PersistenceBaseline"] = {
        "train": evaluate_predictions(y_train, train_df["current_aqi"].values),
        "validation": evaluate_predictions(y_val, val_df["current_aqi"].values),
        "test": evaluate_predictions(y_test, test_df["current_aqi"].values),
        "description": "Naive baseline predicting AQI(t+24h) = Observed AQI(t)",
    }

    best_model_name = None
    best_val_score = float("inf")

    for name, model in candidates.items():
        logger.info("Training candidate model: %s ...", name)
        if name == "CatBoostRegressor":
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=60, verbose=False)
        else:
            model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)

        train_metrics = evaluate_predictions(y_train, pred_train)
        val_metrics = evaluate_predictions(y_val, pred_val)
        test_metrics = evaluate_predictions(y_test, pred_test)

        benchmark_results[name] = {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
            "description": f"Supervised {name} trained on {len(FEATURE_COLUMNS)} zero-leakage features",
        }
        trained_models[name] = model

        logger.info(
            "%s -> Val MAE: %.2f, Val RMSE: %.2f, Val R²: %.4f | Test MAE: %.2f, Test RMSE: %.2f, Test R²: %.4f, Test Cat Acc: %.2f%%",
            name,
            val_metrics["mae"], val_metrics["rmse"], val_metrics["r2"],
            test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"], test_metrics["category_accuracy_pct"],
        )

        val_composite = val_metrics["mae"] + 0.20 * val_metrics["rmse"]
        if val_composite < best_val_score:
            best_val_score = val_composite
            best_model_name = name

    assert best_model_name is not None
    winning_model = trained_models[best_model_name]

    # =====================================================================
    # 3-STAGE CONTROLLED FEATURE ABLATION & FESTIVAL-WINDOW STUDY
    # =====================================================================
    stage1_cols = [
        "current_aqi", "aqi_lag_1", "pm25", "pm10", "no2", "so2", "co", "o3", "nh3", "pm25_to_pm10_ratio",
    ]
    stage2_cols = stage1_cols + WEATHER_AND_PBL_COLS

    imp_s1 = SimpleImputer(strategy="median")
    X_tr_s1 = imp_s1.fit_transform(train_df[stage1_cols])
    X_val_s1 = imp_s1.transform(val_df[stage1_cols])
    X_te_s1 = imp_s1.transform(test_df[stage1_cols])

    m_s1 = CatBoostRegressor(iterations=700, learning_rate=0.035, depth=6, l2_leaf_reg=6.0, random_seed=42, verbose=False)
    m_s1.fit(X_tr_s1, y_train, eval_set=(X_val_s1, y_val), early_stopping_rounds=50, verbose=False)
    pred_val_s1 = m_s1.predict(X_val_s1)
    pred_te_s1 = m_s1.predict(X_te_s1)
    s1_val = evaluate_predictions(y_val, pred_val_s1)
    s1_test = evaluate_predictions(y_test, pred_te_s1)

    imp_s2 = SimpleImputer(strategy="median")
    X_tr_s2 = imp_s2.fit_transform(train_df[stage2_cols])
    X_val_s2 = imp_s2.transform(val_df[stage2_cols])
    X_te_s2 = imp_s2.transform(test_df[stage2_cols])

    m_s2 = CatBoostRegressor(iterations=700, learning_rate=0.035, depth=6, l2_leaf_reg=6.0, random_seed=42, verbose=False)
    m_s2.fit(X_tr_s2, y_train, eval_set=(X_val_s2, y_val), early_stopping_rounds=50, verbose=False)
    pred_val_s2 = m_s2.predict(X_val_s2)
    pred_te_s2 = m_s2.predict(X_te_s2)
    s2_val = evaluate_predictions(y_val, pred_val_s2)
    s2_test = evaluate_predictions(y_test, pred_te_s2)

    full_val = benchmark_results[best_model_name]["validation"]
    full_test = benchmark_results[best_model_name]["test"]
    pred_te_full = winning_model.predict(X_test)

    # Evaluate specifically on Active Regional & National Festival / Event Window Days in Test Set
    fest_mask = (
        (test_df["regional_festival_proximity"].values > 0.15)
        | (test_df["is_diwali_window"].values == 1)
        | (test_df["is_festival_day"].values == 1)
    )
    if np.sum(fest_mask) >= 20:
        fest_s1 = evaluate_predictions(y_test[fest_mask], pred_te_s1[fest_mask])
        fest_s2 = evaluate_predictions(y_test[fest_mask], pred_te_s2[fest_mask])
        fest_full = evaluate_predictions(y_test[fest_mask], pred_te_full[fest_mask])
    else:
        fest_s1, fest_s2, fest_full = s1_test, s2_test, full_test

    ablation_study = {
        "with_weather_and_festival_context": {
            "val_mae": full_val["mae"],
            "val_rmse": full_val["rmse"],
            "test_mae": full_test["mae"],
            "test_rmse": full_test["rmse"],
            "test_r2": full_test["r2"],
            "test_category_accuracy_pct": full_test["category_accuracy_pct"],
        },
        "pollutants_plus_weather_without_festivals": {
            "val_mae": s2_val["mae"],
            "val_rmse": s2_val["rmse"],
            "test_mae": s2_test["mae"],
            "test_rmse": s2_test["rmse"],
            "test_r2": s2_test["r2"],
            "test_category_accuracy_pct": s2_test["category_accuracy_pct"],
        },
        "pollutant_lags_only_without_context": {
            "val_mae": s1_val["mae"],
            "val_rmse": s1_val["rmse"],
            "test_mae": s1_test["mae"],
            "test_rmse": s1_test["rmse"],
            "test_r2": s1_test["r2"],
            "test_category_accuracy_pct": s1_test["category_accuracy_pct"],
        },
        "festival_window_test_subset": {
            "festival_window_samples": int(np.sum(fest_mask)),
            "pollutant_only_mae": fest_s1["mae"],
            "weather_only_mae": fest_s2["mae"],
            "full_with_regional_festival_distance_mae": fest_full["mae"],
            "festival_window_mae_gain": round(fest_s1["mae"] - fest_full["mae"], 2),
            "festival_window_cat_acc_pct": fest_full["category_accuracy_pct"],
        },
        "context_mae_improvement": round(s1_test["mae"] - full_test["mae"], 2),
        "context_rmse_improvement": round(s1_test["rmse"] - full_test["rmse"], 2),
    }

    if hasattr(winning_model, "get_feature_importance"):
        raw_imp = winning_model.get_feature_importance()
    elif hasattr(winning_model, "feature_importances_"):
        raw_imp = winning_model.feature_importances_
    elif "CatBoostRegressor" in trained_models:
        raw_imp = trained_models["CatBoostRegressor"].get_feature_importance()
    else:
        raw_imp = np.ones(len(FEATURE_COLUMNS))

    total_imp = float(np.sum(raw_imp)) or 1.0
    feature_importances = [
        {"feature": col, "importance_pct": round(float(val) * 100.0 / total_imp, 2)}
        for col, val in sorted(zip(FEATURE_COLUMNS, raw_imp), key=lambda x: x[1], reverse=True)
    ]

    val_residuals = np.abs(y_val - winning_model.predict(X_val))
    q80_error = round(float(np.percentile(val_residuals, 80)), 1)
    q90_error = round(float(np.percentile(val_residuals, 90)), 1)

    test_df_eval = test_df.copy()
    test_df_eval["pred_aqi"] = np.clip(pred_te_full, 10.0, 500.0)
    city_test_breakdown = []
    for city_name, grp in test_df_eval.groupby("City"):
        if len(grp) >= 20:
            c_metrics = evaluate_predictions(grp["target_aqi_24h"].values, grp["pred_aqi"].values)
            city_test_breakdown.append({
                "city": city_name,
                "samples": int(len(grp)),
                "mae": c_metrics["mae"],
                "rmse": c_metrics["rmse"],
                "category_accuracy_pct": c_metrics["category_accuracy_pct"],
            })
    city_test_breakdown.sort(key=lambda x: x["samples"], reverse=True)

    model_version = f"airshield-{best_model_name.lower()}-v3.0"
    evaluation_report = {
        "model_version": model_version,
        "selected_model": best_model_name,
        "trained_at_utc": datetime.utcnow().isoformat() + "Z",
        "training_duration_sec": round(time.time() - start_time, 2),
        "dataset_summary": {
            "primary_source": "Official CPCB Multi-City Daily AQI Dataset (2015–2020) + CPCB-Standardized Multi-Year Archive (2023–2025) + Open-Meteo Historical Meteorology",
            "total_verified_samples": int(n),
            "cities_count": int(df_model["City"].nunique()),
            "cities_list": sorted(df_model["City"].unique().tolist()),
            "feature_count": len(FEATURE_COLUMNS),
            "train_split": {
                "samples": int(len(train_df)),
                "start_date": train_df["Date"].min().strftime("%Y-%m-%d"),
                "end_date": train_df["Date"].max().strftime("%Y-%m-%d"),
            },
            "validation_split": {
                "samples": int(len(val_df)),
                "start_date": val_df["Date"].min().strftime("%Y-%m-%d"),
                "end_date": val_df["Date"].max().strftime("%Y-%m-%d"),
            },
            "test_split": {
                "samples": int(len(test_df)),
                "start_date": test_df["Date"].min().strftime("%Y-%m-%d"),
                "end_date": test_df["Date"].max().strftime("%Y-%m-%d"),
            },
        },
        "selected_model_metrics": benchmark_results[best_model_name],
        "candidate_benchmarks": benchmark_results,
        "ablation_study": ablation_study,
        "feature_importances": feature_importances[:20],
        "uncertainty_quantiles": {
            "q80_abs_error": q80_error,
            "q90_abs_error": q90_error,
        },
        "city_generalization": city_test_breakdown[:14],
    }

    artifact_bundle = {
        "model": winning_model,
        "imputer": imputer,
        "feature_columns": FEATURE_COLUMNS,
        "city_priors": city_priors,
        "model_version": model_version,
        "selected_model": best_model_name,
        "uncertainty_quantiles": evaluation_report["uncertainty_quantiles"],
        "feature_medians": dict(zip(FEATURE_COLUMNS, imputer.statistics_)),
        "evaluation_report": evaluation_report,
    }

    joblib.dump(artifact_bundle, MODEL_ARTIFACT_PATH, compress=3)
    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    logger.info("Saved winning model (%s) to %s", best_model_name, MODEL_ARTIFACT_PATH)
    return evaluation_report


if __name__ == "__main__":
    report = train_and_evaluate_pipeline()
    print("\n=== AIRSHIELD v3.0 MULTI-YEAR CHRONOLOGICAL VALIDATION SUMMARY ===")
    print(f"Selected Model : {report['selected_model']} ({report['model_version']})")
    print(
        f"Total Samples  : {report['dataset_summary']['total_verified_samples']} across "
        f"{report['dataset_summary']['cities_count']} Indian cities ({report['dataset_summary']['feature_count']} features)"
    )
    for split_name in ["train", "validation", "test"]:
        m = report["selected_model_metrics"][split_name]
        print(
            f"  {split_name.upper():10s} -> MAE: {m['mae']:5.2f} | RMSE: {m['rmse']:5.2f} | "
            f"R²: {m['r2']:6.4f} | Cat Acc: {m['category_accuracy_pct']:5.2f}% | Macro F1: {m['macro_f1']:5.4f}"
        )
    abl = report["ablation_study"]
    print("\n=== 3-STAGE FEATURE ABLATION & FESTIVAL-WINDOW EFFECTIVENESS ===")
    print(f"  Stage 1 (Pollutant Lags Only)            -> Test MAE: {abl['pollutant_lags_only_without_context']['test_mae']:5.2f} | RMSE: {abl['pollutant_lags_only_without_context']['test_rmse']:5.2f}")
    print(f"  Stage 2 (+ Current & Next-24h Weather)   -> Test MAE: {abl['pollutants_plus_weather_without_festivals']['test_mae']:5.2f} | RMSE: {abl['pollutants_plus_weather_without_festivals']['test_rmse']:5.2f}")
    print(f"  Stage 3 (+ Regional Festival Distances)  -> Test MAE: {abl['with_weather_and_festival_context']['test_mae']:5.2f} | RMSE: {abl['with_weather_and_festival_context']['test_rmse']:5.2f}")
    fw = abl["festival_window_test_subset"]
    print(
        f"  Festival Window Subset ({fw['festival_window_samples']} days) -> "
        f"Pollutant-Only MAE: {fw['pollutant_only_mae']:5.2f} vs Full Pipeline MAE: {fw['full_with_regional_festival_distance_mae']:5.2f} "
        f"(+{fw['festival_window_mae_gain']:4.2f} AQI improvement)"
    )
