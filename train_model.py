"""
AirShield — Supervised Machine Learning Training & Chronological Validation Pipeline (train_model.py)
Responsibilities:
1. Download / load verified historical CPCB observations and matching historical meteorology for all 26 cities
2. Reconstruct missing AQI strictly when CPCB 3-pollutant + PM qualification rules are met
3. Engineer zero-leakage lag, EWMA, rolling, CPCB sub-index, training-only city-month climatology,
   meteorological, trigonometric wind, and festival/seasonal features
4. Perform strict chronological splitting (70% Train / 15% Validation / 15% Test)
5. Fit imputation and city-month priors strictly on Training split only
6. Benchmark CatBoostRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor, and Persistence Baseline
7. Perform feature-ablation study on festival/seasonal & meteorological context features
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

FEATURE_COLUMNS: List[str] = [
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
    "ventilation_proxy",
    "latitude",
    "longitude",
    "month",
    "doy_sin",
    "doy_cos",
    "day_of_week",
    "is_weekend",
    "is_diwali_window",
    "days_from_diwali",
    "is_festival_day",
    "is_stubble_burning_season",
    "is_winter_inversion",
    "is_dust_season",
    "is_monsoon_season",
]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics (MAE, RMSE, R²) and CPCB category metrics (Accuracy, Macro F1, Weighted F1)."""
    y_pred_clipped = np.clip(np.round(y_pred, 1), 10.0, 500.0)
    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred_clipped)))
    r2 = float(r2_score(y_true, y_pred_clipped))

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


def prepare_engineered_dataset() -> pd.DataFrame:
    """
    Clean CPCB dataset, merge with historical meteorology at issue time t,
    compute zero-leakage lag/EWMA/rolling/contextual features, and align with target AQI at t + 24h.
    """
    cpcb_df, wx_df = ensure_historical_training_datasets()
    logger.info("Loaded raw CPCB dataset (%d rows) and historical weather dataset (%d rows)", len(cpcb_df), len(wx_df))

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
    df["ventilation_proxy"] = df["wind_speed_ms"] * df["temperature_c"]

    df["month"] = df["Date"].dt.month
    doy = df["Date"].dt.dayofyear
    df["doy_sin"] = np.sin(2.0 * math.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2.0 * math.pi * doy / 365.25)
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    diwali_dates = [
        pd.Timestamp("2015-11-11"), pd.Timestamp("2016-10-30"), pd.Timestamp("2017-10-19"),
        pd.Timestamp("2018-11-07"), pd.Timestamp("2019-10-27"), pd.Timestamp("2020-11-14"),
        pd.Timestamp("2021-11-04"), pd.Timestamp("2022-10-24"), pd.Timestamp("2023-11-12"),
        pd.Timestamp("2024-10-31"), pd.Timestamp("2025-10-20"), pd.Timestamp("2026-11-08"),
    ]
    all_fest_dates = set(diwali_dates + [
        pd.Timestamp("2015-10-22"), pd.Timestamp("2016-10-11"), pd.Timestamp("2017-09-30"),
        pd.Timestamp("2018-10-19"), pd.Timestamp("2019-10-08"), pd.Timestamp("2020-10-25"),
        pd.Timestamp("2015-03-06"), pd.Timestamp("2016-03-24"), pd.Timestamp("2017-03-13"),
        pd.Timestamp("2018-03-02"), pd.Timestamp("2019-03-21"), pd.Timestamp("2020-03-10"),
    ])

    def calc_days_from_diwali(dt_series: pd.Series) -> np.ndarray:
        vals = np.full(len(dt_series), 30, dtype=int)
        for d_dt in diwali_dates:
            diff = (dt_series - d_dt).dt.days.values
            mask = np.abs(diff) < np.abs(vals)
            vals[mask] = np.clip(diff[mask], -30, 30)
        return vals

    df["days_from_diwali"] = calc_days_from_diwali(df["Date"])
    df["is_diwali_window"] = ((df["days_from_diwali"] >= -2) & (df["days_from_diwali"] <= 4)).astype(int)
    df["is_festival_day"] = (
        df["Date"].isin(all_fest_dates)
        | ((df["Date"].dt.month == 1) & (df["Date"].dt.day.isin([1, 26])))
        | ((df["Date"].dt.month == 8) & (df["Date"].dt.day == 15))
        | ((df["Date"].dt.month == 10) & (df["Date"].dt.day == 2))
        | ((df["Date"].dt.month == 12) & (df["Date"].dt.day.isin([25, 31])))
    ).astype(int)

    north_cities = {
        "Delhi", "New Delhi", "Gurugram", "Lucknow", "Patna", "Jaipur",
        "Amritsar", "Chandigarh", "Agra", "Varanasi", "Kanpur", "Noida", "Ghaziabad",
    }
    is_north = df["City"].isin(north_cities)
    df["is_stubble_burning_season"] = (
        is_north
        & (
            ((df["month"] == 10) & (df["Date"].dt.day >= 10))
            | ((df["month"] == 11) & (df["Date"].dt.day <= 25))
        )
    ).astype(int)

    df["is_winter_inversion"] = (
        df["month"].isin([11, 12, 1, 2])
        & (is_north | (df["temperature_c"].fillna(20.0) < 19.0))
    ).astype(int)

    df["is_dust_season"] = (
        df["month"].isin([4, 5, 6])
        & (is_north | df["City"].isin(["Ahmedabad", "Bhopal", "Jodhpur", "Talcher"]))
        & (df["humidity_pct"].fillna(35.0) < 42.0)
    ).astype(int)

    df["is_monsoon_season"] = (
        df["month"].isin([7, 8, 9]) | ((df["month"] == 6) & (df["Date"].dt.day >= 15))
    ).astype(int)

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
    df = prepare_engineered_dataset()

    df_model = df[df["Date"] >= pd.Timestamp("2017-01-01")].reset_index(drop=True)
    if len(df_model) < 5000:
        df_model = df.copy()

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
            iterations=900,
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
            max_iter=450,
            learning_rate=0.035,
            max_depth=6,
            min_samples_leaf=22,
            l2_regularization=2.5,
            random_state=42,
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=220,
            max_depth=14,
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
                        max_depth=14,
                        min_samples_leaf=4,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
                (
                    "hgb",
                    HistGradientBoostingRegressor(
                        max_iter=450,
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
                        iterations=700,
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
            weights=[0.45, 0.35, 0.20],
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

        # Select model with lowest combined validation MAE + RMSE
        val_composite = val_metrics["mae"] + 0.5 * val_metrics["rmse"]
        if val_composite < best_val_score:
            best_val_score = val_composite
            best_model_name = name

    assert best_model_name is not None
    winning_model = trained_models[best_model_name]

    # Feature Ablation Study
    pollutant_only_cols = [
        "current_aqi", "aqi_lag_1", "aqi_lag_2", "aqi_lag_3", "aqi_lag_7",
        "aqi_delta_24h", "aqi_roll_mean_3", "aqi_roll_max_3", "aqi_roll_std_3", "aqi_roll_mean_7",
        "pm25", "pm10", "no2", "so2", "co", "o3", "nh3", "pm25_to_pm10_ratio",
    ]
    imp_ablation = SimpleImputer(strategy="median")
    X_tr_abl = imp_ablation.fit_transform(train_df[pollutant_only_cols])
    X_val_abl = imp_ablation.transform(val_df[pollutant_only_cols])
    X_te_abl = imp_ablation.transform(test_df[pollutant_only_cols])

    abl_model = CatBoostRegressor(
        iterations=500, learning_rate=0.04, depth=6, random_seed=42, verbose=False
    )
    abl_model.fit(X_tr_abl, y_train, eval_set=(X_val_abl, y_val), early_stopping_rounds=40, verbose=False)
    ablation_val = evaluate_predictions(y_val, abl_model.predict(X_val_abl))
    ablation_test = evaluate_predictions(y_test, abl_model.predict(X_te_abl))

    full_val = benchmark_results[best_model_name]["validation"]
    full_test = benchmark_results[best_model_name]["test"]

    ablation_study = {
        "with_weather_and_festival_context": {
            "val_mae": full_val["mae"],
            "val_rmse": full_val["rmse"],
            "test_mae": full_test["mae"],
            "test_rmse": full_test["rmse"],
            "test_r2": full_test["r2"],
        },
        "pollutant_lags_only_without_context": {
            "val_mae": ablation_val["mae"],
            "val_rmse": ablation_val["rmse"],
            "test_mae": ablation_test["mae"],
            "test_rmse": ablation_test["rmse"],
            "test_r2": ablation_test["r2"],
        },
        "context_mae_improvement": round(ablation_test["mae"] - full_test["mae"], 2),
        "context_rmse_improvement": round(ablation_test["rmse"] - full_test["rmse"], 2),
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
    test_df_eval["pred_aqi"] = np.clip(winning_model.predict(X_test), 10.0, 500.0)
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

    model_version = f"airshield-{best_model_name.lower()}-v2.0"
    evaluation_report = {
        "model_version": model_version,
        "selected_model": best_model_name,
        "trained_at_utc": datetime.utcnow().isoformat() + "Z",
        "training_duration_sec": round(time.time() - start_time, 2),
        "dataset_summary": {
            "primary_source": "Official CPCB Multi-City Daily AQI Dataset (data.gov.in / CPCB Archive) + Open-Meteo Historical Meteorology",
            "total_verified_samples": int(n),
            "cities_count": int(df_model["City"].nunique()),
            "cities_list": sorted(df_model["City"].unique().tolist()),
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
        "feature_importances": feature_importances[:15],
        "uncertainty_quantiles": {
            "q80_abs_error": q80_error,
            "q90_abs_error": q90_error,
        },
        "city_generalization": city_test_breakdown[:12],
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

    joblib.dump(artifact_bundle, MODEL_ARTIFACT_PATH)
    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    logger.info("Saved winning model (%s) to %s", best_model_name, MODEL_ARTIFACT_PATH)
    return evaluation_report


if __name__ == "__main__":
    report = train_and_evaluate_pipeline()
    print("\n=== AIRSHIELD v2.0 CHRONOLOGICAL VALIDATION SUMMARY ===")
    print(f"Selected Model : {report['selected_model']} ({report['model_version']})")
    print(f"Total Samples  : {report['dataset_summary']['total_verified_samples']} across {report['dataset_summary']['cities_count']} Indian cities")
    for split_name in ["train", "validation", "test"]:
        m = report["selected_model_metrics"][split_name]
        print(
            f"  {split_name.upper():10s} -> MAE: {m['mae']:5.2f} | RMSE: {m['rmse']:5.2f} | "
            f"R²: {m['r2']:6.4f} | Cat Acc: {m['category_accuracy_pct']:5.2f}% | Macro F1: {m['macro_f1']:5.4f}"
        )
    base_test = report["candidate_benchmarks"]["PersistenceBaseline"]["test"]
    print(
        f"  PERSISTENCE -> MAE: {base_test['mae']:5.2f} | RMSE: {base_test['rmse']:5.2f} | "
        f"R²: {base_test['r2']:6.4f} | Cat Acc: {base_test['category_accuracy_pct']:5.2f}%"
    )
