#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_FILE = "ev_charging_sessions.csv"  # adjust if Kaggle used a different name
TARGET_COL = "Energy Consumed (kWh)"


def load_data():
    if not os.path.exists(DATA_FILE):
        # try to guess any csv in the current dir
        csvs = [f for f in os.listdir(".") if f.lower().endswith(".csv")]
        if not csvs:
            raise FileNotFoundError(
                f"Could not find {DATA_FILE} or any .csv in {os.getcwd()}"
            )
        print(f"{DATA_FILE} not found, using {csvs[0]} instead")
        fname = csvs[0]
    else:
        fname = DATA_FILE

    df = pd.read_csv(fname)
    print(f"Loaded {fname} with shape {df.shape}")
    return df


def build_feature_sets(df: pd.DataFrame):
    """
    Returns two (X, y) pairs:
      - config_a: 'realistic' feature set (no distance, no duration, no rate, no cost)
      - config_b: 'physics' feature set (includes distance, duration, rate)
    """

    # core numerics
    numeric_common = [
        "Battery Capacity (kWh)",
        "State of Charge (Start %)",
        "Temperature (°C)",
        "Vehicle Age (years)",
    ]

    # these are strong signal but you complained about realism
    numeric_physics_extra = [
        "Distance Driven (since last charge) (km)",
        "Charging Duration (hours)",
        "Charging Rate (kW)",
        "State of Charge (End %)",
        "Charging Cost (USD)",
    ]

    categoricals = [
        "Vehicle Model",
        "Charging Station Location",
        "Time of Day",
        "Day of Week",
        "Charger Type",
        "User Type",
    ]

    # Sanity: check that all columns exist
    for col in numeric_common + numeric_physics_extra + categoricals + [TARGET_COL]:
        if col not in df.columns:
            # don't explode, just warn
            print(f"[WARN] Column missing in data: {col}")

    # ---------- CONFIG A: your 'realistic' setup ----------
    cols_a = numeric_common + categoricals + [TARGET_COL]
    df_a = df[cols_a].copy()

    print("\n[CONFIG A] NaNs per column before dropna:")
    print(df_a.isna().sum())

    df_a = df_a.dropna()
    print("[CONFIG A] Shape after dropna:", df_a.shape)

    df_a_enc = pd.get_dummies(df_a, columns=categoricals, drop_first=True)

    y_a = df_a_enc[TARGET_COL]
    X_a = df_a_enc.drop(columns=[TARGET_COL])

    print("[CONFIG A] X shape:", X_a.shape, " y shape:", y_a.shape)

    # ---------- CONFIG B: 'physics' setup with extra signals ----------
    cols_b = numeric_common + numeric_physics_extra + categoricals + [TARGET_COL]
    cols_b = [c for c in cols_b if c in df.columns]  # only keep existing

    df_b = df[cols_b].copy()

    print("\n[CONFIG B] NaNs per column before dropna:")
    print(df_b.isna().sum())

    df_b = df_b.dropna()
    print("[CONFIG B] Shape after dropna:", df_b.shape)

    df_b_enc = pd.get_dummies(df_b, columns=categoricals, drop_first=True)

    y_b = df_b_enc[TARGET_COL]
    X_b = df_b_enc.drop(columns=[TARGET_COL])

    print("[CONFIG B] X shape:", X_b.shape, " y shape:", y_b.shape)

    return (X_a, y_a), (X_b, y_b)


def eval_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # baseline: predict mean of y_train
    y_base = np.full_like(y_test, y_train.mean(), dtype=float)
    base_mse = mean_squared_error(y_test, y_base)

    print(f"\n=== {name} ===")
    print("MAE:", mae)
    print("MSE:", mse)
    print("R^2:", r2)
    print("Baseline MSE (mean-only):", base_mse)

    return mae, mse, r2


def run_config(label, X, y):
    print(f"\n================ {label} ================")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=2,
    )
    eval_model(f"{label} - RandomForest", rf, X_train, X_test, y_train, y_test)

    # L1-regularized linear regression (Lasso), NOT logistic
    lasso = LassoCV(cv=5, random_state=42, n_jobs=-1)
    eval_model(f"{label} - LassoCV", lasso, X_train, X_test, y_train, y_test)

    # show some feature importance / coefficients for debugging
    if hasattr(rf, "feature_importances_"):
        importances = pd.Series(
            rf.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        print("\nTop 10 RF feature importances:")
        print(importances.head(10))

    if hasattr(lasso, "coef_"):
        coefs = pd.Series(lasso.coef_, index=X.columns).sort_values(
            key=lambda s: s.abs(), ascending=False
        )
        print("\nTop 10 Lasso coefficients (by |value|):")
        print(coefs.head(10))


def main():
    df = load_data()
    (X_a, y_a), (X_b, y_b) = build_feature_sets(df)

    # Config A: your “no distance / no duration / no rate / no cost” setup
    run_config("CONFIG A (realistic, crippled features)", X_a, y_a)

    # Config B: physics-aware setup (includes distance + duration + rate + cost)
    # This is to prove whether the pipeline is actually broken or your feature
    # constraints are just too aggressive.
    run_config("CONFIG B (with distance/duration/rate/cost)", X_b, y_b)


if __name__ == "__main__":
    main()
