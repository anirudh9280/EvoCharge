import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ========= config =========
DATA_PATH = "data/ev_charging_sessions/ev_charging_sessions.csv"

TIME_START_COL = "start_time"
TIME_END_COL   = "end_time"
TARGET_COL     = "energy_kWh"

SESSION_DAY_COL  = "session_day"
SESSION_TYPE_COL = "session_type"
STATION_COL      = "station_id"
# ==========================

df = pd.read_csv(DATA_PATH)

df[TIME_START_COL] = pd.to_datetime(df[TIME_START_COL])
df[TIME_END_COL]   = pd.to_datetime(df[TIME_END_COL])

# keep only positive durations
df = df[df[TIME_END_COL] > df[TIME_START_COL]].copy()

df["duration_hours"] = (df[TIME_END_COL] - df[TIME_START_COL]).dt.total_seconds() / 3600.0
df["start_hour"] = df[TIME_START_COL].dt.hour + df[TIME_START_COL].dt.minute / 60.0

# drop any zero or extremely tiny durations to avoid insane kW
df = df[df["duration_hours"] > 1e-3].copy()

# true energy and true average power
y_energy = df[TARGET_COL].values
y_power = df[TARGET_COL].values / df["duration_hours"].values   # kW target

important_num = ["duration_hours", "start_hour"]
important_cat = [SESSION_DAY_COL, SESSION_TYPE_COL]
other_num = []
other_cat = [STATION_COL]

for col in ["session_id", "user_id", "vehicle_id"]:
    if col in df.columns:
        df[col] = 0

X = df[important_num + important_cat + other_num + other_cat]

# split by index so energy and power stay aligned
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

X_train = X.iloc[train_idx]
X_test  = X.iloc[test_idx]

y_power_train = y_power[train_idx]
y_power_test  = y_power[test_idx]

y_energy_train = y_energy[train_idx]
y_energy_test  = y_energy[test_idx]

transformers = []

if important_num:
    transformers.append(("imp_num", StandardScaler(), important_num))

if important_cat:
    transformers.append(("imp_cat", OneHotEncoder(handle_unknown="ignore"), important_cat))

if other_num:
    transformers.append(("other_num", StandardScaler(), other_num))

if other_cat:
    transformers.append((
        "other_cat",
        OneHotEncoder(handle_unknown="ignore"),
        other_cat,
    ))

preprocessor = ColumnTransformer(transformers=transformers)

# L1-regularized linear model on avg kW
lasso = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LassoCV(alphas=np.logspace(-3, 1, 20), cv=5, random_state=42)),
    ]
)

# less crippled RF, still regularized
rf = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=20,
                random_state=42,
            ),
        ),
    ]
)

# SVR with RBF kernel and small grid search
svr_pipe = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", SVR(kernel="rbf"))
    ]
)

svr_param_grid = {
    "model__C": [1, 10, 100],
    "model__epsilon": [0.5, 1.0, 2.0],
    "model__gamma": ["scale", 0.1, 0.01],
}

svr_cv = GridSearchCV(
    svr_pipe,
    param_grid=svr_param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)

# fit on avg kW
lasso.fit(X_train, y_power_train)
rf.fit(X_train, y_power_train)
svr_cv.fit(X_train, y_power_train)
svr_best = svr_cv.best_estimator_
print("Best SVR params:", svr_cv.best_params_)

# evaluate all models on kWh (energy) metrics
def eval_model(name, model):
    power_pred = model.predict(X_test)                            # kW
    energy_pred = power_pred * X_test["duration_hours"].values    # kWh
    mae = mean_absolute_error(y_energy_test, energy_pred)
    mse = mean_squared_error(y_energy_test, energy_pred)
    r2 = r2_score(y_energy_test, energy_pred)
    print(f"{name}: MAE={mae:.3f}, MSE={mse:.3f}, R^2={r2:.3f}")
    return energy_pred

print("\nModel performance (energy_kWh):")
energy_pred_lasso = eval_model("Lasso(avg_kW)", lasso)
energy_pred_rf    = eval_model("RandomForest(avg_kW)", rf)
energy_pred_svr   = eval_model("SVR(avg_kW)", svr_best)

# baselines
# 1) mean energy baseline
baseline_energy_mean = np.full_like(y_energy_test, y_energy_train.mean())
mse_mean = mean_squared_error(y_energy_test, baseline_energy_mean)
print(f"\nBaseline (mean energy): MSE={mse_mean:.3f}")

# 2) constant-kW baseline
baseline_kW = y_power_train.mean()
baseline_energy_constkW = baseline_kW * X_test["duration_hours"].values
mse_constkW = mean_squared_error(y_energy_test, baseline_energy_constkW)
print(f"Baseline (constant avg kW): MSE={mse_constkW:.3f}")

# write predictions for all rows
df["kWh_pred_lasso"] = lasso.predict(X) * df["duration_hours"].values
df["kWh_pred_rf"]    = rf.predict(X) * df["duration_hours"].values
df["kWh_pred_svr"]   = svr_best.predict(X) * df["duration_hours"].values

df.to_csv("charging_sessions_with_predictions.csv", index=False)
print("\nSaved charging_sessions_with_predictions.csv")
