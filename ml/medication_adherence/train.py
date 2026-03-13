import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# ── Config ───────────────────────────────────────────────
DATA_PATH = "dataset/medication_data.csv"
SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load Data ────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Dataset size: {len(df)} samples")

# ── Feature Engineering ──────────────────────────────────
# encode activity level
activity_map = {"low": 0, "medium": 1, "high": 2}
df["activity_encoded"] = df["activity_level"].map(activity_map)

# encode day of week
day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}
df["day_encoded"] = df["day_of_week"].map(day_map)

# is weekend
df["is_weekend"] = df["day_encoded"].apply(lambda x: 1 if x >= 5 else 0)

# time of day category
def time_category(minutes):
    if minutes < 720:
        return 0  # morning
    elif minutes < 1080:
        return 1  # afternoon
    else:
        return 2  # evening

df["time_category"] = df["scheduled_minutes"].apply(time_category)

# sleep duration
df["sleep_duration"] = df.apply(
    lambda r: (r["sleep_minutes"] - r["wake_minutes"]) % 1440,
    axis=1
)

# ── Features & Target ────────────────────────────────────
FEATURES = [
    "scheduled_minutes",
    "wake_minutes",
    "sleep_minutes",
    "activity_encoded",
    "missed_last_week",
    "day_encoded",
    "is_weekend",
    "time_category",
    "sleep_duration",
    "taken_on_time"
]

TARGET = "optimal_reminder_offset"

X = df[FEATURES]
y = df[TARGET]

# ── Train/Test Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── Train Model ──────────────────────────────────────────
print("\nTraining Random Forest...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nResults:")
print(f"MAE : {mae:.2f} minutes")
print(f"R2  : {r2:.4f}")

# ── Feature Importance ───────────────────────────────────
print("\nFeature Importance:")
importances = model.feature_importances_
for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
    print(f"  {feat:<25} {imp:.4f}")

# ── Save Model ───────────────────────────────────────────
model_path = f"{SAVE_DIR}/medication_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# save feature list and encoders
meta = {
    "features": FEATURES,
    "activity_map": activity_map,
    "day_map": day_map
}
with open(f"{SAVE_DIR}/meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print(f"\nModel saved to {model_path}")
