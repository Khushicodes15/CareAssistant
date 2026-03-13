import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


SAVE_DIR = "saved_model"

with open(f"{SAVE_DIR}/medication_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(f"{SAVE_DIR}/meta.pkl", "rb") as f:
    meta = pickle.load(f)

FEATURES = meta["features"]
activity_map = meta["activity_map"]
day_map = meta["day_map"]


df = pd.read_csv("dataset/medication_data.csv")

df["activity_encoded"] = df["activity_level"].map(activity_map)
df["day_encoded"] = df["day_of_week"].map(day_map)
df["is_weekend"] = df["day_encoded"].apply(lambda x: 1 if x >= 5 else 0)
df["time_category"] = df["scheduled_minutes"].apply(
    lambda m: 0 if m < 720 else (1 if m < 1080 else 2)
)
df["sleep_duration"] = df.apply(
    lambda r: (r["sleep_minutes"] - r["wake_minutes"]) % 1440, axis=1
)

X = df[FEATURES]
y = df["optimal_reminder_offset"]

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE : {mae:.2f} minutes")
print(f"R2  : {r2:.4f}")


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.3, color="steelblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Offset")
plt.ylabel("Predicted Offset")
plt.title("Actual vs Predicted Reminder Offset")

plt.subplot(1, 2, 2)
importances = model.feature_importances_
plt.barh(FEATURES, importances, color="steelblue")
plt.xlabel("Importance")
plt.title("Feature Importance")

plt.tight_layout()
plt.savefig("medication_evaluation.png")
print("Plot saved as medication_evaluation.png")