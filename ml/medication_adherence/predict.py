import os
import pickle
import numpy as np

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_model")

with open(f"{SAVE_DIR}/medication_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(f"{SAVE_DIR}/meta.pkl", "rb") as f:
    meta = pickle.load(f)

FEATURES = meta["features"]
activity_map = meta["activity_map"]
day_map = meta["day_map"]

def predict_reminder_offset(
    scheduled_time_str: str,
    wake_time_str: str,
    sleep_time_str: str,
    activity_level: str,
    missed_last_week: int,
    day_of_week: str,
    taken_on_time: int = 0
) -> dict:

    def to_minutes(t):
        h, m = map(int, t.split(":"))
        return h * 60 + m

    scheduled_minutes = to_minutes(scheduled_time_str)
    wake_minutes = to_minutes(wake_time_str)
    sleep_minutes = to_minutes(sleep_time_str)

    activity_encoded = activity_map.get(activity_level, 1)
    day_encoded = day_map.get(day_of_week, 0)
    is_weekend = 1 if day_encoded >= 5 else 0

    def time_category(m):
        if m < 720: return 0
        elif m < 1080: return 1
        else: return 2

    time_cat = time_category(scheduled_minutes)
    sleep_duration = (sleep_minutes - wake_minutes) % 1440

    features = [[
        scheduled_minutes,
        wake_minutes,
        sleep_minutes,
        activity_encoded,
        missed_last_week,
        day_encoded,
        is_weekend,
        time_cat,
        sleep_duration,
        taken_on_time
    ]]

    offset = model.predict(features)[0]
    offset = max(5, int(round(offset)))

    reminder_minutes = scheduled_minutes - offset
    if reminder_minutes < 0:
        reminder_minutes += 1440
    h = reminder_minutes // 60
    m = reminder_minutes % 60
    reminder_time = f"{h:02d}:{m:02d}"

    return {
        "scheduled_time": scheduled_time_str,
        "reminder_offset_minutes": offset,
        "recommended_reminder_time": reminder_time,
        "message": f"I'll remind you {offset} minutes before your {scheduled_time_str} medicine."
    }


if __name__ == "__main__":
    test_cases = [
        {
            "scheduled_time_str": "20:00",
            "wake_time_str": "07:00",
            "sleep_time_str": "23:00",
            "activity_level": "low",
            "missed_last_week": 5,
            "day_of_week": "Monday",
            "taken_on_time": 0
        },
        {
            "scheduled_time_str": "08:00",
            "wake_time_str": "06:30",
            "sleep_time_str": "22:00",
            "activity_level": "high",
            "missed_last_week": 1,
            "day_of_week": "Wednesday",
            "taken_on_time": 1
        },
        {
            "scheduled_time_str": "13:00",
            "wake_time_str": "08:00",
            "sleep_time_str": "00:00",
            "activity_level": "medium",
            "missed_last_week": 3,
            "day_of_week": "Saturday",
            "taken_on_time": 0
        }
    ]

    print("\n── Medication Reminder Predictions ──\n")
    for case in test_cases:
        result = predict_reminder_offset(**case)
        print(f"Scheduled     : {result['scheduled_time']}")
        print(f"Remind at     : {result['recommended_reminder_time']}")
        print(f"Offset        : {result['reminder_offset_minutes']} mins before")
        print(f"Message       : {result['message']}")
        print("─" * 45)
