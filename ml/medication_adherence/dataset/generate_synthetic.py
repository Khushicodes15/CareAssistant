import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ── Config ───────────────────────────────────────────────
NUM_SAMPLES = 8000

# ── Helpers ──────────────────────────────────────────────
def random_time_minutes(start, end):
    """Return random time in minutes from midnight"""
    return random.randint(start, end)

def minutes_to_str(minutes):
    h = (minutes % 1440) // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"

def activity_level():
    return random.choice(["low", "medium", "high"])

def day_of_week():
    return random.choice(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

# ── Generate ─────────────────────────────────────────────
rows = []

for _ in range(NUM_SAMPLES):
    # scheduled medicine time (in minutes from midnight)
    scheduled_minutes = random_time_minutes(360, 1320)  # 6am to 10pm

    # sleep/wake times
    sleep_minutes = random_time_minutes(1320, 1440) + random_time_minutes(0, 60)  # 10pm - 1am
    wake_minutes = random_time_minutes(360, 480)  # 6am - 8am

    # activity
    activity = activity_level()
    day = day_of_week()

    # missed count last week (0-7)
    missed_last_week = random.randint(0, 7)

    # simulate whether taken on time
    # logic: more likely to miss if high missed_last_week, low activity, late scheduled time
    miss_probability = (missed_last_week / 7) * 0.5
    if activity == "low":
        miss_probability += 0.2
    if scheduled_minutes > 1080:  # after 6pm
        miss_probability += 0.15
    if day in ["Saturday", "Sunday"]:
        miss_probability += 0.1

    miss_probability = min(miss_probability, 0.95)
    taken_on_time = 1 if random.random() > miss_probability else 0

    # optimal reminder offset (minutes before scheduled time)
    # if likely to miss → remind earlier
    if missed_last_week >= 5:
        offset = random.randint(30, 60)
    elif missed_last_week >= 3:
        offset = random.randint(15, 30)
    else:
        offset = random.randint(5, 15)

    # add noise
    offset += random.randint(-5, 5)
    offset = max(5, offset)

    rows.append({
        "scheduled_time": minutes_to_str(scheduled_minutes),
        "scheduled_minutes": scheduled_minutes,
        "sleep_time": minutes_to_str(sleep_minutes % 1440),
        "sleep_minutes": sleep_minutes % 1440,
        "wake_time": minutes_to_str(wake_minutes),
        "wake_minutes": wake_minutes,
        "activity_level": activity,
        "missed_last_week": missed_last_week,
        "day_of_week": day,
        "taken_on_time": taken_on_time,
        "optimal_reminder_offset": offset
    })

df = pd.DataFrame(rows)
df.to_csv("medication_data.csv", index=False)

print(f"Generated {len(df)} samples")
print(df.head())
print(f"\nTaken on time: {df['taken_on_time'].mean():.2%}")
print(f"Avg reminder offset: {df['optimal_reminder_offset'].mean():.1f} minutes")