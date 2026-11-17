import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1000

gender = rng.choice(["Male", "Female"], N)
study_hours = np.clip(rng.normal(3.5, 1.5, N), 0, 9).round(1)
attendance = np.clip(rng.normal(80, 12, N), 35, 100).round(1)
sleep_hours = np.clip(rng.normal(7, 1.2, N), 4, 10).round(1)
parent_edu = rng.choice(
    ["High School", "Bachelor", "Master", "Primary"], N, p=[0.35, 0.35, 0.15, 0.15]
)
internet = rng.choice(["Yes", "No"], N, p=[0.75, 0.25])
extra_classes = rng.choice(["Yes", "No"], N, p=[0.4, 0.6])

edu_bonus = pd.Series(parent_edu).map(
    {"Primary": 0, "High School": 2, "Bachelor": 4, "Master": 6}
).to_numpy()
base = (
    18
    + 5.5 * study_hours
    + 0.28 * attendance
    + 1.5 * (sleep_hours - 7)
    + edu_bonus
    + 3 * (extra_classes == "Yes")
)


def subject(offset, noise):
    return np.clip(base + offset + rng.normal(0, noise, N), 0, 100).round(0)


df = pd.DataFrame(
    {
        "student_id": np.arange(1001, 1001 + N),
        "gender": gender,
        "study_hours_per_day": study_hours,
        "attendance_pct": attendance,
        "sleep_hours": sleep_hours,
        "parental_education": parent_edu,
        "internet_access": internet,
        "extra_classes": extra_classes,
        "math": subject(-3, 7),
        "science": subject(0, 7),
        "english": subject(3, 7),
    }
)

# ---- inject messiness ----
# inconsistent categorical labels
df["gender"] = df["gender"].map(
    lambda g: rng.choice([g, g.lower(), g.upper(), g[0], f" {g} "], p=[0.7, 0.1, 0.05, 0.1, 0.05])
)
df["internet_access"] = df["internet_access"].map(
    lambda v: rng.choice([v, v.lower(), "Y" if v == "Yes" else "N"], p=[0.8, 0.1, 0.1])
)
df["parental_education"] = df["parental_education"].map(
    lambda v: rng.choice([v, v.lower(), v + " "], p=[0.85, 0.1, 0.05])
)

# missing values
for col, frac in [("math", 0.04), ("science", 0.04), ("english", 0.03),
                  ("study_hours_per_day", 0.05), ("attendance_pct", 0.04),
                  ("sleep_hours", 0.03), ("parental_education", 0.02)]:
    df.loc[rng.choice(N, int(N * frac), replace=False), col] = np.nan

# invalid marks / attendance (out of range)
df.loc[rng.choice(N, 8, replace=False), "math"] = rng.choice([105, 120, -5, 999], 8)
df.loc[rng.choice(N, 5, replace=False), "attendance_pct"] = rng.choice([110, 150, -10], 5)

# duplicate rows
dupes = df.sample(25, random_state=1)
df = pd.concat([df, dupes], ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("student_data_raw.csv", index=False)
print(f"Wrote student_data_raw.csv with {len(df)} rows")