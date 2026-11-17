import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("charts", exist_ok=True)
SUBJECTS = ["math", "science", "english"]

# ------------------------------------------------------------------ load
raw = pd.read_csv("student_data_raw.csv")
n_raw = len(raw)
print(f"Raw records: {n_raw}")
print("Missing values per column (raw):")
print(raw.isna().sum()[raw.isna().sum() > 0].to_string(), "\n")

# ------------------------------------------------------------------ clean
df = raw.copy()

# 1. duplicates
n_dupes = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# 2. standardise inconsistent text labels
df["gender"] = (
    df["gender"].str.strip().str.lower().map(
        {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}
    )
)
df["internet_access"] = (
    df["internet_access"].str.strip().str.lower().map(
        {"yes": "Yes", "y": "Yes", "no": "No", "n": "No"}
    )
)
df["parental_education"] = df["parental_education"].str.strip().str.title()

# 3. invalid values -> NaN (marks must be 0-100, attendance 0-100)
invalid_marks = int(((df[SUBJECTS] < 0) | (df[SUBJECTS] > 100)).sum().sum())
invalid_att = int(((df["attendance_pct"] < 0) | (df["attendance_pct"] > 100)).sum())
for c in SUBJECTS:
    df.loc[(df[c] < 0) | (df[c] > 100), c] = np.nan
df.loc[(df["attendance_pct"] < 0) | (df["attendance_pct"] > 100), "attendance_pct"] = np.nan

# 4. missing values: median for numeric, mode for categorical
n_missing_before = int(df.isna().sum().sum())
num_cols = ["study_hours_per_day", "attendance_pct", "sleep_hours"] + SUBJECTS
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
df["parental_education"] = df["parental_education"].fillna(df["parental_education"].mode()[0])

assert df.isna().sum().sum() == 0

# 5. feature engineering
df["avg_marks"] = df[SUBJECTS].mean(axis=1).round(2)
df["grade"] = pd.cut(
    df["avg_marks"], bins=[-1, 39.99, 49.99, 59.99, 74.99, 100],
    labels=["F", "D", "C", "B", "A"],
)
df["performance"] = np.where(df["avg_marks"] >= 60, "High", "Low")
df.to_csv("student_data_clean.csv", index=False)

n_clean = len(df)
print(f"Duplicates removed: {n_dupes}")
print(f"Invalid values corrected: {invalid_marks} marks, {invalid_att} attendance")
print(f"Missing values imputed: {n_missing_before}")
print(f"Clean records: {n_clean}\n")

# ------------------------------------------------------------------ analysis
corr = df[["study_hours_per_day", "attendance_pct", "sleep_hours", "avg_marks"]].corr()["avg_marks"].drop("avg_marks")
print("Correlation with average marks:\n", corr.round(3).to_string(), "\n")

grp = lambda col: df.groupby(col, observed=True)["avg_marks"].mean().round(2)
print("Avg marks by gender:\n", grp("gender").to_string(), "\n")
print("Avg marks by parental education:\n", grp("parental_education").to_string(), "\n")
print("Avg marks by extra classes:\n", grp("extra_classes").to_string(), "\n")
print("Avg marks by internet access:\n", grp("internet_access").to_string(), "\n")
print("Subject means:\n", df[SUBJECTS].mean().round(2).to_string(), "\n")

df["study_band"] = pd.cut(df["study_hours_per_day"], [-0.1, 2, 4, 6, 10], labels=["0-2h", "2-4h", "4-6h", "6h+"])
df["att_band"] = pd.cut(df["attendance_pct"], [0, 60, 75, 90, 100], labels=["<60%", "60-75%", "75-90%", "90%+"])
study_band = df.groupby("study_band", observed=True)["avg_marks"].mean().round(2)
att_band = df.groupby("att_band", observed=True)["avg_marks"].mean().round(2)
print("Avg marks by study band:\n", study_band.to_string(), "\n")
print("Avg marks by attendance band:\n", att_band.to_string(), "\n")

pass_rate = (df[SUBJECTS] >= 40).mean().mul(100).round(1)
print("Pass rate % per subject (>=40):\n", pass_rate.to_string(), "\n")
print("Grade distribution:\n", df["grade"].value_counts().sort_index().to_string(), "\n")

# ------------------------------------------------------------------ charts
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
BLUE, ORANGE, GREEN = "#2f6fb0", "#e08a2e", "#3a9b6b"

# 1 distribution of average marks
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df["avg_marks"], bins=25, color=BLUE, edgecolor="white")
ax.axvline(df["avg_marks"].mean(), color=ORANGE, ls="--", label=f"Mean = {df['avg_marks'].mean():.1f}")
ax.set(title="Distribution of Average Marks", xlabel="Average marks", ylabel="Students")
ax.legend()
fig.tight_layout(); fig.savefig("charts/01_marks_distribution.png"); plt.close(fig)

# 2 subject-wise average
fig, ax = plt.subplots(figsize=(6, 4))
means = df[SUBJECTS].mean()
bars = ax.bar([s.title() for s in SUBJECTS], means, color=[BLUE, GREEN, ORANGE])
ax.bar_label(bars, fmt="%.1f", padding=3)
ax.set(title="Average Marks by Subject", ylabel="Average marks", ylim=(0, 100))
fig.tight_layout(); fig.savefig("charts/02_subject_averages.png"); plt.close(fig)

# 3 study hours vs marks (scatter + trend)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(df["study_hours_per_day"], df["avg_marks"], s=12, alpha=0.4, color=BLUE)
m, b = np.polyfit(df["study_hours_per_day"], df["avg_marks"], 1)
xs = np.linspace(df["study_hours_per_day"].min(), df["study_hours_per_day"].max(), 50)
ax.plot(xs, m * xs + b, color=ORANGE, lw=2, label=f"Trend: +{m:.1f} marks per extra hour")
ax.set(title="Study Hours vs Average Marks", xlabel="Study hours per day", ylabel="Average marks")
ax.legend()
fig.tight_layout(); fig.savefig("charts/03_study_hours_vs_marks.png"); plt.close(fig)

# 4 attendance vs marks
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(df["attendance_pct"], df["avg_marks"], s=12, alpha=0.4, color=GREEN)
m2, b2 = np.polyfit(df["attendance_pct"], df["avg_marks"], 1)
xs = np.linspace(df["attendance_pct"].min(), df["attendance_pct"].max(), 50)
ax.plot(xs, m2 * xs + b2, color=ORANGE, lw=2, label=f"Trend: +{m2:.2f} marks per 1% attendance")
ax.set(title="Attendance vs Average Marks", xlabel="Attendance (%)", ylabel="Average marks")
ax.legend()
fig.tight_layout(); fig.savefig("charts/04_attendance_vs_marks.png"); plt.close(fig)

# 5 parental education
fig, ax = plt.subplots(figsize=(6.5, 4))
order = ["Primary", "High School", "Bachelor", "Master"]
vals = grp("parental_education").reindex(order)
bars = ax.bar(order, vals, color=BLUE)
ax.bar_label(bars, fmt="%.1f", padding=3)
ax.set(title="Average Marks by Parental Education", ylabel="Average marks", ylim=(0, 100))
fig.tight_layout(); fig.savefig("charts/05_parental_education.png"); plt.close(fig)

# 6 correlation heatmap
cols = ["study_hours_per_day", "attendance_pct", "sleep_hours"] + SUBJECTS + ["avg_marks"]
cm = df[cols].corr()
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
labels = [c.replace("_per_day", "").replace("_pct", "").replace("_", " ") for c in cols]
ax.set_xticklabels(labels, rotation=40, ha="right"); ax.set_yticklabels(labels)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center",
                color="white" if cm.iloc[i, j] > 0.6 else "black", fontsize=8)
ax.grid(False); ax.set_title("Correlation Matrix")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout(); fig.savefig("charts/06_correlation_heatmap.png"); plt.close(fig)

# 7 grade distribution
fig, ax = plt.subplots(figsize=(6, 4))
g = df["grade"].value_counts().reindex(["A", "B", "C", "D", "F"])
bars = ax.bar(g.index.astype(str), g.values, color=[GREEN, BLUE, BLUE, ORANGE, "#c0392b"])
ax.bar_label(bars, padding=3)
ax.set(title="Grade Distribution", xlabel="Grade", ylabel="Students")
fig.tight_layout(); fig.savefig("charts/07_grade_distribution.png"); plt.close(fig)

print("Saved 7 charts to charts/")