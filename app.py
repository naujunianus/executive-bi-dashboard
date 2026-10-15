"""Executive BI Dashboard & Predictive Analytics — Flask backend."""
from pathlib import Path
import os
import webbrowser
from threading import Timer

from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
APP = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
REQUIRED = ["Student_ID", "Gender", "Age", "Study_Hours", "Sleep_Hours", "Attendance",
            "Exam_Score", "Monthly_Spending", "Screen_Hours", "Satisfaction"]
NUMERIC = ["Age", "Study_Hours", "Sleep_Hours", "Attendance", "Exam_Score",
           "Monthly_Spending", "Screen_Hours", "Satisfaction"]


def make_dummy_data(n=100, seed=42):
    """Create a reproducible, realistic sample when no CSV is available."""
    rng = np.random.default_rng(seed)
    study = rng.uniform(2, 18, n)
    attendance = rng.uniform(55, 100, n)
    sleep = rng.uniform(4, 9, n)
    screen = rng.uniform(1, 12, n)
    score = np.clip(32 + 2.3 * study + 0.34 * attendance + 1.2 * sleep - 0.72 * screen
                    + rng.normal(0, 5, n), 30, 100)
    return pd.DataFrame({
        "Student_ID": np.arange(1, n + 1), "Gender": rng.choice(["F", "M"], n),
        "Age": rng.integers(17, 25, n), "Study_Hours": study.round(1),
        "Sleep_Hours": sleep.round(1), "Attendance": attendance.round(1),
        "Exam_Score": score.round(1), "Monthly_Spending": rng.integers(400, 2201, n),
        "Screen_Hours": screen.round(1), "Satisfaction": rng.integers(3, 11, n)})


def load_data():
    """Load app-local latihan1.csv or the supplied Downloads CSV; else generate data."""
    candidates = [BASE_DIR / "latihan1.csv", Path.home() / "Downloads" / "latihan1 (1).csv"]
    for path in candidates:
        if path.is_file():
            try:
                frame = pd.read_csv(path)
                if set(REQUIRED).issubset(frame.columns):
                    frame = frame[REQUIRED].copy()
                    frame["Gender"] = frame["Gender"].astype(str).str.upper().str.strip()
                    for col in NUMERIC:
                        frame[col] = pd.to_numeric(frame[col], errors="coerce")
                    frame = frame.dropna(subset=NUMERIC + ["Gender"])
                    if not frame.empty:
                        return frame
            except (OSError, ValueError, pd.errors.ParserError):
                pass
    return make_dummy_data()


def regression(frame):
    """OLS fitted with a pseudoinverse; return coefficients and in-sample R²."""
    features = ["Study_Hours", "Attendance", "Sleep_Hours", "Screen_Hours"]
    x = frame[features].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    y = frame["Exam_Score"].to_numpy(dtype=float)
    beta = np.linalg.pinv(x) @ y
    predicted = x @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(np.sum((y - predicted) ** 2)) / ss_tot if ss_tot else 0.0
    return {"features": features, "intercept": float(beta[0]),
            "coefficients": {name: float(value) for name, value in zip(features, beta[1:])},
            "r2": float(r2), "accuracy_pct": float(max(0, min(100, r2 * 100)))}


def summarize(frame):
    num = frame[NUMERIC]
    weights = regression(frame)
    stats = {col: {key: (None if pd.isna(value) else float(value))
                   for key, value in num[col].describe().to_dict().items()}
             for col in NUMERIC}
    means = frame.groupby("Gender")[NUMERIC].mean(numeric_only=True)
    radar_axes = ["Exam_Score", "Study_Hours", "Sleep_Hours", "Attendance", "Satisfaction"]
    # Normalize each metric to 0–100 against its observed range for a comparable radar.
    low, high = num[radar_axes].min(), num[radar_axes].max()
    radar = {}
    for gender in ["F", "M"]:
        values = means.loc[gender] if gender in means.index else num[radar_axes].mean()
        radar[gender] = {col: (float((values[col] - low[col]) / (high[col] - low[col]) * 100)
                                if high[col] != low[col] else 50.0) for col in radar_axes}
    corr = num.corr().replace([np.inf, -np.inf], np.nan).fillna(0)
    return stats, corr, weights, radar


@APP.get("/")
def index():
    return render_template("index.html")


@APP.get("/api/data")
def api_data():
    gender = request.args.get("gender", "Semua").upper()
    frame = load_data()
    if gender in {"F", "M"}:
        frame = frame[frame["Gender"] == gender]
    if frame.empty:
        return jsonify({"error": "Tidak ada data untuk filter gender ini."}), 404
    stats, corr, weights, radar = summarize(frame)
    raw = frame.replace({np.nan: None}).to_dict(orient="records")
    return jsonify({"raw_data": raw, "stats": stats, "corr_matrix": corr.to_dict(),
                    "numeric_cols": NUMERIC, "weights": weights, "radar_data": radar,
                    "at_risk_count": int((frame["Exam_Score"] < 60).sum()),
                    "total_students": int(len(frame)),
                    "gender_analysis": frame.groupby("Gender")["Exam_Score"].mean().to_dict()})


if __name__ == "__main__":
    # On hosting platforms PORT is supplied by the platform. Locally use 5050.
    port = int(os.environ.get("PORT", "5050"))
    if os.environ.get("RENDER") != "true":
        Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    APP.run(debug=False, host="0.0.0.0", port=port)
