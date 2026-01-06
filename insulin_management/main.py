from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import joblib
import tensorflow as tf

# Initialize app
app = FastAPI(title="AI-Powered Insulin & Kidney Health Predictor")

# Load model and scaler from artifacts
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "fnn_dual_model.keras"
SCALER_PATH = BASE_DIR / "artifacts" / "standard_scaler.pkl"
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
scaler = joblib.load(str(SCALER_PATH))

# Input schema
class PatientData(BaseModel):
    Glucose: float
    BloodPressure: float
    Insulin: float
    BMI: float
    Age: float
    eGFR: float
    Creatinine: float
    CarbsIntake: float
    glucose_trend_mg_dl: Optional[List[float]] = None

@app.get("/")
def home():
    return {"message": "Welcome to the Insulin & Kidney Health Prediction API"}

def _trend_direction(values: Optional[List[float]]) -> Optional[str]:
    if not values or len(values) < 2:
        return None
    delta = values[-1] - values[0]
    if delta > 5:
        return "rising"
    if delta < -5:
        return "falling"
    return "stable"

def _insulin_timing(glucose: float, carbs: float, trend: Optional[str]) -> Tuple[str, str]:
    if glucose < 70:
        return "hold_and_monitor", "glucose below 70 mg/dL"
    if glucose < 90:
        return "delay_until_meal", "glucose below 90 mg/dL"
    if glucose >= 180:
        return "pre_meal_15_min", "glucose above 180 mg/dL"
    if trend == "rising" and carbs >= 150:
        return "pre_meal_10_min", "rising trend with high carbs"
    if carbs >= 200:
        return "pre_meal_10_min", "high carb intake"
    return "with_meal", "standard timing"

def _adjust_insulin_dose(
    predicted: float,
    glucose: float,
    egfr: float,
    creatinine: float,
    trend: Optional[str],
) -> Tuple[float, List[str]]:
    factor = 1.0
    reasons: List[str] = []
    if glucose >= 180 or trend == "rising":
        factor += 0.1
        reasons.append("rising glucose")
    if glucose <= 80 or trend == "falling":
        factor -= 0.1
        reasons.append("falling glucose")
    if egfr < 60 or creatinine > 1.3:
        factor -= 0.1
        reasons.append("reduced renal function")
    factor = max(0.7, min(1.3, factor))
    adjusted = round(max(0.0, predicted * factor), 2)
    return adjusted, reasons

def _kidney_alerts(egfr: float, creatinine: float, kidney_status: str) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    if kidney_status == "At Risk" or egfr < 60 or creatinine > 1.3:
        severity = "critical" if egfr < 45 or creatinine > 2.0 else "warning"
        alerts.append(
            {
                "alert_type": "KidneyRisk",
                "severity": severity,
                "reason": "eGFR below 60 or creatinine elevated",
                "recommended_action": "Review renal labs and monitor eGFR/creatinine.",
                "escalate_to": ["Clinician"],
            }
        )
    return alerts

def _recommendations(trend: Optional[str], kidney_status: str) -> List[str]:
    recs: List[str] = []
    if trend == "rising":
        recs.append("Tighten carb targets for rising glucose.")
    if trend == "falling":
        recs.append("Avoid aggressive correction during falling glucose.")
    if kidney_status == "At Risk":
        recs.append("Review renal biomarkers and follow up with clinician.")
    return recs

@app.post("/predict")
def predict(data: PatientData):
    trend = _trend_direction(data.glucose_trend_mg_dl)

    # Convert input to numpy array
    input_data = np.array([[data.Glucose, data.BloodPressure, data.Insulin,
                            data.BMI, data.Age, data.eGFR,
                            data.Creatinine, data.CarbsIntake]])

    # Scale input
    scaled_input = scaler.transform(input_data)

    # Predict
    pred_reg, pred_cls = model.predict(scaled_input)
    predicted_insulin_dose = float(pred_reg[0][0])

    # Get model prediction
    kidney_health_label = int(np.argmax(pred_cls[0]))
    kidney_risk = kidney_health_label == 1 or data.eGFR < 60 or data.Creatinine > 1.3
    kidney_status = "At Risk" if kidney_risk else "Normal"

    adjusted_dose, dose_adjustment_reasons = _adjust_insulin_dose(
        predicted_insulin_dose,
        data.Glucose,
        data.eGFR,
        data.Creatinine,
        trend,
    )
    insulin_timing, insulin_timing_reason = _insulin_timing(
        data.Glucose, data.CarbsIntake, trend
    )
    alerts = _kidney_alerts(data.eGFR, data.Creatinine, kidney_status)
    recommendations = _recommendations(trend, kidney_status)

    return {
        "predicted_insulin_dose": round(predicted_insulin_dose, 2),
        "adjusted_insulin_dose": adjusted_dose,
        "dose_adjustment_reasons": dose_adjustment_reasons,
        "insulin_timing": insulin_timing,
        "insulin_timing_reason": insulin_timing_reason,
        "predicted_kidney_health": kidney_status,
        "glucose_trend": trend,
        "alerts": alerts,
        "recommendations": recommendations,
    }
