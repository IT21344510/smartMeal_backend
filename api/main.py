from __future__ import annotations

import logging
import os
import re
from datetime import date
from uuid import uuid4
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, conlist

import numpy as np
import pandas as pd
from api import alert_engine
from api.meal_service import FoodPrediction, FoodRecognitionEngine
from api.meal_optimizer import MealOptimizer

try:
    import joblib
except ImportError:  # pragma: no cover - optional dependency
    joblib = None
try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - optional dependency
    tf = None

logger = logging.getLogger("smartmeal")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class PredictRequest(BaseModel):
    Glucose: float = Field(..., description="Current glucose reading (mg/dL)")
    BloodPressure: float = Field(..., description="Systolic blood pressure (mmHg)")
    Insulin: float = Field(..., description="Current insulin level or last dose (units)")
    BMI: float = Field(..., description="Body Mass Index")
    Age: int = Field(..., ge=0, description="Age in years")
    eGFR: float = Field(..., description="Estimated glomerular filtration rate")
    Creatinine: float = Field(..., description="Serum creatinine")
    CarbsIntake: float = Field(..., description="Daily carbohydrate intake (grams)")
    glucose_trend_mg_dl: Optional[List[float]] = Field(
        None, description="Optional CGM trend values to refine timing adjustments."
    )


class KidneyAlert(BaseModel):
    alert_type: str
    severity: str
    reason: str
    recommended_action: str
    escalate_to: List[str]


class PredictResponse(BaseModel):
    predicted_insulin_dose: float
    adjusted_insulin_dose: float
    dose_adjustment_reasons: List[str]
    insulin_timing: str
    insulin_timing_reason: str
    predicted_kidney_health: str
    kidney_risk_explanations: List[str]
    glucose_trend: Optional[str]
    alerts: List[KidneyAlert]
    recommendations: List[str]


class GlucoseSignals(BaseModel):
    fasting_mg_dl: float
    post_meal_mg_dl: float
    daily_trend_mg_dl: conlist(float, min_items=1)


class HbA1cEntry(BaseModel):
    date: date
    value: float


class Lifestyle(BaseModel):
    carb_intake_g_per_day: float
    activity_minutes_per_day: float


AlertPermission = Literal["all", "emergency_only", "weekly_summary", "high_risk_only"]


class AlertContact(BaseModel):
    role: Literal["family", "doctor"]
    email: str = Field(..., description="Notification email for the contact.")
    name: Optional[str] = Field(None, description="Optional display name for the contact.")
    permission: AlertPermission = Field(
        "all",
        description="Notification mode: all, emergency_only, weekly_summary, or high_risk_only.",
    )


class RiskExplanation(BaseModel):
    factor: str
    impact: str


class Alert(BaseModel):
    alert_type: str
    risk_level: Optional[str] = None
    risk_score: Optional[int] = None
    reason: str
    explanation_summary: Optional[str] = None
    explanation: str
    recommended_action: str
    suggested_next_action: Optional[str] = None


class IntegratedRequest(BaseModel):
    glucose: GlucoseSignals
    hba1c_history: conlist(HbA1cEntry, min_items=1)
    lifestyle: Lifestyle
    medication_adherence_pct: float
    age_years: int
    bmi: float
    behavior_consistency_pct: float
    patient_id: Optional[str] = Field(None, description="Optional patient identifier for alert logging.")
    patient_name: Optional[str] = Field(None, description="Optional patient display name for alert emails.")
    notify_email: Optional[str] = Field(None, description="Optional email for alert notifications.")
    notify_phone: Optional[str] = Field(None, description="Optional phone for SMS alerts (if enabled).")
    alert_contacts: List[AlertContact] = Field(
        default_factory=list,
        description="Optional alert contacts (family/doctor) with notification permissions.",
    )
    raw_features_override: Optional[Dict[str, Any]] = Field(
        None, description="Optional direct feature overrides for the XAI joblib pipeline."
    )


class IntegratedResult(BaseModel):
    stage: str
    stage_confidence: float
    risk_score: int
    risk_level: str
    risk_explanations: List[RiskExplanation]
    forecast_3_months: float
    forecast_6_months: float
    trend: str
    alerts: List[Alert]
    future_treatment_recommendations: List[str] = Field(default_factory=list)
    checkup_schedule: Optional[str] = None


class IntegratedResponse(BaseModel):
    model: str = "integrated_patient_health"
    patient_id: Optional[str] = None
    result: IntegratedResult


class FeedAlert(BaseModel):
    patient_id: str
    alert_probability: float
    reasons: List[str]
    channel: str


class FeedProcessResponse(BaseModel):
    triggered: int
    alerts: List[FeedAlert]


class NutrientBreakdown(BaseModel):
    calories: float
    carbs_g: float
    protein_g: float
    fat_g: float
    fiber_g: float


class FoodAnalysis(BaseModel):
    detected_items: List[str]
    food_confidence: float
    portion_size: str
    calories: float
    carbs: str
    protein: str
    fat: str
    fiber: str
    carb_load_g: float


class RecentFoodInput(BaseModel):
    label: str
    confidence: float
    nutrients: NutrientBreakdown
    meal_health_score: Optional[str] = None


class MealPlanDay(BaseModel):
    day: str
    meals: List[str]


class MealFeedback(BaseModel):
    meal: str
    action: Literal["accept", "reject"]
    notes: Optional[str] = None


class MealOptimizationRequest(BaseModel):
    glucose_category: Literal["low", "normal", "high"] = Field(
        ..., description="Current glucose category to guide carb targets."
    )
    daily_calorie_target: int = Field(..., gt=0, description="Daily calorie goal for the weekly plan.")
    age_years: int = Field(..., ge=0)
    bmi: float
    gender: Optional[str] = None
    preferences: List[str] = Field(default_factory=list, description="Cuisine or macro preferences, e.g. ['low-gi'].")
    allergies: List[str] = Field(default_factory=list, description="Allergy keywords to exclude from meal options.")
    avoid_items: List[str] = Field(default_factory=list, description="Specific foods to avoid.")
    past_meals: List[str] = Field(default_factory=list, description="Recent meal names to improve rotation.")
    feedback: List[MealFeedback] = Field(
        default_factory=list, description="User feedback on meals (accept/reject)."
    )
    glucose_trend_mg_dl: Optional[conlist(float, min_items=1)] = Field(
        None, description="Optional glucose trend values to adapt carb targets."
    )
    recent_food: Optional[RecentFoodInput] = Field(
        None,
        description="Optional result from Model A to adapt planning (label, confidence, nutrients).",
    )


class MealOptimizationResponse(BaseModel):
    meal_recommendations: List[str]
    healthy_alternatives: List[str]
    weekly_meal_calendar: List[MealPlanDay]
    grocery_list: List[str]
    updated_preferences: List[str]
    model_adjustments: str


STAGE_MAP = {0: "Pre-diabetic", 1: "Managed", 2: "Uncontrolled"}
INSULIN_MAP = {0: "No", 1: "Steady", 2: "Up", 3: "Down", 4: "Ch"}
XAI_BASE_DIR_ENV = "SMARTMEAL_XAI_DIR"
DEFAULT_XAI_DIR = Path(__file__).resolve().parent.parent / "Explainable-AI-system"
INSULIN_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "insulin_management" / "artifacts"
DEFAULT_TF_INSULIN_MODEL = INSULIN_ARTIFACTS_DIR / "fnn_dual_model.keras"
DEFAULT_TF_SCALER = INSULIN_ARTIFACTS_DIR / "standard_scaler.pkl"

# Defaults pulled from median/mode values in cleaned_diabetes_dataset.csv so we can
# fill missing columns before handing rows to the preprocessing pipeline.
DEFAULT_FEATURE_ROW: Dict[str, Any] = {
    "race": "Caucasian",
    "gender": "Female",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "payer_code": "Unknown",
    "medical_specialty": "Unknown",
    "num_lab_procedures": 44,
    "num_procedures": 1,
    "num_medications": 15,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 8,
    "max_glu_serum": "Norm",
    "A1Cresult": ">8",
    "metformin": 0.0,
    "repaglinide": 0.0,
    "nateglinide": 0.0,
    "chlorpropamide": 0.0,
    "glimepiride": 0.0,
    "acetohexamide": 0.0,
    "glipizide": 0.0,
    "glyburide": 0.0,
    "tolbutamide": 0.0,
    "pioglitazone": 0.0,
    "rosiglitazone": 0.0,
    "acarbose": 0.0,
    "miglitol": 0.0,
    "troglitazone": 0.0,
    "tolazamide": 0.0,
    "examide": 0.0,
    "citoglipton": 0.0,
    "glyburide-metformin": 0.0,
    "glipizide-metformin": 0.0,
    "glimepiride-pioglitazone": 0.0,
    "metformin-rosiglitazone": 0.0,
    "metformin-pioglitazone": 0.0,
    "change": 0.0,
    "diabetesMed": "Yes",
    "age_mid": 65.0,
    "diag_1_group": "Circulatory",
    "diag_2_group": "Circulatory",
    "diag_3_group": "Circulatory",
    "med_intensity": 1.0,
    "chronic_count": 2,
    "acute_count": 0,
    "encounter_burden": 0,
    "comorbidity_index": 0.4999642882651239,
    "elderly": 0,
    "adult": 1,
    "med_diag_interaction": 108,
}


def _load_optional_model(env_var: str) -> Optional[object]:
    """Load a joblib model when SMARTMEAL_*_MODEL env vars are set."""
    if joblib is None:
        return None

    path_value = os.getenv(env_var)
    if not path_value:
        return None

    path = Path(path_value).expanduser()
    if not path.exists():
        logger.warning("Model path %s does not exist.", path)
        return None

    try:
        return joblib.load(path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Could not load model from %s (%s).", path, exc)
        return None


class ModelBundle:
    def __init__(self) -> None:
        self.insulin_model = _load_optional_model("SMARTMEAL_INSULIN_MODEL")
        self.kidney_model = _load_optional_model("SMARTMEAL_KIDNEY_MODEL")
        self.tf_model = None
        self.tf_scaler = None
        self._load_tf_fallback()

    def _load_tf_fallback(self) -> None:
        if self.insulin_model is not None and self.kidney_model is not None:
            return
        if tf is None or joblib is None:
            logger.warning("TensorFlow/joblib not available; TF insulin fallback disabled.")
            return
        if not DEFAULT_TF_INSULIN_MODEL.exists() or not DEFAULT_TF_SCALER.exists():
            logger.warning(
                "TF insulin artifacts not found at %s.", INSULIN_ARTIFACTS_DIR
            )
            return
        try:
            self.tf_model = tf.keras.models.load_model(str(DEFAULT_TF_INSULIN_MODEL), compile=False)
            self.tf_scaler = joblib.load(str(DEFAULT_TF_SCALER))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to load TF insulin/kidney artifacts (%s).", exc)

    @staticmethod
    def _kidney_label_from_pred(raw: Any) -> int:
        arr = np.asarray(raw)
        if arr.ndim == 0:
            return int(arr >= 0.5)
        if arr.size == 1:
            return int(arr.reshape(-1)[0] >= 0.5)
        return int(np.argmax(arr))

    def _predict_tf(self, features: List[float]) -> Tuple[float, str]:
        if self.tf_model is None or self.tf_scaler is None:
            raise HTTPException(status_code=503, detail="Insulin model not loaded.")
        scaled = self.tf_scaler.transform([features])
        pred_reg, pred_cls = self.tf_model.predict(scaled)
        insulin_dose = round(float(pred_reg[0][0]), 2)
        kidney_label = self._kidney_label_from_pred(pred_cls[0])
        kidney_health = "Risk" if kidney_label == 1 else "Good"
        return insulin_dose, kidney_health

    def predict_pair(self, features: List[float]) -> Tuple[float, str]:
        insulin_dose: Optional[float] = None
        kidney_health: Optional[str] = None

        if self.insulin_model is not None:
            insulin_dose = self._predict_joblib_insulin(features)
        if self.kidney_model is not None:
            kidney_health = self._predict_joblib_kidney(features)

        if (insulin_dose is None or kidney_health is None) and self.tf_model is not None:
            tf_insulin, tf_kidney = self._predict_tf(features)
            if insulin_dose is None:
                insulin_dose = tf_insulin
            if kidney_health is None:
                kidney_health = tf_kidney

        if insulin_dose is None:
            raise HTTPException(status_code=503, detail="Insulin model not loaded.")
        if kidney_health is None:
            raise HTTPException(status_code=503, detail="Kidney model not loaded.")
        return insulin_dose, kidney_health

    def _predict_joblib_insulin(self, features: List[float]) -> float:
        try:
            value = float(self.insulin_model.predict([features])[0])
            return round(value, 2)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Insulin model failed (%s).", exc)
            raise HTTPException(status_code=500, detail="Insulin model prediction failed.")

    def _predict_joblib_kidney(self, features: List[float]) -> str:
        try:
            raw = self.kidney_model.predict([features])[0]
            return "Good" if int(raw) == 0 else "Risk"
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Kidney model failed (%s).", exc)
            raise HTTPException(status_code=500, detail="Kidney model prediction failed.")

    def predict_insulin(self, features: List[float]) -> float:
        return self.predict_pair(features)[0]

    def predict_kidney(self, features: List[float]) -> str:
        return self.predict_pair(features)[1]


class ExplainableAIBundle:
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        base = Path(os.getenv(XAI_BASE_DIR_ENV, DEFAULT_XAI_DIR))
        self.base_dir = base if base.is_absolute() else (Path.cwd() / base)
        self.preprocess = None
        self.stage_model = None
        self.risk_model = None
        self.los_model = None
        self.alert_model = None
        self.insulin_model = None
        self.hba1c3_model = None
        self.hba1c6_model = None
        self.x_columns: List[str] = []
        self.available = False
        self._load_assets()

    def _load_assets(self) -> None:
        if joblib is None:
            logger.warning("joblib is not installed; explainable models disabled.")
            return

        try:
            self.preprocess = joblib.load(self.base_dir / "preprocess_pipeline.joblib")
            self.stage_model = joblib.load(self.base_dir / "model_stage.joblib")
            self.risk_model = joblib.load(self.base_dir / "model_risk.joblib")
            self.los_model = joblib.load(self.base_dir / "model_los.joblib")
            self.insulin_model = joblib.load(self.base_dir / "model_insulin.joblib")
            self.hba1c3_model = joblib.load(self.base_dir / "model_hba1c_3m.joblib")
            self.hba1c6_model = joblib.load(self.base_dir / "model_hba1c_6m.joblib")
            alert_model_path = self.base_dir / "model_alert.joblib"
            if alert_engine.USE_ALERT_MODEL and alert_model_path.exists():
                self.alert_model = joblib.load(alert_model_path)
            self.x_columns = list(getattr(self.preprocess, "feature_names_in_", []))
            self.available = bool(
                self.preprocess
                and self.stage_model
                and self.risk_model
                and self.los_model
                and self.hba1c3_model
                and self.hba1c6_model
                and self.x_columns
            )
        except Exception as exc:
            logger.error("Explainable AI joblib failed to load (%s).", exc)
            self.available = False

    @staticmethod
    def _risk_level_from_model(score: int) -> str:
        if score < 34:
            return "Low"
        if score < 67:
            return "Medium"
        return "High"

    @staticmethod
    def _impact_label(value: int) -> str:
        return f"+{value}" if value >= 0 else str(value)

    def _risk_reason_breakdown(self, req: IntegratedRequest, risk_score: int) -> List[RiskExplanation]:
        candidates: List[Tuple[str, int]] = []

        def add_reason(factor: str, impact: int) -> None:
            candidates.append((factor, impact))

        fasting = req.glucose.fasting_mg_dl
        post_meal = req.glucose.post_meal_mg_dl
        trend_values = req.glucose.daily_trend_mg_dl
        latest_hba1c = req.hba1c_history[-1].value
        carbs = req.lifestyle.carb_intake_g_per_day
        activity = req.lifestyle.activity_minutes_per_day
        med_adherence = req.medication_adherence_pct
        behavior_consistency = req.behavior_consistency_pct
        bmi = req.bmi
        age = req.age_years

        if fasting >= 130:
            add_reason("High fasting glucose (>=130 mg/dL)", 15)
        elif fasting >= 110:
            add_reason("Elevated fasting glucose (110-129 mg/dL)", 8)
        elif fasting <= 90:
            add_reason("Controlled fasting glucose (<=90 mg/dL)", -6)

        if post_meal >= 180:
            add_reason("High post-meal glucose (>=180 mg/dL)", 14)
        elif post_meal >= 140:
            add_reason("Elevated post-meal glucose (140-179 mg/dL)", 8)
        elif post_meal <= 130:
            add_reason("Controlled post-meal glucose (<=130 mg/dL)", -5)

        if trend_values:
            delta = trend_values[-1] - trend_values[0]
            spike_count = sum(value >= 180 for value in trend_values)
            if spike_count >= 3:
                add_reason("Frequent glucose spikes (>=180 mg/dL)", 12)
            elif delta >= 15:
                add_reason(f"Rising glucose trend (+{int(delta)} mg/dL)", 10)
            elif delta <= -15:
                add_reason(f"Falling glucose trend ({int(delta)} mg/dL)", -6)
            elif spike_count == 0 and abs(delta) < 5:
                add_reason("Stable glucose trend", -4)

        if latest_hba1c >= 8:
            add_reason("High HbA1c (>=8%)", 18)
        elif latest_hba1c >= 7:
            add_reason("Elevated HbA1c (7-7.9%)", 10)
        elif latest_hba1c <= 6.5:
            add_reason("Controlled HbA1c (<=6.5%)", -8)

        if carbs >= 220:
            add_reason("High carb intake (>=220 g/day)", 10)
        elif carbs >= 180:
            add_reason("Elevated carb intake (180-219 g/day)", 6)
        elif carbs <= 150:
            add_reason("Lower carb intake (<=150 g/day)", -4)

        if activity < 20:
            add_reason("Low activity (<20 min/day)", 12)
        elif activity < 30:
            add_reason("Low activity (20-29 min/day)", 8)
        elif activity >= 45:
            add_reason("Active lifestyle (>=45 min/day)", -6)

        if med_adherence < 60:
            add_reason("Low medication adherence (<60%)", 15)
        elif med_adherence < 80:
            add_reason("Suboptimal medication adherence (60-79%)", 10)
        elif med_adherence >= 90:
            add_reason("Strong medication adherence (>=90%)", -6)

        if behavior_consistency < 50:
            add_reason("Low behavior consistency (<50%)", 10)
        elif behavior_consistency < 70:
            add_reason("Inconsistent logging/plan adherence (50-69%)", 6)
        elif behavior_consistency >= 85:
            add_reason("Consistent logging/plan adherence (>=85%)", -4)

        if bmi >= 35:
            add_reason("High BMI (>=35)", 10)
        elif bmi >= 30:
            add_reason("Elevated BMI (30-34.9)", 6)
        elif bmi <= 25:
            add_reason("Healthy BMI (<=25)", -4)

        if age >= 65:
            add_reason("Older age (>=65)", 5)

        if not candidates:
            candidates.extend(
                [
                    ("Model probability", risk_score),
                    ("Stable glucose trend", -2),
                ]
            )

        if risk_score >= 67:
            primary = [item for item in candidates if item[1] > 0]
            secondary = [item for item in candidates if item[1] <= 0]
        elif risk_score <= 33:
            primary = [item for item in candidates if item[1] < 0]
            secondary = [item for item in candidates if item[1] >= 0]
        else:
            primary = [item for item in candidates if item[1] > 0]
            secondary = [item for item in candidates if item[1] <= 0]

        primary = sorted(primary, key=lambda item: abs(item[1]), reverse=True)
        secondary = sorted(secondary, key=lambda item: abs(item[1]), reverse=True)

        selected: List[Tuple[str, int]] = []
        for item in primary:
            if item not in selected:
                selected.append(item)
            if len(selected) >= 3:
                break
        if len(selected) < 2:
            for item in secondary:
                if item not in selected:
                    selected.append(item)
                if len(selected) >= 2:
                    break
        if len(selected) < 3:
            for item in secondary:
                if item not in selected:
                    selected.append(item)
                if len(selected) >= 3:
                    break
        if len(selected) < 2:
            fallback = ("Model probability", risk_score)
            if fallback not in selected:
                selected.append(fallback)
        if len(selected) < 2:
            selected.append(("Stable glucose trend", -2))

        return [
            RiskExplanation(factor=factor, impact=self._impact_label(impact))
            for factor, impact in selected
        ]

    def _build_feature_frame(self, req: IntegratedRequest) -> pd.DataFrame:
        # Start with dataset medians/modes so the preprocessor receives complete columns.
        row: Dict[str, Any] = DEFAULT_FEATURE_ROW.copy()
        latest_hba1c = req.hba1c_history[-1].value
        trend_delta = req.glucose.daily_trend_mg_dl[-1] - req.glucose.daily_trend_mg_dl[0]
        max_glucose = max(
            req.glucose.daily_trend_mg_dl + [req.glucose.fasting_mg_dl, req.glucose.post_meal_mg_dl]
        )

        chronic_count = max(1, int(req.bmi >= 30) + int(latest_hba1c >= 7.0) + 1)
        acute_count = int(abs(trend_delta) >= 10)
        encounter_burden = max(row["encounter_burden"], min(10, len(req.glucose.daily_trend_mg_dl)))
        comorbidity_index = round((req.bmi / 35) + (latest_hba1c / 9), 3)
        num_lab_procs = max(row["num_lab_procedures"], len(req.glucose.daily_trend_mg_dl) * 3)
        num_meds = max(1, int(round(req.medication_adherence_pct / 10)))
        med_intensity = round(max(0.0, min(5.0, req.medication_adherence_pct / 20)), 2)
        number_diagnoses = max(row["number_diagnoses"], chronic_count + acute_count)

        row.update(
            {
                "age_mid": float(req.age_years),
                "elderly": int(req.age_years >= 65),
                "adult": int(18 <= req.age_years < 65),
                "num_lab_procedures": num_lab_procs,
                "num_medications": num_meds,
                "med_intensity": med_intensity,
                "comorbidity_index": comorbidity_index,
                "chronic_count": chronic_count,
                "acute_count": acute_count,
                "encounter_burden": encounter_burden,
                "number_diagnoses": number_diagnoses,
                "max_glu_serum": ">200" if max_glucose >= 200 else "Norm",
                "A1Cresult": ">8" if latest_hba1c >= 8 else ">7" if latest_hba1c >= 7 else "Norm",
                "diabetesMed": "Yes" if req.medication_adherence_pct > 0 else "No",
                "diag_1_group": "Diabetes",
                "diag_2_group": "Diabetes" if trend_delta >= 5 else row["diag_2_group"],
                "diag_3_group": "Circulatory" if req.bmi >= 32 else row["diag_3_group"],
            }
        )

        if req.raw_features_override:
            row.update(req.raw_features_override)

        normalized = {col: row.get(col) for col in self.x_columns} if self.x_columns else row
        return pd.DataFrame([normalized], columns=self.x_columns or None)

    def predict(
        self, req: IntegratedRequest, return_meta: bool = False
    ) -> Union[IntegratedResult, Tuple[IntegratedResult, Dict[str, Any]]]:
        if not self.available:
            raise HTTPException(status_code=503, detail="Explainable AI models not loaded.")

        try:
            x_df = self._build_feature_frame(req)
            x_proc = self.preprocess.transform(x_df)

            stage_probs = self.stage_model.predict_proba(x_proc)[0]
            stage_idx = int(np.argmax(stage_probs))
            stage_label = STAGE_MAP.get(stage_idx, f"Stage-{stage_idx}")
            stage_conf = float(stage_probs[stage_idx])

            risk_probs = self.risk_model.predict_proba(x_proc)[0]
            risk_score = int(round(float(risk_probs[1]) * 100))
            risk_level = self._risk_level_from_model(risk_score)

            los_pred = int(self.los_model.predict(x_proc)[0])
            trend = "Stable" if los_pred == 0 else "Slightly rising" if los_pred == 1 else "Rising"

            hba1c_3 = float(self.hba1c3_model.predict(x_proc)[0])
            hba1c_6 = float(self.hba1c6_model.predict(x_proc)[0])

            insulin_label: Optional[str] = None
            if self.insulin_model is not None:
                try:
                    insulin_code = int(self.insulin_model.predict(x_proc)[0])
                    insulin_label = INSULIN_MAP.get(insulin_code, "Unknown")
                except Exception:
                    insulin_label = None

            result = IntegratedResult(
                stage=stage_label,
                stage_confidence=round(stage_conf, 2),
                risk_score=risk_score,
                risk_level=risk_level,
                risk_explanations=self._risk_reason_breakdown(req, risk_score),
                forecast_3_months=round(hba1c_3, 2),
                forecast_6_months=round(hba1c_6, 2),
                trend=trend,
                alerts=[],
            )
            if return_meta:
                meta = {
                    "risk_probability": float(risk_probs[1]),
                    "stage_idx": stage_idx,
                    "los_bucket": los_pred,
                }
                return result, meta
            return result
        except Exception as exc:
            logger.error("XAI joblib prediction failed (%s).", exc)
            raise HTTPException(status_code=500, detail="Explainable AI prediction failed.")


model_bundle = ModelBundle()
xai_bundle = ExplainableAIBundle()
alert_service: Optional[alert_engine.AlertEngine] = None
food_engine = FoodRecognitionEngine()
meal_optimizer = MealOptimizer()


def _init_alert_engine() -> Optional[alert_engine.AlertEngine]:
    if not xai_bundle.available:
        return None

    try:
        return alert_engine.AlertEngine(
            base_dir=xai_bundle.base_dir,
            preprocess=xai_bundle.preprocess,
            risk_model=xai_bundle.risk_model,
            stage_model=xai_bundle.stage_model,
            los_model=xai_bundle.los_model,
            alert_model=xai_bundle.alert_model,
            log_path=alert_engine.ALERT_LOG_CSV,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Alert engine failed to initialize (%s).", exc)
        return None


def _get_alert_engine() -> Optional[alert_engine.AlertEngine]:
    global alert_service
    if alert_service is None:
        alert_service = _init_alert_engine()
    return alert_service


app = FastAPI(
    title="SmartMeal Backend",
    description="FastAPI service for insulin dose, kidney risk, and integrated patient health predictions.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/model-a/analyze",
    response_model=FoodAnalysis,
    tags=["Food Model A"],
    summary="Detect food items and estimate nutrients (Model A)",
    response_description="Detected food items, confidence, portion, and nutrient estimates.",
)
async def analyze_food(file: UploadFile = File(...)) -> FoodAnalysis:
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="File must be an image (jpg or png).")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    prediction = food_engine.predict(data)
    return _analysis_from_prediction(prediction)


@app.post(
    "/model-b/optimize",
    response_model=MealOptimizationResponse,
    tags=["Meal Model B"],
    summary="Generate personalized meal plan and grocery list (Model B)",
    response_description="Meal recommendations, alternatives, weekly plan, and grocery list.",
)
def optimize_meals(body: MealOptimizationRequest = Body(...)) -> MealOptimizationResponse:
    result = meal_optimizer.optimize(body)
    return MealOptimizationResponse(**result)


@app.post(
    "/model-b/optimize-from-image",
    response_model=MealOptimizationResponse,
    tags=["Meal Model B"],
    summary="Generate meal plan using Model A image input + Model B optimizer",
    response_description="Meal recommendations, alternatives, weekly plan, and grocery list.",
)
async def optimize_meals_from_image(
    payload: str = Form(..., description="JSON payload matching MealOptimizationRequest."),
    file: UploadFile = File(...),
) -> MealOptimizationResponse:
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="File must be an image (jpg or png).")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        body = MealOptimizationRequest.parse_raw(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload for meal optimization.") from exc

    prediction = food_engine.predict(data)
    body = body.copy(update={"recent_food": _recent_food_from_prediction(prediction)})
    result = meal_optimizer.optimize(body)
    return MealOptimizationResponse(**result)


def _to_feature_vector(body: PredictRequest) -> List[float]:
    return [
        body.Glucose,
        body.BloodPressure,
        body.Insulin,
        body.BMI,
        float(body.Age),
        body.eGFR,
        body.Creatinine,
        body.CarbsIntake,
    ]


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


def _kidney_alerts(
    egfr: float, creatinine: float, kidney_status: str
) -> List[KidneyAlert]:
    alerts: List[KidneyAlert] = []
    if kidney_status != "Good" or egfr < 60 or creatinine > 1.3:
        severity = "critical" if egfr < 45 or creatinine > 2.0 else "warning"
        alerts.append(
            KidneyAlert(
                alert_type="KidneyRisk",
                severity=severity,
                reason="eGFR below 60 or creatinine elevated",
                recommended_action="Review renal labs and monitor eGFR/creatinine.",
                escalate_to=["Clinician"],
            )
        )
    return alerts


def _kidney_risk_reason_breakdown(
    egfr: float,
    creatinine: float,
    kidney_status: str,
    glucose_trend_mg_dl: Optional[List[float]],
) -> List[str]:
    reasons: List[str] = []
    spike_count = 0
    if glucose_trend_mg_dl:
        spike_count = sum(value >= 180 for value in glucose_trend_mg_dl)

    if kidney_status != "Good":
        if creatinine > 2.0:
            reasons.append(f"Very high creatinine ({creatinine:.1f} mg/dL)")
        elif creatinine > 1.3:
            reasons.append(f"High creatinine ({creatinine:.1f} mg/dL)")
        if egfr < 45:
            reasons.append(f"Low eGFR ({egfr:.0f})")
        elif egfr < 60:
            reasons.append(f"Reduced eGFR ({egfr:.0f})")
        if spike_count >= 2:
            reasons.append(f"Frequent glucose spikes ({spike_count} days >=180 mg/dL)")
        elif spike_count == 1:
            reasons.append("Recent glucose spike (>=180 mg/dL)")
        if len(reasons) < 2:
            reasons.append("Kidney model flagged elevated risk")
    else:
        reasons.append(f"Creatinine within range ({creatinine:.1f} mg/dL)")
        reasons.append(f"eGFR within range ({egfr:.0f})")
        if glucose_trend_mg_dl and spike_count == 0:
            reasons.append("No recent glucose spikes")
        if len(reasons) < 2:
            reasons.append("Kidney indicators within expected range")

    return reasons[:3]


def _recommendations(trend: Optional[str], kidney_status: str) -> List[str]:
    recs: List[str] = []
    if trend == "rising":
        recs.append("Tighten carb targets for rising glucose.")
    if trend == "falling":
        recs.append("Avoid aggressive correction during falling glucose.")
    if kidney_status != "Good":
        recs.append("Review renal biomarkers and follow up with clinician.")
    return recs


def _collect_alert_recipients(body: IntegratedRequest) -> List[Dict[str, str]]:
    recipients: List[Dict[str, str]] = []
    for contact in body.alert_contacts:
        email = contact.email.strip() if isinstance(contact.email, str) else ""
        if not email:
            continue
        recipients.append({"email": email, "permission": contact.permission})
    if body.notify_email and isinstance(body.notify_email, str):
        email = body.notify_email.strip()
        if email:
            recipients.append({"email": email, "permission": "all"})

    seen = set()
    unique: List[Dict[str, str]] = []
    for entry in recipients:
        key = entry["email"].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def _ensure_patient_id(value: Optional[str]) -> str:
    if value and str(value).strip():
        return str(value).strip()
    stamp = date.today().strftime("%Y%m%d")
    token = uuid4().hex[:6]
    return f"patient-{stamp}-{token}"


def _future_care_plan(
    *,
    stage: str,
    risk_level: str,
    trend: str,
    medication_adherence_pct: float,
    behavior_consistency_pct: float,
) -> Tuple[List[str], str]:
    treatments: List[str] = []
    stage_label = stage.lower()
    risk_label = risk_level.lower()
    trend_label = trend.lower() if trend else ""

    high_risk = risk_label == "high" or "uncontrolled" in stage_label
    rising = "rising" in trend_label

    if high_risk or rising:
        checkup = "Clinician follow-up within 1-2 weeks; repeat labs in 4-6 weeks."
        treatments.extend(
            [
                "Review medication regimen with clinician; consider therapy intensification.",
                "Prioritize nutrition coaching and daily glucose checks.",
            ]
        )
    elif risk_label == "medium" or "managed" in stage_label or "slightly rising" in trend_label:
        checkup = "Follow up in 4-6 weeks; HbA1c check in ~3 months."
        treatments.extend(
            [
                "Adjust meal plan and activity targets; re-evaluate dose timing.",
                "Schedule a mid-cycle check-in to review trends.",
            ]
        )
    else:
        checkup = "Routine checkup in 3-6 months; HbA1c check in ~6 months."
        treatments.append("Maintain current plan and continue monitoring trends.")

    if medication_adherence_pct < 80:
        treatments.append("Review adherence barriers; consider reminders or simplification.")
    if behavior_consistency_pct < 70:
        treatments.append("Add coaching or logging support to improve consistency.")

    return treatments, checkup


def _analysis_from_prediction(pred: FoodPrediction) -> FoodAnalysis:
    return FoodAnalysis(
        detected_items=pred.detected_items,
        food_confidence=pred.food_confidence,
        portion_size=pred.portion_size,
        calories=pred.calories,
        carbs=pred.carbs,
        protein=pred.protein,
        fat=pred.fat,
        fiber=pred.fiber,
        carb_load_g=pred.carb_load_g,
    )


def _parse_grams(value: Optional[str]) -> float:
    if not value:
        return 0.0
    match = re.search(r"[-+]?\d*\.?\d+", str(value))
    return float(match.group(0)) if match else 0.0


def _recent_food_from_prediction(pred: FoodPrediction) -> RecentFoodInput:
    label = pred.detected_items[0] if pred.detected_items else "Unknown"
    return RecentFoodInput(
        label=label,
        confidence=pred.food_confidence,
        nutrients=NutrientBreakdown(
            calories=pred.calories,
            carbs_g=pred.carb_load_g,
            protein_g=_parse_grams(pred.protein),
            fat_g=_parse_grams(pred.fat),
            fiber_g=_parse_grams(pred.fiber),
        ),
    )


@app.post(
    "/predict/risk",
    response_model=PredictResponse,
    tags=["Insulin & Kidney"],
    summary="Predict insulin dose and kidney health",
    response_description="Predicted insulin dose and kidney health label.",
)
def predict(
    body: PredictRequest = Body(...)
) -> PredictResponse:
    features = _to_feature_vector(body)
    trend = _trend_direction(body.glucose_trend_mg_dl)
    insulin_dose, kidney_health = model_bundle.predict_pair(features)
    adjusted_dose, dose_adjustment_reasons = _adjust_insulin_dose(
        insulin_dose,
        body.Glucose,
        body.eGFR,
        body.Creatinine,
        trend,
    )
    insulin_timing, insulin_timing_reason = _insulin_timing(
        body.Glucose, body.CarbsIntake, trend
    )
    alerts = _kidney_alerts(body.eGFR, body.Creatinine, kidney_health)
    kidney_risk_explanations = _kidney_risk_reason_breakdown(
        body.eGFR,
        body.Creatinine,
        kidney_health,
        body.glucose_trend_mg_dl,
    )
    recommendations = _recommendations(trend, kidney_health)
    return PredictResponse(
        predicted_insulin_dose=insulin_dose,
        adjusted_insulin_dose=adjusted_dose,
        dose_adjustment_reasons=dose_adjustment_reasons,
        insulin_timing=insulin_timing,
        insulin_timing_reason=insulin_timing_reason,
        predicted_kidney_health=kidney_health,
        kidney_risk_explanations=kidney_risk_explanations,
        glucose_trend=trend,
        alerts=alerts,
        recommendations=recommendations,
    )


@app.post(
    "/integrated/predict",
    response_model=IntegratedResponse,
    tags=["Integrated Model"],
    summary="Predict stage, risk, forecasts, and alerts (Model 1)",
    response_description="Integrated patient health outputs.",
)
def integrated_predict(
    body: IntegratedRequest = Body(...)
) -> IntegratedResponse:
    patient_id = _ensure_patient_id(body.patient_id)
    joblib_result, meta = xai_bundle.predict(body, return_meta=True)

    alerts_payloads: List[Alert] = []
    service = _get_alert_engine()
    if service:
        recipients = _collect_alert_recipients(body)
        alert_dicts = service.alerts_for_prediction(
            patient_id=patient_id,
            patient_name=body.patient_name,
            risk_probability=meta["risk_probability"],
            stage_idx=meta["stage_idx"],
            stage_label=joblib_result.stage,
            los_bucket=meta["los_bucket"],
            risk_level=joblib_result.risk_level,
            risk_score=joblib_result.risk_score,
            email=None,
            phone=body.notify_phone,
            recipients=recipients,
        )
        alerts_payloads = [Alert(**item) for item in alert_dicts]

    future_treatments, checkup_schedule = _future_care_plan(
        stage=joblib_result.stage,
        risk_level=joblib_result.risk_level,
        trend=joblib_result.trend,
        medication_adherence_pct=body.medication_adherence_pct,
        behavior_consistency_pct=body.behavior_consistency_pct,
    )

    joblib_result = joblib_result.copy(
        update={
            "alerts": alerts_payloads,
            "future_treatment_recommendations": future_treatments,
            "checkup_schedule": checkup_schedule,
        }
    )
    return IntegratedResponse(patient_id=patient_id, result=joblib_result)


@app.post(
    "/alerts/process-feed",
    response_model=FeedProcessResponse,
    tags=["Alerts"],
    summary="Run the automatic alert engine on incoming_patients.csv once",
    response_description="Alerts triggered from the incoming CSV feed.",
)
def process_alert_feed() -> FeedProcessResponse:
    service = _get_alert_engine()
    if service is None:
        raise HTTPException(status_code=503, detail="Alert engine not available (models not loaded).")

    alerts = service.process_feed_once(xai_bundle.x_columns or list(DEFAULT_FEATURE_ROW.keys()), DEFAULT_FEATURE_ROW)
    feed_alerts = [FeedAlert(**item) for item in alerts]
    return FeedProcessResponse(triggered=len(feed_alerts), alerts=feed_alerts)
