# SmartMeal Backend

FastAPI service that hosts the insulin dose and kidney-risk predictors for the SmartMeal Flutter app, plus a single explainable stage/risk/forecast/alert system derived from `Explainable-AI-system/diabetic.ipynb`.

## Quick start
1) `python -m venv .venv` then activate it.
2) `pip install -r requirements.txt`
3) Run the API: `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`
   - Place the joblib files from `Explainable-AI-system/` (not `model2/`) in that directory or point `SMARTMEAL_XAI_DIR` to their location.
   - If you need separate insulin/kidney models, set `SMARTMEAL_INSULIN_MODEL` and `SMARTMEAL_KIDNEY_MODEL` to joblib paths; otherwise the endpoints will return 503.

Models now load solely from the single multi-task pipeline in `Explainable-AI-system/diabetic.ipynb` (stage + risk + HbA1c forecast + alerts). The old Model 2 alert module is deprecated; keep the base joblibs in `Explainable-AI-system/` for loading and remove any reliance on `Explainable-AI-system/model2/`. There is no rule-based fallback—requests are served directly from the joblib models.

## API contract (current insulin/kidney endpoint)
- **POST** `/predict`
- **Body (JSON)**
  ```json
  {
    "Glucose": 120,
    "BloodPressure": 80,
    "Insulin": 15,
    "BMI": 27.5,
    "Age": 45,
    "eGFR": 95,
    "Creatinine": 0.9,
    "CarbsIntake": 180
  }
  ```
- **Response**
  ```json
  {
    "predicted_insulin_dose": 12.3,
    "predicted_kidney_health": "Good",
    "kidney_risk_explanations": [
      "Creatinine within range (0.9 mg/dL)",
      "eGFR within range (95)",
      "No recent glucose spikes"
    ]
  }
  ```
- **Health check**: `GET /health` returns `{"status":"ok"}`.

### Flutter wiring snippet
```dart
final uri = Uri.parse('http://<api-host>:8000/predict');
final response = await http.post(
  uri,
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({
    'Glucose': glucose,
    'BloodPressure': bloodPressure,
    'Insulin': insulin,
    'BMI': bmi,
    'Age': age,
    'eGFR': egfr,
    'Creatinine': creatinine,
    'CarbsIntake': carbsIntake,
  }),
);
```

## New endpoints: Food model (A) + meal optimizer (B)
- **POST** `/model-a/analyze` (`multipart/form-data` with `file` = JPG/PNG). Runs the bundled CNN at `ai-powered-personalized-meal-planning-dietary-optimization/food_types/model/keras_model.h5` and returns label, confidence, portion size, nutrient estimate, and health score.
  ```bash
  curl -X POST -F "file=@pizza.jpg" http://localhost:8000/model-a/analyze
  ```
- **POST** `/model-b/optimize` (JSON) uses `ai-powered-personalized-meal-planning-dietary-optimization/model_b/trained_model.joblib` plus `epi_r_preprocessed.csv` to rank meals, generate a weekly plan, and compute a grocery list. Override paths with `SMARTMEAL_MODEL_B_PATH` and `SMARTMEAL_MODEL_B_DATASET` if needed. The artifact was trained with scikit-learn 1.7.2, so keep the backend on that version to avoid load errors.
  ```json
  {
    "glucose_category": "high",
    "daily_calorie_target": 1800,
    "age_years": 52,
    "bmi": 29.4,
    "preferences": ["low-gi", "high-fiber"],
    "allergies": ["peanut"],
    "feedback": [
      {"meal": "Grilled Chicken Salad", "action": "accept"},
      {"meal": "Fried Rice", "action": "reject"}
    ],
    "glucose_trend_mg_dl": [140, 145, 150, 155],
    "recent_food": {
      "label": "Pizza",
      "confidence": 0.92,
      "nutrients": {"calories": 510, "carbs_g": 55, "protein_g": 22, "fat_g": 20, "fiber_g": 4}
    }
  }
  ```
- **POST** `/model-b/optimize-from-image` (multipart/form-data) chains Model A + Model B. Provide `file` (image) and `payload` (JSON string for MealOptimizationRequest). The image is analyzed by Model A and injected into `recent_food` automatically.
  ```bash
  curl -X POST http://localhost:8000/model-b/optimize-from-image ^
    -F "file=@pizza.jpg" ^
    -F "payload={\"glucose_category\":\"high\",\"daily_calorie_target\":1800,\"age_years\":52,\"bmi\":29.4}"
  ```

## Model 1: Integrated patient health analysis (diabetic.ipynb)
Single multi-task model (joblib-only) that performs stage detection, daily risk scoring, HbA1c forecasting, and alert/recommendation generation. Processes: pattern extraction from glucose and behavior, multivariate predictive modeling, multi-task output mapping (stage + risk + forecast + alerts), and explainability via SHAP/LIME. Inputs from the API are mapped into the same feature set expected by the notebook pipeline; no rule-based logic remains.

### User-facing inputs (clear names)
- `fasting_glucose_mg_dl`: Latest pre-breakfast glucose (mg/dL).
- `post_meal_glucose_mg_dl`: Average 2-hour post-meal glucose over the last 3 days (mg/dL).
- `daily_glucose_trend_mg_dl`: Array of the last 7 daily mean glucose values (mg/dL) for trend detection.
- `hba1c_history`: List of `{ "date": "YYYY-MM-DD", "value": <percent> }` objects sorted oldest to newest.
- `carb_intake_g_per_day`: Average grams of carbs per day over the last 7 days.
- `activity_minutes_per_day`: Average daily moderate/vigorous activity minutes over the last 7 days.
- `medication_adherence_pct`: Percent of prescribed doses taken over the last 7 days.
- `age_years`: Patient age in years.
- `bmi`: Body mass index (kg/m^2).
- `behavior_consistency_pct`: Percent of days (last 30) with complete logging/plan adherence; captures long-term behavior.
- `patient_id`: Optional patient identifier used for alert logging. If omitted, the API generates one like `patient-YYYYMMDD-abcdef` and returns it in the response.
- `notify_email`: Optional single email for alert notifications (legacy).
- `alert_contacts`: Optional list of family/doctor contacts with per-contact permission settings.

These replace opaque fields like `race`, `admission_type_id`, and `encounter_burden`. Map them inside the notebook/preprocessor so users only see the friendly names above.

### Sample request (Model 1)
```json
{
  "glucose": {
    "fasting_mg_dl": 112,
    "post_meal_mg_dl": 168,
    "daily_trend_mg_dl": [124, 127, 131, 136, 138, 141, 145]
  },
  "hba1c_history": [
    {"date": "2024-06-01", "value": 7.2},
    {"date": "2024-09-01", "value": 7.0}
  ],
  "lifestyle": {
    "carb_intake_g_per_day": 230,
    "activity_minutes_per_day": 25
  },
  "medication_adherence_pct": 92,
  "age_years": 48,
  "bmi": 29.4,
  "behavior_consistency_pct": 68,
  "patient_id": "patient-123",
  "alert_contacts": [
    {
      "role": "family",
      "name": "Alex Lee",
      "email": "alex.lee@example.com",
      "permission": "high_risk_only"
    },
    {
      "role": "doctor",
      "name": "Dr. Kumar",
      "email": "dr.kumar@example.com",
      "permission": "emergency_only"
    }
  ]
}
```

### Alert contacts and permissions
- Use `alert_contacts` to add family members and doctors for alert notifications.
- `permission` options: `all`, `emergency_only`, `weekly_summary`, `high_risk_only`.
- `notify_email` is still supported for a single recipient if you do not supply `alert_contacts`.

### Email notifications
- Set `SMTP_HOST`, `SMTP_USER`, `SMTP_PASS`, `SMTP_PORT` (default 587), and optional `ALERT_FROM_EMAIL`.
- Emails are sent when SMTP is configured and the selected permission allows immediate alerts.

### Sample response (Model 1 - stage + risk + forecast + alerts)
```json
{
  "model": "integrated_patient_health",
  "patient_id": "patient-20260101-a1b2c3",
  "result": {
    "stage": "Managed",
    "stage_confidence": 0.84,
    "risk_score": 72,
    "risk_level": "High",
    "risk_explanations": [
      {"factor": "Elevated HbA1c (7-7.9%)", "impact": "+10"},
      {"factor": "High carb intake (>=220 g/day)", "impact": "+10"},
      {"factor": "Low activity (20-29 min/day)", "impact": "+8"}
    ],
    "forecast_3_months": 7.1,
    "forecast_6_months": 7.4,
    "trend": "Rising",
    "alerts": [
      {
        "alert_type": "Critical",
        "risk_level": "High",
        "risk_score": 72,
        "reason": "High glucose trend for 3+ days with rising HbA1c forecast",
        "explanation_summary": "High glucose trend for 3+ days with rising HbA1c forecast",
        "explanation": "Glucose trend and carb load pushed the risk score above 70 while activity stayed low.",
        "recommended_action": "Add a 15-minute walk after dinner and cut 20-30g carbs from evening meals.",
        "suggested_next_action": "Add a 15-minute walk after dinner and cut 20-30g carbs from evening meals."
      }
    ],
    "future_treatment_recommendations": [
      "Review medication regimen with clinician; consider therapy intensification.",
      "Prioritize nutrition coaching and daily glucose checks."
    ],
    "checkup_schedule": "Clinician follow-up within 1-2 weeks; repeat labs in 4-6 weeks."
  }
}
```

- If no alert is needed, return `"alerts": []`.
- Alert generation now lives inside Model 1; the separate Model 2 module has been removed.

### Output fields in plain language
- `stage`: Clinical stage classification (for example, Managed, Pre-diabetic/Normal, Unmanaged).
- `stage_confidence`: Probability for the predicted stage.
- `risk_score`: 0-100 daily risk index (higher = higher risk).
- `risk_level`: Discrete bucket mapped from the score (Low/Medium/High).
- `risk_explanations`: Top drivers from SHAP/LIME (or heuristic fallback) using glucose, HbA1c, carb intake, activity, adherence, BMI, and behavior consistency. Low-risk outputs may include protective factors with negative impact values.
- `forecast_3_months` / `forecast_6_months`: HbA1c forecasts in %.
- `trend`: Direction of predicted HbA1c movement (Rising/Stable/Falling).
- `alerts`: Inline alerts and recommendations driven by the same factors; includes type, risk level/score, explanation summary, and suggested next action.
- `future_treatment_recommendations`: Automated care suggestions based on stage/risk trends.
- `checkup_schedule`: Suggested timing for the next clinical follow-up and labs.

## Current model descriptions (share with frontend)
- **Insulin dose regressor**: Predicts a suggested insulin dose (continuous value) using glucose, blood pressure, insulin, BMI, age, eGFR, creatinine, and daily carb intake. Output: numeric dose in the same unit used during training.
- **Kidney health classifier**: Predicts kidney status as `Good` vs `Risk` using the same feature set. Models may output `0/1` internally; the API maps these to `Good` or `Risk` strings.
- **Integrated patient health model (Model 1, diabetic.ipynb)**: Single pipeline covering stage detection, daily risk scoring, HbA1c forecasting, and built-in alert/recommendation generation using the user-friendly inputs listed above.

### Sample integrated outputs (from notebooks)
- Multi-day row with inline alert
  ```json
  {
    "date": "2024-09-01",
    "result": {
      "stage": "Pre-diabetic/Normal",
      "stage_confidence": 0.52,
      "risk_score": 46,
      "risk_level": "Medium",
      "risk_explanations": [
        {"factor": "Mean glucose (7-day)", "impact": "+12"},
        {"factor": "Carbs total (7-day)", "impact": "+8"}
      ],
      "forecast_3_months": 6.78,
      "forecast_6_months": 6.83,
      "trend": "Stable",
      "alerts": [
        {
          "alert_type": "Warning",
          "risk_level": "Medium",
          "risk_score": 46,
          "reason": "High carbohydrate intake; Low physical activity",
          "explanation_summary": "High carbohydrate intake; Low physical activity",
          "explanation": "High carbohydrate intake; Low physical activity",
          "recommended_action": "Reduce evening carbohydrate intake / Increase walking by 15-30 minutes per day",
          "suggested_next_action": "Reduce evening carbohydrate intake / Increase walking by 15-30 minutes per day"
        }
      ],
      "future_treatment_recommendations": [
        "Adjust meal plan and activity targets; re-evaluate dose timing.",
        "Schedule a mid-cycle check-in to review trends."
      ],
      "checkup_schedule": "Follow up in 4-6 weeks; HbA1c check in ~3 months."
    }
  }
  ```
- Single prediction row
  ```json
  {
    "stage": "Managed",
    "stage_confidence": 120,
    "risk_score": 31,
    "risk_level": "Low",
    "risk_explanations": [
      {"factor": "Model probability", "impact": "+31"},
      {"factor": "Stable glucose trend", "impact": "+0"}
    ],
    "forecast_3_months": 9.6,
    "forecast_6_months": 9.8,
    "trend": "Stable",
    "alerts": [],
    "future_treatment_recommendations": [
      "Maintain current plan and continue monitoring trends."
    ],
    "checkup_schedule": "Routine checkup in 3-6 months; HbA1c check in ~6 months."
  }
  ```

### Frontend contract for each current output
- **Insulin dose**
  - Field: `predicted_insulin_dose` (float)
  - Display: numeric dose with unit label and optional client-side confidence band.
  - Graphs: small bar vs patient baseline; time series if history is stored.
- **Kidney health**
  - Field: `predicted_kidney_health` (string: Good|Risk)
  - Reasons: `kidney_risk_explanations` (list of strings for top drivers/protective factors)
  - Display: badge with color (green for Good, amber/red for Risk) plus a short tip.
  - Graphs: eGFR and creatinine mini-trends with normal ranges; donut gauge for risk.
- **Integrated patient health (Model 1)**
  - Fields: `stage`, `stage_confidence`, `risk_score`, `risk_level`, `risk_explanations`, `forecast_3_months`, `forecast_6_months`, `trend`, `alerts`, `future_treatment_recommendations`, `checkup_schedule`.
  - Display: timeline cards for stage/risk, forecast tiles, and an alerts lane fed directly from `alerts` (no separate model).
  - Graphs: 7-day glucose trend and forecast arrows; chips showing top explanations.

## UI blueprint for the Explainable AI system (Flutter)
- **Inputs drawer**: Collapsible sheet with the Model 1 fields (fasting/post-meal glucose, trend array, HbA1c history, carbs/day, activity minutes/day, medication adherence %, age, BMI, behavior consistency) with unit helpers and preset chips.
- **Hero insights row**:
  - Card 1: Insulin dose result with a slim meter indicating position vs usual range; CTA to log/confirm dose.
  - Card 2: Kidney status badge (Good/Risk) with a one-line action (for example, "Monitor eGFR weekly" when Risk).
- **Explainability strip (XAI-lite)**: Horizontal chips like "High glucose: pushing risk up" or "Consistent meds: lowering risk," derived from the Model 1 `risk_explanations` until SHAP is wired in.
- **AI risk reason breakdown**: Expandable card under the risk score that lists the top 2-3 `risk_explanations`, each with a colored impact pill (+/-) and a one-line factor summary. Use red/orange for drivers, green for protective factors, and show raw values (glucose, HbA1c, carbs, activity) in a compact subline when expanded.
- **Trend canvas**: Multi-series line chart (glucose, insulin dose, eGFR) with shaded normal bands; toggle carb-intake bars; pinch/scroll to navigate multi-day windows.
- **Risk and stage timeline**: Scrollable daily cards using the Model 1 structure (stage, confidence, risk score/level, risk explanations) with inline sparkline of glucose trend and badges for top factors.
- **Forecast tiles**: Two compact tiles for 3-month and 6-month HbA1c forecasts with arrow indicators (Rising/Stable/Falling) and color-coded confidence.
- **Alerts lane**: Cards driven directly by `alerts` from Model 1 (type, risk level/score, explanation summary, suggested next action); swipe to acknowledge/retain.
- **Future care plan**: Compact card listing `checkup_schedule` and top `future_treatment_recommendations`.
- **Action bar**: Quick actions like "Share with doctor," "Download PDF summary," and "Set reminder" keyed off the latest alert severity.

## Single-model alignment (stage + risk + forecast + alerts)
Model 1 is the only explainable module. Alert and recommendation generation now rides inside the same multi-task pipeline, so there is no Model 2 to maintain or expose.
