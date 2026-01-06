from __future__ import annotations

import hashlib
import json
import os
import smtplib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Directories/paths can be overridden via env so the engine can run outside FastAPI too.
DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent / "Explainable-AI-system"
INCOMING_CSV = Path(os.getenv("INCOMING_CSV", DEFAULT_BASE_DIR / "incoming_patients.csv"))
ALERT_LOG_CSV = Path(os.getenv("ALERT_LOG_CSV", DEFAULT_BASE_DIR / "alerts_log.csv"))
ALERT_MODEL_PATH = Path(os.getenv("ALERT_MODEL_PATH", DEFAULT_BASE_DIR / "model_alert.joblib"))
USE_ALERT_MODEL = os.getenv("USE_ALERT_MODEL", "0") == "1"
ALERT_COOLDOWN_MINUTES = int(os.getenv("ALERT_COOLDOWN_MINUTES", "360"))

PATIENT_ID_COL = os.getenv("PATIENT_ID_COL", "patient_id")
EMAIL_COL = os.getenv("EMAIL_COL", "notify_email")
PHONE_COL = os.getenv("PHONE_COL", "notify_phone")
ALERT_PERMISSION_MODES = {"all", "emergency_only", "weekly_summary", "high_risk_only"}


@dataclass
class AlertDecision:
    alert: bool
    probability: float
    reasons: List[str]


def _risk_level_from_score(score: int) -> str:
    if score < 34:
        return "Low"
    if score < 67:
        return "Medium"
    return "High"


def _normalize_permission(value: Optional[str]) -> str:
    permission = str(value or "all").strip().lower()
    return permission if permission in ALERT_PERMISSION_MODES else "all"


def _normalize_recipients(
    recipients: Optional[Sequence[Dict[str, Any]]], fallback_email: Optional[str]
) -> List[Tuple[str, str]]:
    normalized: List[Tuple[str, str]] = []
    if recipients:
        for entry in recipients:
            email = str(entry.get("email", "")).strip()
            if not email:
                continue
            permission = _normalize_permission(entry.get("permission"))
            normalized.append((email, permission))
    if fallback_email and isinstance(fallback_email, str) and fallback_email.strip():
        normalized.append((fallback_email.strip(), "all"))

    seen = set()
    unique: List[Tuple[str, str]] = []
    for email, permission in normalized:
        key = email.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append((email, permission))
    return unique


def _should_notify(permission: str, alert_type: str, risk_level: str) -> bool:
    mode = _normalize_permission(permission)
    if mode == "weekly_summary":
        return False
    if mode == "emergency_only":
        return alert_type.strip().lower() == "critical"
    if mode == "high_risk_only":
        return risk_level.strip().lower() == "high" or alert_type.strip().lower() == "critical"
    return True


def _send_email_if_configured(to_email: str, subject: str, body: str) -> bool:
    """Best-effort SMTP sender. Returns True when an email was dispatched."""
    smtp_host = os.getenv("SMTP_HOST")
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    if not (smtp_host and smtp_user and smtp_pass):
        return False

    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    from_email = os.getenv("ALERT_FROM_EMAIL", smtp_user)

    msg = EmailMessage()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as smtp:
            smtp.starttls()
            smtp.login(smtp_user, smtp_pass)
            smtp.send_message(msg)
        return True
    except Exception:
        # Do not raise inside FastAPI flow; just report failure to caller.
        return False


def rule_based_alert(
    pred: Dict[str, Any], *, risk_level: Optional[str] = None, risk_score: Optional[int] = None
) -> AlertDecision:
    """
    Thin port of the notebook rule engine.
    Trigger when: elevated risk, medium risk, stage >= 2, or los_bucket >= 2.
    Medium risk now emits a Warning so the API surfaces alerts for non-low cases.
    """
    reasons: List[str] = []

    risk_prob = float(pred.get("risk_probability", 0.0))
    stage = int(pred.get("stage", 0))
    los_bucket = int(pred.get("los_bucket", 0))
    score_val = risk_score if risk_score is not None else int(round(risk_prob * 100))
    level_val = risk_level.lower() if isinstance(risk_level, str) else _risk_level_from_score(score_val)

    high_risk = risk_prob >= 0.80 or score_val >= 70 or level_val in {"high", "critical"}
    medium_risk = level_val == "medium" or risk_prob >= 0.50 or score_val >= 35

    if high_risk:
        reasons.append(f"High readmission risk (score={score_val}, p={risk_prob:.2f})")
    elif medium_risk:
        reasons.append(f"Moderate readmission risk (score={score_val}, p={risk_prob:.2f})")
    if stage >= 2:
        reasons.append("High stage severity (stage=2)")
    if los_bucket >= 2:
        reasons.append("Predicted long length-of-stay (bucket=2)")

    alert = len(reasons) > 0
    probability = max(risk_prob, score_val / 100 if risk_score is not None else risk_prob, 0.8 if high_risk else 0.6 if medium_risk else 0.0)
    return AlertDecision(alert=alert, probability=probability, reasons=reasons)


def model_based_alert(alert_model: Any, features_for_alert: np.ndarray) -> AlertDecision:
    """Optional meta-model path; expects predict_proba."""
    proba = float(alert_model.predict_proba(features_for_alert)[0, 1])
    alert = proba >= 0.5
    reasons = [f"Alert-model probability {proba:.2f} (threshold 0.50)"]
    return AlertDecision(alert=alert, probability=proba, reasons=reasons)


def ensure_log_exists(log_path: Path) -> None:
    if not log_path.exists():
        df = pd.DataFrame(
            columns=["timestamp_utc", "patient_id", "alert_key", "channel", "alert_probability", "reasons_json", "message"]
        )
        df.to_csv(log_path, index=False)


def make_alert_key(patient_id: str, decision: AlertDecision) -> str:
    base = f"{patient_id}|{decision.alert}|{round(decision.probability, 3)}|" + "|".join(decision.reasons)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]


def recently_alerted(log_path: Path, patient_id: str, cooldown_minutes: int) -> bool:
    if not log_path.exists():
        return False

    df = pd.read_csv(log_path)
    if df.empty:
        return False

    df_patient = df[df["patient_id"].astype(str) == str(patient_id)]
    if df_patient.empty:
        return False

    df_patient["timestamp_utc"] = pd.to_datetime(df_patient["timestamp_utc"], utc=True, errors="coerce")
    latest = df_patient["timestamp_utc"].max()
    if pd.isna(latest):
        return False

    return (datetime.now(timezone.utc) - latest) < timedelta(minutes=cooldown_minutes)


def log_alert(log_path: Path, patient_id: str, alert_key: str, channel: str, decision: AlertDecision, message: str) -> None:
    ensure_log_exists(log_path)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "patient_id": str(patient_id),
        "alert_key": alert_key,
        "channel": channel,
        "alert_probability": float(decision.probability),
        "reasons_json": json.dumps(decision.reasons, ensure_ascii=False),
        "message": message,
    }
    df = pd.read_csv(log_path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(log_path, index=False)


def build_message(
    patient_name: Optional[str],
    decision: AlertDecision,
    alert_payload: Dict[str, Any],
    *,
    risk_level: str,
    stage_label: str,
) -> str:
    display_name = str(patient_name or "").strip() or "Patient"
    summary = alert_payload.get("explanation_summary") or alert_payload.get("reason", "Automated risk alert")
    suggested = alert_payload.get("suggested_next_action") or alert_payload.get("recommended_action", "")
    risk_score = alert_payload.get("risk_score")
    risk_score_line = f"Risk score: {risk_score}\n" if risk_score is not None else ""
    return (
        f"ALERT for {display_name}\n"
        f"Risk level: {risk_level}\n"
        f"{risk_score_line}"
        f"Stage: {stage_label}\n"
        f"Probability: {decision.probability:.2f}\n"
        f"Summary: {summary}\n"
        f"Suggested next action: {suggested}\n"
        f"Reasons: {', '.join(decision.reasons)}\n"
    )


class AlertEngine:
    """
    Reusable alert engine that can run inline (FastAPI) or on CSV feeds.
    Accepts the already-loaded preprocess + models from ExplainableAIBundle.
    """

    def __init__(
        self,
        base_dir: Path,
        preprocess: Any,
        risk_model: Any,
        stage_model: Any,
        los_model: Any,
        alert_model: Optional[Any] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self.base_dir = base_dir
        self.preprocess = preprocess
        self.risk_model = risk_model
        self.stage_model = stage_model
        self.los_model = los_model
        self.alert_model = alert_model
        self.log_path = Path(log_path or ALERT_LOG_CSV)

    def _combine_decisions(
        self,
        risk_probability: float,
        stage: int,
        los_bucket: int,
        *,
        risk_score: Optional[int] = None,
        risk_level: Optional[str] = None,
    ) -> AlertDecision:
        pred = {"risk_probability": risk_probability, "stage": stage, "los_bucket": los_bucket}
        decision = rule_based_alert(pred, risk_level=risk_level, risk_score=risk_score)

        if self.alert_model is not None:
            m_dec = model_based_alert(
                self.alert_model, np.array([[risk_probability, stage, los_bucket]], dtype=float)
            )
            if m_dec.alert or decision.alert:
                decision.alert = True
                decision.probability = max(decision.probability, m_dec.probability)
                for reason in m_dec.reasons:
                    if reason not in decision.reasons:
                        decision.reasons.append(reason)

        return decision

    @staticmethod
    def _format_alert(
        decision: AlertDecision, risk_level: str, stage_label: str, risk_score: int
    ) -> Dict[str, Any]:
        alert_type = "Critical" if decision.probability >= 0.8 or risk_level.lower() == "high" else "Warning"
        reason = decision.reasons[0] if decision.reasons else "Automated risk alert"
        explanation = "; ".join(decision.reasons) if decision.reasons else "Automated alert triggered by risk engine."
        if alert_type == "Critical":
            recommended_action = "Escalate to care team, tighten monitoring for 24-48 hours."
        elif "stage" in reason.lower():
            recommended_action = "Review medication and schedule clinician follow-up."
        else:
            recommended_action = "Increase activity and reduce carbohydrate load; recheck within 24 hours."

        return {
            "alert_type": alert_type,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "reason": reason,
            "explanation_summary": reason,
            "explanation": f"{explanation} (stage={stage_label}, risk={risk_level})",
            "recommended_action": recommended_action,
            "suggested_next_action": recommended_action,
        }

    def alerts_for_prediction(
        self,
        *,
        patient_id: Optional[str],
        patient_name: Optional[str] = None,
        risk_probability: float,
        stage_idx: int,
        stage_label: str,
        los_bucket: int,
        risk_level: str,
        risk_score: int,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        recipients: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        decision = self._combine_decisions(
            risk_probability, stage_idx, los_bucket, risk_score=risk_score, risk_level=risk_level
        )
        if not decision.alert:
            return []

        alert_payload = self._format_alert(decision, risk_level, stage_label, risk_score)

        if patient_id:
            # Deduplication and optional outbound notifications.
            if not recently_alerted(self.log_path, patient_id, ALERT_COOLDOWN_MINUTES):
                message = build_message(
                    patient_name,
                    decision,
                    alert_payload,
                    risk_level=risk_level,
                    stage_label=stage_label,
                )
                alert_key = make_alert_key(patient_id, decision)

                sent_email = False
                skipped_weekly = False
                for to_email, permission in _normalize_recipients(recipients, email):
                    if not _should_notify(permission, alert_payload["alert_type"], risk_level):
                        if _normalize_permission(permission) == "weekly_summary":
                            skipped_weekly = True
                        continue
                    sent = _send_email_if_configured(
                        to_email,
                        subject=f"Diabetes Risk Alert: {patient_id}",
                        body=message,
                    )
                    if sent:
                        sent_email = True
                        log_alert(self.log_path, patient_id, alert_key, "email", decision, message)

                # SMS is intentionally omitted here to avoid adding new deps; log fallback instead.
                if not sent_email:
                    channel = "summary" if skipped_weekly else "api"
                    log_alert(self.log_path, patient_id, alert_key, channel, decision, message)

        return [alert_payload]

    def process_feed_once(
        self,
        x_columns: Sequence[str],
        default_row: Dict[str, Any],
        *,
        incoming_csv: Path = INCOMING_CSV,
    ) -> List[Dict[str, Any]]:
        """
        Batch process the CSV feed (incoming_patients.csv by default) and return triggered alerts.
        Each row must contain the model feature columns; missing columns are filled from defaults.
        """
        if self.preprocess is None or self.risk_model is None or self.stage_model is None or self.los_model is None:
            raise RuntimeError("Explainable AI models are not loaded; cannot process feed.")

        if not incoming_csv.exists():
            incoming_csv.write_text("")
            return []

        df = pd.read_csv(incoming_csv)
        if df.empty:
            return []

        feature_rows: List[Dict[str, Any]] = []
        patient_meta: List[Tuple[str, Optional[str], Optional[str], Optional[str]]] = []

        for idx, row in df.iterrows():
            base = default_row.copy()
            for col in x_columns:
                if col in row and pd.notna(row[col]):
                    base[col] = row[col]
            feature_rows.append({col: base.get(col) for col in x_columns})
            patient_name = None
            if "patient_name" in row and pd.notna(row["patient_name"]):
                patient_name = str(row.get("patient_name")).strip() or None
            patient_meta.append(
                (
                    str(row.get(PATIENT_ID_COL, f"row_{idx}")),
                    row.get(EMAIL_COL),
                    row.get(PHONE_COL),
                    patient_name,
                )
            )

        x_df = pd.DataFrame(feature_rows, columns=list(x_columns))
        x_proc = self.preprocess.transform(x_df)

        alerts: List[Dict[str, Any]] = []
        risk_probas = self.risk_model.predict_proba(x_proc)
        stages = self.stage_model.predict_proba(x_proc)
        los_preds = self.los_model.predict(x_proc)

        for idx, (risk_row, stage_row, los_pred) in enumerate(zip(risk_probas, stages, los_preds)):
            risk_probability = float(risk_row[1])
            risk_score = int(round(risk_probability * 100))
            risk_level = _risk_level_from_score(risk_score)
            stage_idx = int(np.argmax(stage_row))
            los_bucket = int(los_pred)

            decision = self._combine_decisions(
                risk_probability, stage_idx, los_bucket, risk_score=risk_score, risk_level=risk_level
            )
            if not decision.alert:
                continue

            pid, email, phone, patient_name = patient_meta[idx]
            if recently_alerted(self.log_path, pid, ALERT_COOLDOWN_MINUTES):
                continue

            alert_key = make_alert_key(pid, decision)
            stage_label = f"Stage-{stage_idx}"
            alert_payload = self._format_alert(decision, risk_level, stage_label, risk_score)
            message = build_message(
                patient_name,
                decision,
                alert_payload,
                risk_level=risk_level,
                stage_label=stage_label,
            )
            sent_email = False
            if email and isinstance(email, str) and email.strip():
                sent_email = _send_email_if_configured(
                    email.strip(), subject=f"Diabetes Risk Alert: {pid}", body=message
                )
                if sent_email:
                    log_alert(self.log_path, pid, alert_key, "email", decision, message)

            channel = "email" if sent_email else "feed"
            log_alert(self.log_path, pid, alert_key, channel, decision, message)

            alerts.append(
                {
                    "patient_id": pid,
                    "alert_probability": decision.probability,
                    "reasons": decision.reasons,
                    "channel": channel,
                }
            )

        return alerts
