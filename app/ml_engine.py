import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

ARTIFACT_DIR = "/srv/app/models"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "supervised_active_model.joblib")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.joblib")


class OvertimeMLEngine:
    def __init__(self, min_decisions_to_train: int = 5):
        self.min_decisions_to_train = min_decisions_to_train
        self.scaler = StandardScaler()
        self.iforest = IsolationForest(n_estimators=100, contamination=0.08, random_state=42)
        self.lof = LocalOutlierFactor(n_neighbors=5, contamination=0.08, novelty=True)
        self.supervised_model = LogisticRegression(class_weight="balanced", max_iter=1000)

        self.is_supervised_trained = False
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        self._init_cold_start_baseline()
        self._load_persisted_artifacts()

    @staticmethod
    def hash_identifier(identifier: str) -> str:
        """Privacy by design: Pseudonymize raw employee/device codes using SHA-256."""
        if not identifier:
            return ""
        return hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:16]

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        trust_weights = {"GREEN": 1.0, "AMBER": 0.6, "YELLOW": 0.6, "RED": 0.1}
        df = df.copy()

        df["trust_num"] = df["trust_val"].map(trust_weights).fillna(0.5)
        df["ot_ratio"] = df["overtime_hours"] / (df["total_hours"] + 1e-4)

        # Dynamic rolling baseline deviation
        if "historical_median_ot" not in df.columns:
            df["historical_median_ot"] = 1.0
        df["ot_deviation"] = df["overtime_hours"] - df["historical_median_ot"]

        # Verification mode fallback flag: 1 if downgraded to password/card; 0 if biometric
        if "verify_mode_fallback" not in df.columns:
            df["verify_mode_fallback"] = 0
        df["verify_mode_fallback"] = df["verify_mode_fallback"].astype(int)

        cols = [
            "overtime_hours",
            "total_hours",
            "sensor_flags_count",
            "trust_num",
            "ot_ratio",
            "ot_deviation",
            "verify_mode_fallback",
        ]
        return df[cols].values

    def _init_cold_start_baseline(self):
        synthetic_shifts = pd.DataFrame([
            {"overtime_hours": 0.0, "total_hours": 8.0, "sensor_flags_count": 0, "trust_val": "GREEN", "historical_median_ot": 0.5, "verify_mode_fallback": 0},
            {"overtime_hours": 0.5, "total_hours": 8.5, "sensor_flags_count": 0, "trust_val": "GREEN", "historical_median_ot": 0.5, "verify_mode_fallback": 0},
            {"overtime_hours": 1.0, "total_hours": 9.0, "sensor_flags_count": 0, "trust_val": "GREEN", "historical_median_ot": 1.0, "verify_mode_fallback": 0},
            {"overtime_hours": 1.5, "total_hours": 9.5, "sensor_flags_count": 0, "trust_val": "GREEN", "historical_median_ot": 1.0, "verify_mode_fallback": 0},
            {"overtime_hours": 2.0, "total_hours": 10.0, "sensor_flags_count": 0, "trust_val": "GREEN", "historical_median_ot": 1.5, "verify_mode_fallback": 0},
            {"overtime_hours": 2.0, "total_hours": 10.0, "sensor_flags_count": 1, "trust_val": "GREEN", "historical_median_ot": 1.5, "verify_mode_fallback": 0},
        ])
        X = self._extract_features(synthetic_shifts)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.iforest.fit(X_scaled)
        self.lof.fit(X_scaled)

    def _load_persisted_artifacts(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                self.supervised_model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.is_supervised_trained = True
            except Exception:
                self.is_supervised_trained = False

    def train_from_decisions(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        records = []
        for d in decisions:
            raw_fv = d.get("featureVectorJson")
            if not raw_fv:
                continue
            try:
                features = json.loads(raw_fv) if isinstance(raw_fv, str) else raw_fv
            except Exception:
                continue
            features["label"] = 1 if d.get("decision") == "REJECTED" else 0
            records.append(features)

        if len(records) < self.min_decisions_to_train:
            return {
                "active_learning_trained": self.is_supervised_trained,
                "model_architecture": "SUPERVISED_ACTIVE_LOGREG" if self.is_supervised_trained else "COLD_START_WAITING_FOR_DATA",
                "supervisor_decisions_recorded": len(records),
            }

        df = pd.DataFrame(records)
        y = df["label"].values

        if len(np.unique(y)) < 2:
            return {
                "active_learning_trained": self.is_supervised_trained,
                "model_architecture": "SUPERVISED_ACTIVE_LOGREG" if self.is_supervised_trained else "AWAITING_CLASS_BALANCE",
                "supervisor_decisions_recorded": len(records),
            }

        candidate_scaler = StandardScaler()
        X = self._extract_features(df)
        X_scaled = candidate_scaler.fit_transform(X)

        candidate_model = LogisticRegression(class_weight="balanced", max_iter=1000)
        candidate_model.fit(X_scaled, y)

        preds = candidate_model.predict(X_scaled)
        score = f1_score(y, preds, average="weighted")
        if score < 0.50:
            return {
                "active_learning_trained": self.is_supervised_trained,
                "model_architecture": "FAILED_VALIDATION_GATE",
                "supervisor_decisions_recorded": len(records),
                "validation_f1": round(score, 3),
            }

        self.supervised_model = candidate_model
        self.scaler = candidate_scaler
        self.is_supervised_trained = True

        joblib.dump(self.supervised_model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)

        return {
            "active_learning_trained": True,
            "model_architecture": "SUPERVISED_ACTIVE_LOGREG",
            "supervisor_decisions_recorded": len(records),
            "validation_f1": round(score, 3),
        }

    def score(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        df_claim = pd.DataFrame([claim])
        reasons = []

        ot = float(claim.get("overtime_hours", 0.0))
        flags = int(claim.get("sensor_flags_count", 0))
        trust = str(claim.get("trust_val", "GREEN"))
        fallback = int(claim.get("verify_mode_fallback", 0))

        if ot > 4.0:
            reasons.append(f"Excessive overtime ({ot:.1f} hrs > 4.0h limit)")
        if flags > 0:
            reasons.append(f"Hardware sensor flags detected ({flags} events)")
        if trust in ["RED", "AMBER", "YELLOW"]:
            reasons.append(f"Terminal trust degraded ({trust})")
        if fallback == 1:
            reasons.append("Verification mode fallback detected (PIN/Password used instead of biometric)")

        X = self._extract_features(df_claim)
        X_scaled = self.scaler.transform(X)

        if self.is_supervised_trained:
            prob_fraud = float(self.supervised_model.predict_proba(X_scaled)[0][1])
            is_anomalous = bool(prob_fraud > 0.50)
            if is_anomalous:
                reasons.append(f"Active learning model scored fraud risk at {prob_fraud:.1%}")

            return {
                "isMlAnomalous": is_anomalous,
                "mlAnomalyScore": float(round(prob_fraud, 3)),
                "mlBaselineModel": "SUPERVISED_ACTIVE_LOGREG",
                "mlExplainability": str("; ".join(reasons) if reasons else "Claim matches supervisor-approved patterns."),
            }

        # Cold-Start Unsupervised Scoring
        iforest_raw = float(-self.iforest.score_samples(X_scaled)[0])
        norm_iforest = float(1.0 / (1.0 + np.exp(-3 * (iforest_raw - 0.5))))
        trust_weight = {"GREEN": 1.0, "AMBER": 0.6, "YELLOW": 0.6, "RED": 0.1}.get(trust, 0.5)

        final_score = (
            (0.4 * norm_iforest)
            + (0.3 * (1.0 - trust_weight))
            + (0.2 * (1.0 if fallback == 1 else 0.0))
            + (0.1 * (1.0 if len(reasons) > 0 else 0.0))
        )
        final_score = float(min(max(final_score, 0.0), 1.0))
        is_anomalous = bool(final_score > 0.60)

        return {
            "isMlAnomalous": is_anomalous,
            "mlAnomalyScore": float(round(final_score, 3)),
            "mlBaselineModel": "ENSEMBLE_IFOREST_RULES_V1",
            "mlExplainability": str("; ".join(reasons) if reasons else "Within baseline operational tolerance."),
        }

    def run_benchmark(self, n_samples: int = 100) -> Dict[str, Any]:
        """Fault and fraud injection benchmark (Item 11). Evaluates model robustness against known attacks."""
        rng = np.random.default_rng(42)
        n_clean = int(n_samples * 0.7)
        n_fraud = n_samples - n_clean

        # Clean claims (Normal shifts, biometric verified)
        clean_df = pd.DataFrame({
            "overtime_hours": rng.uniform(0.0, 2.5, n_clean),
            "total_hours": rng.uniform(8.0, 10.5, n_clean),
            "sensor_flags_count": np.zeros(n_clean, dtype=int),
            "trust_val": ["GREEN"] * n_clean,
            "historical_median_ot": rng.uniform(0.5, 2.0, n_clean),
            "verify_mode_fallback": np.zeros(n_clean, dtype=int),
            "label": np.zeros(n_clean, dtype=int),
        })

        # Fraud claims (Padded OT, buddy punching fallback, clock tampering)
        fraud_df = pd.DataFrame({
            "overtime_hours": rng.uniform(4.5, 9.0, n_fraud),
            "total_hours": rng.uniform(12.5, 17.0, n_fraud),
            "sensor_flags_count": rng.integers(1, 4, n_fraud),
            "trust_val": rng.choice(["AMBER", "RED"], n_fraud),
            "historical_median_ot": rng.uniform(0.5, 1.5, n_fraud),
            "verify_mode_fallback": rng.choice([0, 1], n_fraud, p=[0.3, 0.7]),
            "label": np.ones(n_fraud, dtype=int),
        })

        test_data = pd.concat([clean_df, fraud_df], ignore_index=True).sample(frac=1.0, random_state=42)
        y_true = test_data["label"].values

        y_preds = []
        y_scores = []
        for _, row in test_data.iterrows():
            eval_res = self.score(row.to_dict())
            y_preds.append(1 if eval_res["isMlAnomalous"] else 0)
            y_scores.append(eval_res["mlAnomalyScore"])

        y_preds = np.array(y_preds)
        y_scores = np.array(y_scores)

        tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()

        return {
            "total_injected_test_shifts": int(n_samples),
            "clean_shifts": int(n_clean),
            "fraud_injected_shifts": int(n_fraud),
            "active_model_evaluated": "SUPERVISED_ACTIVE_LOGREG" if self.is_supervised_trained else "ENSEMBLE_IFOREST_COLD_START",
            "metrics": {
                "precision": float(round(precision_score(y_true, y_preds, zero_division=0), 3)),
                "recall": float(round(recall_score(y_true, y_preds, zero_division=0), 3)),
                "f1_score": float(round(f1_score(y_true, y_preds, zero_division=0), 3)),
                "roc_auc": float(round(roc_auc_score(y_true, y_scores), 3)),
            },
            "confusion_matrix": {
                "true_negatives_clean_passed": int(tn),
                "false_positives_clean_flagged": int(fp),
                "false_negatives_fraud_missed": int(fn),
                "true_positives_fraud_quarantined": int(tp),
            },
        }
