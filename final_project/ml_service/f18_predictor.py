from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_DIR / "models" / "f18_canonical_huber_010"


@dataclass(frozen=True)
class PredictionResult:
    model_version: str
    predicted_price_per_m2: float
    predicted_deal_amount: float | None
    predicted_price_per_pyeong: float
    raw_residual_log: float
    predicted_log_price_per_m2: float
    interval_low: float | None
    interval_high: float | None
    interval_basis: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "modelVersion": self.model_version,
            "predictedPricePerM2": self.predicted_price_per_m2,
            "predictedDealAmount": self.predicted_deal_amount,
            "predictedPricePerPyeong": self.predicted_price_per_pyeong,
            "rawResidualLog": self.raw_residual_log,
            "predictedLogPricePerM2": self.predicted_log_price_per_m2,
            "intervalLow": self.interval_low,
            "intervalHigh": self.interval_high,
            "intervalBasis": self.interval_basis,
        }


class F18Predictor:
    def __init__(self, artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        self.metadata = self._load_json("metadata.json")
        self.schema = self._load_json("feature_schema.json")
        self.medians = {key: float(value) for key, value in self._load_json("numeric_medians.json").items()}
        self.model = tf.keras.models.load_model(self.artifact_dir / "keras_model.keras", compile=False)
        self.numeric_features = list(self.schema["numeric_features"])
        self.embedding_features = list(self.schema["embedding_features"])
        self.base_log_feature = str(self.schema["base_log_feature"])
        self.model_version = str(self.metadata["model_version"])

    def _load_json(self, name: str) -> dict[str, Any]:
        path = self.artifact_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def make_inputs(self, numeric_features: dict[str, Any], embedding_features: dict[str, Any]) -> dict[str, Any]:
        numeric_values = []
        for feature in self.numeric_features:
            value = numeric_features.get(feature, self.medians.get(feature))
            if value is None or (isinstance(value, float) and math.isnan(value)):
                value = self.medians.get(feature, 0.0)
            numeric_values.append(float(value))
        inputs: dict[str, Any] = {
            "numeric_input": np.asarray([numeric_values], dtype="float32"),
        }
        for feature in self.embedding_features:
            value = embedding_features.get(feature, "missing")
            if value is None:
                value = "missing"
            inputs[f"{feature}_input"] = tf.convert_to_tensor([[str(value)]], dtype=tf.string)
        return inputs

    def predict(
        self,
        numeric_features: dict[str, Any],
        embedding_features: dict[str, Any],
        *,
        base_log_value: float | None = None,
        area_m2: float | None = None,
        interval_pct: float | None = None,
        interval_basis: str | None = None,
    ) -> PredictionResult:
        if base_log_value is None:
            base_value = numeric_features.get(self.base_log_feature, self.medians.get(self.base_log_feature))
            if base_value is None:
                raise ValueError(f"base_log_value or {self.base_log_feature} is required")
            base_log_value = float(base_value)
        raw_residual = float(self.model.predict(self.make_inputs(numeric_features, embedding_features), verbose=0).reshape(-1)[0])
        pred_log = float(base_log_value + raw_residual)
        pred_ppm = float(math.exp(pred_log))
        pred_total = float(pred_ppm * area_m2) if area_m2 is not None else None
        low = high = None
        if pred_total is not None and interval_pct is not None:
            interval = max(0.0, float(interval_pct))
            low = float(pred_total * max(0.0, 1.0 - interval))
            high = float(pred_total * (1.0 + interval))
        return PredictionResult(
            model_version=self.model_version,
            predicted_price_per_m2=pred_ppm,
            predicted_deal_amount=pred_total,
            predicted_price_per_pyeong=pred_ppm * 3.305785,
            raw_residual_log=raw_residual,
            predicted_log_price_per_m2=pred_log,
            interval_low=low,
            interval_high=high,
            interval_basis=interval_basis,
        )

    def predict_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.predict(
            payload.get("numeric_features", {}),
            payload.get("embedding_features", {}),
            base_log_value=payload.get("base_log_value"),
            area_m2=payload.get("area_m2"),
            interval_pct=payload.get("interval_pct"),
            interval_basis=payload.get("interval_basis"),
        )
        out = result.to_dict()
        if "transaction_id" in payload:
            out["transactionId"] = payload["transaction_id"]
        return out

