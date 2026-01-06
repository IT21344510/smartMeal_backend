from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from fastapi import HTTPException
from PIL import Image

logger = logging.getLogger("smartmeal.meal")

FOOD_MODEL_ENV = "SMARTMEAL_FOOD_MODEL"
FOOD_NUTRIENTS_MODEL_ENV = "SMARTMEAL_FOOD_NUTRIENTS_MODEL"
FOOD_LABELS_ENV = "SMARTMEAL_FOOD_LABELS"
DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "ai-powered-personalized-meal-planning-dietary-optimization"
    / "model"
)
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "keras_model.h5"
DEFAULT_NUTRIENTS_MODEL_PATH = DEFAULT_MODEL_DIR / "food_nutrients_auto.h5"
DEFAULT_LABELS_PATH = DEFAULT_MODEL_DIR / "labels.txt"
DEFAULT_PORTION_SIZE = "Medium"


@dataclass
class FoodPrediction:
    detected_items: List[str]
    food_confidence: float
    portion_size: str
    calories: float
    carbs: str
    protein: str
    fat: str
    fiber: str
    carb_load_g: float


def _load_labels(path: Path) -> List[str]:
    """
    Parse the label file produced alongside the Keras model.
    Matches notebook normalization so outputs align with training labels.
    """
    labels: List[str] = []
    if not path.exists():
        logger.warning("Label file %s does not exist; falling back to numeric labels.", path)
        return labels

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\s*\d+\s+(.*)$", line)
        label = (match.group(1) if match else line).strip()
        label = label.lower().replace("_", " ")
        label = re.sub(r"[^a-z0-9\s]+", " ", label)
        label = re.sub(r"\s+", " ", label).strip()
        labels.append(label)
    return labels


class FoodRecognitionEngine:
    """
    Handles Model A: food classification + nutrient estimation using trained models only.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        nutrient_model_path: Optional[Path] = None,
        labels_path: Optional[Path] = None,
        portion_size: str = DEFAULT_PORTION_SIZE,
    ) -> None:
        self.model_path = Path(model_path) if model_path else Path(DEFAULT_MODEL_PATH)
        self.nutrient_model_path = (
            Path(nutrient_model_path) if nutrient_model_path else Path(DEFAULT_NUTRIENTS_MODEL_PATH)
        )
        self.labels_path = Path(labels_path) if labels_path else Path(DEFAULT_LABELS_PATH)
        self.labels = _load_labels(self.labels_path)
        self.default_portion = portion_size
        self._tf = None
        self._class_model = None
        self._nutrient_model = None
        self._target_size_class: Tuple[int, int] = (224, 224)
        self._target_size_nutrient: Tuple[int, int] = (224, 224)

    @staticmethod
    def _target_size_from_model(model: object) -> Optional[Tuple[int, int]]:
        shape = getattr(model, "input_shape", None)
        if shape and len(shape) >= 3 and shape[1] and shape[2]:
            # Keras input is (batch, height, width, channels); PIL resize expects (width, height).
            return (int(shape[2]), int(shape[1]))
        return None

    def _ensure_models(self) -> None:
        if self._class_model is not None and self._nutrient_model is not None:
            return
        try:
            import tensorflow as tf  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise HTTPException(
                status_code=503,
                detail="TensorFlow is required for food recognition. Install tensorflow(-cpu) and retry.",
            ) from exc

        class_model_path = Path(self.model_path)
        if not class_model_path.exists():
            raise HTTPException(status_code=503, detail=f"Food model not found at {class_model_path}.")

        nutrient_model_path = Path(self.nutrient_model_path)
        if not nutrient_model_path.exists():
            raise HTTPException(status_code=503, detail=f"Nutrient model not found at {nutrient_model_path}.")

        self._tf = tf
        try:
            self._class_model = tf.keras.models.load_model(class_model_path, compile=False)
        except Exception as exc:  # pragma: no cover - defensive load
            logger.error("Failed to load food model from %s (%s).", class_model_path, exc)
            raise HTTPException(status_code=500, detail="Food model failed to load.") from exc

        try:
            self._nutrient_model = tf.keras.models.load_model(nutrient_model_path, compile=False)
        except Exception as exc:  # pragma: no cover - defensive load
            logger.error("Failed to load nutrient model from %s (%s).", nutrient_model_path, exc)
            raise HTTPException(status_code=500, detail="Nutrient model failed to load.") from exc

        class_size = self._target_size_from_model(self._class_model)
        if class_size:
            self._target_size_class = class_size
        nutrient_size = self._target_size_from_model(self._nutrient_model)
        if nutrient_size:
            self._target_size_nutrient = nutrient_size

    @staticmethod
    def _load_image(data: bytes) -> Image.Image:
        try:
            return Image.open(BytesIO(data)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid image file; must be a readable JPG/PNG.") from exc

    @staticmethod
    def _image_to_array(
        image: Image.Image, target_size: Tuple[int, int], normalize: bool
    ) -> np.ndarray:
        resized = image.resize(target_size)
        arr = np.asarray(resized, dtype="float32")
        if normalize:
            arr = arr / 255.0
        return arr

    def _extract_nutrient_output(self, output: object) -> np.ndarray:
        if isinstance(output, dict):
            nutrients = output.get("nutrients")
            if nutrients is None:
                raise HTTPException(status_code=500, detail="Nutrient model output missing nutrients head.")
            return np.asarray(nutrients)

        if isinstance(output, (list, tuple)):
            output_names = getattr(self._nutrient_model, "output_names", []) if self._nutrient_model else []
            if output_names:
                for name, value in zip(output_names, output):
                    if name == "nutrients":
                        return np.asarray(value)
            if len(output) == 1:
                return np.asarray(output[0])
            if len(output) >= 2:
                return np.asarray(output[1])

        return np.asarray(output)

    def _predict_nutrients(self, arr: np.ndarray) -> np.ndarray:
        assert self._nutrient_model is not None  # for type checkers
        output = self._nutrient_model.predict(arr[None, ...], verbose=0)
        nutrients = np.asarray(self._extract_nutrient_output(output))
        if nutrients.ndim > 1:
            nutrients = nutrients[0]
        if nutrients.ndim != 1:
            nutrients = nutrients.flatten()
        if nutrients.size < 5:
            raise HTTPException(status_code=500, detail="Nutrient model output missing expected values.")
        return nutrients

    def predict(self, image_bytes: bytes) -> FoodPrediction:
        self._ensure_models()
        image = self._load_image(image_bytes)
        class_arr = self._image_to_array(image, self._target_size_class, normalize=True)
        nutrient_arr = self._image_to_array(image, self._target_size_nutrient, normalize=False)

        assert self._class_model is not None  # for type checkers
        class_output = self._class_model.predict(class_arr[None, ...], verbose=0)
        if isinstance(class_output, (list, tuple)):
            class_output = class_output[0]
        class_probs = np.asarray(class_output)
        if class_probs.ndim != 1:
            class_probs = class_probs.flatten()

        idx = int(np.argmax(class_probs)) if class_probs.size else 0
        confidence = float(class_probs[idx]) if class_probs.size else 0.0
        label_raw = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
        label_pretty = label_raw.title() if label_raw else f"Class {idx}"

        nutrients = self._predict_nutrients(nutrient_arr)

        return FoodPrediction(
            detected_items=[label_pretty],
            food_confidence=round(confidence, 4),
            portion_size=self.default_portion,
            calories=float(nutrients[0]),
            carbs=f"{float(nutrients[1]):.1f}g",
            protein=f"{float(nutrients[2]):.1f}g",
            fat=f"{float(nutrients[3]):.1f}g",
            fiber=f"{float(nutrients[4]):.1f}g",
            carb_load_g=float(nutrients[1]),
        )


def load_food_engine() -> FoodRecognitionEngine:
    model_path = Path(os.getenv(FOOD_MODEL_ENV, str(DEFAULT_MODEL_PATH)))
    nutrient_model_path = Path(os.getenv(FOOD_NUTRIENTS_MODEL_ENV, str(DEFAULT_NUTRIENTS_MODEL_PATH)))
    labels_path = Path(os.getenv(FOOD_LABELS_ENV, str(DEFAULT_LABELS_PATH)))

    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    if not nutrient_model_path.is_absolute():
        nutrient_model_path = Path.cwd() / nutrient_model_path
    if not labels_path.is_absolute():
        labels_path = Path.cwd() / labels_path

    return FoodRecognitionEngine(
        model_path=model_path,
        nutrient_model_path=nutrient_model_path,
        labels_path=labels_path,
    )
