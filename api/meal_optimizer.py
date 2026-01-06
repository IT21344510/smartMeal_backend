from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import pandas as pd
from fastapi import HTTPException

try:
    import joblib
except ImportError:  # pragma: no cover - optional dependency
    joblib = None

logger = logging.getLogger("smartmeal.model_b")

MODEL_B_ENV = "SMARTMEAL_MODEL_B_PATH"
MODEL_B_DATASET_ENV = "SMARTMEAL_MODEL_B_DATASET"

BASE_DIR = (
    Path(__file__).resolve().parent.parent
    / "ai-powered-personalized-meal-planning-dietary-optimization"
    / "model_b"
)
DEFAULT_MODEL_PATH = BASE_DIR / "trained_model.joblib"
DEFAULT_DATASET_PATH = BASE_DIR / "epi_r_preprocessed.csv"

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MEALS_PER_DAY = 3

GROCERY_KEYWORDS = [
    "chicken",
    "fish",
    "salmon",
    "tuna",
    "egg",
    "lentil",
    "bean",
    "chickpea",
    "rice",
    "quinoa",
    "oats",
    "bread",
    "pasta",
    "tomato",
    "spinach",
    "broccoli",
    "carrot",
    "onion",
    "garlic",
    "pepper",
    "mushroom",
    "yogurt",
    "cheese",
    "tofu",
    "potato",
    "sweet potato",
    "salad",
    "vegetable",
    "fruit",
]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


class MealOptimizer:
    def __init__(self, model_path: Optional[Path] = None, dataset_path: Optional[Path] = None) -> None:
        self.model_path = self._resolve_path(
            Path(os.getenv(MODEL_B_ENV, str(model_path or DEFAULT_MODEL_PATH)))
        )
        self.dataset_path = self._resolve_path(
            Path(os.getenv(MODEL_B_DATASET_ENV, str(dataset_path or DEFAULT_DATASET_PATH)))
        )
        self.model: Optional[Any] = None
        self.model_error: Optional[str] = None
        self.data: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        self.tag_columns: List[str] = []
        self._column_lookup: Dict[str, List[str]] = {}
        self._load_assets()

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        path = path.expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def _load_assets(self) -> None:
        self._load_model()
        self._load_dataset()
        if self.model is not None and self.data is not None:
            self._attach_model_predictions()

    def _load_model(self) -> None:
        if joblib is None:
            self.model_error = "joblib is not installed."
            return

        if not self.model_path.exists():
            self.model_error = f"Model path does not exist: {self.model_path}"
            return

        try:
            self.model = joblib.load(self.model_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.model_error = f"Failed to load model: {exc}"
            logger.warning("Model B joblib failed to load (%s).", exc)

    def _load_dataset(self) -> None:
        if not self.dataset_path.exists():
            logger.warning("Model B dataset not found at %s.", self.dataset_path)
            return

        df = pd.read_csv(self.dataset_path)
        if "title" not in df.columns or "calories" not in df.columns:
            logger.warning("Model B dataset missing title/calories columns.")
            return

        df = df.dropna(subset=["title", "calories"]).copy()
        df["title"] = df["title"].astype(str)
        df["title_norm"] = df["title"].str.lower()
        df["title_key"] = df["title"].apply(_normalize)
        df = self._ensure_glucose_category(df)

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        self.feature_columns = [c for c in numeric_cols if c != "calories"]
        tag_columns: List[str] = []
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            if series.min() >= 0 and series.max() <= 1:
                tag_columns.append(col)
        self.tag_columns = tag_columns
        self.data = df
        self._column_lookup = self._build_column_lookup(self.tag_columns)

    @staticmethod
    def _glucose_category_from_calories(calories: float) -> str:
        if calories < 400:
            return "low"
        if calories <= 700:
            return "normal"
        return "high"

    def _ensure_glucose_category(self, df: pd.DataFrame) -> pd.DataFrame:
        if "glucose_category" in df.columns:
            df["glucose_category"] = df["glucose_category"].astype(str).str.lower()
            missing = df["glucose_category"].isna() | (df["glucose_category"] == "nan")
            if missing.any():
                df.loc[missing, "glucose_category"] = df.loc[missing, "calories"].apply(
                    self._glucose_category_from_calories
                )
            return df

        df["glucose_category"] = df["calories"].apply(self._glucose_category_from_calories)
        return df

    def _attach_model_predictions(self) -> None:
        assert self.data is not None
        try:
            preds = self.model.predict(self.data[self.feature_columns])
            self.data["_model_pred"] = pd.Series(preds).astype(str).str.lower()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.model_error = f"Model prediction failed: {exc}"
            self.model = None
            logger.warning("Model B prediction failed (%s).", exc)

    @staticmethod
    def _build_column_lookup(columns: Sequence[str]) -> Dict[str, List[str]]:
        lookup: Dict[str, List[str]] = {}
        for col in columns:
            norm = _normalize(str(col))
            lookup.setdefault(norm, []).append(col)
        return lookup

    def _columns_for_term(self, term: str) -> List[str]:
        term_norm = _normalize(term)
        if not term_norm:
            return []
        matches: List[str] = []
        for norm, cols in self._column_lookup.items():
            if term_norm in norm:
                matches.extend(cols)
        return matches

    def _exclude_terms(self, df: pd.DataFrame, terms: Sequence[str]) -> pd.DataFrame:
        if not terms:
            return df
        banned_mask = pd.Series(False, index=df.index)
        for term in terms:
            term_norm = _normalize(term)
            if not term_norm:
                continue
            title_mask = df["title_key"].str.contains(re.escape(term_norm), na=False)
            cols = self._columns_for_term(term_norm)
            if cols:
                tag_mask = (df[cols] > 0).any(axis=1)
                title_mask = title_mask | tag_mask
            banned_mask = banned_mask | title_mask
        return df[~banned_mask]

    def _preference_score(self, df: pd.DataFrame, terms: Sequence[str]) -> pd.Series:
        if not terms:
            return pd.Series(0, index=df.index)
        score = pd.Series(0, index=df.index)
        for term in terms:
            term_norm = _normalize(term)
            if not term_norm:
                continue
            title_mask = df["title_key"].str.contains(re.escape(term_norm), na=False)
            cols = self._columns_for_term(term_norm)
            if cols:
                tag_mask = (df[cols] > 0).any(axis=1)
                title_mask = title_mask | tag_mask
            score += title_mask.astype(int)
        return score

    def _filter_by_category(self, df: pd.DataFrame, category: str, min_count: int = 80) -> pd.DataFrame:
        category = category.lower()
        category_col = "_model_pred" if "_model_pred" in df.columns else "glucose_category"
        if category_col not in df.columns:
            return df

        subset = df[df[category_col].astype(str).str.lower() == category]
        if len(subset) >= min_count:
            return subset

        order = ["low", "normal", "high"]
        if category not in order:
            return df
        idx = order.index(category)
        neighbors = [order[i] for i in (idx - 1, idx + 1) if 0 <= i < len(order)]
        expanded = df[df[category_col].astype(str).str.lower().isin([category] + neighbors)]
        return expanded if not expanded.empty else df

    @staticmethod
    def _filter_by_calories(df: pd.DataFrame, per_meal_target: float, min_count: int = 80) -> pd.DataFrame:
        tolerances = [0.2, 0.35, 0.5]
        for tol in tolerances:
            lower = max(0, per_meal_target * (1 - tol))
            upper = per_meal_target * (1 + tol)
            subset = df[(df["calories"] >= lower) & (df["calories"] <= upper)]
            if len(subset) >= min_count:
                return subset
        return df

    @staticmethod
    def _score_candidates(
        df: pd.DataFrame, per_meal_target: float, pref_score: pd.Series, glucose_category: str
    ) -> pd.DataFrame:
        rating = df.get("rating", pd.Series(0, index=df.index)).fillna(0)
        rating_max = max(float(rating.max()), 1.0)
        rating_score = rating / rating_max

        cal_diff = (df["calories"] - per_meal_target).abs()
        cal_score = 1 - (cal_diff / max(per_meal_target, 1.0)).clip(0, 1)

        pref_norm = pref_score / max(pref_score.max(), 1)

        protein = df.get("protein", pd.Series(0, index=df.index)).fillna(0)
        fat = df.get("fat", pd.Series(0, index=df.index)).fillna(0)
        sodium = df.get("sodium", pd.Series(0, index=df.index)).fillna(0)
        protein_score = protein / max(float(protein.max()), 1.0)
        fat_score = 1 - (fat / max(float(fat.max()), 1.0))
        sodium_score = 1 - (sodium / max(float(sodium.max()), 1.0))
        macro_score = (protein_score + fat_score + sodium_score) / 3

        if glucose_category == "high":
            score = (2.0 * pref_norm) + (1.2 * cal_score) + (1.0 * rating_score) + (0.8 * macro_score)
        elif glucose_category == "low":
            score = (1.5 * pref_norm) + (1.0 * cal_score) + (1.0 * rating_score) + (0.4 * macro_score)
        else:
            score = (1.8 * pref_norm) + (1.0 * cal_score) + (1.0 * rating_score) + (0.6 * macro_score)

        scored = df.copy()
        scored["_score"] = score
        return scored.sort_values("_score", ascending=False)

    @staticmethod
    def _build_week_plan(meals: Sequence[str]) -> List[Dict[str, List[str]]]:
        if not meals:
            return [{"day": day, "meals": []} for day in DAY_NAMES]
        plan: List[Dict[str, List[str]]] = []
        idx = 0
        for day in DAY_NAMES:
            day_meals: List[str] = []
            for _ in range(MEALS_PER_DAY):
                day_meals.append(meals[idx % len(meals)])
                idx += 1
            plan.append({"day": day, "meals": day_meals})
        return plan

    def _grocery_list(self, df: pd.DataFrame, meals: Sequence[str]) -> List[str]:
        if df.empty:
            return []
        meals_norm = [_normalize(item) for item in meals]
        items: List[str] = []
        for term in GROCERY_KEYWORDS:
            term_norm = _normalize(term)
            if any(term_norm in name for name in meals_norm):
                items.append(term)
                continue
            cols = self._columns_for_term(term_norm)
            if cols and (df[cols] > 0).any().any():
                items.append(term)
        if "vegetable" not in items:
            items.append("vegetable")
        if "whole grains" not in items and "whole" not in items:
            items.append("whole grains")
        if "olive oil" not in items:
            items.append("olive oil")
        return _dedupe_keep_order(items)

    @staticmethod
    def _trend_direction(values: Optional[Sequence[float]]) -> Optional[str]:
        if not values:
            return None
        if len(values) < 2:
            return None
        return "rising" if values[-1] > values[0] else "falling" if values[-1] < values[0] else "stable"

    def optimize(self, body: Any) -> Dict[str, Any]:
        if self.data is None:
            raise HTTPException(status_code=503, detail="Meal optimizer dataset not available.")

        preferences = [item.strip() for item in getattr(body, "preferences", []) if item and item.strip()]
        allergies = [item.strip() for item in getattr(body, "allergies", []) if item and item.strip()]
        avoid_items = [item.strip() for item in getattr(body, "avoid_items", []) if item and item.strip()]
        past_meals = [item.strip() for item in getattr(body, "past_meals", []) if item and item.strip()]
        feedback = getattr(body, "feedback", []) or []
        glucose_trend = getattr(body, "glucose_trend_mg_dl", None)

        rejected_meals = [item.meal for item in feedback if getattr(item, "action", "") == "reject"]
        accepted_meals = [item.meal for item in feedback if getattr(item, "action", "") == "accept"]

        exclude_terms = allergies + avoid_items + rejected_meals + past_meals
        df = self._exclude_terms(self.data, exclude_terms)

        glucose_category = str(getattr(body, "glucose_category", "normal")).lower()
        df = self._filter_by_category(df, glucose_category)

        per_meal_target = float(getattr(body, "daily_calorie_target", 1800)) / MEALS_PER_DAY
        df = self._filter_by_calories(df, per_meal_target)

        pref_score = self._preference_score(df, preferences + accepted_meals)
        scored = self._score_candidates(df, per_meal_target, pref_score, glucose_category)
        scored = scored.drop_duplicates(subset=["title"])

        if scored.empty:
            raise HTTPException(status_code=404, detail="No meal recommendations matched the filters.")

        meal_pool = scored["title"].head(25).tolist()
        meal_recommendations = scored["title"].head(5).tolist()
        weekly_plan = self._build_week_plan(meal_pool[: MEALS_PER_DAY * len(DAY_NAMES)])
        grocery_list = self._grocery_list(scored.head(25), meal_pool)

        recent_food = getattr(body, "recent_food", None)
        healthy_alternatives: List[str] = []
        updated_preferences: List[str] = []
        adjustments: List[str] = []

        if glucose_category == "high":
            healthy_alternatives.append("Swap refined grains for whole grains.")
            healthy_alternatives.append("Add a non-starchy vegetable side.")
            adjustments.append("Increase high-fiber meals, reduce high-carb options.")
        elif glucose_category == "low":
            healthy_alternatives.append("Add a balanced snack with protein.")
            adjustments.append("Slightly increase calorie density to avoid lows.")
        else:
            healthy_alternatives.append("Balance each meal with lean protein.")
            adjustments.append("Keep carbs distributed evenly across meals.")

        if recent_food and getattr(recent_food, "nutrients", None):
            nutrients = recent_food.nutrients
            if getattr(nutrients, "carbs_g", 0) >= 60:
                healthy_alternatives.append("Reduce refined carbs at the next meal.")
                updated_preferences.append("Lower refined carbs")
            if getattr(nutrients, "fiber_g", 0) <= 5:
                healthy_alternatives.append("Add a fiber-rich side (beans or greens).")
                updated_preferences.append("More fiber")
            if getattr(nutrients, "calories", 0) >= per_meal_target * 1.4:
                updated_preferences.append("Smaller portions at dinner")

        bmi_value = float(getattr(body, "bmi", 0.0))
        if bmi_value >= 30:
            updated_preferences.append("Lower calorie density")
            healthy_alternatives.append("Choose grilled or baked instead of fried.")
            adjustments.append("Reduce calorie density to support weight goals.")

        trend = self._trend_direction(glucose_trend)
        if trend == "rising":
            updated_preferences.append("More low-gi meals")
            adjustments.append("Tighten carb targets for rising glucose trend.")
        elif trend == "falling":
            adjustments.append("Maintain steady carb intake to prevent lows.")

        for meal in accepted_meals[:2]:
            updated_preferences.append(f"More meals like {meal}")
        for meal in rejected_meals[:2]:
            updated_preferences.append(f"Less {meal}")

        updated_preferences = _dedupe_keep_order(updated_preferences or preferences)
        healthy_alternatives = _dedupe_keep_order(healthy_alternatives)[:4]

        if preferences:
            adjustments.append("Boosted cuisine and tag preferences.")
        if rejected_meals:
            adjustments.append("Downranked rejected meals.")

        model_adjustments = "; ".join(_dedupe_keep_order(adjustments)) or "Balanced meals based on inputs."

        return {
            "meal_recommendations": meal_recommendations,
            "healthy_alternatives": healthy_alternatives,
            "weekly_meal_calendar": weekly_plan,
            "grocery_list": grocery_list,
            "updated_preferences": updated_preferences,
            "model_adjustments": model_adjustments,
        }
