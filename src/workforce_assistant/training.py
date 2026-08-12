"""Model training, evaluation, explainability, and employee risk scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .features import ENGINEERED_FEATURES, model_features


DEFAULT_LOW_THRESHOLD = 0.30
DEFAULT_HIGH_THRESHOLD = 0.60
RANDOM_STATE = 42


def categorize_probability(
    probability: float, low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD
) -> str:
    """Convert one probability into the required low/medium/high categories."""
    if probability < low_threshold:
        return "Low"
    if probability <= high_threshold:
        return "Medium"
    return "High"


def _build_preprocessor(inputs: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = inputs.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = inputs.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def _candidate_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def _evaluate_model(
    model: Pipeline, test_inputs: pd.DataFrame, test_target: pd.Series
) -> dict[str, float]:
    probability = model.predict_proba(test_inputs)[:, 1]
    prediction = (probability >= 0.50).astype(int)
    return {
        "Accuracy": accuracy_score(test_target, prediction),
        "Precision": precision_score(test_target, prediction, zero_division=0),
        "Recall": recall_score(test_target, prediction, zero_division=0),
        "F1Score": f1_score(test_target, prediction, zero_division=0),
        "ROCAUC": roc_auc_score(test_target, probability),
    }


def _feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = np.abs(estimator.coef_[0])
    else:
        importance = np.zeros(len(feature_names))

    cleaned_names = [
        name.replace("numeric__", "").replace("categorical__", "")
        for name in feature_names
    ]
    result = pd.DataFrame(
        {"Feature": cleaned_names, "Importance": np.asarray(importance, dtype=float)}
    )
    total = result["Importance"].sum()
    if total > 0:
        result["Importance"] = result["Importance"] / total
    return result.sort_values("Importance", ascending=False).reset_index(drop=True)


def employee_reason_codes(row: pd.Series, income_median: float) -> list[str]:
    """Return understandable observed indicators for an employee profile."""
    reasons: list[str] = []
    if row["OverTime"] == "Yes":
        reasons.append("Works overtime")
    if int(row["JobSatisfaction"]) <= 2:
        reasons.append("Low job satisfaction")
    if int(row["EnvironmentSatisfaction"]) <= 2:
        reasons.append("Low environment satisfaction")
    if int(row["WorkLifeBalance"]) <= 2:
        reasons.append("Low work-life balance")
    if int(row["YearsSinceLastPromotion"]) >= 4:
        reasons.append("Long period since promotion")
    if row["BusinessTravel"] == "Travel_Frequently":
        reasons.append("Frequent business travel")
    if int(row["DistanceFromHome"]) >= 15:
        reasons.append("Long distance from home")
    if int(row["StockOptionLevel"]) == 0:
        reasons.append("No stock options")
    if float(row["MonthlyIncome"]) < income_median:
        reasons.append("Income below workforce median")
    return reasons[:4] or ["No prominent rule-based indicators"]


def score_employees(
    data: pd.DataFrame,
    bundle: dict[str, Any],
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> pd.DataFrame:
    """Score every employee and attach risk categories and reason codes."""
    probabilities = bundle["model"].predict_proba(model_features(data))[:, 1]
    scored = data.copy()
    scored["AttritionProbability"] = probabilities
    scored["RiskCategory"] = [
        categorize_probability(value, low_threshold, high_threshold)
        for value in probabilities
    ]
    income_median = float(bundle["income_median"])
    scored["ReasonCodes"] = [
        "; ".join(employee_reason_codes(row, income_median))
        for _, row in scored.iterrows()
    ]
    return scored


def train_models(data: pd.DataFrame) -> dict[str, Any]:
    """Train all required models and return the selected model bundle."""
    inputs = model_features(data)
    target = data["Attrition"].astype(int)
    train_inputs, test_inputs, train_target, test_target = train_test_split(
        inputs,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    preprocessor = _build_preprocessor(train_inputs)
    candidates = _candidate_models(preprocessor)
    trained_models: dict[str, Pipeline] = {}
    metric_rows: list[dict[str, Any]] = []
    balanced_weights = compute_sample_weight("balanced", train_target)

    for name, model in candidates.items():
        if name == "Gradient Boosting":
            model.fit(train_inputs, train_target, model__sample_weight=balanced_weights)
        else:
            model.fit(train_inputs, train_target)
        trained_models[name] = model
        metric_rows.append({"Model": name, **_evaluate_model(model, test_inputs, test_target)})

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["ROCAUC", "F1Score"], ascending=False
    ).reset_index(drop=True)
    best_model_name = str(metrics.iloc[0]["Model"])
    best_model = trained_models[best_model_name]

    bundle: dict[str, Any] = {
        "model": best_model,
        "model_name": best_model_name,
        "metrics": metrics,
        "feature_importance": _feature_importance(best_model),
        "income_median": float(data["MonthlyIncome"].median()),
        "training_rows": int(len(train_inputs)),
        "testing_rows": int(len(test_inputs)),
        "positive_training_rows": int(train_target.sum()),
        "engineered_features": ENGINEERED_FEATURES,
        "excluded_sensitive_features": ["Gender", "MaritalStatus"],
        "random_state": RANDOM_STATE,
    }
    bundle["risk_scores"] = score_employees(data, bundle)
    return bundle


def save_model_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(Path(path))

