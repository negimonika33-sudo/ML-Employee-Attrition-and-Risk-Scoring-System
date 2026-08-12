"""Dataset loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "Age",
    "Attrition",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
}


def validate_dataset(data: pd.DataFrame) -> dict[str, Any]:
    """Validate the fields needed by the dashboard and return quality details."""
    if data.empty:
        raise ValueError("The dataset is empty.")

    missing_columns = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing_columns:
        raise ValueError(
            "The dataset is missing required columns: " + ", ".join(missing_columns)
        )

    if data["Attrition"].isna().any():
        raise ValueError("Attrition contains missing values.")

    attrition_values = set(pd.to_numeric(data["Attrition"], errors="coerce").dropna())
    if not attrition_values.issubset({0, 1}):
        raise ValueError("Attrition must contain only 0 and 1 values.")

    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "missing_cells": int(data.isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()),
    }


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, validate, and prepare a workforce CSV file."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = pd.read_csv(dataset_path)
    quality = validate_dataset(data)
    prepared = data.copy()
    prepared["Attrition"] = pd.to_numeric(
        prepared["Attrition"], errors="raise"
    ).astype(int)
    prepared["AttritionStatus"] = prepared["Attrition"].map(
        {0: "Stayed", 1: "Left"}
    )
    prepared.insert(
        0,
        "EmployeeID",
        [f"EMP-{number:04d}" for number in range(1, len(prepared) + 1)],
    )
    return prepared, quality

