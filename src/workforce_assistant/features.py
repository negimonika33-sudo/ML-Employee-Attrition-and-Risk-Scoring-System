"""Feature engineering used consistently for training and prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


ENGINEERED_FEATURES = [
    "IncomeToExperienceRatio",
    "PromotionDelayIndicator",
    "EngagementCompositeScore",
    "WorkloadStressFlag",
]

# These personal attributes are kept in the source data for auditing but are not
# used by the model to make employment-risk predictions.
EXCLUDED_MODEL_COLUMNS = {
    "Attrition",
    "AttritionStatus",
    "EmployeeID",
    "Gender",
    "MaritalStatus",
}


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the four features required by the project specification."""
    featured = data.copy()
    experience_denominator = featured["TotalWorkingYears"].astype(float) + 1.0
    company_years_denominator = featured["YearsAtCompany"].astype(float) + 1.0

    featured["IncomeToExperienceRatio"] = (
        featured["MonthlyIncome"].astype(float) / experience_denominator
    )
    featured["PromotionDelayIndicator"] = (
        featured["YearsSinceLastPromotion"].astype(float)
        / company_years_denominator
    )
    featured["EngagementCompositeScore"] = featured[
        [
            "EnvironmentSatisfaction",
            "JobInvolvement",
            "JobSatisfaction",
            "RelationshipSatisfaction",
            "WorkLifeBalance",
        ]
    ].mean(axis=1)
    featured["WorkloadStressFlag"] = (
        featured["OverTime"].eq("Yes")
        & (
            featured["WorkLifeBalance"].le(2)
            | featured["JobSatisfaction"].le(2)
        )
    ).astype(int)

    featured.replace([np.inf, -np.inf], np.nan, inplace=True)
    return featured


def model_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return the model-ready input fields without target or protected columns."""
    featured = add_engineered_features(data)
    return featured.drop(
        columns=[column for column in EXCLUDED_MODEL_COLUMNS if column in featured],
        errors="ignore",
    )

