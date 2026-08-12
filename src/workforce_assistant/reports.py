"""Generate the required research paper and executive summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _metrics_markdown(metrics: pd.DataFrame) -> str:
    header = "| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n|---|---:|---:|---:|---:|---:|"
    rows = [
        "| {Model} | {Accuracy:.3f} | {Precision:.3f} | {Recall:.3f} | {F1Score:.3f} | {ROCAUC:.3f} |".format(
            **row
        )
        for row in metrics.to_dict("records")
    ]
    return "\n".join([header, *rows])


def _attrition_breakdown(data: pd.DataFrame, column: str) -> str:
    """Return a compact EDA table for one categorical field."""
    summary = (
        data.groupby(column, observed=True)["Attrition"]
        .agg(Employees="size", AttritionCases="sum", AttritionRate="mean")
        .sort_values("AttritionRate", ascending=False)
    )
    rows = [
        f"| {category} | {int(row.Employees):,} | {int(row.AttritionCases):,} | {row.AttritionRate:.1%} |"
        for category, row in summary.iterrows()
    ]
    header = "| Category | Employees | Attrition cases | Attrition rate |\n|---|---:|---:|---:|"
    return "\n".join([header, *rows])


def build_research_paper(data: pd.DataFrame, bundle: dict[str, Any]) -> str:
    scores = bundle["risk_scores"]
    distribution = scores["RiskCategory"].value_counts()
    top_features = bundle["feature_importance"].head(10)
    feature_lines = "\n".join(
        f"- {row.Feature}: {row.Importance:.1%}"
        for row in top_features.itertuples(index=False)
    )
    return f"""# Machine Learning-Based Employee Attrition Prediction and Risk Scoring System

## Abstract

This project uses the supplied employee dataset to estimate attrition probability, group employees into practical risk categories, and present explainable findings through a Streamlit dashboard. It is an educational decision-support prototype and must not be used as the sole basis for employment decisions.

## Dataset overview

- Records: {len(data):,}
- Input fields: 31
- Historical attrition cases: {int(data['Attrition'].sum()):,}
- Historical attrition rate: {data['Attrition'].mean():.1%}
- Missing cells: {int(data.isna().sum().sum()):,}
- Duplicate rows: {int(data.drop(columns=['EmployeeID', 'AttritionStatus']).duplicated().sum()):,}

## Exploratory data analysis

### Overtime

{_attrition_breakdown(data, 'OverTime')}

### Business travel

{_attrition_breakdown(data, 'BusinessTravel')}

### Department

{_attrition_breakdown(data, 'Department')}

### Job role

{_attrition_breakdown(data, 'JobRole')}

The descriptive analysis shows higher historical attrition among employees working overtime, frequent travelers, and selected job roles. These are associations rather than proof of cause.

## Methodology

Categorical variables are one-hot encoded, numerical fields are median-imputed and standardized, and the target is split into 80% training and 20% testing data using stratification. Class imbalance is handled with class weights for Logistic Regression and Random Forest and balanced sample weights for Gradient Boosting.

The required engineered features are:

1. Income-to-experience ratio.
2. Promotion-delay indicator.
3. Engagement composite score.
4. Workload-stress flag.

Gender and marital status are excluded from model training. The remaining fields and engineered features are evaluated using Logistic Regression, Random Forest, and Gradient Boosting.

## Model evaluation

{_metrics_markdown(bundle['metrics'])}

The selected model is **{bundle['model_name']}**, chosen by test-set ROC-AUC with F1-Score as a secondary comparison.

## Risk-scoring framework

- Low risk: probability below 30%.
- Medium risk: probability from 30% through 60%.
- High risk: probability above 60%.

Current distribution:

- Low: {int(distribution.get('Low', 0)):,}
- Medium: {int(distribution.get('Medium', 0)):,}
- High: {int(distribution.get('High', 0)):,}

## Model explainability

Top model features:

{feature_lines}

The dashboard also provides employee-level reason codes based on observable conditions such as overtime, low satisfaction, promotion delay, and frequent travel. These reason codes aid interpretation but are not causal explanations.

## Recommendations

1. Use the dashboard to prioritize voluntary retention conversations at group level.
2. Review recurring overtime and work-life-balance concerns.
3. Investigate department and role patterns with employee feedback.
4. Reassess thresholds and model performance before any real deployment.
5. Require human review and fairness monitoring for every proposed intervention.

## Limitations and responsible use

The dataset is small and may not represent a current workforce. Predictions describe statistical patterns, not certainty. Risk scores must never trigger automatic adverse action. This project requires validation, privacy review, fairness testing, and governance before real-world use.
"""


def build_executive_summary(data: pd.DataFrame, bundle: dict[str, Any]) -> str:
    scores = bundle["risk_scores"]
    best = bundle["metrics"].set_index("Model").loc[bundle["model_name"]]
    high_risk = int(scores["RiskCategory"].eq("High").sum())
    return f"""# Executive Summary for Government Stakeholders

## Purpose

The project demonstrates how machine learning can support earlier, more consistent workforce-retention planning using the supplied educational employee dataset.

## Headline results

- Employees analyzed: {len(data):,}
- Historical attrition rate: {data['Attrition'].mean():.1%}
- Selected model: {bundle['model_name']}
- Test ROC-AUC: {best['ROCAUC']:.3f}
- Test recall: {best['Recall']:.3f}
- Employees above the default 60% high-risk threshold: {high_risk:,}

## Operational value

The dashboard provides group-level risk distribution, department and role summaries, individual profiles, contributing indicators, adjustable thresholds, and what-if exploration. It can help authorized users identify where supportive retention conversations may be most useful.

## Safeguards

- Gender and marital status are excluded from the model.
- Predictions are advisory and require human review.
- The system does not automate employment decisions.
- Scores must be protected as sensitive workforce information.
- Real deployment requires privacy, legal, fairness, and model-governance review.

## Recommendation

Treat this project as an educational prototype. A controlled pilot should validate data quality, predictive performance, fairness, and employee-impact safeguards before any operational use.
"""


def write_reports(
    data: pd.DataFrame, bundle: dict[str, Any], output_directory: str | Path
) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "research_paper.md").write_text(
        build_research_paper(data, bundle), encoding="utf-8"
    )
    (output_path / "executive_summary.md").write_text(
        build_executive_summary(data, bundle), encoding="utf-8"
    )
