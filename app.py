"""Streamlit application for employee attrition prediction and risk scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from workforce_assistant import (  # noqa: E402
    categorize_probability,
    load_dataset,
    load_model_bundle,
    model_features,
)


DATASET_PATH = PROJECT_DIR / "data" / "Palo Alto Networks.csv"
MODEL_PATH = PROJECT_DIR / "artifacts" / "model_bundle.joblib"

st.set_page_config(
    page_title="Employee Attrition Risk System",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.7rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: #f5f7fb;
        border: 1px solid #e1e7ef;
        border-radius: 12px;
        padding: 15px;
    }
    .risk-note {
        background: #eef6f5;
        border-left: 4px solid #16796f;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 8px 0 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def read_model(path: str):
    return load_model_bundle(path)


@st.cache_data
def read_data(path: str):
    return load_dataset(path)


def risk_distribution(data: pd.DataFrame) -> pd.DataFrame:
    counts = data["RiskCategory"].value_counts().reindex(
        ["Low", "Medium", "High"], fill_value=0
    )
    return counts.rename_axis("Risk category").reset_index(name="Employees")


def group_risk(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        data.groupby(column)
        .agg(
            Employees=("EmployeeID", "size"),
            AverageRisk=("AttritionProbability", "mean"),
            HighRisk=("RiskCategory", lambda values: int(values.eq("High").sum())),
        )
        .reset_index()
        .sort_values(["AverageRisk", "Employees"], ascending=[False, False])
    )


st.title("Machine Learning Employee Attrition Risk System")
st.caption("Predictive workforce analytics with explainable employee risk scoring")

if not MODEL_PATH.exists():
    st.error("The trained model is missing. Run `python train_model.py` once, then restart the app.")
    st.stop()

try:
    workforce, quality = read_data(str(DATASET_PATH))
    bundle = read_model(str(MODEL_PATH))
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"The project data or model could not be loaded: {exc}")
    st.stop()

scored = bundle["risk_scores"].copy()

with st.sidebar:
    st.header("Dashboard controls")
    selected_departments = st.multiselect(
        "Department", sorted(scored["Department"].unique())
    )
    selected_roles = st.multiselect("Job role", sorted(scored["JobRole"].unique()))
    st.subheader("Risk thresholds")
    low_threshold = st.slider(
        "Low/medium threshold", 0.10, 0.45, 0.30, 0.05,
        help="Probabilities below this value are Low risk.",
    )
    high_threshold = st.slider(
        "Medium/high threshold", low_threshold + 0.05, 0.90, 0.60, 0.05,
        help="Probabilities above this value are High risk.",
    )
    st.divider()
    st.caption(f"Selected model: {bundle['model_name']}")
    st.caption(f"Dataset: {quality['rows']:,} rows · {quality['missing_cells']} missing cells")

scored["RiskCategory"] = [
    categorize_probability(value, low_threshold, high_threshold)
    for value in scored["AttritionProbability"]
]
filtered = scored.copy()
if selected_departments:
    filtered = filtered[filtered["Department"].isin(selected_departments)]
if selected_roles:
    filtered = filtered[filtered["JobRole"].isin(selected_roles)]

if filtered.empty:
    st.warning("No employees match the selected filters. Remove a filter to continue.")
    st.stop()

high_risk_count = int(filtered["RiskCategory"].eq("High").sum())
st.markdown(
    f'<div class="risk-note"><strong>Current view:</strong> {len(filtered):,} employees, '
    f'{high_risk_count:,} above the {high_threshold:.0%} high-risk threshold, and an average '
    f'predicted attrition probability of {filtered["AttritionProbability"].mean():.1%}.</div>',
    unsafe_allow_html=True,
)

metric_columns = st.columns(4)
metric_columns[0].metric("Employees", f"{len(filtered):,}")
metric_columns[1].metric("High-risk employees", f"{high_risk_count:,}")
metric_columns[2].metric(
    "Average predicted risk", f"{filtered['AttritionProbability'].mean():.1%}"
)
metric_columns[3].metric("Historical attrition", f"{filtered['Attrition'].mean():.1%}")

risk_tab, employee_tab, department_tab, explain_tab, what_if_tab, submission_tab = st.tabs(
    [
        "Risk dashboard",
        "Employee profile",
        "Department view",
        "Explainability",
        "What-if explorer",
        "Submission files",
    ]
)

with risk_tab:
    left, right = st.columns([1, 1.5])
    with left:
        st.subheader("Overall risk distribution")
        distribution = risk_distribution(filtered)
        st.bar_chart(distribution.set_index("Risk category"), color="#16796f")
        st.dataframe(distribution, hide_index=True, width="stretch")
    with right:
        st.subheader("Highest predicted-risk employees")
        risk_table = filtered[
            [
                "EmployeeID",
                "Department",
                "JobRole",
                "AttritionProbability",
                "RiskCategory",
                "ReasonCodes",
            ]
        ].sort_values("AttritionProbability", ascending=False)
        st.dataframe(
            risk_table.style.format({"AttritionProbability": "{:.1%}"}),
            hide_index=True,
            width="stretch",
            height=430,
        )
        st.download_button(
            "Download filtered risk scores",
            risk_table.to_csv(index=False).encode("utf-8"),
            "employee_risk_scores.csv",
            "text/csv",
        )

with employee_tab:
    st.subheader("Employee risk profile")
    employee_id = st.selectbox("Employee ID", filtered["EmployeeID"].tolist())
    employee = filtered.loc[filtered["EmployeeID"].eq(employee_id)].iloc[0]
    profile_columns = st.columns(4)
    profile_columns[0].metric("Attrition probability", f"{employee['AttritionProbability']:.1%}")
    profile_columns[1].metric("Risk category", employee["RiskCategory"])
    profile_columns[2].metric("Department", employee["Department"])
    profile_columns[3].metric("Job role", employee["JobRole"])
    st.markdown("#### Key contributing indicators")
    for reason in str(employee["ReasonCodes"]).split("; "):
        st.markdown(f"- {reason}")
    st.caption(
        "Reason codes are observable profile indicators, not proof that a factor caused attrition."
    )
    detail_fields = [
        "Age",
        "BusinessTravel",
        "DistanceFromHome",
        "EnvironmentSatisfaction",
        "JobInvolvement",
        "JobSatisfaction",
        "MonthlyIncome",
        "OverTime",
        "StockOptionLevel",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsSinceLastPromotion",
    ]
    details = pd.DataFrame(
        {
            "Field": detail_fields,
            "Value": [str(employee[field]) for field in detail_fields],
        }
    )
    st.dataframe(details, hide_index=True, width="stretch")

with department_tab:
    st.subheader("Aggregated risk by department and role")
    department_columns = st.columns(2)
    with department_columns[0]:
        department_summary = group_risk(filtered, "Department")
        st.caption("Department-level risk")
        st.dataframe(
            department_summary.style.format({"AverageRisk": "{:.1%}"}),
            hide_index=True,
            width="stretch",
        )
        st.bar_chart(
            department_summary.set_index("Department")[["AverageRisk"]],
            horizontal=True,
            color="#375a7f",
        )
    with department_columns[1]:
        role_summary = group_risk(filtered, "JobRole")
        st.caption("Role-level risk")
        st.dataframe(
            role_summary.style.format({"AverageRisk": "{:.1%}"}),
            hide_index=True,
            width="stretch",
        )
        st.bar_chart(
            role_summary.set_index("JobRole")[["AverageRisk"]],
            horizontal=True,
            color="#16796f",
        )

with explain_tab:
    st.subheader("Model evaluation")
    metrics = bundle["metrics"].copy()
    st.dataframe(
        metrics.style.format(
            {
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1Score": "{:.3f}",
                "ROCAUC": "{:.3f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"{bundle['model_name']} was selected using test-set ROC-AUC, with F1-Score as a secondary comparison."
    )
    st.subheader("Feature importance")
    top_importance = bundle["feature_importance"].head(15)
    st.bar_chart(
        top_importance.set_index("Feature")[["Importance"]],
        horizontal=True,
        color="#7a5195",
    )
    st.dataframe(
        top_importance.style.format({"Importance": "{:.1%}"}),
        hide_index=True,
        width="stretch",
    )
    st.info(
        "Gender and marital status are excluded from model training. Importance indicates model influence, not causation."
    )

with what_if_tab:
    st.subheader("What-if scenario exploration")
    scenario_employee_id = st.selectbox(
        "Starting employee ID", filtered["EmployeeID"].tolist(), key="scenario_employee"
    )
    source_row = workforce.loc[workforce["EmployeeID"].eq(scenario_employee_id)].iloc[0]
    control_columns = st.columns(3)
    with control_columns[0]:
        scenario_overtime = st.selectbox(
            "Overtime", ["No", "Yes"],
            index=0 if source_row["OverTime"] == "No" else 1,
            key="scenario_overtime",
        )
        scenario_job_satisfaction = st.slider(
            "Job satisfaction", 1, 4, int(source_row["JobSatisfaction"]),
            key="scenario_job_satisfaction",
        )
        scenario_environment = st.slider(
            "Environment satisfaction", 1, 4,
            int(source_row["EnvironmentSatisfaction"]), key="scenario_environment",
        )
    with control_columns[1]:
        scenario_balance = st.slider(
            "Work-life balance", 1, 4, int(source_row["WorkLifeBalance"]),
            key="scenario_balance",
        )
        scenario_promotion = st.slider(
            "Years since promotion", 0,
            int(workforce["YearsSinceLastPromotion"].max()),
            int(source_row["YearsSinceLastPromotion"]), key="scenario_promotion",
        )
        scenario_distance = st.slider(
            "Distance from home", int(workforce["DistanceFromHome"].min()),
            int(workforce["DistanceFromHome"].max()),
            int(source_row["DistanceFromHome"]), key="scenario_distance",
        )
    with control_columns[2]:
        scenario_income = st.slider(
            "Monthly income", int(workforce["MonthlyIncome"].min()),
            int(workforce["MonthlyIncome"].max()), int(source_row["MonthlyIncome"]),
            step=100, key="scenario_income",
        )
        travel_options = sorted(workforce["BusinessTravel"].unique())
        scenario_travel = st.selectbox(
            "Business travel", travel_options,
            index=travel_options.index(source_row["BusinessTravel"]), key="scenario_travel",
        )

    scenario = workforce.loc[workforce["EmployeeID"].eq(scenario_employee_id)].copy()
    scenario.loc[:, "OverTime"] = scenario_overtime
    scenario.loc[:, "JobSatisfaction"] = scenario_job_satisfaction
    scenario.loc[:, "EnvironmentSatisfaction"] = scenario_environment
    scenario.loc[:, "WorkLifeBalance"] = scenario_balance
    scenario.loc[:, "YearsSinceLastPromotion"] = scenario_promotion
    scenario.loc[:, "DistanceFromHome"] = scenario_distance
    scenario.loc[:, "MonthlyIncome"] = scenario_income
    scenario.loc[:, "BusinessTravel"] = scenario_travel
    scenario_probability = float(
        bundle["model"].predict_proba(model_features(scenario))[0, 1]
    )
    original_probability = float(
        scored.loc[scored["EmployeeID"].eq(scenario_employee_id), "AttritionProbability"].iloc[0]
    )
    scenario_category = categorize_probability(
        scenario_probability, low_threshold, high_threshold
    )
    result_columns = st.columns(3)
    result_columns[0].metric("Original probability", f"{original_probability:.1%}")
    result_columns[1].metric(
        "Scenario probability",
        f"{scenario_probability:.1%}",
        delta=f"{scenario_probability - original_probability:+.1%}",
        delta_color="inverse",
    )
    result_columns[2].metric("Scenario category", scenario_category)
    st.caption(
        "What-if results illustrate model behavior. They do not prescribe employment actions or prove causation."
    )

with submission_tab:
    st.subheader("Required submission deliverables")
    research_path = PROJECT_DIR / "reports" / "research_paper.md"
    summary_path = PROJECT_DIR / "reports" / "executive_summary.md"
    if research_path.exists() and summary_path.exists():
        research_text = research_path.read_text(encoding="utf-8")
        summary_text = summary_path.read_text(encoding="utf-8")
        st.download_button(
            "Download research paper",
            research_text.encode("utf-8"),
            "research_paper.md",
            "text/markdown",
        )
        st.download_button(
            "Download executive summary",
            summary_text.encode("utf-8"),
            "executive_summary.md",
            "text/markdown",
        )
        with st.expander("Research paper preview"):
            st.markdown(research_text)
        with st.expander("Executive summary preview"):
            st.markdown(summary_text)
    else:
        st.warning("Submission reports are missing. Run `python train_model.py`.")

    st.subheader("Method and responsible-use notes")
    st.markdown(
        f"""
        - Training rows: {bundle['training_rows']:,}; testing rows: {bundle['testing_rows']:,}.
        - Train/test splitting is stratified and reproducible with random state {bundle['random_state']}.
        - Numerical fields are median-imputed and standardized.
        - Categorical fields are one-hot encoded.
        - Class imbalance is handled with class or sample weights.
        - Engineered features: {', '.join(bundle['engineered_features'])}.
        - Predictions are educational estimates, not certain outcomes.
        - No automated adverse employment decision should be based on this dashboard.
        """
    )
