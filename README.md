# Employee Attrition Prediction and Risk Scoring System

An internship-level machine-learning project that trains three classification models,
assigns explainable attrition-risk scores, and presents the results in Streamlit.

New to coding or machine learning? Start with
[`BEGINNER_GUIDE.md`](BEGINNER_GUIDE.md) for a plain-language explanation and a short
demonstration script.

## What the project asked for

The assignment required a machine-learning system that uses the supplied employee CSV to
predict employee attrition. It asked for data preprocessing, four engineered features,
class-imbalance handling, a stratified train/test split, and a comparison of Logistic
Regression, Random Forest, and Gradient Boosting or XGBoost using five evaluation metrics.

It also required a probability and Low/Medium/High risk category for every employee,
feature importance and individual reasons, and an interactive Streamlit dashboard with
filters, employee profiles, department and role analysis, adjustable thresholds, and
what-if predictions. The final submission also needed a research paper containing EDA,
insights, and recommendations, plus an executive summary for government stakeholders.

## Requirements covered

- Categorical encoding, numerical scaling, and missing-value handling.
- Class-imbalance handling and a stratified 80/20 train/test split.
- Income-to-experience, promotion-delay, engagement, and workload-stress features.
- Logistic Regression, Random Forest, and Gradient Boosting models.
- Accuracy, precision, recall, F1-Score, and ROC-AUC comparison.
- Employee attrition probability and Low/Medium/High risk categories.
- Feature importance and individual reason codes.
- Risk distribution, employee profile, department/role aggregation, and what-if modules.
- Department/role filters, threshold sliders, and employee selectors.
- Research paper and executive summary for government stakeholders.

The project deliberately does not use deep learning, external APIs, databases, cloud
services, XGBoost, or SHAP. Everything runs locally in memory.

## Project structure

```text
workforce-assistant/
├── BEGINNER_GUIDE.md               Plain-language project explanation
├── app.py                          Streamlit dashboard
├── train_model.py                  Reproducible training entry point
├── run_app.py                      Dashboard launcher
├── data/                           Supplied CSV
├── reference/                      Authoritative requirements document
├── src/workforce_assistant/        Data, features, models, and reports
├── artifacts/                      Model, metrics, importance, and risk scores
├── reports/                        Research paper and executive summary
├── tests/                          Automated project checks
└── requirements.txt                Python libraries
```

## How it works

The training script loads and validates the employee CSV, prepares the data, and creates
four useful features related to experience, promotion delay, engagement, and workload.
It then trains three machine-learning models and compares them using accuracy, precision,
recall, F1-Score, and ROC-AUC.

The best-performing model calculates an attrition probability for each employee. The
probability is converted into Low, Medium, or High risk and saved with simple reason codes.
The Streamlit app reads these results and displays dashboards, employee profiles,
department and role summaries, model explanations, and what-if predictions. Everything
runs locally; no API, database, cloud service, or internet connection is required.

## Setup

```bash
cd CBABU/workforce-assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train the models

```bash
python train_model.py
```

Training takes only a few seconds and regenerates everything under `artifacts/` and
`reports/`. The supplied trained artifacts are already included, but this command proves
that the project is reproducible.

## Run the dashboard

```bash
python run_app.py
```

Open the local address printed by Streamlit, normally `http://localhost:8501`. Stop the
dashboard with `Ctrl+C`.

## Run the tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Five-minute explanation

1. Load and validate the 1,470 employee records.
2. Create four business-friendly features from existing fields.
3. Split the data into stratified training and testing groups.
4. Encode categories, scale numbers, and balance the minority attrition class.
5. Train Logistic Regression, Random Forest, and Gradient Boosting.
6. Compare all five required test metrics and choose the highest ROC-AUC model.
7. Convert probabilities into Low, Medium, and High risk categories.
8. Display group and employee results with reason codes and what-if controls.

## Responsible-use boundary

This is an educational prototype. Gender and marital status are excluded from training.
Predictions are uncertain statistical estimates and must not automate employment decisions.
Any real use would require privacy, legal, fairness, security, and model-governance review.
