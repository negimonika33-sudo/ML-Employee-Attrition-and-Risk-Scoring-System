# Machine Learning-Based Employee Attrition Prediction and Risk Scoring System

## Abstract

This project uses the supplied employee dataset to estimate attrition probability, group employees into practical risk categories, and present explainable findings through a Streamlit dashboard. It is an educational decision-support prototype and must not be used as the sole basis for employment decisions.

## Dataset overview

- Records: 1,470
- Input fields: 31
- Historical attrition cases: 237
- Historical attrition rate: 16.1%
- Missing cells: 0
- Duplicate rows: 0

## Exploratory data analysis

### Overtime

| Category | Employees | Attrition cases | Attrition rate |
|---|---:|---:|---:|
| Yes | 416 | 127 | 30.5% |
| No | 1,054 | 110 | 10.4% |

### Business travel

| Category | Employees | Attrition cases | Attrition rate |
|---|---:|---:|---:|
| Travel_Frequently | 277 | 69 | 24.9% |
| Travel_Rarely | 1,043 | 156 | 15.0% |
| Non-Travel | 150 | 12 | 8.0% |

### Department

| Category | Employees | Attrition cases | Attrition rate |
|---|---:|---:|---:|
| Sales | 446 | 92 | 20.6% |
| Human Resources | 63 | 12 | 19.0% |
| Research & Development | 961 | 133 | 13.8% |

### Job role

| Category | Employees | Attrition cases | Attrition rate |
|---|---:|---:|---:|
| Sales Representative | 83 | 33 | 39.8% |
| Laboratory Technician | 259 | 62 | 23.9% |
| Human Resources | 52 | 12 | 23.1% |
| Sales Executive | 326 | 57 | 17.5% |
| Research Scientist | 292 | 47 | 16.1% |
| Manufacturing Director | 145 | 10 | 6.9% |
| Healthcare Representative | 131 | 9 | 6.9% |
| Manager | 102 | 5 | 4.9% |
| Research Director | 80 | 2 | 2.5% |

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

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.755 | 0.346 | 0.596 | 0.438 | 0.783 |
| Random Forest | 0.837 | 0.488 | 0.426 | 0.455 | 0.768 |
| Gradient Boosting | 0.776 | 0.377 | 0.617 | 0.468 | 0.765 |

The selected model is **Logistic Regression**, chosen by test-set ROC-AUC with F1-Score as a secondary comparison.

## Risk-scoring framework

- Low risk: probability below 30%.
- Medium risk: probability from 30% through 60%.
- High risk: probability above 60%.

Current distribution:

- Low: 744
- Medium: 395
- High: 331

## Model explainability

Top model features:

- JobRole_Research Director: 7.7%
- JobRole_Laboratory Technician: 6.4%
- JobRole_Sales Representative: 5.8%
- BusinessTravel_Non-Travel: 5.1%
- OverTime_No: 4.7%
- EducationField_Other: 4.6%
- BusinessTravel_Travel_Frequently: 4.1%
- OverTime_Yes: 4.0%
- EducationField_Human Resources: 3.7%
- JobRole_Healthcare Representative: 3.3%

The dashboard also provides employee-level reason codes based on observable conditions such as overtime, low satisfaction, promotion delay, and frequent travel. These reason codes aid interpretation but are not causal explanations.

## Recommendations

1. Use the dashboard to prioritize voluntary retention conversations at group level.
2. Review recurring overtime and work-life-balance concerns.
3. Investigate department and role patterns with employee feedback.
4. Reassess thresholds and model performance before any real deployment.
5. Require human review and fairness monitoring for every proposed intervention.

## Limitations and responsible use

The dataset is small and may not represent a current workforce. Predictions describe statistical patterns, not certainty. Risk scores must never trigger automatic adverse action. This project requires validation, privacy review, fairness testing, and governance before real-world use.
