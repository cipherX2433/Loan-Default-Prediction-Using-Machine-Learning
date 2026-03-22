# 🏦 Loan Approval Prediction — ML Research

A comprehensive machine learning pipeline for predicting loan approval outcomes using classical and gradient-boosted classifiers. The project covers the full data science lifecycle: exploratory data analysis, feature engineering, class imbalance handling, hyperparameter tuning, model comparison, and rigorous statistical evaluation.

---

## 📌 Problem Statement

Given an applicant's personal, financial, and credit profile, predict whether a loan will be **approved (1)** or **rejected (0)**. This is a binary classification problem with real-world class imbalance, where false negatives (missed defaults) carry significant business risk.

---

## 📁 Dataset

**File:** `loan_data.csv`

**Features:**

| Type | Features |
|---|---|
| Numerical | `person_age`, `person_income`, `person_emp_exp`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`, `credit_score` |
| Categorical | `person_gender`, `person_education`, `person_home_ownership`, `loan_intent`, `previous_loan_defaults_on_file` |
| Target | `loan_status` (0 = Rejected, 1 = Approved) |

---

## 🔬 Methodology

### 1. Data Preprocessing
- Stratified train/test split (75/25)
- Outlier removal: filtered applicants with `person_age > 80`
- IQR-based clipping on skewed numeric columns (`person_income`, `loan_amnt`, `loan_percent_income`, `person_emp_exp`)
- Standard scaling of all numerical features

### 2. Exploratory Data Analysis
- Loan approval class distribution
- Box plots and KDE plots per feature, split by loan status
- Count plots for categorical features vs. target
- Pearson correlation heatmap

### 3. Feature Engineering

| New Feature | Description |
|---|---|
| `person_education_ord` | Ordinal encoding of education level (High School → Doctorate) |
| `is_renter` | Binary flag: 1 if home ownership is RENT |
| `is_high_risk_intent` | Binary flag: 1 if loan intent is DEBTCONSOLIDATION or MEDICAL |
| `interest_to_income_ratio` | `(loan_amnt × loan_int_rate) / person_income`, clipped at 2.0 |

### 4. Feature Selection
- **VIF analysis** to detect multicollinearity; `person_age` dropped due to high VIF
- **Weight of Evidence (WoE) / Information Value (IV)** analysis; features with IV < 0.02 excluded
- One-hot encoding of remaining categorical features

### 5. Class Imbalance Handling
- **SMOTETomek** (combined over- and under-sampling) applied to training data
- Threshold tuning considered as an alternative to preserve real data distributions

### 6. Models Trained

| Model | Tuning Strategy |
|---|---|
| Logistic Regression | RandomizedSearchCV (C, solver, penalty) |
| Random Forest | RandomizedSearchCV (n_estimators, max_depth, features) |
| XGBoost | RandomizedSearchCV (depth, lr, subsample, regularization) |
| LightGBM | RandomizedSearchCV + **Optuna** (30 trials, 5-fold CV) |
| CatBoost | RandomizedSearchCV (iterations, depth, l2, bagging) |

### 7. Evaluation & Model Selection
- Classification report (precision, recall, F1)
- ROC-AUC and AUC-PR curves per model
- Overlapping multi-model ROC and Precision–Recall curve comparison
- Permutation importance (LightGBM)
- SHAP TreeExplainer values (XGBoost)
- Confusion matrices

### 8. Statistical Validation
- 5-fold Stratified Cross-Validation (ROC-AUC)
- **Paired t-test** with Bonferroni correction across all model pairs
- **Cohen's d** effect sizes for pairwise differences
- **Bootstrap confidence intervals** (n=1000) for Macro F1, ROC-AUC, and AUC-PR

---

## 📊 Results Summary

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.86 | — |
| Random Forest | 0.92 | — |
| XGBoost | 0.92 | — |
| CatBoost | 0.93 | — |
| **LightGBM** | **0.93** | **Best** |

> LightGBM (tuned with Optuna) achieved the highest overall performance and was selected as the final model based on ROC-AUC, AUC-PR, and bootstrap confidence intervals.

---

## 🗂️ Project Structure

```
├── loan_data.csv               # Raw dataset
├── research_notebook.ipynb     # Full analysis notebook (123 cells)
└── README.md
```

---

## ⚙️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn \
            xgboost lightgbm catboost imbalanced-learn \
            shap optuna statsmodels
```

**Tested versions:**

| Library | Version |
|---|---|
| Python | 3.x |
| scikit-learn | logged in notebook |
| XGBoost | logged in notebook |
| LightGBM | logged in notebook |
| imbalanced-learn | logged in notebook |
| SHAP | logged in notebook |

> Run cell 110 in the notebook to print all exact library versions used.

---

## 🚀 Getting Started

1. Clone the repository and place `loan_data.csv` in the working directory (or update the path in cell 1).
2. Open the notebook in **Google Colab** or a local Jupyter environment.
3. Run cells sequentially from top to bottom.
4. Final model comparison plots and bootstrap CIs are generated in the last section.

---

## 🔑 Key Insights

- **Loan-to-income ratio** and **credit score** are among the strongest predictors of approval.
- Applicants with `DEBTCONSOLIDATION` or `MEDICAL` loan intent are flagged as higher risk.
- Renters, despite not owning property, show competitive approval patterns due to smaller, safer loan requests and better credit behavior.
- Approved borrowers paradoxically carry higher interest burdens, suggesting lenders accept higher-risk profiles in some approved cases.
- SMOTETomek significantly improved recall for the minority class without substantially degrading precision.

---

## 📄 License

This project is for research and educational purposes.
