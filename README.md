# Histological Inflammation Severity Prediction

This project investigates whether machine learning models trained on quantitative histological features can accurately predict inflammation severity scores (0–5) and binary inflammation status (<3 vs ≥3).

The project was developed as part of an MSc-level machine learning coursework and follows best practices in patient-level evaluation and reproducible experimentation.

---

## Objectives
- Predict continuous inflammation severity using regression models
- Evaluate binary inflammation classification (<3 vs ≥3)
- Compare linear, bagging, and boosting-based models
- Ensure patient-level generalisation using GroupKFold cross-validation

---

## Dataset
The dataset consists of quantitative histological features extracted from tissue samples, along with pathologist-assigned inflammation severity scores.

⚠️ Raw data is not included due to usage restrictions.  
See `data/README.md` for details.

---

## Models Evaluated
- Logistic Regression (binary baseline)
- Random Forest (classification and regression)
- Gradient Boosting Regression
- XGBoost Regression

---

## Evaluation Strategy
- Patient-level GroupKFold cross-validation (k = 5)
- Mean Absolute Error (MAE) for regression tasks
- Accuracy, Sensitivity, Specificity, ROC-AUC for binary classification

---

## Key Results

| Model | Task | Performance |
|------|------|-------------|
| Logistic Regression | Binary | ROC-AUC ≈ 0.98 |
| Random Forest | Regression | MAE ≈ 0.61 |
| Gradient Boosting | Regression | MAE ≈ 0.72 |
| XGBoost | Regression | MAE ≈ 0.70 |

Random Forest achieved the lowest error and most stable performance across folds.

---

## Repository Structure
- `notebooks/` – model development and experiments  
- `src/` – reusable preprocessing and evaluation code  
- `results/` – final tables and figures  
- `report/` – research-paper-style PDF  
- `data/` – dataset description (no raw data)  

---

## Reproducibility

All experiments were implemented in Python using scikit-learn and XGBoost.

Install dependencies:

```bash
pip install -r requirements.txt

---

## Author
Muhib Ul Aziz
MSc Data Science

## License
MIT License
