# Maternal Health Risk Prediction

Classification pipeline for maternal health risk stratification using physiological indicators, with a deployed Streamlit inference interface.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-UCI_Repository-4C8CBF?style=flat-square)](https://archive.ics.uci.edu/dataset/863/maternal+health+risk)

---

## Problem Statement

Maternal mortality remains a critical public health challenge, particularly in regions with limited clinical infrastructure. This project builds a supervised classification system to stratify patients into risk levels — **Low**, **Mid**, or **High** — based on routine physiological measurements, enabling earlier clinical intervention without specialist dependency.

---

## Pipeline

```
Raw Physiological Data (CSV)
    │
    ▼
Exploratory Data Analysis
  Feature distributions · Correlation analysis · Class balance
    │
    ▼
Preprocessing
  Missing value handling · Feature scaling · Train/test split
    │
    ▼
Model Training
  ├── K-Nearest Neighbors (KNN)
  └── Random Forest Classifier
    │
    ▼
Evaluation
  Accuracy · Cross-validation · RMSE · Classification report
    │
    ▼
Streamlit Inference App (app.py)
  Real-time risk prediction from user-input vitals
```

---

## Features

- Multi-class risk classification: Low / Mid / High risk stratification
- Comparative model evaluation between KNN and Random Forest on identical train/test splits
- Cross-validation for generalization assessment beyond single-split accuracy
- EDA with feature correlation and class distribution analysis
- Streamlit app for real-time inference without code execution

---

## Dataset

Source: [UCI Machine Learning Repository — Maternal Health Risk](https://archive.ics.uci.edu/dataset/863/maternal+health+risk)

| Feature | Description |
|---|---|
| Age | Patient age in years |
| SystolicBP | Systolic blood pressure (mmHg) |
| DiastolicBP | Diastolic blood pressure (mmHg) |
| BS | Blood sugar level (mmol/L) |
| BodyTemp | Body temperature (°F) |
| HeartRate | Resting heart rate (bpm) |
| RiskLevel | Target — Low / Mid / High |

---

## Results

| Model | Accuracy | Notes |
|---|---|---|
| K-Nearest Neighbors | 83.70% | Sensitive to feature scale; scaled input required |
| Random Forest | 83.74% | Better consistency across folds; lower RMSE |

Random Forest is the selected production model due to ensemble stability, better handling of feature interactions, and lower variance across cross-validation folds.

---

## Project Structure

```
Maternal-Health-Risk-Prediction/
│
├── Maternal_Health_Risk.ipynb     # Full ML pipeline: EDA, training, evaluation
├── Maternal Health Risk Data Set.csv  # UCI dataset
├── app.py                         # Streamlit inference app
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/Nilay-20/Maternal-Health-Risk-Prediction.git
cd Maternal-Health-Risk-Prediction

# Run inference app
streamlit run app.py
```

To explore the full training pipeline, open `Maternal_Health_Risk.ipynb` in Jupyter or VS Code.

---

## Roadmap

- Add XGBoost and LightGBM for gradient boosting comparison
- SHAP-based feature importance for clinical interpretability
- Hyperparameter tuning via GridSearchCV for KNN and RF
- Expand dataset with additional physiological markers
