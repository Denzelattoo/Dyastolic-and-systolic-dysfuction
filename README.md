# Dyastolic-and-systolic-dysfuction
ML pipeline for systolic/diastolic dysfunction screening on highly imbalanced medical data (1–7% positive cases). Implements ensemble voting, isotonic calibration, patient-level validation, and threshold optimization via weighted Youden index to maximize recall while controlling false positives.
# 🫀 Kvark: Machine Learning for Myocardial Dysfunction Prediction

## 📖 Overview
This repository contains a complete machine learning pipeline for the prediction of **systolic**, **diastolic**, and **combined myocardial dysfunction**. The project is built on the clinical "Kvark" dataset and implements modern ensemble methods, probability calibration, and decision threshold optimization tailored for medical screening tasks.

## 🎯 Target Variables
The model is trained independently for four clinical scenarios:

| Variable | Description |
|----------|-------------|
| `DyastDisf_outcome` | Diastolic dysfunction (Grade 2–3) |
| `SystDisf_outcome_AV` | Systolic dysfunction by sex-specific LVEF norms (<52% M / <54% F) |
| `SystDisf_outcome_AW` | Severe systolic dysfunction (LVEF < 40%) |
| `Syst_DyastDisf_outcome` | Combined systolic-diastolic dysfunction |

## 🛠 Methodology & Architecture
The project follows a rigorous clinical-ML workflow:

| Component | Implementation |
|-----------|---------------|
| **Data Leakage Prevention** | `GroupShuffleSplit` ensures no patient overlap between train/test sets |
| **Preprocessing** | Median imputation (`SimpleImputer`) + `RobustScaler` within `Pipeline` |
| **Base Estimators** | `LogisticRegressionCV` (elasticnet/saga) + `ExtraTreesClassifier` |
| **Ensembling** | Soft voting (`VotingClassifier`, `voting='soft'`) |
| **Probability Calibration** | `CalibratedClassifierCV` with isotonic regression for reliable probabilities |
| **Hyperparameter Tuning** | `HalvingGridSearchCV` with `StratifiedKFold` |
| **Threshold Optimization** | **Weighted Youden Index**: `(w × TPR) − FPR` to prioritize sensitivity (recall) |

## 📊 Performance on Held-Out Test Set
| Target Variable | ROC-AUC | Sensitivity | Specificity | PPV | NPV | Balanced Acc |
|-----------------|---------|-------------|-------------|-----|-----|--------------|
| Diastolic Dysfunction | `0.837` | `0.733` | `0.752` | `0.241` | `0.963` | `0.743` |
| Systolic Dysfunction (AV) | `0.908` | `0.729` | `0.888` | `0.347` | `0.976` | `0.808` |
| Severe Systolic (LVEF <40%) | `0.856` | `0.682` | `0.838` | `0.109` | `0.989` | `0.760` |
| Combined Dysfunction | `0.816` | `0.720` | `0.753` | `0.341` | `0.938` | `0.737` |
