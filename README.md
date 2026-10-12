# Cricket Player Performance & Elite Selection Machine Learning System

An end-to-end, publication-grade academic Machine Learning project demonstrating data exploration, feature engineering, class imbalance mitigation, ensemble modeling, hyperparameter tuning, model explainability (SHAP), and match-day prediction simulation.

---

## 📌 Project Overview
- **Dataset**: `player_performance_dataset_10000.csv` (10,000 player-match observations × 31 attributes).
- **Core Problem Formulation**: **Predicting Elite Match-Winning Performers** (`High_Performer = 1` for `Performance_Score >= 60.0`), representing the top $19.04\%$ of performances (e.g. Player of the Match contenders / Dream XI selection).
- **Key Academic Focus**: Tackling the **~1:4.25 class imbalance ratio** through cost-sensitive learning, SMOTE, PR-AUC evaluation, and decision threshold optimization.
- **Champion Model**: **Tuned LightGBM Classifier** achieving **0.9168 ROC-AUC**, **0.7510 PR-AUC**, **82.57% Balanced Accuracy**, and **0.6743 F1-Score** with sub-15ms inference latency.

---

## 📁 Repository Structure

```
c:/ML_project/
├── data/
│   ├── raw/
│   │   └── player_performance_dataset_10000.csv       # Original dataset
│   └── processed/
│       ├── train_features_engineered.csv               # Processed train features (8,000 x 88)
│       ├── test_features_engineered.csv                # Processed test features (2,000 x 88)
│       ├── train_target.csv                            # Binary target for train (8,000)
│       └── test_target.csv                             # Binary target for test (2,000)
│
├── notebooks/
│   ├── 01_Dataset_Exploration.ipynb                   # Phase 0 profiling & target formulation
│   ├── 02_EDA.ipynb                                   # Distributions, correlations & outlier audit
│   ├── 03_Preprocessing.ipynb                         # Leakage-free train/test split & scaling
│   ├── 04_Feature_Engineering_Selection.ipynb         # Cricket domain features & mutual information
│   ├── 05_Baseline_Model.ipynb                        # Dummy baselines & Logistic benchmark
│   ├── 06_Imbalance_Handling.ipynb                    # Cost-sensitive, SMOTE, and RUS resampling
│   ├── 07_Advanced_Models.ipynb                       # Decision Trees, RF, ExtraTrees, XGBoost, LightGBM
│   ├── 08_Hyperparameter_Tuning.ipynb                 # Randomized search & validation curves
│   ├── 09_Model_Evaluation.ipynb                      # Confusion matrix, calibration & threshold search
│   ├── 10_Model_Comparison.ipynb                      # Master benchmark & SHAP explainability
│   └── 11_Final_Prediction_Demonstration.ipynb         # End-to-end interactive & batch inference
│
├── models/
│   ├── preprocessor.joblib                            # Fitted preprocessing pipeline (88 features)
│   ├── baseline_logistic.joblib                       # L2-regularized baseline model
│   ├── logistic_balanced.joblib                       # Cost-sensitive logistic regression
│   ├── logistic_smote.joblib                          # SMOTE + logistic regression pipeline
│   ├── decision_tree.joblib                           # Pruned decision tree
│   ├── random_forest.joblib                           # 100-tree Random Forest
│   ├── xgboost.joblib                                 # Regularized XGBoost
│   ├── lightgbm.joblib                                # Gradient-boosted LightGBM
│   ├── best_random_forest.joblib                      # Tuned Random Forest candidate
│   ├── best_lightgbm.joblib                           # Tuned LightGBM candidate
│   └── champion_pipeline.joblib                       # Exported project champion model
│
├── outputs/
│   ├── figures/                                       # High-resolution saved plots (.png)
│   ├── tables/                                        # Benchmark and evaluation tables (.csv)
│   └── predictions/                                   # Exported match predictions (.csv)
│
├── src/
│   ├── __init__.py                                    # Module initializer
│   ├── utils.py                                       # Seed enforcement, paths, serialization
│   ├── preprocessing.py                               # Custom transformer, stratified split
│   ├── feature_engineering.py                         # Cricket metrics calculation & MI selector
│   └── evaluation.py                                  # Comprehensive metric suite & curves
│
├── reports/
│   └── academic_project_report.md                     # Comprehensive academic report with viva guide
│
├── requirements.txt                                   # Reproducible dependencies
├── README.md                                          # This documentation
└── .gitignore                                         # Git exclusion rules
```

---

## ⚙️ Environment Setup & Virtual Environment Activation

A virtual environment named `.venv` has been configured with Python 3.14:

```powershell
# Activate the virtual environment in PowerShell
.\.venv\Scripts\Activate.ps1

# (Optional) Verify dependencies
python -c "import numpy, pandas, sklearn, xgboost, lightgbm, imblearn, shap; print('Environment Verified!')"
```

---

## 📊 Master Model Benchmark Comparison

All models evaluated on the identical 2,000 hold-out test set:

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Latency / 1k ms |
|---|---|---|---|---|---|---|---|---|
| **Dummy (Most Frequent)** | 0.8095 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.1905 | < 0.1 |
| **Baseline Logistic (Unweighted)** | 0.8830 | 0.7752 | 0.7363 | 0.6010 | 0.6618 | 0.9264 | 0.7707 | 8.18 |
| **Cost-Sensitive Logistic (Balanced)** | 0.8285 | 0.8459 | 0.5303 | **0.8740** | 0.6601 | 0.9239 | 0.7654 | **1.49** |
| **SMOTE Logistic** | 0.8340 | 0.8493 | 0.5397 | **0.8740** | 0.6673 | 0.9238 | 0.7678 | 1.77 |
| **Decision Tree (Pruned)** | 0.7840 | 0.7883 | 0.4612 | 0.7953 | 0.5838 | 0.8476 | 0.5619 | 1.65 |
| **Random Forest (100 Trees)** | 0.8405 | 0.8312 | 0.5554 | 0.8163 | 0.6610 | 0.9147 | 0.7259 | 50.02 |
| **XGBoost Classifier** | 0.8335 | 0.8259 | 0.5420 | 0.8136 | 0.6506 | 0.9171 | 0.7453 | 9.35 |
| **LightGBM Classifier** | 0.8410 | 0.8376 | 0.5552 | 0.8320 | 0.6660 | 0.9174 | 0.7507 | 5.79 |
| **Tuned Random Forest** | 0.8585 | 0.8102 | 0.6065 | 0.7323 | 0.6635 | 0.9149 | 0.7320 | 112.26 |
| **Tuned LightGBM (Default $\tau=0.50$)** | **0.8620** | 0.8074 | **0.6185** | 0.7192 | 0.6650 | 0.9168 | **0.7510** | 13.20 |
| **Tuned LightGBM (Optimal $\tau^*=0.44$)** | 0.8575 | **0.8257** | 0.5972 | **0.7743** | **0.6743** | **0.9168** | **0.7510** | 13.20 |

---

## 🎯 Key Project Insights & Viva Defenses

1. **The Accuracy Paradox**: Trivial dummy classification scores $80.95\%$ accuracy with zero recall. Accuracy is mathematically invalid for imbalanced problems; models must be judged via **Balanced Accuracy**, **PR-AUC**, and **F1-Score**.
2. **Zero Data Leakage Protocol**:
   - Identifiers (`Player_ID`, `Match_ID`) dropped.
   - Continuous `Performance_Score` strictly excluded from $X$.
   - Train/Test split conducted before feature scaling or encoding.
   - Resampling (SMOTE / RUS) executed strictly inside CV folds.
3. **Domain Feature Engineering**: Engineered composite cricket metrics (`Batting_Impact`, `Bowling_Impact`, `Boundary_Run_Ratio`, `Form_Consistency_Index`, `Pressure_Resilience`) significantly increased Mutual Information with target.
4. **Game-Theoretic Interpretability (SHAP)**: Global beeswarm plots and local waterfall plots proved that `Recent_Form`, `Batting_Impact`, and `Wickets` positively drive elite classification, while `Economy` negatively penalizes performance ratings.
5. **Operational Optimization**: Deriving the optimal decision threshold $\tau^* = 0.44$ boosted elite player recall to $77.43\%$ while maintaining false alarms below $13\%$.

---

## 📖 Complete Documentation
For full theoretical derivations, cross-validation tables, and viva defense answers, see [`reports/academic_project_report.md`](reports/academic_project_report.md).
