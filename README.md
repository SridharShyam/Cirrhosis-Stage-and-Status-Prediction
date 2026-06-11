# Cirrhosis Stage & Status Prediction

> Predicting liver disease severity and patient survival outcomes using clinical biomarkers from the Mayo Clinic PBC Trial — upgraded from an in-plant training project to a production-grade ML pipeline.



## 📌 Project Overview

This project was originally developed during my **in-plant training at CIRF** and has since been rebuilt end-to-end with production-grade practices — fixing data leakage, recovering lost data through imputation, and replacing single-model accuracy evaluation with rigorous multi-model cross-validation.

**Two prediction targets:**
- `Stage` — Histological disease stage (1→4), representing severity of liver scarring
- `Status` — Patient outcome (Censored / Liver Transplant / Deceased)

**Dataset:** [UCI Cirrhosis Patient Survival Dataset](https://www.kaggle.com/datasets/joebeachcapital/cirrhosis-patient-survival-prediction) — 418 patients from the Mayo Clinic Primary Biliary Cirrhosis (PBC) randomized trial (1974–1984), 17 clinical features.



## 🔄 Before vs After (What Changed)

| Dimension | Original (CIRF training) | Upgraded | 
| ----------- | -------------------------- | ---------- | 
| Missing data | `dropna()` — lost 34% of rows (418 → 276) | `SimpleImputer` median/mode — all **418 rows retained** | 
| Encoding | `LabelEncoder` applied before split **(data leakage)** | Manual `map()` after imputation, zero leakage | 
| Status encoding | Encoded **twice** — silent label corruption | Encoded **once**, correctly | 
| Models | RandomForest only, default params | LR · SVM · RandomForest · XGBoost | 
| Validation | Single train/test split | **5-fold cross-validation** with F1-macro | 
| Imbalance handling | Ignored | `stratify=y` split + `class_weight='balanced'` | 
| Evaluation metric | `accuracy_score` only | F1-macro · ROC-AUC · Confusion Matrix | 
| Hyperparameter tuning | None | `RandomizedSearchCV` (20 iter, 5-fold) | 
| Feature analysis | None | Top-10 importances with **clinical interpretation** | 
| Reproducibility | No `random_state` on classifier | `RANDOM_STATE = 42` globally | 
| EDA | 5 plots, no interpretation | 6 plots, each followed by clinical insight markdown | 



## 🏗️ Project Structure

```	ext
Cirrhosis-Stage-and-Status-Prediction/
│
├── Liver_Cirrhosis.ipynb       # Upgraded production-grade notebook
├── cirrhosis.csv               # Dataset (UCI / Mayo Clinic PBC Trial)
├── README.md                   # This file
│
└── outputs/
    ├── stage_confusion_matrix.png
    ├── status_confusion_matrix.png
    ├── stage_roc_curve.png
    ├── stage_feature_importance.png
    └── status_feature_importance.png
```



## ⚙️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-189B4A?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-11557C?style=flat)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-4C72B0?style=flat)



## 🔬 Methodology

### 1. Data Preprocessing
- **Smart imputation** — median strategy for 7 numerical lab values (Cholesterol, Copper, Alk_Phos, SGOT, Tryglicerides, Platelets, Prothrombin); most-frequent for 5 categorical clinical signs
- **Manual encoding** — binary signs (Sex, Ascites, Hepatomegaly, Spiders) mapped to 0/1; Edema ordinal (N→0, S→1, Y→2); Status (C→0, CL→1, D→2)
- **Stratified split** — 80/20 split with `stratify=y` to preserve class proportions for both targets

### 2. Model Benchmarking
Four models evaluated with **5-fold cross-validation, F1-macro scoring**:
- Logistic Regression (`class_weight='balanced'`)
- Support Vector Classifier (`class_weight='balanced'`)
- Random Forest Classifier (`class_weight='balanced'`)
- XGBoost Classifier

### 3. Hyperparameter Tuning
`RandomizedSearchCV` on best-performing model:
```python
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# n_iter=20, cv=5, scoring='f1_macro'
```

### 4. Evaluation
- Classification report with named labels
- `ConfusionMatrixDisplay` heatmap
- Multi-class ROC curves (one-vs-rest per class) with AUC
- Macro ROC-AUC score



## 📊 Key Results

### Stage Prediction
| Stage | AUC | 
| ------- | ----- | 
| Stage 1 | 0.59 | 
| Stage 2 | 0.63 | 
| Stage 3 | 0.55 | 
| Stage 4 | **0.81** | 

> Stage 4 (end-stage cirrhosis) is predicted most reliably — clinically consistent, as its biomarker signature is most distinct.

### Status Prediction
- Model correctly identifies **Censored** (41/47) and **Deceased** (23/32) patients
- Transplant class (rare, ~6%) remains the hardest to distinguish — a known limitation of class-imbalanced survival datasets

### Top Clinical Predictors

**For Stage:** `N_Days` · `Platelets` · `Age` · `Albumin` · `Bilirubin`

**For Status:** `Bilirubin` · `Age` · `Prothrombin` · `N_Days` · `Platelets`

> These align precisely with what hepatologists use in clinical practice — Bilirubin (waste processing), Prothrombin (clotting), Albumin (protein synthesis), Platelets (portal hypertension proxy).



## 🧠 Clinical Context

Primary Biliary Cirrhosis (PBC) is a chronic autoimmune liver disease where the immune system destroys small bile ducts, causing bile accumulation, progressive fibrosis, and ultimately cirrhosis. The Mayo Clinic conducted a landmark RCT (1974–1984) testing D-penicillamine — the drug showed no significant benefit, but the dataset became a benchmark for medical ML and survival analysis.

The features in this dataset are **real clinical lab values**, not engineered proxies. When the model identifies Bilirubin and Prothrombin as top predictors, it's agreeing with decades of hepatology literature.



## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/SridharShyam/Cirrhosis-Stage-and-Status-Prediction.git
cd Cirrhosis-Stage-and-Status-Prediction

# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn

# Open the notebook
jupyter notebook Liver_Cirrhosis.ipynb
```

Or open directly in Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SridharShyam/Cirrhosis-Stage-and-Status-Prediction/blob/main/Liver_Cirrhosis.ipynb)



## 📚 References

- **Dataset:** Dickson et al. (1989). *Prognosis in Primary Biliary Cirrhosis: Model for Decision Making*. Hepatology, 10(1), 1–7.
- **UCI Repository:** [Cirrhosis Patient Survival Prediction](https://archive.ics.uci.edu/dataset/878/cirrhosis+patient+survival+prediction)
- **Kaggle:** [Joe Beach Capital — Cirrhosis Dataset](https://www.kaggle.com/datasets/joebeachcapital/cirrhosis-patient-survival-prediction)



## 👤 Author

**Sridhar Shyam**
B.Tech AI & ML · Saveetha Engineering College · Chennai

[![LinkedIn](https://img.shields.io/badge/LinkedIn-shyam--2005--ds--ml-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/shyam-2005-ds-ml)
[![GitHub](https://img.shields.io/badge/GitHub-SridharShyam-181717?style=flat&logo=github)](https://github.com/SridharShyam)
[![Portfolio](https://img.shields.io/badge/Portfolio-shyam--portfolio-00C7B7?style=flat&logo=vercel)](https://shyam-portfolio-chi.vercel.app)



---

Originally developed during in-plant training at CIRF · Upgraded June 2026
