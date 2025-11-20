#   Material Hardness & Oxidation Prediction  
### Intelligent Microstructure–Property Modeling for Materials Engineering  

<p align="center">

  <!-- Python version -->
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python 3.10">
  </a>

  <!-- Framework -->
  <a href="https://flask.palletsprojects.com/">
    <img src="https://img.shields.io/badge/Framework-Flask-000000.svg?logo=flask" alt="Flask">
  </a>

  <!-- Machine Learning -->
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/ML-Scikit--Learn-FCBA03.svg?logo=scikitlearn" alt="scikit-learn">
  </a>

  <!-- SHAP -->
  <a href="https://shap.readthedocs.io/en/latest/">
    <img src="https://img.shields.io/badge/Explainability-SHAP-EA4C89.svg" alt="SHAP">
  </a>

  <!-- Render Deployment -->
  <a href="https://render.com/">
    <img src="https://img.shields.io/badge/Deploy-Render-46E3B7.svg?logo=render" alt="Render">
  </a>

  <!-- License -->
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
  </a>

  <!-- Last Commit -->
  <img src="https://img.shields.io/github/last-commit/TheComputationalCore/Material-Hardness-Oxidation-Prediction" alt="Last Commit">

  <!-- Repo Size -->
  <img src="https://img.shields.io/github/repo-size/TheComputationalCore/Material-Hardness-Oxidation-Prediction" alt="Repo Size">

  <!-- Stars -->
  <img src="https://img.shields.io/github/stars/TheComputationalCore/Material-Hardness-Oxidation-Prediction?style=social" alt="Stars">

</p>

**Live Demo:** https://material-hardness-oxidation-prediction.onrender.com  

**Research Backing:**  
[Experimental Studies of Stellite-6 Hardfaced Layer on Ferrous Materials by Tig Surfacing Process — IOP Conference Series](https://iopscience.iop.org/article/10.1088/1757-899X/998/1/012061)

---

## 🚀 Overview  

This project delivers a **high-fidelity machine learning system** for predicting:

1. **Material Hardness**  
2. **Oxidation Rate**

It integrates advanced ML pipelines, automated input validation, SHAP‑based interpretability, and a modern browser interface — bridging **materials science** with **production‑grade ML engineering**.

The system enables researchers and engineers to:

- Predict microstructure-driven properties instantly  
- Understand governing factors using explainable AI  
- Experiment with process variables digitally  
- Accelerate materials & process optimization  

---

## 🧪 Scientific Foundation  

Hardness and oxidation behavior strongly influence:

- Heat treatment outcomes  
- Wear and corrosion resistance  
- Component lifetime  
- Surface engineering performance  
- High‑temperature reliability  

Physical experiments are **expensive and time‑consuming**, motivating AI surrogate modeling.

This system extends ideas from:

**Dinesh Chandra (2020)**  
*Experimental Studies of Stellite-6 Hardfaced Layer on Ferrous Materials by Tig Surfacing Process*  
IOP Conference Series: Materials Science and Engineering  
https://doi.org/10.1088/1757-899X/998/1/012061  

---

## 🏗 Architecture  

```
material-hardness-oxidation-prediction/
│
├── data/                     # Datasets
├── models/                   # Trained ML models + metadata
├── src/
│   ├── app/                  # Flask app (UI, routes, templates)
│   ├── inference/            # Prediction + schema validation
│   ├── models/               # ML pipelines + training scripts
│   └── utils/                # Shared utilities
│
├── screenshots/              # UI previews & SHAP visuals
├── tests/                    # Pytest suite
├── requirements.txt
├── render.yaml
├── Procfile
└── runtime.txt
```

---

## 🌐 UI Preview  

### **Home Interface**
<img src="screenshots/demo-01-home.png" width="750">

### **Prediction Workflow**
<img src="screenshots/demo-02-predict.png" width="750">

### **Hardness Explainability (SHAP)**
<img src="screenshots/demo-03-hardness-shap.png" width="750">

### **Oxidation Explainability (SHAP)**
<img src="screenshots/demo-04-oxidation-shap.png" width="750">

---

## 📊 Exploratory Data Analysis (EDA)

<details>
<summary><strong>Expand EDA Visualizations</strong></summary>

### Hardness Dataset
<img src="src/app/static/plots/eda_hardness_correlation.png" width="420">
<img src="src/app/static/plots/eda_hardness_hist.png" width="420">

### Oxidation Dataset
<img src="src/app/static/plots/eda_oxidation_correlation.png" width="420">
<img src="src/app/static/plots/eda_oxidation_hist.png" width="420">

</details>

---

## 📈 Model Performance & Diagnostics

<details>
<summary><strong>Expand Performance Visuals</strong></summary>

### Hardness Model
<img src="src/app/static/plots/perf_hardness_actual_vs_pred.png" width="420">
<img src="src/app/static/plots/perf_hardness_residuals.png" width="420">
<img src="src/app/static/plots/fi_hardness_coefficients.png" width="420">

### Oxidation Model
<img src="src/app/static/plots/perf_oxidation_actual_vs_pred.png" width="420">
<img src="src/app/static/plots/perf_oxidation_residuals.png" width="420">
<img src="src/app/static/plots/fi_oxidation_importances.png" width="420">

</details>

---

## 🧠 Machine Learning Pipelines  

Each model includes:

- Schema validation  
- Preprocessing & feature engineering  
- Scikit‑learn pipelines  
- Regression models (Linear Regression, Random Forest)  
- SHAP‑based explainability  
- Metadata for reproducibility  

### **Training Scripts**
```
src/models/train_hardness.py
src/models/train_oxidation.py
```

### **Evaluation**
```
src/models/evaluate.py
```

---

## 🛠 Local Development  

### **1. Clone repo**
```bash
git clone https://github.com/TheComputationalCore/Material-Hardness-Oxidation-Prediction
cd Material-Hardness-Oxidation-Prediction
```

### **2. Create environment**
```bash
conda create -n mhoc python=3.10
conda activate mhoc
pip install -r requirements.txt
```

### **3. Run app**
```bash
python src/app/app.py
```

Visit: http://localhost:5000  

---

## 🧪 Testing  
```bash
pytest -q
```

---

## 🚀 Deployment (Render)

### Build Command
```
pip install -r requirements.txt
```

### Start Command
```
gunicorn "app.app:app" --chdir src --bind 0.0.0.0:$PORT --workers 2
```

---

## 📘 Documentation  

- `docs/MODEL_CARD.md`  
- `docs/ARCHITECTURE.md`  
- `docs/API_REFERENCE.md`  

---

## 👤 Author  

**Dinesh Chandra — TheComputationalCore**  
GitHub: https://github.com/TheComputationalCore  
YouTube: https://www.youtube.com/@TheComputationalCore  

---

## 📦 License  
MIT License — Open for academic & professional use.
