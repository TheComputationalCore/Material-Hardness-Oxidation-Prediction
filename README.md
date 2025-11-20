# 🔥 Material Hardness & Oxidation Prediction  
### Intelligent Microstructure–Property Modeling for Materials Engineering  

**Live Demo:** https://material-hardness-oxidation-prediction.onrender.com  

**Research Backing:**  
[Machine learning–assisted prediction of mechanical properties of EN‑8 alloy steel — IOP Conference Series](https://iopscience.iop.org/article/10.1088/1757-899X/998/1/012061)

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
*Experimental Studies of Stellite‑6 hardfacing layer on EN‑8 & Mild Steel*  
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
