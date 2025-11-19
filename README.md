# Material Hardness & Oxidation Prediction
### Intelligent Microstructure–Property Modeling for Materials Engineering  
**Author:** Dinesh Chandra  
**Live Demo:** https://your-deployment-url.com  
**Research Backing:**  
*[Machine learning–assisted prediction of mechanical properties of EN-8 alloy steel — IOP Conference Series](https://iopscience.iop.org/article/10.1088/1757-899X/998/1/012061)*

---

<div align="center">
  <img src="screenshots/demo-01-home.png" width="750">
</div>

---

## 🚀 Overview

This project delivers a **high-fidelity machine learning system** for predicting:

1. **Material Hardness**  
2. **Oxidation Rate**

It integrates advanced ML pipelines, automated input validation, SHAP-based explainability, and a modern browser-based interface — bridging **materials science** with **production-grade ML engineering**.

The system enables researchers and engineers to:

- Predict microstructure-driven material properties within seconds  
- Understand governing factors via explainable AI  
- Experiment with digital process variations  
- Accelerate materials & process optimization  

---

## 🧪 Scientific Foundation

Hardness and oxidation behavior strongly influence:

- Heat treatment performance  
- Wear and corrosion resistance  
- Component lifetime  
- Structural reliability  
- Surface engineering outcomes  

Experiments are **expensive and time-intensive**, motivating the need for **AI surrogate models**.

This system extends the ideas from:

**Dinesh Chandra (2020). Machine learning–assisted prediction of mechanical properties of EN-8 alloy steel.**  
IOP Conference Series: Materials Science and Engineering.  
https://doi.org/10.1088/1757-899X/998/1/012061

---

## 🏗 Architecture

material-hardness-oxidation-prediction/
│
├── data/                     # Datasets + documentation
├── models/                   # Trained ML models + metadata
├── src/
│   ├── app/                  # Flask app (UI, routes, HTML templates, static files)
│   ├── inference/            # Prediction + schema validation logic
│   ├── models/               # ML pipelines (training & evaluation)
│   └── utils/                # Config & common utilities
│
├── screenshots/              # UI and SHAP visualization images
├── tests/                    # pytest suite
├── requirements.txt
├── render.yaml
├── Procfile
└── runtime.txt

---

## 🌐 UI Preview

### **Home Interface**
<div align="center">
  <img src="screenshots/demo-01-home.png" width="750">
</div>

---

### **Prediction Workflow**
<div align="center">
  <img src="screenshots/demo-02-predict.png" width="750">
</div>

---

### **Hardness Explainability (SHAP)**
<div align="center">
  <img src="screenshots/demo-03-hardness-shap.png" width="750">
</div>

---

### **Oxidation Explainability (SHAP)**
<div align="center">
  <img src="screenscreenshots/demo-04-oxidation-shap.png" width="750">
</div>

---

## 📊 Exploratory Data Analysis (EDA)

<details>
<summary><strong>Click to expand EDA visualizations</strong></summary>

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
<summary><strong>Click to expand performance plots</strong></summary>

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

Each model provides:

- Data validation  
- Preprocessing & feature engineering  
- Scikit-learn regression pipelines  
- Hyperparameter tuning  
- SHAP-based explainability  
- Metadata for reproducibility  

### Training Scripts
src/models/train_hardness.py  
src/models/train_oxidation.py

### Evaluation
src/models/evaluate.py

---

## 🛠 Setup (Local Development)

### **1. Clone the repository**
git clone https://github.com/TheComputationalCore/Material-Hardness-Oxidation-Prediction
cd Material-Hardness-Oxidation-Prediction

### **2. Create environment**
conda create -n mhoc python=3.10
conda activate mhoc
pip install -r requirements.txt

### **3. Run application**
python src/app/app.py

Open in browser:  
http://localhost:5000

---

## 🧪 Testing
pytest -q

---

## 🚀 Deployment (Render)

### **Build Command**
pip install -r requirements.txt

### **Start Command**
gunicorn "app.app:app" --chdir src --bind 0.0.0.0:$PORT --workers 2

---

## 📘 Documentation

- MODEL_CARD.md  
- ARCHITECTURE.md  
- API_REFERENCE.md  

---

## 📄 Citation

Dinesh Chandra (2020),  
Machine learning–assisted prediction of mechanical properties of EN-8 alloy steel,  
IOP Conference Series: Materials Science and Engineering, 998 (1), 012061.  
https://doi.org/10.1088/1757-899X/998/1/012061

---

## ⭐ Author

**Dinesh Chandra**  
Machine Learning Engineer & Materials Science Researcher

---

## 📦 License

MIT License — open for academic and professional use.
