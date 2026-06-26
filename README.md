# 📦 RapiEntregas Predictive Analytics & Optimization

An end-to-end Data Science and Machine Learning project focused on analyzing delivery operations, addressing data quality challenges, and deploying a predictive classification pipeline to identify delivery delays for **RapiEntregas**.

---

## 📈 Project Overview
Operational efficiency is critical for delivery platforms. This project leverages synthetic data representing 5,000 deliveries across 5 major Colombian cities (Bogotá, Medellín, Cali, Barranquilla, and Bucaramanga) deployed by 150 couriers with diverse experience and vehicle assets. 

The primary business goal is to mitigate slow deliveries (`delivery_time > 60 minutes`) by uncovering underlying constraints during peak hours, handling intentional messy real-world data quality anomalies, and training predictive models to enable preemptive logistics handling.

## 🛠️ Data Quality & Engineering Challenges Addressed
To emulate real-world environments, intentional data flaws were injected and resolved during the Exploratory Data Analysis (EDA) phase:
- **Missing Value Handling:** Handled missingness in `calificacion_cliente` utilizing mean imputation through `SimpleImputer` prior to normalization.
- **Deduplication:** Dropped structural row duplicates injected during the simulation pipeline.
- **Categorical Harmonization:** Resolved lexical inconsistencies within spatial markers (e.g., merging variations like *'Bogota'*, *'Bta'*, *'BOGOTA'* into standard representations).
- **Outlier Mitigation:** Handled hard upper-bound anomalies injected into delivery completion times.

---

## 🤖 Modeling Pipeline & Performance

### 1. Feature Engineering & Preprocessing
- **Target Variable Definition:** Generated a robust classification target `es_entrega_lenta` (defined as `1` if `tiempo_entrega_minutos > 60`, otherwise `0`).
- **Standardization:** Applied `StandardScaler` to uniform continuous scales across numerical matrices.
- **Categorical Encoding:** Processed nominal dimensions (`ciudad`, `tipo_vehiculo`, `dia_semana`) leveraging k-1 dummy transformation (`drop_first=True`) to secure structural independence.
- **Data Splitting:** Executed a structured 80/20 train/test divide stratified along the target class vector to retain proper imbalance distributions.

### 2. Model Performance Summary
Two architectures were optimized and evaluated on the test subset:

* **Logistic Regression Baseline:** Demonstrated high stability, scoring an overall **96% Evaluation Accuracy**.
* **Random Forest Classifier (Optimized):** Fine-tuned via 5-fold cross-validated Hyperparameter distribution sampling (`RandomizedSearchCV`), yielding highly reliable validation metrics across multiple configuration bounds:
    - **Overall Model Accuracy:** `~96%`
    - **Precision (Class 1 - Delayed):** `0.79`
    - **Recall (Class 1 - Delayed):** `0.66`
    - **F1-Score (Class 1 - Delayed):** `0.72`

---

## 🚀 Tech Stack Used
- **Core Language:** Python 3
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning & Preprocessing:** Scikit-Learn (`StandardScaler`, `SimpleImputer`, `train_test_split`, `LogisticRegression`, `RandomForestClassifier`, `RandomizedSearchCV`, `KFold`)
- **Model Serialization:** Joblib

---

## 📦 How to Run This Project Locally

1. **Clone this repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/rapientregas-predictive-analytics.git](https://github.com/YOUR_USERNAME/rapientregas-predictive-analytics.git)
   cd rapientregas-predictive-analytics
