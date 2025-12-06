# 🌍 Climate Event Predictor

### Machine Learning Models for Severity Classification & Economic Impact Forecasting

This project analyzes a global dataset of major climate events (2020–Sept 2025) and builds predictive models to estimate both event severity and economic impact. The dataset includes 3,000+ events across 51 countries, capturing critical information about disasters and their consequences.

The goal of this project is to explore patterns in climate-related events and develop machine learning models that enable early warning, risk assessment, and economic planning.

## 📁 Dataset Overview

The dataset tracks worldwide climate events with detailed attributes such as:

 - Event type (e.g., flood, wildfire, hurricane, drought)
 - Location (country, latitude, longitude)
 - Severity score (1–10)
 - Duration of event
 - Casualties and affected population
 - Infrastructure damage score
 - Economic impact (in millions USD)
 - Response time
 - Date and temporal information

Total events: 3,000+

Countries: 51

Time span: 2020 – September 2025

### 📌 Primary Use Case

Build machine learning models to predict:

 - Severity level (Low / Medium / High)
 - Economic impact (in million USD)

## 🔧 Project Structure
### ✔️ Data Exploration & Visualization

The project performs extensive EDA, including:

- Distribution of event types
- Severity histogram
- Economic impact distribution
- Missing value checks
- Duplicate checks
- Visual plots using Matplotlib and Seaborn

## 🛠️ Data Preprocessing & Feature Engineering

Key preprocessing steps include:

### 🔹 Encoding & Scaling

- Label encoding for countries
- One-hot encoding for event types
- Standard scaling for ML training

### 🔹 Temporal Features

Extracted from event date:

- Quarter
- Day of year

### 🔹 Engineered Features

Created to enhance predictive power:

- casualty_rate
- response_efficiency
- severity_x_duration
- deaths_per_affected
- lat_long_interaction
- damage_per_day
- population_density_proxy
- casualties_x_damage

### 🔹 Outlier Treatment (Crucial Step)

Economic impact contained extreme outliers (up to $718M).
These severely harmed regression performance.

A 99.5th percentile cap was applied to retain natural variability while removing distortions.

**Impact of outlier treatment:**

- R² improved from 0.0864 → 0.7960
- Model predictions became stable and meaningful

## 🤖 Machine Learning Models
### 1. Severity Classification (Random Forest Classifier)

Target labels:

- 0 → Low
- 1 → Medium
- 2 → High

Model Performance:
- Accuracy: 90.33%
- Strong separation between severity classes
- Feature importance analysis highlights key drivers (duration, casualties, damage score, etc.)

### 2. Economic Impact Prediction (Gradient Boosting Regressor)

Target transformed using log1p()

**Model Performance (after outlier handling):**

| Metric       | Value                 |
| ------------ | --------------------- |
| MAE          | *Low error (scaled)*  |
| RMSE         | *Stable across range* |
| **R² Score** | **0.7960**            |


Plots included:
- Predicted vs Actual
<img width="589" height="390" alt="Predicted vs Actual Economic Impact" src="https://github.com/user-attachments/assets/fe1b4301-c6f0-48d2-90b0-515f35dc3424" />

- Residual analysis
<img width="589" height="390" alt="Residual Plot - Economic Impact" src="https://github.com/user-attachments/assets/6f2f9601-6f90-4ba3-9838-48f2ead07400" />


## 📊 Results Summary
| Task                           | Model                       | Performance         |
| ------------------------------ | --------------------------- | ------------------- |
| **Severity Prediction**        | Random Forest Classifier    | **90.33% accuracy** |
| **Economic Impact Prediction** | Gradient Boosting Regressor | **R² = 0.80**       |

**Key Insight**

Economic impact is highly sensitive to extreme values. Proper handling of outliers is essential for producing reliable and actionable forecasts.

## 🗂️ Technologies Used
- Python
- Pandas / NumPy
- Matplotlib / Seaborn
- Scikit-Learn
- Jupyter Notebook

## 🙌 Acknowledgments

Dataset created and curated for climate event research and machine learning experimentation.
