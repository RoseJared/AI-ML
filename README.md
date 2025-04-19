# AQI Prediction Using Machine Learning

This project focuses on predicting the Air Quality Index (AQI) using various machine learning regression models. The primary dataset consists of hourly air pollution data from Beijing, including pollutant levels and meteorological measurements. The models are implemented in Python using libraries such as `scikit-learn`, `xgboost`, and `tensorflow.keras`.

## Files

- FinalProject.ipynb: A Jupyter notebook containing all preprocessing steps, model implementations, evaluations, visualizations, and commentary.
- finalproject.py: A Python script version of the notebook for streamlined execution or integration.

## Objectives

- Clean and preprocess AQI data.
- Convert pollutant units and calculate AQI using the Chinese Ministry of Environmental Protection (MEP) standards.
- Engineer features including AQI categories.
- Evaluate and compare multiple machine learning models:
  - Linear Regression (with interaction & polynomial features)
  - Decision Tree Regressor
  - Random Forest Regressor
  - Support Vector Regression (SVR)
  - Gradient Boosting Regressor
  - XGBoost Regressor
  - Neural Networks (Keras)

## Technologies

- Python 3.x
- `pandas`, `numpy` for data manipulation
- `matplotlib`, `seaborn` for visualization
- `scikit-learn` for ML models & hyperparameter tuning
- `xgboost` for gradient boosting
- `tensorflow.keras` for neural network modeling

## Performance Metrics

Each model is evaluated using:
- Mean Squared Error (MSE)
- R² Score

Visualizations are provided to compare predicted vs. actual AQI across models.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/aqi-prediction.git
   cd aqi-prediction
