# 🚲 UCI Bike Sharing Demand: The Ultimate Regression Benchmark

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![Algorithms](https://img.shields.io/badge/Algorithms-11_Tested-success)

## 📖 Project Overview
This repository serves as a comprehensive, end-to-end benchmarking laboratory for supervised machine learning regression. Utilizing the legendary **UCI Bike Sharing Demand (Hourly)** dataset, this project stress-tests 11 distinct regression architectures—ranging from foundational parametric equations to state-of-the-art gradient boosting ensembles.

The objective is not just to achieve the lowest error, but to empirically demonstrate the unique mathematical behaviors, strengths, and critical flaws of each algorithm when confronted with severe multicollinearity, non-linear cyclic patterns, and deep interaction effects.

## 🎯 The Problem Statement
**Objective:** To architect a continuous regression framework capable of predicting the exact number of bicycles (`cnt`) rented during any given hour in a city, utilizing strictly temporal data, weather telemetry, and seasonal indicators.

**Why this Dataset?**
Sitting in the absolute architectural "Goldilocks Zone" (**17,379 rows and 16 features**), this dataset is large enough to prevent instantaneous overfitting but small enough to execute computationally heavy algorithms (like KNN and AdaBoost) in seconds. It contains hidden traps—like the multicollinearity between temperature features and the highly non-linear, bimodal distributions of daily rush hours—that perfectly expose algorithmic limitations.

---

## 📊 Dataset Description & Feature Matrix
The data represents two years of hourly historical logs from the Capital Bikeshare system in Washington D.C.

### The Features ($X$)
* **Temporal & Cyclic:** `hr` (Hour of day: 0-23), `mnth` (Month: 1-12), `weekday` (Day of week: 0-6).
* **Categorical Flags:** `season` (1-4), `holiday` (0/1), `workingday` (0/1), `weathersit` (1-4: Clear to Heavy Rain/Snow).
* **Continuous Meteorological:** `temp` (Normalized temperature), `atemp` (Normalized "feels-like" temperature), `hum` (Humidity), `windspeed`.

### Target Variable ($y$)
* **`cnt`**: The total count of rented bikes (casual + registered) within that specific hour.

---

## 🔬 The 11-Algorithm Stress Test

This project systematically runs the data through four categories of machine learning algorithms to observe their distinct mathematical paradigms:

### 1. The Parametric Baselines
* **Simple Linear Regression:** Tests bivariate correlation (e.g., `temp` vs. `cnt`). Predictably underfits due to lack of context.
* **Multiple Linear Regression (OLS):** Tests multidimensional linear mapping. Fails dramatically on the `hr` feature because demand peaks twice a day (bimodal), which a straight line cannot capture.
* **Polynomial Regression:** Executes basis expansion on temporal features, successfully allowing the linear model to mathematically curve and capture morning/evening rush-hour spikes.

### 2. Regularized Shrinkage (The Multicollinearity Trap)
* **Ridge Regression ($L_2$):** Confronts the heavy correlation between `temp` and `atemp`. Ridge applies a quadratic penalty to stabilize the wildly fluctuating coefficients caused by this multicollinearity.
* **Lasso Regression ($L_1$):** Applies an absolute-value penalty, realizing `atemp` is redundant and aggressively shrinking its coefficient to exactly $0$, effectively executing automated feature selection.

### 3. Spatial & Instance-Based Frameworks
* **K-Nearest Neighbors (KNN) Regressor:** Predicts demand by finding historical hours with identical weather/time signatures. Demonstrates the strict necessity of applying `StandardScaler` to prevent feature magnitude bias.
* **Decision Tree Regressor:** Abandons linear equations for spatial partitioning. Solves the interaction problem (e.g., *IF workingday=1 AND hr=17 THEN demand=High*), but demonstrates severe pathological overfitting if depth is unconstrained.

### 4. The Heavyweight Ensembles (Bagging & Boosting)
* **Random Forest Regressor:** Averages predictions across 100 decorrelated decision trees, immediately neutralizing the single tree's variance and establishing a formidable predictive baseline.
* **AdaBoost Regressor:** Sequentially trains weak learners, dynamically boosting the mathematical weights of poorly predicted anomalous hours (e.g., sudden thunderstorms) to force the model to adapt to outliers.
* **XGBoost (Extreme Gradient Boosting):** Utilizes second-order Taylor expansion and internal regularization to map highly complex multi-dimensional interactions without overfitting.
* **LightGBM:** The ultimate performance champion. Natively handles the categorical seasonal/weather flags via histogram-based binning, executing the entire matrix exponentially faster than standard boosting algorithms.

---
