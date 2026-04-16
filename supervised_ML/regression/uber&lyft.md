# 🚕 Dynamic Ride-Sharing Pricing: A Comprehensive Regression Showdown

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![Dataset](https://img.shields.io/badge/Dataset-Uber%20%26%20Lyft%20Boston-lightgrey)

## 📖 Project Overview
This repository functions as a massive, end-to-end benchmarking laboratory designed to empirically evaluate eleven distinct regression architectures. By utilizing a high-volume, real-world dataset detailing dynamic ride-sharing prices (Uber & Lyft), this project transitions from foundational parametric equations to state-of-the-art gradient boosting ensembles, systematically dissecting how different mathematical paradigms handle non-linear surges, multicollinear weather data, and strict linear baseline rules.

## 🎯 The Problem Statement
**Objective:** To architect and benchmark predictive regression models capable of accurately forecasting the exact `Price` of a ride-share trip by synthesizing spatial constraints (distance), temporal dynamics (hour, day), market mechanics (surge multipliers), and environmental variables (real-time meteorological data).

**The Mathematical Challenge:** Ride-sharing prices are fundamentally anchored to a deterministic linear equation (Base Fare + [Rate × Distance]). However, this linear foundation is aggressively distorted by non-linear surge multipliers and localized demand spikes, establishing an environment where linear baselines establish a floor, but advanced ensembles are strictly required to map the complex, multi-dimensional topological interactions.

---

## 📊 Dataset Description: Uber & Lyft Boston (600,000+ Records)
The dataset comprises over half a million recorded ride-share transactions coupled with granular weather telemetry. 

### Core Feature Matrix ($X$)
* **Spatial & Categorical:** `Distance` (miles), `Source`, `Destination`, `Cab_Type` (Uber/Lyft), `Name` (e.g., Black, SUV, Shared).
* **Temporal:** `Hour`, `Day`, `Month`.
* **Market Mechanics:** `Surge_Multiplier` (The critical non-linear distortion factor).
* **Meteorological:** `Temperature`, `Apparent_Temperature`, `Humidity`, `Precipitation_Probability`, `Cloud_Cover`.

### Target Variable ($y$)
* **`Price`**: The continuous numerical fiat value charged for the completed trip.

---

## 🔬 Algorithmic Deconstruction & Empirical Setup

This repository executes a rigorous evaluation of the following eleven algorithms, mapped specifically to the aerodynamic and temporal realities of dynamic pricing.

### 1. The Parametric Baselines
* **Simple Linear Regression**
    * **Theoretical Setup:** Utilizing exclusively `Distance` to predict `Price`.
    * **Contextual Insight:** Establishes the absolute mathematical baseline. It proves that longer rides cost progressively more but catastrophically fails to explain price variance for identical routes during distinct temporal rush hours.
* **Multiple Linear Regression (OLS)**
    * **Theoretical Setup:** Expanding the feature space to include `Distance`, `Surge_Multiplier`, and `Temperature`.
    * **Contextual Insight:** Illustrates multidimensional optimization. While $R^2$ improves, the algorithm inherently assumes a strict, unyielding linear relationship between weather variables and pricing, completely missing the threshold effects of severe weather.
* **Polynomial Regression (Basis Expansion)**
    * **Theoretical Setup:** Projecting the `Hour` feature into a 4th-degree polynomial space prior to linear fitting.
    * **Contextual Insight:** Solves the temporal non-linearity problem. Standard linear models draw a flat line across a 24-hour period; this basis expansion mathematically curves the regression hyperplane to perfectly encapsulate the cyclic spikes of morning and evening rush-hour demand.

### 2. Regularized Shrinkage Architectures
* **Ridge Regression ($L_2$ Regularization)**
    * **Theoretical Setup:** Ingesting all highly correlated weather variables (`Temperature` and `Apparent_Temperature`).
    * **Contextual Insight:** Mitigates the catastrophic variance inflation (multicollinearity) that destroys standard OLS models by applying a quadratic penalty constraint, thereby stabilizing the coefficient weights without discarding the meteorological telemetry entirely.
* **Lasso Regression ($L_1$ Regularization)**
    * **Theoretical Setup:** Executing automated feature selection across dozens of noisy environmental features.
    * **Contextual Insight:** By applying an absolute-value geometric penalty, Lasso ruthlessly forces the mathematical coefficients of statistically redundant variables (e.g., highly correlated humidity metrics) precisely to zero, extracting a sparse, highly interpretable pricing model.

### 3. Instance-Based & Spatial Partitioning
* **K-Nearest Neighbors (KNN) Regressor**
    * **Theoretical Setup:** Predicting a new ride's price by mathematically averaging the cost of the $K$ most spatially and environmentally proximate historical rides using Minkowski distance metrics.
    * **Contextual Insight:** Demonstrates the absolute necessity of rigorous feature scaling (`StandardScaler`); otherwise, the algorithm illogically conflates a 5-mile distance metric with a 50-degree temperature variance. It also explicitly highlights the severe latency (Curse of Dimensionality) when executing distance matrices against 600,000+ rows.
* **Decision Tree Regressor (CART)**
    * **Theoretical Setup:** Recursively fracturing the multi-dimensional feature space into mutually exclusive orthogonal regions by minimizing variance.
    * **Contextual Insight:** Yields immensely interpretable Boolean logic chains (e.g., *IF distance > 5 AND surge > 1.5 THEN price = $45*). However, if structurally unpruned, it demonstrates severe pathological overfitting, meticulously memorizing the pricing anomalies of singular trips.

### 4. Bagging & Boosting Ensembles (The Heavyweights)
* **Random Forest Regressor**
    * **Theoretical Setup:** Constructing a massive ensemble of uncorrelated, bootstrapped decision trees operating on stochastically selected feature subspaces.
    * **Contextual Insight:** Comprehensively rectifies the individual Decision Tree's overfitting vulnerabilities by enforcing structural variance reduction via prediction averaging, establishing an extraordinarily robust, general-purpose predictive baseline.
* **AdaBoost (Adaptive Boosting) Regressor**
    * **Theoretical Setup:** Sequentially training shallow decision stumps, dynamically re-weighting the probability distribution of the dataset to heavily penalize historical prediction failures.
    * **Contextual Insight:** By mathematically forcing subsequent iterations to prioritize previously miscalculated instances, this architecture becomes highly specialized at identifying and mapping the complex, non-linear mechanics governing extreme, anomalous surge multiplier events.
* **XGBoost (Extreme Gradient Boosting)**
    * **Theoretical Setup:** Expanding gradient boosting through the deployment of second-order Taylor expansion approximations (incorporating the Hessian) and intrinsic $L_1$/$L_2$ algorithmic penalization.
    * **Contextual Insight:** Functions as the optimal balance between aggressive gradient descent optimization and rigorous variance suppression, effectively capturing the most intricate interactions between spatial locations and temporal weather events without capitulating to noise.
* **LightGBM (Light Gradient Boosting Machine)**
    * **Theoretical Setup:** Executing asymmetric, leaf-wise tree expansion coupled with histogram-based feature discretization and Gradient-based One-Side Sampling (GOSS).
    * **Contextual Insight:** The ultimate architectural solution for datasets of this immense magnitude. By fundamentally altering how continuous features are binned and split, LightGBM trains on the 600,000+ row matrix exponentially faster than Random Forest or standard XGBoost while maintaining parity or superiority in predictive accuracy.

---

## ⚙️ Repository Pipeline & Execution

To effectively benchmark these topologies, the pipeline enforces strict uniformity in preprocessing:
1. **Data Sanitization:** Removal of NaN rows (specifically missing price tags on canceled rides).
2. **Categorical Encoding:** One-Hot Encoding applied to nominal geographical coordinates (`Source`, `Destination`) and `Cab_Type`.
3. **Dimensional Scaling:** Robust transformation of all continuous features ensuring optimal convergence velocities for gradient-dependent and distance-dependent estimators.
4. **Iterative Benchmarking:** Executing each algorithm against an isolated 20% holdout matrix.

