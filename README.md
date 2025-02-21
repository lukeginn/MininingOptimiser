# Kaggle Mining Optimiser

[![License](https://img.shields.io/badge/license-CC--BY--NC--ND-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/username/repo/releases)

This repository contains the code and configuration files for the Kaggle Mining Optimiser project. The project aims to optimize mining operations using machine learning models.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Configuration](#configuration)
- [Discussion](#discussion)
- [License](#license)

## Introduction

The Kaggle Mining Optimiser project uses machine learning models to predict and optimize the concentration of iron and silica in mining operations. The models are trained using historical data and various features related to the mining process.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Configuration

You can configure the project by editing the following files in the `config` folder: `general.yaml`, `data.yaml`, `model.yaml`, `clustering.yaml`, `simulation.yaml`, and `optimisation.yaml`.

## Discussion

There are four stages to this project:

### 1. Data Preprocessing

In this stage, we clean and prepare the data for analysis. This includes handling missing values, normalizing data, and feature engineering to create new variables that can improve model performance. The preprocessing steps are as follows:

- **Initial Preprocessing**: Standardizes column names, cleans data by replacing commas with dots, converts specific columns to datetime and numeric types, and aggregates data by grouping it from half-secondly to hourly.
- **Missing Data Processing**: Identifies and corrects missing data by deleting rows with missing values, interpolating time series, and replacing missing values with specified values.
- **Outlier Processing**: Identifies and handles outliers using various methods such as IQR, Z-score, MAD, DBSCAN, Isolation Forest, and LOF.
- **Lag Introduction**: Introduces lags for specified features and automatically optimizes lags for the target feature.
- **Data Aggregation**: Performs rolling aggregation on the data. This ensures the data volume is lost, whilst the lag propagation times in the processing facility are effectively negligible.
- **Feature Engineering**: Creates new features that can improve model performance or can aid in the optimisation procedure.
- **Shutdown Filtering**: Filters out shutdown periods from the data.

These steps ensure that the data is cleaned, transformed, and ready for model training, evaluation and optimisation.

### 2. Machine Learning Model Training

Here, we train various machine learning models using the preprocessed data. We experiment with different algorithms and hyperparameters to find the best-performing model for predicting the concentration of iron and silica. The modeling steps are as follows:

- **Model Training**: The `ModelTrainer` class handles loading data, splitting data, training models, hyperparameter tuning, and evaluating models. Various machine learning algorithms such as Linear Regression, Decision Trees, Random Forests, and Gradient Boosting are used.
- **Model Evaluation**: The `ModelEvaluator` class handles loading models, evaluating performance on a test set using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared, and generating evaluation reports.
- **Model Persistence**: The `ModelPersistence` class handles saving the trained models to disk and loading them for future use.
- **Model Inference**: The `ModelInference` class handles loading models, making predictions on new data, and generating output files or visualizations based on the predictions.

These steps ensure that the models are trained, evaluated, saved, and ready for deployment in the mining optimization process.

### 3. Clustering

We use clustering techniques to group similar data points together. This helps in identifying patterns and trends in the data, which can be useful for further analysis and decision-making. The clustering steps are as follows:

- **Clustering Processor**: The `ClusteringProcessor` class handles loading data, selecting features, applying clustering algorithms, evaluating clusters, and generating artifacts. Various clustering algorithms such as K-Means, DBSCAN, and Agglomerative Clustering are used.
- **Evaluating Clusters**: The quality of the clusters is evaluated using metrics such as Silhouette Score, Davies-Bouldin Index, and Inertia.
- **Generating Artifacts**: Artifacts such as cluster labels and visualizations are generated to help understand the clustering results.

These steps ensure that the data is effectively clustered, providing valuable insights for further analysis and decision-making in the mining optimization process.

### 4. Simulation and Optimisation

In this final stage, we simulate different mining scenarios using the trained models and optimize the operations to achieve the best possible outcomes. This involves running simulations, analyzing results, and making adjustments to improve efficiency and productivity. The simulation and optimization steps are as follows: The simulation and optimization steps are as follows:

- **Simulation Processor**: The `SimulationProcessor` class handles loading models, running simulations, analyzing results, and generating reports. This helps in predicting the outcomes of different mining scenarios.
- **Optimization Processor**: The `OptimizationProcessor` class handles loading data and models, defining objectives, applying optimization algorithms, evaluating solutions, and generating artifacts. Various optimization algorithms such as Genetic Algorithms, Particle Swarm Optimization, and Gradient Descent are used.

These steps ensure that the mining operations are effectively simulated and optimized, providing valuable insights for improving efficiency and productivity in the mining process.These steps ensure that the mining operations are effectively simulated and optimized, providing valuable insights for improving efficiency and productivity in the mining process.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. This license only allows for downloading and sharing of the work, as long as credit is made. It cannot be changed in any way or used commercially. See the LICENSE file for more details.

```plaintextplaintext
Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International Public LicenseCreative Commons Attribution-NonCommercial-NoDerivs 4.0 International Public License






For the full license text, please refer to https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode...By exercising the Licensed Rights (defined below), You accept and agree to be bound by the terms and conditions of this Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International Public License ("Public License"). To the extent this Public License may be interpreted as a contract, You are granted the Licensed Rights in consideration of Your acceptance of these terms and conditions, and the Licensor grants You such rights in consideration of benefits the Licensor receives from making the Licensed Material available under these terms and conditions.By exercising the Licensed Rights (defined below), You accept and agree to be bound by the terms and conditions of this Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International Public License ("Public License"). To the extent this Public License may be interpreted as a contract, You are granted the Licensed Rights in consideration of Your acceptance of these terms and conditions, and the Licensor grants You such rights in consideration of benefits the Licensor receives from making the Licensed Material available under these terms and conditions.

...

For the full license text, please refer to https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode