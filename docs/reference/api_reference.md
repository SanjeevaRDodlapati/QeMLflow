# API Reference for ChemML

## Overview

This document provides an API reference for the ChemML project, detailing the functions and classes available in the various modules. The ChemML project focuses on machine learning and quantum computing techniques for molecular modeling and drug design.

## Modules

### Data Processing

#### molecular_preprocessing.py

- **Functions:**
  - `clean_data(data)`: Cleans the molecular data by handling missing values and normalizing the data.
  - `normalize_data(data)`: Normalizes the molecular data to a standard scale.

#### feature_extraction.py

- **Functions:**
  - `extract_descriptors(molecule)`: Extracts molecular descriptors from the given molecule.
  - `generate_fingerprints(molecule)`: Generates molecular fingerprints for the given molecule.

### Models

#### classical_ml/regression_models.py

- **Classes:**
  - `LinearRegressionModel`: Implements a linear regression model.
    - `train(X, y)`: Trains the model on the provided features and target.
    - `predict(X)`: Predicts the target for the given features.
    - `evaluate(y_true, y_pred)`: Evaluates the model performance using metrics.

#### quantum_ml/quantum_circuits.py

- **Classes:**
  - `QuantumCircuit`: Defines a quantum circuit for quantum machine learning tasks.
    - `simulate()`: Simulates the quantum circuit.
    - `evaluate()`: Evaluates the performance of the quantum model.

### Drug Design

#### molecular_generation.py

- **Functions:**
  - `generate_molecule()`: Generates a new molecular structure using generative models.

#### property_prediction.py

- **Functions:**
  - `predict_properties(molecule)`: Predicts molecular properties using machine learning techniques.

### Utilities

#### visualization.py

- **Functions:**
  - `plot_molecule(molecule)`: Visualizes the molecular structure.
  - `plot_results(results)`: Plots the results of model evaluations.

#### metrics.py

- **Functions:**
  - `calculate_accuracy(y_true, y_pred)`: Calculates the accuracy of the model predictions.
  - `calculate_precision(y_true, y_pred)`: Calculates the precision of the model predictions.
  - `calculate_recall(y_true, y_pred)`: Calculates the recall of the model predictions.

## Usage

Refer to the respective module documentation for detailed usage examples and additional information on each function and class.
