"""
ChemML Drug Discovery - Property Prediction
==========================================

Comprehensive molecular property prediction tools for drug discovery.

This module provides:
- Molecular descriptor calculation
- Physicochemical property prediction
- Custom property model training
- Property prediction pipelines
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors

    RDKIT_AVAILABLE = True
except ImportError:
    logging.warning("RDKit not available. Property prediction will be limited.")
    RDKIT_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning("Scikit-learn not available. Property prediction will be limited.")
    SKLEARN_AVAILABLE = False


class MolecularPropertyPredictor:
    """
    Predict various molecular properties relevant to drug discovery.
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.trained_properties = set()

    def calculate_molecular_descriptors(self, smiles: str) -> Dict[str, float]:
        """
        Calculate molecular descriptors for property prediction.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary of calculated descriptors
        """
        if not RDKIT_AVAILABLE:
            return {
                "molecular_weight": 0.0,
                "logp": 0.0,
                "tpsa": 0.0,
                "hbd": 0,
                "hba": 0,
                "rotatable_bonds": 0,
                "aromatic_rings": 0,
            }

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {
                    desc: 0.0
                    for desc in [
                        "molecular_weight",
                        "logp",
                        "tpsa",
                        "hbd",
                        "hba",
                        "rotatable_bonds",
                        "aromatic_rings",
                    ]
                }

            descriptors = {
                "molecular_weight": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "tpsa": Descriptors.TPSA(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "aromatic_rings": Descriptors.NumAromaticRings(mol),
                "heavy_atoms": Descriptors.HeavyAtomCount(mol),
                "rings": Descriptors.RingCount(mol),
                "molar_refractivity": Crippen.MolMR(mol),
                "num_heterocycles": Descriptors.NumHeterocycles(mol),
                "num_saturated_rings": Descriptors.NumSaturatedRings(mol),
                "fraction_csp3": Descriptors.FractionCsp3(mol),
                "num_aliphatic_carbocycles": Descriptors.NumAliphaticCarbocycles(mol),
                "num_aliphatic_heterocycles": Descriptors.NumAliphaticHeterocycles(mol),
            }

            return descriptors

        except Exception as e:
            logging.warning(f"Error calculating descriptors for SMILES {smiles}: {e}")
            return {
                desc: 0.0
                for desc in [
                    "molecular_weight",
                    "logp",
                    "tpsa",
                    "hbd",
                    "hba",
                    "rotatable_bonds",
                    "aromatic_rings",
                ]
            }

    def predict_physicochemical_properties(self, smiles: str) -> Dict[str, float]:
        """
        Predict basic physicochemical properties.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary with predicted properties
        """
        if not RDKIT_AVAILABLE:
            return {
                "solubility": 0.0,
                "permeability": 0.5,
                "stability": 0.7,
                "bioavailability": 0.5,
            }

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {
                    "solubility": 0.0,
                    "permeability": 0.0,
                    "stability": 0.0,
                    "bioavailability": 0.0,
                }

            # Get basic descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            # Rule-based predictions (simplified)
            # Solubility prediction (LogS approximation)
            solubility = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * tpsa - 0.74 * hbd

            # Permeability prediction (simplified Caco-2)
            permeability = 1.0 if (logp > 1 and logp < 3 and tpsa < 60) else 0.3

            # Stability prediction (simplified)
            stability = 0.8 if (mw < 500 and logp < 5) else 0.4

            # Bioavailability prediction (simplified)
            bioavailability = (
                0.85 if (mw < 500 and logp < 5 and hbd <= 5 and hba <= 10) else 0.3
            )

            return {
                "solubility": max(0.0, min(1.0, solubility)),
                "permeability": max(0.0, min(1.0, permeability)),
                "stability": max(0.0, min(1.0, stability)),
                "bioavailability": max(0.0, min(1.0, bioavailability)),
            }

        except Exception as e:
            logging.warning(f"Error predicting properties for SMILES {smiles}: {e}")
            return {
                "solubility": 0.0,
                "permeability": 0.0,
                "stability": 0.0,
                "bioavailability": 0.0,
            }

    def train_property_model(
        self,
        training_data: pd.DataFrame,
        property_name: str,
        smiles_column: str = "SMILES",
        target_column: str = None,
    ) -> Dict[str, float]:
        """
        Train a model for predicting a specific molecular property.

        Args:
            training_data: DataFrame with SMILES and target values
            property_name: Name of the property to predict
            smiles_column: Name of the SMILES column
            target_column: Name of the target column (defaults to property_name)

        Returns:
            Dictionary with training metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for model training")

        if target_column is None:
            target_column = property_name

        if target_column not in training_data.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in training data"
            )

        # Calculate descriptors for all SMILES
        descriptor_list = []
        target_values = []

        for idx, row in training_data.iterrows():
            smiles = row[smiles_column]
            target = row[target_column]

            if pd.isna(target) or pd.isna(smiles):
                continue

            descriptors = self.calculate_molecular_descriptors(smiles)
            if descriptors:
                descriptor_list.append(descriptors)
                target_values.append(target)

        if len(descriptor_list) == 0:
            raise ValueError("No valid SMILES/target pairs found")

        # Convert to DataFrame
        X = pd.DataFrame(descriptor_list)
        y = np.array(target_values)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # Store model and scaler
        self.models[property_name] = model
        self.scalers[property_name] = scaler
        self.trained_properties.add(property_name)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
        }

        logging.info(f"Trained model for {property_name}. Metrics: {metrics}")
        return metrics

    def predict_property(
        self, smiles: Union[str, List[str]], property_name: str
    ) -> Union[float, List[float]]:
        """
        Predict a specific property for given SMILES.

        Args:
            smiles: Single SMILES string or list of SMILES
            property_name: Name of the property to predict

        Returns:
            Predicted value(s)
        """
        if property_name not in self.trained_properties:
            raise ValueError(f"No trained model found for property '{property_name}'")

        single_smiles = isinstance(smiles, str)
        if single_smiles:
            smiles = [smiles]

        predictions = []
        for smi in smiles:
            descriptors = self.calculate_molecular_descriptors(smi)
            if descriptors:
                X = pd.DataFrame([descriptors])
                X_scaled = self.scalers[property_name].transform(X)
                pred = self.models[property_name].predict(X_scaled)[0]
                predictions.append(pred)
            else:
                predictions.append(np.nan)

        return predictions[0] if single_smiles else predictions

    def predict_multiple_properties(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Predict multiple properties for a list of SMILES.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            DataFrame with predictions for all trained properties
        """
        results = {"SMILES": smiles_list}

        for prop_name in self.trained_properties:
            try:
                predictions = self.predict_property(smiles_list, prop_name)
                results[f"predicted_{prop_name}"] = predictions
            except Exception as e:
                logging.warning(f"Error predicting {prop_name}: {e}")
                results[f"predicted_{prop_name}"] = [np.nan] * len(smiles_list)

        # Add physicochemical properties
        for i, smiles in enumerate(smiles_list):
            if i == 0:  # Initialize columns on first iteration
                phys_props = self.predict_physicochemical_properties(smiles)
                for prop_name in phys_props:
                    results[prop_name] = []

            phys_props = self.predict_physicochemical_properties(smiles)
            for prop_name, value in phys_props.items():
                results[prop_name].append(value)

        return pd.DataFrame(results)


class TrainedPropertyModel:
    """Wrapper class for trained property prediction models."""

    def __init__(self, model_dict: Dict):
        self.model_dict = model_dict
        self.model = model_dict["model"]
        self.scaler = model_dict.get("scaler")
        self.metrics = model_dict["metrics"]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def get_metrics(self) -> Dict:
        """Get training metrics."""
        return self.metrics


# Module-level convenience functions


def predict_properties(
    molecular_data: Union[pd.DataFrame, List[str]],
    model: Optional[MolecularPropertyPredictor] = None,
    smiles_column: str = "SMILES",
) -> pd.DataFrame:
    """
    Predict molecular properties for a dataset.

    Args:
        molecular_data: DataFrame containing SMILES or list of SMILES strings
        model: Trained property predictor (optional, creates new if None)
        smiles_column: Name of the SMILES column (when DataFrame input)

    Returns:
        DataFrame with predicted properties
    """
    if model is None:
        model = MolecularPropertyPredictor()

    # Handle different input types
    if isinstance(molecular_data, list):
        smiles_list = molecular_data
    elif isinstance(molecular_data, pd.DataFrame):
        smiles_list = molecular_data[smiles_column].tolist()
    else:
        raise TypeError("molecular_data must be a DataFrame or list of SMILES strings")

    predictions = model.predict_multiple_properties(smiles_list)

    return predictions


def preprocess_data(molecular_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess molecular data for property prediction.

    Args:
        molecular_data: DataFrame with molecular data

    Returns:
        Preprocessed DataFrame
    """
    cleaned_data = handle_missing_values(molecular_data)
    normalized_data = normalize_data(cleaned_data)
    return normalized_data


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in molecular data.

    Args:
        data: DataFrame with potential missing values

    Returns:
        DataFrame with missing values handled
    """
    # For numeric columns, fill with mean
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    # For string columns, fill with 'Unknown'
    string_columns = data.select_dtypes(include=["object"]).columns
    data[string_columns] = data[string_columns].fillna("Unknown")

    return data


def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numeric data for better model performance.

    Args:
        data: DataFrame to normalize

    Returns:
        Normalized DataFrame
    """
    numeric_columns = data.select_dtypes(include=[np.number]).columns

    # Min-Max normalization
    for col in numeric_columns:
        if data[col].max() != data[col].min():  # Avoid division by zero
            data[col] = (data[col] - data[col].min()) / (
                data[col].max() - data[col].min()
            )

    return data


def evaluate_model(
    predictions: np.ndarray, true_values: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate model performance with multiple metrics.

    Args:
        predictions: Predicted values
        true_values: True values

    Returns:
        Dictionary of evaluation metrics
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "sklearn not available"}

    metrics = calculate_metrics(predictions, true_values)

    # Add additional metrics
    metrics["mse"] = mean_squared_error(true_values, predictions)
    metrics["r2"] = r2_score(true_values, predictions)

    return metrics


def calculate_metrics(
    predictions: np.ndarray, true_values: np.ndarray
) -> Dict[str, float]:
    """
    Calculate basic evaluation metrics.

    Args:
        predictions: Predicted values
        true_values: True values

    Returns:
        Dictionary with MAE and RMSE
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "sklearn not available"}

    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)

    return {"MAE": mae, "RMSE": rmse}


def train_property_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "regression",
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainedPropertyModel:
    """
    Train a property prediction model.

    Args:
        X: Feature matrix
        y: Target values
        model_type: Type of model ('regression' or 'classification')
        test_size: Fraction of data to use for testing
        random_state: Random state for reproducibility

    Returns:
        TrainedPropertyModel object with predict() method
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("Scikit-learn is required for model training")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Choose model based on type
    if model_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }
    else:  # regression
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
        }

    model_dict = {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "predictions": y_pred,
        "true_values": y_test,
    }

    return TrainedPropertyModel(model_dict)


def evaluate_property_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, task_type: str = "regression"
) -> Dict[str, float]:
    """
    Evaluate property prediction performance.

    Args:
        y_true: True values
        y_pred: Predicted values
        task_type: Type of task ('regression' or 'classification')

    Returns:
        Dictionary of evaluation metrics
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "sklearn not available"}

    if task_type == "classification":
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
    else:  # regression
        return {
            "r2_score": r2_score(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
        }
