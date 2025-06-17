"""
QeMLflow DeepChem Integration
==========================

Seamless integration with DeepChem library.
Provides wrappers and utilities for using DeepChem models with QeMLflow.

Key Features:
- DeepChem model wrappers with consistent APIs
- Data format conversions between QeMLflow and DeepChem
- Hybrid workflows combining custom and DeepChem featurizers
- Model ensemble capabilities
"""

import warnings

import numpy as np
import pandas as pd

# DeepChem imports
try:
    import deepchem as dc

    HAS_DEEPCHEM = True
except ImportError:
    HAS_DEEPCHEM = False


class DeepChemModelWrapper:
    """
    Wrapper for DeepChem models to provide consistent QeMLflow interface.
    """

    def __init__(self, model_type: str = "multitask_regressor", **kwargs) -> None:
        """
        Initialize DeepChem model wrapper.

        Args:
            model_type: Type of DeepChem model
            **kwargs: Model-specific parameters
        """
        if not HAS_DEEPCHEM:
            raise ImportError(
                "DeepChem not available. Install with: pip install deepchem"
            )

        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs) -> Dict[str, float]:
        """
        Fit DeepChem model to data.

        Args:
            X: Feature matrix
            y: Target values
            **fit_kwargs: Additional fitting parameters

        Returns:
            Dictionary with training metrics
        """
        # Convert to DeepChem dataset format
        dataset = dc.data.NumpyDataset(X=X, y=y)

        # Create model based on type
        if self.model_type == "multitask_regressor":
            self.model = dc.models.MultitaskRegressor(
                n_tasks=y.shape[1] if y.ndim > 1 else 1,
                n_features=X.shape[1],
                **self.kwargs,
            )
        elif self.model_type == "multitask_classifier":
            self.model = dc.models.MultitaskClassifier(
                n_tasks=y.shape[1] if y.ndim > 1 else 1,
                n_features=X.shape[1],
                **self.kwargs,
            )
        elif self.model_type == "graph_conv":
            self.model = dc.models.GraphConvModel(
                n_tasks=y.shape[1] if y.ndim > 1 else 1, **self.kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Fit model
        self.model.fit(dataset, **fit_kwargs)
        self.is_fitted = True

        # Evaluate on training data
        train_scores = self.model.evaluate(
            dataset, metrics=[dc.metrics.Metric(dc.metrics.mean_squared_error)]
        )

        return {"train_mse": train_scores["mean_squared_error"]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        dataset = dc.data.NumpyDataset(X=X)
        predictions = self.model.predict(dataset)

        return predictions

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        dataset = dc.data.NumpyDataset(X=X, y=y)

        # Choose metrics based on model type
        if "classifier" in self.model_type:
            metrics = [dc.metrics.Metric(dc.metrics.accuracy_score)]
        else:
            metrics = [
                dc.metrics.Metric(dc.metrics.mean_squared_error),
                dc.metrics.Metric(dc.metrics.r2_score),
            ]

        scores = self.model.evaluate(dataset, metrics=metrics)
        return scores


class HybridFeaturizer:
    """
    Combine QeMLflow custom featurizers with DeepChem featurizers.
    """

    def __init__(
        self, qemlflow_featurizers: List = None, deepchem_featurizers: List = None
    ):
        """
        Initialize hybrid featurizer.

        Args:
            qemlflow_featurizers: List of QeMLflow featurizer instances
            deepchem_featurizers: List of DeepChem featurizer instances
        """
        if not HAS_DEEPCHEM:
            warnings.warn(
                "DeepChem not available. Only QeMLflow featurizers will be used."
            )

        # Set up default featurizers if none provided
        if qemlflow_featurizers is None:
            from ..core.featurizers import DescriptorCalculator, MorganFingerprint

            qemlflow_featurizers = [
                MorganFingerprint(radius=2, n_bits=1024),
                DescriptorCalculator(["MolWt", "LogP", "TPSA"]),
            ]

        if deepchem_featurizers is None and HAS_DEEPCHEM:
            import deepchem as dc

            deepchem_featurizers = [
                dc.feat.CircularFingerprint(size=1024),
                dc.feat.RDKitDescriptors(),
            ]

        self.qemlflow_featurizers = qemlflow_featurizers or []
        self.deepchem_featurizers = deepchem_featurizers or []

    def featurize(self, smiles_list: List[str]) -> np.ndarray:
        """
        Generate combined features using both QeMLflow and DeepChem featurizers.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Combined feature matrix
        """
        all_features = []

        # QeMLflow featurizers
        for featurizer in self.qemlflow_featurizers:
            features = featurizer.featurize(smiles_list)
            all_features.append(features)

        # DeepChem featurizers
        if HAS_DEEPCHEM:
            for featurizer in self.deepchem_featurizers:
                try:
                    features = featurizer.featurize(smiles_list)
                    # Handle potential failed featurizations
                    if hasattr(features, "X"):
                        features = features.X
                    all_features.append(features)
                except Exception as e:
                    warnings.warn(f"DeepChem featurizer failed: {e}")

        # Combine all features
        if all_features:
            return np.hstack(all_features)
        else:
            raise ValueError("No valid features generated")


def create_deepchem_dataset(
    smiles_list: List[str],
    labels: Optional[np.ndarray] = None,
    featurizer: str = "ECFP",
) -> "dc.data.Dataset":
    """
    Create DeepChem dataset from SMILES and labels.

    Args:
        smiles_list: List of SMILES strings
        labels: Optional labels
        featurizer: Type of featurizer to use

    Returns:
        DeepChem dataset object
    """
    if not HAS_DEEPCHEM:
        raise ImportError("DeepChem not available")

    # Select featurizer
    if featurizer == "ECFP":
        featurizer_obj = dc.feat.CircularFingerprint(size=1024)
    elif featurizer == "GraphConv":
        featurizer_obj = dc.feat.ConvMolFeaturizer()
    elif featurizer == "Weave":
        featurizer_obj = dc.feat.WeaveFeaturizer()
    else:
        raise ValueError(f"Unknown featurizer: {featurizer}")

    # Create dataset
    loader = dc.data.CSVLoader(
        tasks=[] if labels is None else ["target"],
        feature_field="smiles",
        featurizer=featurizer_obj,
    )

    # Create temporary DataFrame
    data_dict = {"smiles": smiles_list}
    if labels is not None:
        data_dict["target"] = labels

    df = pd.DataFrame(data_dict)

    # Save temporarily and load (DeepChem expects file input)
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name

    try:
        dataset = loader.create_dataset(temp_file)
        return dataset
    finally:
        os.unlink(temp_file)


def deepchem_to_qemlflow_format(
    dataset: "dc.data.Dataset",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert DeepChem dataset to QeMLflow format.

    Args:
        dataset: DeepChem dataset

    Returns:
        Tuple of (features, labels)
    """
    if not HAS_DEEPCHEM:
        raise ImportError("DeepChem not available")

    features = dataset.X
    labels = dataset.y if dataset.y is not None else None

    return features, labels


def create_deepchem_model_ensemble(
    model_configs: List[Dict],
) -> List[DeepChemModelWrapper]:
    """
    Create ensemble of DeepChem models.

    Args:
        model_configs: List of model configuration dictionaries

    Returns:
        List of model wrappers
    """
    ensemble = []

    for config in model_configs:
        model_type = config.pop("model_type", "multitask_regressor")
        wrapper = DeepChemModelWrapper(model_type=model_type, **config)
        ensemble.append(wrapper)

    return ensemble


def benchmark_deepchem_featurizers(smiles_list: List[str]) -> Dict[str, Dict]:
    """
    Benchmark different DeepChem featurizers.

    Args:
        smiles_list: List of SMILES for benchmarking

    Returns:
        Dictionary with benchmark results
    """
    if not HAS_DEEPCHEM:
        warnings.warn("DeepChem not available. Cannot benchmark featurizers.")
        return {}

    import time

    featurizers = {
        "ECFP": dc.feat.CircularFingerprint(size=1024),
        "MACCS": dc.feat.MACCSKeysFingerprint(),
        "RDKitDescriptors": dc.feat.RDKitDescriptors(),
        "ConvMol": dc.feat.ConvMolFeaturizer(),
    }

    results = {}

    for name, featurizer in featurizers.items():
        try:
            start_time = time.time()
            features = featurizer.featurize(smiles_list)
            end_time = time.time()

            # Extract feature array
            if hasattr(features, "X"):
                feature_array = features.X
            else:
                feature_array = features

            # Handle GraphConv features (list of graph objects)
            if isinstance(feature_array, list):
                valid_count = sum(1 for f in feature_array if f is not None)
                feature_shape = "variable_graph"
            else:
                valid_count = len(feature_array)
                feature_shape = (
                    feature_array.shape
                    if feature_array.ndim > 1
                    else (len(feature_array),)
                )

            results[name] = {
                "execution_time": end_time - start_time,
                "features_shape": feature_shape,
                "valid_molecules": valid_count,
                "success_rate": valid_count / len(smiles_list),
            }

        except Exception as e:
            results[name] = {"error": str(e), "success_rate": 0.0}

    return results


# Convenience functions
def quick_deepchem_model(
    X: np.ndarray, y: np.ndarray, model_type: str = "multitask_regressor"
) -> DeepChemModelWrapper:
    """Quickly create and train a DeepChem model."""
    model = DeepChemModelWrapper(model_type=model_type)
    model.fit(X, y)
    return model


def deepchem_molecular_transformer(
    smiles_list: List[str],
) -> Optional["dc.models.TorchModel"]:
    """Create DeepChem molecular transformer model."""
    if not HAS_DEEPCHEM:
        warnings.warn("DeepChem not available.")
        return None

    try:
        # This would create a transformer model in newer DeepChem versions
        # For now, return a placeholder
        warnings.warn("Molecular transformer not implemented in this DeepChem version.")
        return None
    except Exception as e:
        warnings.warn(f"Error creating molecular transformer: {e}")
        return None


# Export main classes and functions
__all__ = [
    "DeepChemModelWrapper",
    "HybridFeaturizer",
    "create_deepchem_dataset",
    "deepchem_to_qemlflow_format",
    "create_deepchem_model_ensemble",
    "benchmark_deepchem_featurizers",
    "quick_deepchem_model",
]
