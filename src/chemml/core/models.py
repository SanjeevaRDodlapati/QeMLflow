"""
ChemML Core Models
=================

Machine learning models optimized for chemistry and drug discovery applications.
Provides both classical ML and deep learning approaches with consistent APIs.

Key Features:
- Unified interface for all model types
- Built-in evaluation metrics for chemistry tasks
- Integration with popular ML libraries
- Experiment tracking and reproducibility
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class BaseModel(ABC):
    """Abstract base class for all ChemML models."""

    def __init__(self, task_type: str = "regression", **kwargs) -> None:
        """
        Initialize base model.

        Args:
            task_type: 'regression' or 'classification'
        """
        self.task_type = task_type
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.feature_names = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(X)
        if self.task_type == "regression":
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)
            rmse = np.sqrt(mse)
            return {"mse": mse, "rmse": rmse, "r2": r2}
        else:
            accuracy = accuracy_score(y, predictions)
            metrics = {"accuracy": accuracy}
            try:
                if len(np.unique(y)) == 2:
                    auc = roc_auc_score(y, predictions)
                    metrics["auc"] = auc
            except Exception:
                pass
            return metrics


class LinearModel(BaseModel):
    """
    Linear regression models with regularization options.

    Supports standard linear regression, Ridge, and Lasso regularization.
    """

    def __init__(
        self,
        regularization: str = "none",
        alpha: float = 1.0,
        normalize_features: bool = True,
        **kwargs,
    ):
        """
        Initialize linear model.

        Args:
            regularization: 'none', 'ridge', or 'lasso'
            alpha: Regularization strength
            normalize_features: Whether to normalize input features
        """
        super().__init__(**kwargs)
        self.regularization = regularization
        self.alpha = alpha
        self.normalize_features = normalize_features
        if regularization == "ridge":
            self.model = Ridge(alpha=alpha)
        elif regularization == "lasso":
            self.model = Lasso(alpha=alpha)
        else:
            self.model = LinearRegression()
        if normalize_features:
            self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Fit linear model to data."""
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        val_metrics = self.evaluate(X_val, y_val)
        return val_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        return self.model.predict(X_scaled)


class RandomForestModel(BaseModel):
    """
    Random Forest model for regression and classification tasks.

    Provides excellent baseline performance for molecular property prediction.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize Random Forest model.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            random_state: Random seed for reproducibility
        """
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        if self.task_type == "regression":
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Fit Random Forest model."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        val_metrics = self.evaluate(X_val, y_val)
        return val_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.feature_importances_


class SVMModel(BaseModel):
    """
    Support Vector Machine for regression and classification.

    Good for non-linear relationships in molecular data.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        normalize_features: bool = True,
        **kwargs,
    ):
        """
        Initialize SVM model.

        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly')
            C: Regularization parameter
            gamma: Kernel coefficient
            normalize_features: Whether to normalize features
        """
        super().__init__(**kwargs)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.normalize_features = normalize_features
        if self.task_type == "regression":
            self.model = SVR(kernel=kernel, C=C, gamma=gamma)
        else:
            self.model = SVC(kernel=kernel, C=C, gamma=gamma)
        if normalize_features:
            self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Fit SVM model."""
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        val_metrics = self.evaluate(X_val, y_val)
        return val_metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        return self.model.predict(X_scaled)


if HAS_TORCH:

    class NeuralNetwork(BaseModel, nn.Module):
        """
        Feed-forward neural network for molecular property prediction.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int] = [256, 128, 64],
            output_dim: int = 1,
            dropout_rate: float = 0.2,
            **kwargs,
        ):
            """
            Initialize neural network.

            Args:
                input_dim: Number of input features
                hidden_dims: List of hidden layer dimensions
                output_dim: Number of outputs
                dropout_rate: Dropout probability
            """
            BaseModel.__init__(self, **kwargs)
            nn.Module.__init__(self)
            self.input_dim = input_dim
            self.hidden_dims = hidden_dims
            self.output_dim = output_dim
            self.dropout_rate = dropout_rate
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                    ]
                )
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)
            self.scaler = StandardScaler()

        def forward(self, x) -> Any:
            """Forward pass."""
            return self.network(x)

        def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            **kwargs,
        ) -> Dict[str, float]:
            """Fit neural network."""
            X_scaled = self.scaler.fit_transform(X)
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
            X_val_tensor = torch.FloatTensor(X_val)
#_y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1))
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            criterion = (
                nn.MSELoss()
                if self.task_type == "regression"
                else nn.CrossEntropyLoss()
            )
            self.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.forward(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
            self.is_fitted = True
            self.eval()
            with torch.no_grad():
                val_outputs = self.forward(X_val_tensor)
                val_predictions = val_outputs.numpy().flatten()
            val_metrics = {}
            if self.task_type == "regression":
                mse = mean_squared_error(y_val, val_predictions)
                r2 = r2_score(y_val, val_predictions)
                val_metrics = {"mse": mse, "r2": r2, "rmse": np.sqrt(mse)}
            return val_metrics

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Make predictions."""
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            self.eval()
            with torch.no_grad():
                outputs = self.forward(X_tensor)
                return outputs.numpy().flatten()


def setup_experiment_tracking(experiment_name: str, config: Dict = None) -> Any:
    """Setup experiment tracking with Weights & Biases."""
    if not HAS_WANDB:
        print("Warning: wandb not available. Install with: pip install wandb")
        return None
    try:
        run = wandb.init(
            project="chemml-experiments",
            name=experiment_name,
            config=config or {},
            tags=["chemml"],
        )
        print(f"✅ Experiment tracking started: {run.url}")
        return run
    except Exception as e:
        print(f"⚠️ Experiment tracking setup failed: {e}")
        return None


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to experiment tracker."""
    if HAS_WANDB and wandb.run is not None:
        wandb.log(metrics, step=step)


def compare_models(
    models: Dict[str, BaseModel], X: np.ndarray, y: np.ndarray
) -> pd.DataFrame:
    """
    Compare performance of multiple models.

    Args:
        models: Dictionary of model_name -> model_instance
        X: Feature matrix
        y: Target values

    Returns:
        DataFrame with comparison results
    """
    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            metrics = model.fit(X, y)
            metrics["model"] = name
            results.append(metrics)
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
    return pd.DataFrame(results)


def create_linear_model(regularization: str = "none", **kwargs) -> LinearModel:
    """Create a linear regression model."""
    return LinearModel(regularization=regularization, **kwargs)


def create_rf_model(task_type: str = "regression", **kwargs) -> RandomForestModel:
    """Create a Random Forest model."""
    return RandomForestModel(task_type=task_type, **kwargs)


def create_svm_model(task_type: str = "regression", **kwargs) -> SVMModel:
    """Create an SVM model."""
    return SVMModel(task_type=task_type, **kwargs)


if HAS_TORCH:

    def create_neural_network(input_dim: int, **kwargs) -> NeuralNetwork:
        """Create a neural network model."""
        return NeuralNetwork(input_dim=input_dim, **kwargs)


# Alias for backward compatibility and generic usage
Model = BaseModel

#__all__ = [
    "BaseModel",
    "Model",  # Add the alias to exports
    "LinearModel",
    "RandomForestModel",
    "SVMModel",
    "setup_experiment_tracking",
    "log_metrics",
    "compare_models",
    "create_linear_model",
    "create_rf_model",
    "create_svm_model",
]
if HAS_TORCH:
    __all__.extend(["NeuralNetwork", "create_neural_network"])
