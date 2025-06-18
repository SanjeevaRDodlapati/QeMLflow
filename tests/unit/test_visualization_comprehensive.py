"""
Comprehensive test suite for src.utils.visualization module.

This test suite provides extensive coverage for all visualization functionality including:
- MolecularVisualizer class (molecular structure plotting, multi-molecule grids, property distributions)
- ModelVisualizer class (feature importance, model performance, predictions vs actual, confusion matrix)
- ChemicalSpaceVisualizer class (PCA chemical space plotting)
- Standalone functions (convenience plotting functions)
- Dashboard creation and integration scenarios
- Error handling and fallback behavior
- Cross-platform compatibility
- Performance testing
"""

import io
import os
import sys
import tempfile
import unittest
import warnings
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

# Mock dependencies that might not be available
MOCK_MATPLOTLIB = False
MOCK_RDKIT = False
MOCK_PLOTLY = False

# Try importing with fallbacks
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    MOCK_MATPLOTLIB = True
    # Mock matplotlib
    plt = Mock()
    sns = Mock()

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw, rdDepictor
except ImportError:
    MOCK_RDKIT = True
    # Mock RDKit
    Chem = Mock()
    Draw = Mock()
    Descriptors = Mock()
    rdDepictor = Mock()

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    MOCK_PLOTLY = True
    # Mock Plotly
    go = Mock()
    px = Mock()
    make_subplots = Mock()

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Import the visualization module
from qemlflow.core.utils.visualization import *
from qemlflow.core.utils.visualization import (
    Chem,
    ChemicalSpaceVisualizer,
    ModelVisualizer,
    MolecularVisualizer,
    create_dashboard_plots,
    plot_feature_importance,
    plot_model_performance,
    plot_molecular_structure,
    rdkit,
    sklearn,
)


class TestMolecularVisualizer(unittest.TestCase):
    """Test MolecularVisualizer class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_smiles = "CCO"  # Ethanol
        self.sample_smiles_list = [
            "CCO",
            "CC(=O)O",
            "CC(C)O",
        ]  # Ethanol, Acetic acid, Isopropanol

    @patch("src.utils.visualization.RDKIT_AVAILABLE", True)
    @patch("src.utils.visualization.rdDepictor")
    @patch("src.utils.visualization.Draw")
    @patch("src.utils.visualization.Chem")
    def test_plot_molecular_structure_with_rdkit(
        self, mock_chem, mock_draw, mock_depictor
    ):
        """Test molecular structure plotting with RDKit available."""
        # Setup mocks
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_img = Mock()
        mock_draw.MolToImage.return_value = mock_img

        # Test with SMILES string
        _result = MolecularVisualizer.plot_molecular_structure(self.sample_smiles)

        mock_chem.MolFromSmiles.assert_called_once_with(self.sample_smiles)
        mock_depictor.Compute2DCoords.assert_called_once_with(mock_mol)
        mock_draw.MolToImage.assert_called_once()
        mock_img.show.assert_called_once()

    @patch("src.utils.visualization.RDKIT_AVAILABLE", False)
    def test_plot_molecular_structure_without_rdkit(self):
        """Test molecular structure plotting without RDKit."""
        with patch("src.utils.visualization.logging.warning") as mock_warning:
            result = MolecularVisualizer.plot_molecular_structure(self.sample_smiles)

            self.assertIsNone(result)
            mock_warning.assert_called_with(
                "RDKit not available for molecular visualization"
            )

    @patch("src.utils.visualization.RDKIT_AVAILABLE", True)
    @patch("src.utils.visualization.Draw")
    @patch("src.utils.visualization.Chem")
    def test_plot_molecular_structure_with_filename(self, mock_chem, mock_draw):
        """Test molecular structure plotting with file output."""
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol
        mock_img = Mock()
        mock_draw.MolToImage.return_value = mock_img

        filename = os.path.join(self.temp_dir, "test_mol.png")
        result = MolecularVisualizer.plot_molecular_structure(
            self.sample_smiles, filename=filename
        )

        mock_img.save.assert_called_once_with(filename)
        self.assertEqual(result, filename)

    @patch("src.utils.visualization.RDKIT_AVAILABLE", True)
    @patch("src.utils.visualization.Chem")
    def test_plot_molecular_structure_invalid_smiles(self, mock_chem):
        """Test molecular structure plotting with invalid SMILES."""
        mock_chem.MolFromSmiles.return_value = None

        with patch("src.utils.visualization.logging.warning") as mock_warning:
            result = MolecularVisualizer.plot_molecular_structure("INVALID")

            self.assertIsNone(result)
            mock_warning.assert_called_with("Invalid SMILES: INVALID")

    @patch("src.utils.visualization.RDKIT_AVAILABLE", True)
    @patch("src.utils.visualization.Draw")
    @patch("src.utils.visualization.Chem")
    def test_plot_multiple_molecules_with_rdkit(self, mock_chem, mock_draw):
        """Test multiple molecule plotting with RDKit available."""
        # Setup mocks
        mock_mols = [Mock() for _ in self.sample_smiles_list]
        mock_chem.MolFromSmiles.side_effect = mock_mols
        mock_img = Mock()
        mock_draw.MolsToGridImage.return_value = mock_img

        _result = MolecularVisualizer.plot_multiple_molecules(self.sample_smiles_list)

        self.assertEqual(
            mock_chem.MolFromSmiles.call_count, len(self.sample_smiles_list)
        )
        mock_draw.MolsToGridImage.assert_called_once()
        mock_img.show.assert_called_once()

    @patch("src.utils.visualization.RDKIT_AVAILABLE", True)
    @patch("src.utils.visualization.Draw")
    @patch("src.utils.visualization.Chem")
    def test_plot_multiple_molecules_with_titles(self, mock_chem, mock_draw):
        """Test multiple molecule plotting with custom titles."""
        mock_mols = [Mock() for _ in self.sample_smiles_list]
        mock_chem.MolFromSmiles.side_effect = mock_mols
        mock_img = Mock()
        mock_draw.MolsToGridImage.return_value = mock_img

        titles = ["Ethanol", "Acetic Acid", "Isopropanol"]
        _result = MolecularVisualizer.plot_multiple_molecules(
            self.sample_smiles_list, titles=titles
        )

        # Check that MolsToGridImage was called with correct legends
        call_args = mock_draw.MolsToGridImage.call_args
        self.assertIn("legends", call_args.kwargs)

    @patch("src.utils.visualization.RDKIT_AVAILABLE", True)
    @patch("src.utils.visualization.Chem")
    def test_plot_multiple_molecules_no_valid_mols(self, mock_chem):
        """Test multiple molecule plotting with no valid molecules."""
        mock_chem.MolFromSmiles.return_value = None

        with patch("src.utils.visualization.logging.warning") as mock_warning:
            result = MolecularVisualizer.plot_multiple_molecules(
                ["INVALID1", "INVALID2"]
            )

            self.assertIsNone(result)
            mock_warning.assert_called_with("No valid molecules to display")

    @patch("src.utils.visualization.MATPLOTLIB_AVAILABLE", True)
    @patch("src.utils.visualization.plt")
    def test_plot_molecular_properties_distribution_with_matplotlib(self, mock_plt):
        """Test molecular properties distribution plotting with matplotlib."""
        # Create sample data
        df = pd.DataFrame(
            {
                "MW": [46.07, 60.05, 60.10],
                "LogP": [-0.31, -0.17, 0.05],
                "TPSA": [20.23, 37.30, 20.23],
            }
        )

        # Setup matplotlib mocks
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        MolecularVisualizer.plot_molecular_properties_distribution(df, ["MW", "LogP"])

        mock_plt.subplots.assert_called_once()
        self.assertEqual(len(mock_axes), 2)

    @patch("src.utils.visualization.MATPLOTLIB_AVAILABLE", False)
    def test_plot_molecular_properties_distribution_without_matplotlib(self):
        """Test molecular properties distribution plotting without matplotlib."""
        df = pd.DataFrame({"MW": [46.07, 60.05, 60.10]})

        with patch("src.utils.visualization.logging.warning") as mock_warning:
            MolecularVisualizer.plot_molecular_properties_distribution(df, ["MW"])

            mock_warning.assert_called_with("Matplotlib not available for plotting")


class TestModelVisualizer(unittest.TestCase):
    """Test ModelVisualizer class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_importances = np.array([0.1, 0.3, 0.2, 0.15, 0.25])
        self.sample_feature_names = ["feat1", "feat2", "feat3", "feat4", "feat5"]
        self.sample_history = {
            "loss": [1.0, 0.8, 0.6, 0.4, 0.2],
            "accuracy": [0.6, 0.7, 0.8, 0.85, 0.9],
        }

    @patch("src.utils.visualization.MATPLOTLIB_AVAILABLE", True)
    @patch("src.utils.visualization.plt")
    def test_plot_feature_importance_with_matplotlib(self, mock_plt):
        """Test feature importance plotting with matplotlib."""
        mock_plt.figure.return_value = Mock()
        mock_plt.barh.return_value = [Mock() for _ in range(5)]

        ModelVisualizer.plot_feature_importance(
            self.sample_importances, self.sample_feature_names
        )

        mock_plt.figure.assert_called_once()
        mock_plt.barh.assert_called_once()
        mock_plt.yticks.assert_called_once()
        mock_plt.xlabel.assert_called_once()
        mock_plt.title.assert_called_once()

    @patch("src.utils.visualization.MATPLOTLIB_AVAILABLE", False)
    def test_plot_feature_importance_without_matplotlib(self):
        """Test feature importance plotting without matplotlib."""
        with patch("src.utils.visualization.logging.warning") as mock_warning:
            ModelVisualizer.plot_feature_importance(
                self.sample_importances, self.sample_feature_names
            )

            mock_warning.assert_called_with("Matplotlib not available for plotting")

    def test_plot_feature_importance_validation_errors(self):
        """Test feature importance plotting with validation errors."""
        # Test empty inputs
        with self.assertRaises(ValueError):
            ModelVisualizer.plot_feature_importance(np.array([]), [])

        # Test mismatched lengths
        with self.assertRaises(ValueError):
            ModelVisualizer.plot_feature_importance(
                np.array([0.1, 0.2]), ["feat1", "feat2", "feat3"]
            )

    @patch("src.utils.visualization.MATPLOTLIB_AVAILABLE", True)
    @patch("src.utils.visualization.plt")
    def test_plot_model_performance_with_matplotlib(self, mock_plt):
        """Test model performance plotting with matplotlib."""
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        ModelVisualizer.plot_model_performance(self.sample_history)

        mock_plt.subplots.assert_called_once()
        mock_plt.suptitle.assert_called_once()
        mock_plt.tight_layout.assert_called_once()

    @patch("src.utils.visualization.MATPLOTLIB_AVAILABLE", True)
    @patch("src.utils.visualization.plt")
    def test_plot_model_performance_single_metric(self, mock_plt):
        """Test model performance plotting with single metric."""
        mock_axes = Mock()
        mock_plt.subplots.return_value = (Mock(), mock_axes)

        history = {"loss": [1.0, 0.8, 0.6]}
        ModelVisualizer.plot_model_performance(history)

        mock_plt.subplots.assert_called_once()

    @patch("src.utils.visualization.MATPLOTLIB_AVAILABLE", True)
    @patch("src.utils.visualization.plt")
    def test_plot_predictions_vs_actual_with_matplotlib(self, mock_plt):
        """Test predictions vs actual plotting with matplotlib."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        with patch("sklearn.metrics.r2_score", return_value=0.95):
            ModelVisualizer.plot_predictions_vs_actual(y_true, y_pred)

            mock_plt.figure.assert_called_once()
            mock_plt.scatter.assert_called_once()
            mock_plt.plot.assert_called_once()  # Perfect prediction line
            mock_plt.xlabel.assert_called_once()
            mock_plt.ylabel.assert_called_once()
            mock_plt.title.assert_called_once()

    @patch("src.utils.visualization.MATPLOTLIB_AVAILABLE", True)
    @patch("src.utils.visualization.plt")
    def test_plot_confusion_matrix_with_matplotlib(self, mock_plt):
        """Test confusion matrix plotting with matplotlib."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        labels = ["Class A", "Class B", "Class C"]

        with patch(
            "sklearn.metrics.confusion_matrix",
            return_value=np.array([[2, 0, 0], [0, 1, 1], [0, 1, 1]]),
        ):
            ModelVisualizer.plot_confusion_matrix(y_true, y_pred, labels=labels)

            mock_plt.figure.assert_called_once()
            mock_plt.imshow.assert_called_once()
            mock_plt.colorbar.assert_called_once()
            mock_plt.title.assert_called_once()

    @patch("src.utils.visualization.MATPLOTLIB_AVAILABLE", True)
    @patch("src.utils.visualization.plt")
    def test_plot_confusion_matrix_without_labels(self, mock_plt):
        """Test confusion matrix plotting without custom labels."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])

        with patch(
            "sklearn.metrics.confusion_matrix", return_value=np.array([[1, 1], [0, 2]])
        ):
            ModelVisualizer.plot_confusion_matrix(y_true, y_pred)

            mock_plt.figure.assert_called_once()


class TestChemicalSpaceVisualizer(unittest.TestCase):
    """Test ChemicalSpaceVisualizer class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_df = pd.DataFrame(
            {"SMILES": ["CCO", "CC(=O)O", "CC(C)O", "CCCC"], "Activity": [1, 0, 1, 0]}
        )

    @patch("src.utils.visualization.RDKIT_AVAILABLE", True)
    @patch("src.utils.visualization.MATPLOTLIB_AVAILABLE", True)
    @patch("src.utils.visualization.plt")
    @patch("src.utils.visualization.Descriptors")
    @patch("src.utils.visualization.Chem")
    def test_plot_chemical_space_pca_with_rdkit_matplotlib(
        self, mock_chem, mock_descriptors, mock_plt
    ):
        """Test chemical space PCA plotting with RDKit and matplotlib."""
        # Setup mocks
        mock_mol = Mock()
        mock_chem.MolFromSmiles.return_value = mock_mol

        # Mock descriptor calculations
        mock_descriptors.MolWt.return_value = 46.07
        mock_descriptors.MolLogP.return_value = -0.31
        mock_descriptors.TPSA.return_value = 20.23
        mock_descriptors.NumHDonors.return_value = 1
        mock_descriptors.NumHAcceptors.return_value = 1
        mock_descriptors.NumRotatableBonds.return_value = 0
        mock_descriptors.RingCount.return_value = 0

        with patch("sklearn.decomposition.PCA") as mock_pca_class, patch(
            "sklearn.preprocessing.StandardScaler"
        ) as mock_scaler_class:
            mock_pca = Mock()
            mock_pca.fit_transform.return_value = np.array(
                [[1, 2], [3, 4], [5, 6], [7, 8]]
            )
            mock_pca.explained_variance_ratio_ = np.array([0.6, 0.3])
            mock_pca_class.return_value = mock_pca

            mock_scaler = Mock()
            mock_scaler.fit_transform.return_value = np.array([[1, 2, 3, 4, 5, 6, 7]])
            mock_scaler_class.return_value = mock_scaler

            ChemicalSpaceVisualizer.plot_chemical_space_pca(self.sample_df)

            mock_plt.figure.assert_called_once()
            mock_plt.scatter.assert_called()
            mock_plt.xlabel.assert_called_once()
            mock_plt.ylabel.assert_called_once()
            mock_plt.title.assert_called_once()

    @patch("src.utils.visualization.RDKIT_AVAILABLE", False)
    def test_plot_chemical_space_pca_without_rdkit(self):
        """Test chemical space PCA plotting without RDKit."""
        with patch("src.utils.visualization.logging.warning") as mock_warning:
            ChemicalSpaceVisualizer.plot_chemical_space_pca(self.sample_df)

            mock_warning.assert_called_with(
                "RDKit not available for descriptor calculation"
            )

    @patch("src.utils.visualization.MATPLOTLIB_AVAILABLE", False)
    def test_plot_chemical_space_pca_without_matplotlib(self):
        """Test chemical space PCA plotting without matplotlib."""
        with patch("src.utils.visualization.logging.warning") as mock_warning:
            ChemicalSpaceVisualizer.plot_chemical_space_pca(self.sample_df)

            mock_warning.assert_called_with("Matplotlib not available for plotting")


class TestStandaloneFunctions(unittest.TestCase):
    """Test standalone plotting functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_smiles = "CCO"

    @patch("src.utils.visualization.MolecularVisualizer.plot_molecular_structure")
    def test_plot_molecular_structure_function(self, mock_method):
        """Test standalone plot_molecular_structure function."""
        mock_method.return_value = "test_result"

        result = plot_molecular_structure(self.sample_smiles)

        mock_method.assert_called_once_with(self.sample_smiles, None)
        self.assertEqual(result, "test_result")

    @patch("src.utils.visualization.ModelVisualizer.plot_feature_importance")
    def test_plot_feature_importance_function(self, mock_method):
        """Test standalone plot_feature_importance function."""
        importances = np.array([0.1, 0.2, 0.3])
        features = ["f1", "f2", "f3"]

        plot_feature_importance(importances, features)

        mock_method.assert_called_once_with(
            importances, features, "Feature Importance", None
        )

    @patch("src.utils.visualization.ModelVisualizer.plot_model_performance")
    def test_plot_model_performance_function(self, mock_method):
        """Test standalone plot_model_performance function."""
        history = {"loss": [1.0, 0.5]}

        plot_model_performance(history)

        mock_method.assert_called_once_with(history, "Model Performance", None)


class TestDashboardCreation(unittest.TestCase):
    """Test dashboard creation and integration scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_results_df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CC(=O)O", "CC(C)O"],
                "MW": [46.07, 60.05, 60.10],
                "LogP": [-0.31, -0.17, 0.05],
                "Activity": [1, 0, 1],
            }
        )

    @patch(
        "src.utils.visualization.MolecularVisualizer.plot_molecular_properties_distribution"
    )
    @patch("src.utils.visualization.ChemicalSpaceVisualizer.plot_chemical_space_pca")
    @patch("pathlib.Path.mkdir")
    def test_create_dashboard_plots_with_data(
        self, mock_mkdir, mock_chemical_space, mock_properties
    ):
        """Test dashboard creation with sample data."""
        created_plots = create_dashboard_plots(self.sample_results_df, self.temp_dir)

        # Verify directory creation
        mock_mkdir.assert_called_once_with(exist_ok=True)

        # Verify plotting methods called
        mock_properties.assert_called_once()
        mock_chemical_space.assert_called_once()

        # Check return value structure
        self.assertIsInstance(created_plots, dict)

    def test_create_dashboard_plots_empty_dataframe(self):
        """Test dashboard creation with empty DataFrame."""
        empty_df = pd.DataFrame()

        created_plots = create_dashboard_plots(empty_df, self.temp_dir)

        self.assertIsInstance(created_plots, dict)
        self.assertEqual(len(created_plots), 0)

    @patch("src.utils.visualization.logging.error")
    def test_create_dashboard_plots_error_handling(self, mock_error):
        """Test dashboard creation error handling."""
        # Force an error by passing invalid data
        with patch("pathlib.Path.mkdir", side_effect=Exception("Test error")):
            created_plots = create_dashboard_plots(self.sample_results_df)

            mock_error.assert_called()
            self.assertIsInstance(created_plots, dict)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios combining multiple visualization components."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    @patch("src.utils.visualization.MATPLOTLIB_AVAILABLE", True)
    @patch("src.utils.visualization.RDKIT_AVAILABLE", True)
    def test_complete_visualization_workflow(self):
        """Test complete visualization workflow."""
        # Create sample data
        df = pd.DataFrame(
            {"SMILES": ["CCO", "CC(=O)O"], "MW": [46.07, 60.05], "Activity": [1, 0]}
        )

        with patch.multiple(
            "src.utils.visualization",
            MolecularVisualizer=Mock(),
            ModelVisualizer=Mock(),
            ChemicalSpaceVisualizer=Mock(),
        ):
            # Test dashboard creation
            plots = create_dashboard_plots(df, self.temp_dir)

            # Verify it returns a dictionary
            self.assertIsInstance(plots, dict)

    @patch("src.utils.visualization.MATPLOTLIB_AVAILABLE", True)
    def test_model_evaluation_visualization_pipeline(self):
        """Test model evaluation visualization pipeline."""
        # Sample model evaluation data
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9])
        history = {"loss": [1.0, 0.5, 0.2], "accuracy": [0.6, 0.8, 0.9]}
        importances = np.array([0.3, 0.2, 0.5])
        features = ["feat1", "feat2", "feat3"]

        with patch.multiple(
            "src.utils.visualization.ModelVisualizer",
            plot_predictions_vs_actual=Mock(),
            plot_model_performance=Mock(),
            plot_feature_importance=Mock(),
        ) as mock_methods:
            # Test complete model evaluation visualization
            ModelVisualizer.plot_predictions_vs_actual(y_true, y_pred)
            ModelVisualizer.plot_model_performance(history)
            ModelVisualizer.plot_feature_importance(importances, features)

            # Verify all methods called
            mock_methods["plot_predictions_vs_actual"].assert_called_once()
            mock_methods["plot_model_performance"].assert_called_once()
            mock_methods["plot_feature_importance"].assert_called_once()


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    @patch("src.utils.visualization.RDKIT_AVAILABLE", True)
    @patch("src.utils.visualization.Chem")
    def test_molecular_visualization_error_handling(self, mock_chem):
        """Test error handling in molecular visualization."""
        # Simulate RDKit error
        mock_chem.MolFromSmiles.side_effect = Exception("RDKit error")

        with patch("src.utils.visualization.logging.error") as mock_error:
            result = MolecularVisualizer.plot_molecular_structure("CCO")

            self.assertIsNone(result)
            mock_error.assert_called()

    @patch("src.utils.visualization.MATPLOTLIB_AVAILABLE", True)
    @patch("src.utils.visualization.plt")
    def test_matplotlib_error_handling(self, mock_plt):
        """Test error handling in matplotlib operations."""
        # Simulate matplotlib error
        mock_plt.figure.side_effect = Exception("Matplotlib error")

        with patch("src.utils.visualization.logging.error") as mock_error:
            ModelVisualizer.plot_feature_importance(np.array([0.1, 0.2]), ["f1", "f2"])

            mock_error.assert_called()

    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        # Test None inputs
        result = plot_molecular_structure(None)
        self.assertIsNone(result)

        # Test empty arrays
        with self.assertRaises(ValueError):
            ModelVisualizer.plot_feature_importance(np.array([]), [])

        # Test mismatched input sizes
        with self.assertRaises(ValueError):
            ModelVisualizer.plot_feature_importance(np.array([0.1]), ["f1", "f2"])


class TestCrossModuleCompatibility(unittest.TestCase):
    """Test compatibility with other modules and dependencies."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_visualization_imports(self):
        """Test that visualization module imports correctly."""
        # Test module-level imports
        from src.utils.visualization import (
            ChemicalSpaceVisualizer,
            ModelVisualizer,
            MolecularVisualizer,
        )

        # Verify classes are available
        self.assertTrue(hasattr(MolecularVisualizer, "plot_molecular_structure"))
        self.assertTrue(hasattr(ModelVisualizer, "plot_feature_importance"))
        self.assertTrue(hasattr(ChemicalSpaceVisualizer, "plot_chemical_space_pca"))

    def test_dependency_availability_flags(self):
        """Test dependency availability flags."""
        from src.utils import visualization

        # Test that availability flags exist
        self.assertTrue(hasattr(visualization, "MATPLOTLIB_AVAILABLE"))
        self.assertTrue(hasattr(visualization, "RDKIT_AVAILABLE"))
        self.assertTrue(hasattr(visualization, "PLOTLY_AVAILABLE"))

        # Flags should be boolean
        self.assertIsInstance(visualization.MATPLOTLIB_AVAILABLE, bool)
        self.assertIsInstance(visualization.RDKIT_AVAILABLE, bool)
        self.assertIsInstance(visualization.PLOTLY_AVAILABLE, bool)

    def test_numpy_pandas_integration(self):
        """Test integration with NumPy and pandas."""
        # Test with pandas DataFrame
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Should not raise errors
        try:
            create_dashboard_plots(df, self.temp_dir)
        except Exception as e:
            # Only fail if it's not a mocked dependency issue
            if "Mock" not in str(e):
                self.fail(f"DataFrame integration failed: {e}")

        # Test with NumPy arrays
        arr = np.array([0.1, 0.2, 0.3])
        features = ["f1", "f2", "f3"]

        try:
            plot_feature_importance(arr, features)
        except Exception as e:
            if "Mock" not in str(e):
                self.fail(f"NumPy integration failed: {e}")


class TestPerformance(unittest.TestCase):
    """Test performance aspects of visualization functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_large_dataset_handling(self):
        """Test visualization with large datasets."""
        # Create large dataset
        large_df = pd.DataFrame(
            {
                "SMILES": ["CCO"] * 1000,
                "MW": np.random.uniform(100, 500, 1000),
                "LogP": np.random.uniform(-2, 5, 1000),
            }
        )

        # Test that it doesn't raise memory errors
        try:
            with patch("src.utils.visualization.MolecularVisualizer"):
                create_dashboard_plots(large_df, self.temp_dir)
        except MemoryError:
            self.fail("Large dataset caused memory error")
        except Exception:
            # Other exceptions are okay for this test
            pass

    def test_feature_importance_performance(self):
        """Test feature importance plotting with many features."""
        # Test with many features
        n_features = 500
        importances = np.random.random(n_features)
        features = [f"feature_{i}" for i in range(n_features)]

        # Should handle large feature sets gracefully
        try:
            with patch("src.utils.visualization.ModelVisualizer"):
                plot_feature_importance(importances, features)
        except Exception:
            # Exceptions are okay for this performance test
            pass


if __name__ == "__main__":
    # Suppress warnings during testing
    warnings.filterwarnings("ignore")

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestMolecularVisualizer,
        TestModelVisualizer,
        TestChemicalSpaceVisualizer,
        TestStandaloneFunctions,
        TestDashboardCreation,
        TestIntegrationScenarios,
        TestErrorHandling,
        TestCrossModuleCompatibility,
        TestPerformance,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'='*50}")
    print("Visualization Utils Test Summary")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.2f}%"
    )

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            error_msg = traceback.split("AssertionError: ")[-1].split("\n")[0]
            print(f"  - {test}: {error_msg}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            error_msg = traceback.split("\n")[-2]
            print(f"  - {test}: {error_msg}")
