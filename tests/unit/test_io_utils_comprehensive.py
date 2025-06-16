"""
Comprehensive test suite for io_utils module.

This module tests data loading, saving, configuration management,
and file operations functionality.
"""

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from src.utils.io_utils import (
    Chem,
    ConfigManager,
    DataLoader,
    FileManager,
    ImportError:,
    ResultsExporter,
    _json_serializer,
    except,
    export_results,
    from,
    import,
    load_experiment_results,
    load_molecular_data,
    pass,
    rdkit,
    save_model_results,
    save_molecular_data,
    setup_logging,
    try:,
    validate_data_integrity,
)


class TestDataLoader(unittest.TestCase):
    """Test DataLoader class functionality"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.loader = DataLoader(self.temp_dir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test DataLoader initialization"""
        loader = DataLoader()
        self.assertEqual(str(loader.data_dir), "data")

        loader_with_dir = DataLoader("/custom/path")
        self.assertEqual(str(loader_with_dir.data_dir), "/custom/path")

    def test_load_csv(self):
        """Test CSV loading"""
        # Create test CSV file
        test_data = pd.DataFrame(
            {"SMILES": ["CCO", "CC", "CCC"], "activity": [1.5, 2.0, 1.8]}
        )
        csv_path = Path(self.temp_dir) / "test.csv"
        test_data.to_csv(csv_path, index=False)

        # Test loading
        loaded_data = self.loader.load_csv("test.csv")
        pd.testing.assert_frame_equal(loaded_data, test_data)

    def test_load_sdf_with_rdkit(self):
        """Test SDF loading with RDKit available"""
        with patch("rdkit.Chem") as mock_chem:
            # Mock RDKit functionality
            mock_mol = MagicMock()
            mock_mol.GetPropsAsDict.return_value = {"MW": 46.07}
            mock_chem.MolToSmiles.return_value = "CCO"
            mock_chem.SDMolSupplier.return_value = [mock_mol]

            with patch("src.utils.io_utils.Chem", mock_chem):
                result = self.loader.load_sdf("test.sdf")

                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]["smiles"], "CCO")
                self.assertEqual(result[0]["properties"]["MW"], 46.07)

    def test_load_sdf_without_rdkit(self):
        """Test SDF loading without RDKit"""
        with patch.dict("sys.modules", {"rdkit.Chem": None}):
            try:
                self.loader.load_sdf("test.sdf")
                self.fail("Should have raised ImportError")
            except ImportError:
                pass  # Expected

    def test_load_smiles_file_csv(self):
        """Test loading SMILES from CSV file"""
        # Create test CSV file
        test_data = pd.DataFrame(
            {"SMILES": ["CCO", "CC", "CCC"], "ID": ["mol1", "mol2", "mol3"]}
        )
        csv_path = Path(self.temp_dir) / "smiles.csv"
        test_data.to_csv(csv_path, index=False)

        loaded_data = self.loader.load_smiles_file("smiles.csv")
        pd.testing.assert_frame_equal(loaded_data, test_data)

    def test_load_smiles_file_tab_delimited(self):
        """Test loading SMILES from tab-delimited file"""
        # Create test tab-delimited file
        test_data = pd.DataFrame(
            {"SMILES": ["CCO", "CC", "CCC"], "ID": ["mol1", "mol2", "mol3"]}
        )
        tab_path = Path(self.temp_dir) / "smiles.txt"
        test_data.to_csv(tab_path, index=False, sep="\t")

        loaded_data = self.loader.load_smiles_file("smiles.txt")
        pd.testing.assert_frame_equal(loaded_data, test_data)

    def test_load_protein_fasta_with_biopython(self):
        """Test FASTA loading with Biopython available"""
        with patch("Bio.SeqIO.SeqIO") as mock_seqio:
            # Mock Biopython functionality
            mock_record = MagicMock()
            mock_record.id = "protein1"
            mock_record.seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
            mock_seqio.parse.return_value = [mock_record]

            with patch("src.utils.io_utils.SeqIO", mock_seqio):
                result = self.loader.load_protein_fasta("test.fasta")

                self.assertEqual(len(result), 1)
                self.assertIn("protein1", result)
                self.assertEqual(result["protein1"], str(mock_record.seq))

    def test_load_protein_fasta_without_biopython(self):
        """Test FASTA loading without Biopython"""
        with patch.dict("sys.modules", {"Bio.SeqIO": None}):
            try:
                self.loader.load_protein_fasta("test.fasta")
                self.fail("Should have raised ImportError")
            except ImportError:
                pass  # Expected

    def test_load_chembl_data_with_client(self):
        """Test ChEMBL data loading with client available"""
        with patch("chembl_webresource_client.new_client.new_client") as mock_client:
            # Mock ChEMBL client
            mock_activity_client = MagicMock()
            mock_activities = [
                {
                    "target_chembl_id": "CHEMBL123",
                    "standard_value": 100,
                    "standard_type": "IC50",
                },
                {
                    "target_chembl_id": "CHEMBL123",
                    "standard_value": 200,
                    "standard_type": "IC50",
                },
            ]
            mock_activity_client.filter.return_value = mock_activities
            mock_client.activity = mock_activity_client

            with patch("src.utils.io_utils.new_client", mock_client):
                result = self.loader.load_chembl_data("CHEMBL123")

                self.assertEqual(len(result), 2)
                self.assertEqual(result.iloc[0]["standard_value"], 100)

    def test_load_chembl_data_without_client(self):
        """Test ChEMBL data loading without client"""
        with patch.dict("sys.modules", {"chembl_webresource_client.new_client": None}):
            try:
                self.loader.load_chembl_data("CHEMBL123")
                self.fail("Should have raised ImportError")
            except ImportError:
                pass  # Expected

    def test_load_json(self):
        """Test JSON loading"""
        test_data = {"key": "value", "number": 42}
        json_path = Path(self.temp_dir) / "test.json"
        with open(json_path, "w") as f:
            json.dump(test_data, f)

        loaded_data = self.loader.load_json("test.json")
        self.assertEqual(loaded_data, test_data)

    def test_load_pickle(self):
        """Test pickle loading"""
        test_data = {"key": "value", "array": np.array([1, 2, 3])}
        pickle_path = Path(self.temp_dir) / "test.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(test_data, f)

        loaded_data = self.loader.load_pickle("test.pkl")
        self.assertEqual(loaded_data["key"], test_data["key"])
        np.testing.assert_array_equal(loaded_data["array"], test_data["array"])


class TestStandaloneFunctions(unittest.TestCase):
    """Test standalone functions"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_molecular_data_csv(self):
        """Test loading molecular data from CSV"""
        test_data = pd.DataFrame(
            {"SMILES": ["CCO", "CC", "CCC"], "activity": [1.5, 2.0, 1.8]}
        )
        csv_path = Path(self.temp_dir) / "molecules.csv"
        test_data.to_csv(csv_path, index=False)

        loaded_data = load_molecular_data(csv_path)
        self.assertEqual(len(loaded_data), 3)
        self.assertIn("smiles", loaded_data.columns)

    def test_load_molecular_data_different_smiles_column_names(self):
        """Test loading with different SMILES column names"""
        test_cases = ["SMILES", "smiles", "Smiles", "SMILE", "smile"]

        for col_name in test_cases:
            test_data = pd.DataFrame({col_name: ["CCO", "CC"], "activity": [1.5, 2.0]})
            csv_path = Path(self.temp_dir) / f"molecules_{col_name}.csv"
            test_data.to_csv(csv_path, index=False)

            loaded_data = load_molecular_data(csv_path)
            self.assertIn("smiles", loaded_data.columns)

    def test_load_molecular_data_missing_smiles_column(self):
        """Test error when SMILES column is missing"""
        test_data = pd.DataFrame({"compound": ["CCO", "CC"], "activity": [1.5, 2.0]})
        csv_path = Path(self.temp_dir) / "no_smiles.csv"
        test_data.to_csv(csv_path, index=False)

        with self.assertRaises(ValueError):
            load_molecular_data(csv_path)

    def test_load_molecular_data_file_not_found(self):
        """Test error when file doesn't exist"""
        with self.assertRaises(FileNotFoundError):
            load_molecular_data("/nonexistent/file.csv")

    def test_load_molecular_data_empty_file(self):
        """Test error when file is empty"""
        empty_data = pd.DataFrame()
        csv_path = Path(self.temp_dir) / "empty.csv"
        empty_data.to_csv(csv_path, index=False)

        with self.assertRaises(ValueError):
            load_molecular_data(csv_path)

    def test_save_molecular_data_csv(self):
        """Test saving molecular data to CSV"""
        test_data = pd.DataFrame(
            {"smiles": ["CCO", "CC", "CCC"], "activity": [1.5, 2.0, 1.8]}
        )
        csv_path = Path(self.temp_dir) / "output.csv"

        save_molecular_data(test_data, csv_path)

        self.assertTrue(csv_path.exists())
        loaded_data = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(loaded_data, test_data)

    def test_save_molecular_data_json(self):
        """Test saving molecular data to JSON"""
        test_data = pd.DataFrame({"smiles": ["CCO", "CC"], "activity": [1.5, 2.0]})
        json_path = Path(self.temp_dir) / "output.json"

        save_molecular_data(test_data, json_path)

        self.assertTrue(json_path.exists())

    def test_export_results_json(self):
        """Test exporting results to JSON"""
        results = {
            "accuracy": 0.95,
            "predictions": [1, 0, 1, 1],
            "model_params": {"lr": 0.01},
        }
        json_path = Path(self.temp_dir) / "results.json"

        export_results(results, str(json_path), format="json")

        self.assertTrue(json_path.exists())
        with open(json_path, "r") as f:
            loaded_results = json.load(f)
        self.assertEqual(loaded_results["accuracy"], 0.95)

    def test_export_results_unsupported_format(self):
        """Test error with unsupported format"""
        results = {"key": "value"}

        with self.assertRaises(ValueError):
            export_results(results, "output.xyz", format="xyz")

    def test_save_model_results(self):
        """Test saving model results"""
        model_results = {"accuracy": 0.95, "loss": 0.1, "epochs": 100}

        result_path = save_model_results(
            model_results, "test_experiment", self.temp_dir
        )

        self.assertTrue(Path(result_path).exists())
        self.assertIn("test_experiment_results.json", result_path)

    def test_load_experiment_results(self):
        """Test loading experiment results"""
        model_results = {"accuracy": 0.95, "loss": 0.1}

        # Save results first
        save_model_results(model_results, "test_exp", self.temp_dir)

        # Load results
        loaded_results = load_experiment_results("test_exp", self.temp_dir)

        self.assertEqual(loaded_results["accuracy"], 0.95)
        self.assertEqual(loaded_results["loss"], 0.1)

    def test_load_experiment_results_file_not_found(self):
        """Test error when results file doesn't exist"""
        with self.assertRaises(FileNotFoundError):
            load_experiment_results("nonexistent_exp", self.temp_dir)

    def test_json_serializer(self):
        """Test custom JSON serializer"""
        # Test numpy array
        arr = np.array([1, 2, 3])
        result = _json_serializer(arr)
        self.assertEqual(result, [1, 2, 3])

        # Test numpy scalar
        scalar = np.int64(42)
        result = _json_serializer(scalar)
        self.assertEqual(result, 42)

        # Test pandas DataFrame
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = _json_serializer(df)
        self.assertEqual(result, [{"a": 1, "b": 3}, {"a": 2, "b": 4}])


class TestResultsExporter(unittest.TestCase):
    """Test ResultsExporter class"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = ResultsExporter(self.temp_dir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test ResultsExporter initialization"""
        self.assertTrue(Path(self.temp_dir).exists())

    def test_save_dataframe_csv(self):
        """Test saving DataFrame to CSV"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        self.exporter.save_dataframe(df, "test.csv", format="csv")

        saved_path = Path(self.temp_dir) / "test.csv"
        self.assertTrue(saved_path.exists())

    def test_save_dataframe_unsupported_format(self):
        """Test error with unsupported format"""
        df = pd.DataFrame({"a": [1, 2]})

        with self.assertRaises(ValueError):
            self.exporter.save_dataframe(df, "test.xyz", format="xyz")

    def test_save_model_results(self):
        """Test saving model results"""
        results = {
            "accuracy": 0.95,
            "predictions": np.array([1, 0, 1]),
            "nested": {"param": np.float64(0.1)},
        }

        self.exporter.save_model_results(results, "model_results.json")

        saved_path = Path(self.temp_dir) / "model_results.json"
        self.assertTrue(saved_path.exists())

    def test_export_smiles_list_with_index(self):
        """Test exporting SMILES list with index"""
        smiles_list = ["CCO", "CC", "CCC"]

        self.exporter.export_smiles_list(
            smiles_list, "molecules.csv", include_index=True
        )

        saved_path = Path(self.temp_dir) / "molecules.csv"
        self.assertTrue(saved_path.exists())

        df = pd.read_csv(saved_path)
        self.assertEqual(len(df), 3)
        self.assertIn("ID", df.columns)
        self.assertIn("SMILES", df.columns)

    def test_export_smiles_list_without_index(self):
        """Test exporting SMILES list without index"""
        smiles_list = ["CCO", "CC"]

        self.exporter.export_smiles_list(
            smiles_list, "molecules.txt", include_index=False
        )

        saved_path = Path(self.temp_dir) / "molecules.txt"
        self.assertTrue(saved_path.exists())

    def test_save_molecular_descriptors(self):
        """Test saving molecular descriptors"""
        descriptors = pd.DataFrame(
            {"SMILES": ["CCO", "CC"], "MW": [46.07, 30.07], "LogP": [-0.31, 0.6]}
        )

        self.exporter.save_molecular_descriptors(descriptors)

        saved_path = Path(self.temp_dir) / "molecular_descriptors.csv"
        self.assertTrue(saved_path.exists())


class TestConfigManager(unittest.TestCase):
    """Test ConfigManager class"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test ConfigManager initialization"""
        self.assertTrue(Path(self.temp_dir).exists())

    def test_save_and_load_yaml_config(self):
        """Test saving and loading YAML config"""
        config = {
            "model": {"type": "RandomForest", "n_estimators": 100},
            "data": {"train_size": 0.8},
            "features": ["MW", "LogP"],
        }

        self.config_manager.save_config(config, "test_config.yaml")
        loaded_config = self.config_manager.load_config("test_config.yaml")

        self.assertEqual(loaded_config, config)

    def test_save_and_load_json_config(self):
        """Test saving and loading JSON config"""
        config = {"model": {"type": "SVM", "C": 1.0}, "preprocessing": {"scale": True}}

        self.config_manager.save_config(config, "test_config.json")
        loaded_config = self.config_manager.load_config("test_config.json")

        self.assertEqual(loaded_config, config)

    def test_load_config_file_not_found(self):
        """Test error when config file doesn't exist"""
        with self.assertRaises(FileNotFoundError):
            self.config_manager.load_config("nonexistent.yaml")

    def test_save_config_unsupported_format(self):
        """Test error with unsupported config format"""
        config = {"key": "value"}

        with self.assertRaises(ValueError):
            self.config_manager.save_config(config, "config.txt")

    def test_load_config_unsupported_format(self):
        """Test error when loading unsupported format"""
        # Create a file with unsupported extension
        config_path = Path(self.temp_dir) / "config.txt"
        config_path.write_text("some content")

        with self.assertRaises(ValueError):
            self.config_manager.load_config("config.txt")

    def test_create_experiment_config(self):
        """Test creating comprehensive experiment config"""
        model_params = {"type": "RandomForest", "n_estimators": 100}
        data_params = {"file": "molecules.csv", "target": "activity"}
        training_params = {"epochs": 50, "batch_size": 32}

        filename = self.config_manager.create_experiment_config(
            "test_experiment", model_params, data_params, training_params
        )

        self.assertEqual(filename, "test_experiment_config.yaml")

        # Load and verify the config
        config = self.config_manager.load_config(filename)
        self.assertEqual(config["model"], model_params)
        self.assertEqual(config["data"], data_params)
        self.assertEqual(config["training"], training_params)


class TestFileManager(unittest.TestCase):
    """Test FileManager class"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_directory_structure(self):
        """Test creating directory structure"""
        subdirs = ["data", "models", "results", "configs"]

        FileManager.create_directory_structure(self.temp_dir, subdirs)

        for subdir in subdirs:
            self.assertTrue((Path(self.temp_dir) / subdir).exists())

    def test_list_files(self):
        """Test listing files in directory"""
        # Create some test files
        test_files = ["file1.txt", "file2.csv", "file3.json"]
        for filename in test_files:
            (Path(self.temp_dir) / filename).touch()

        # List all files
        all_files = FileManager.list_files(self.temp_dir)
        self.assertEqual(len(all_files), 3)

        # List CSV files only
        csv_files = FileManager.list_files(self.temp_dir, "*.csv")
        self.assertEqual(len(csv_files), 1)

    def test_get_file_info_existing_file(self):
        """Test getting info for existing file"""
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("hello world")

        info = FileManager.get_file_info(str(test_file))

        self.assertTrue(info["exists"])
        self.assertTrue(info["is_file"])
        self.assertFalse(info["is_dir"])
        self.assertEqual(info["suffix"], ".txt")
        self.assertGreater(info["size_bytes"], 0)

    def test_get_file_info_nonexistent_file(self):
        """Test getting info for nonexistent file"""
        info = FileManager.get_file_info("/nonexistent/file.txt")

        self.assertFalse(info["exists"])

    def test_backup_file(self):
        """Test creating file backup"""
        test_file = Path(self.temp_dir) / "original.txt"
        test_file.write_text("original content")

        backup_path = FileManager.backup_file(str(test_file))

        self.assertTrue(Path(backup_path).exists())
        self.assertIn("_backup", backup_path)

    def test_backup_file_nonexistent(self):
        """Test error when backing up nonexistent file"""
        with self.assertRaises(FileNotFoundError):
            FileManager.backup_file("/nonexistent/file.txt")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""

    def test_setup_logging(self):
        """Test logging setup"""
        with patch("logging.basicConfig") as mock_config:
            setup_logging(log_level="DEBUG")
            mock_config.assert_called_once()

    def test_setup_logging_with_file(self):
        """Test logging setup with file"""
        with tempfile.NamedTemporaryFile() as tmp_file:
            with patch("logging.basicConfig") as mock_config:
                setup_logging(log_file=tmp_file.name)
                mock_config.assert_called_once()

    def test_validate_data_integrity(self):
        """Test data integrity validation"""
        df = pd.DataFrame(
            {
                "SMILES": ["CCO", "CC", None, "CCC"],
                "activity": [1.5, 2.0, 1.8, 2.2],
                "ID": ["mol1", "mol2", "mol3", "mol1"],  # Duplicate
            }
        )

        required_columns = ["SMILES", "activity", "target"]
        issues = validate_data_integrity(df, required_columns)

        # Check missing columns
        self.assertIn("target", issues["missing_columns"])

        # Check missing values
        self.assertEqual(issues["missing_values"]["SMILES"], 1)

        # Check duplicates
        self.assertEqual(issues["duplicate_rows"], 1)

        # Check summary
        self.assertEqual(issues["summary"]["total_rows"], 4)
        self.assertEqual(issues["summary"]["total_columns"], 3)


class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_molecular_data_corrupted_file(self):
        """Test handling corrupted file"""
        corrupted_file = Path(self.temp_dir) / "corrupted.csv"
        corrupted_file.write_text("invalid,csv,content\nno,proper,structure")

        # Should not raise exception but might return unexpected results
        try:
            result = load_molecular_data(corrupted_file)
            # If it loads, check that it handles missing SMILES gracefully
            if "smiles" not in result.columns:
                self.fail("Should have raised ValueError for missing SMILES")
        except (ValueError, pd.errors.EmptyDataError):
            # This is expected for corrupted files
            pass

    def test_save_molecular_data_permission_error(self):
        """Test handling permission errors during save"""
        df = pd.DataFrame({"smiles": ["CCO"], "activity": [1.5]})

        # Try to save to a nonexistent directory that won't be created
        invalid_path = "/nonexistent_root/readonly/data.csv"

        # This should handle the error gracefully
        try:
            save_molecular_data(df, invalid_path)
            self.fail("Should have raised an error for invalid path")
        except (PermissionError, FileNotFoundError, OSError):
            # This is expected for invalid paths
            pass


class TestPerformance(unittest.TestCase):
    """Test performance with larger datasets"""

    def test_large_dataframe_operations(self):
        """Test operations with larger DataFrames"""
        # Create a moderately large dataset
        large_df = pd.DataFrame(
            {
                "smiles": ["CCO"] * 1000,
                "activity": np.random.random(1000),
                "property1": np.random.random(1000),
                "property2": np.random.random(1000),
            }
        )

        # Test data integrity validation
        issues = validate_data_integrity(large_df, ["smiles", "activity"])

        self.assertEqual(issues["summary"]["total_rows"], 1000)
        self.assertEqual(issues["summary"]["total_columns"], 4)

    def test_json_serialization_performance(self):
        """Test JSON serialization with complex data"""
        complex_data = {
            "arrays": [np.random.random(10) for _ in range(5)],  # Smaller arrays
            "scalars": [np.float64(x) for x in range(10)],  # Fewer scalars
            "nested": {
                "data": np.random.random((5, 5)),  # Smaller matrix
                "metadata": {"version": 1.0},
            },
        }

        # This should complete without errors
        serialized = _json_serializer(complex_data)
        self.assertIsInstance(serialized, dict)


if __name__ == "__main__":
    unittest.main()
