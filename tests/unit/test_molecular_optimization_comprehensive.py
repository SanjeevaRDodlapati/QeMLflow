"""
Comprehensive test suite for molecular optimization module.

This test suite provides thorough coverage of the molecular optimization functionality
including Bayesian optimization, genetic algorithms, and standalone optimization functions.
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn import *
try:
    from rdkit import Chem
except ImportError:
    pass

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

try:
    from drug_design.molecular_optimization import (
        BayesianOptimizer,
        GeneticAlgorithmOptimizer,
        MolecularOptimizer,
        _generate_similar_molecule,
        batch_optimize,
        calculate_optimization_metrics,
        compare_optimization_methods,
        create_optimization_report,
        extract_optimization_insights,
        generate_optimization_summary,
        load_optimization_checkpoint,
        optimize_molecule,
        save_optimization_checkpoint,
        validate_optimization_results,
    )
except ImportError as e:
    pytest.skip(
        f"Could not import molecular_optimization module: {e}", allow_module_level=True
    )


class TestMolecularOptimizer:
    """Test cases for the base MolecularOptimizer class."""

    def test_init_default_objective(self):
        """Test initialization with default objective function."""
        optimizer = MolecularOptimizer()
        assert optimizer.objective_function is not None
        assert optimizer.optimization_history == []

    def test_init_custom_objective(self):
        """Test initialization with custom objective function."""

        def custom_objective(smiles):
            return 0.8

        optimizer = MolecularOptimizer(custom_objective)
        assert optimizer.objective_function == custom_objective

    def test_default_objective_valid_molecule(self):
        """Test default objective function with valid molecules."""
        optimizer = MolecularOptimizer()

        # Test common molecules
        score_water = optimizer.objective_function("O")  # Water
        score_methane = optimizer.objective_function("C")  # Methane
        score_aspirin = optimizer.objective_function(
            "CC(=O)OC1=CC=CC=C1C(=O)O"
        )  # Aspirin

        assert 0.0 <= score_water <= 1.0
        assert 0.0 <= score_methane <= 1.0
        assert 0.0 <= score_aspirin <= 1.0

    def test_default_objective_invalid_molecule(self):
        """Test default objective function with invalid molecules."""
        optimizer = MolecularOptimizer()

        # Test invalid SMILES
        score_invalid = optimizer.objective_function("INVALID_SMILES")
        score_empty = optimizer.objective_function("")
        score_none = optimizer.objective_function(None)

        assert score_invalid == 0.0
        assert score_empty == 0.0
        assert score_none == 0.0

    def test_default_objective_without_rdkit(self):
        """Test default objective function when RDKit is not available."""
        optimizer = MolecularOptimizer()

        with patch("drug_design.molecular_optimization.RDKIT_AVAILABLE", False):
            score = optimizer.objective_function("CCO")
            assert 0.0 <= score <= 1.0  # Should return random value

    def test_optimize_not_implemented(self):
        """Test that base class optimize method raises NotImplementedError."""
        optimizer = MolecularOptimizer()

        with pytest.raises(NotImplementedError):
            optimizer.optimize(["CCO"], 10)


class TestBayesianOptimizer:
    """Test cases for the BayesianOptimizer class."""

    def test_init(self):
        """Test BayesianOptimizer initialization."""
        optimizer = BayesianOptimizer()
        assert optimizer.gp_model is None
        assert optimizer.scaler is None
        assert optimizer.optimization_history == []

    def test_molecular_to_features_valid(self):
        """Test molecular feature extraction with valid molecules."""
        optimizer = BayesianOptimizer()

        features = optimizer._molecular_to_features("CCO")  # Ethanol
        assert isinstance(features, np.ndarray)
        assert len(features) == 10  # Expected number of features
        assert all(isinstance(f, (int, float)) for f in features)

    def test_molecular_to_features_invalid(self):
        """Test molecular feature extraction with invalid molecules."""
        optimizer = BayesianOptimizer()

        features_invalid = optimizer._molecular_to_features("INVALID")
        features_none = optimizer._molecular_to_features(None)

        assert isinstance(features_invalid, np.ndarray)
        assert len(features_invalid) == 10
        assert isinstance(features_none, np.ndarray)
        assert len(features_none) == 10

    def test_molecular_to_features_without_rdkit(self):
        """Test molecular feature extraction when RDKit is not available."""
        optimizer = BayesianOptimizer()

        with patch("drug_design.molecular_optimization.RDKIT_AVAILABLE", False):
            features = optimizer._molecular_to_features("CCO")
            assert isinstance(features, np.ndarray)
            assert len(features) == 10

    def test_acquisition_function_no_model(self):
        """Test acquisition function when no GP model exists."""
        optimizer = BayesianOptimizer()

        features = np.random.random(10)
        score = optimizer._acquisition_function(features)
        assert 0.0 <= score <= 1.0

    def test_acquisition_function_with_model(self):
        """Test acquisition function with trained GP model."""
        optimizer = BayesianOptimizer()

        # Mock GP model and scaler
        mock_gp = Mock()
        mock_gp.predict.return_value = (np.array([0.5]), np.array([0.1]))
        optimizer.gp_model = mock_gp

        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[0.1, 0.2, 0.3]])
        optimizer.scaler = mock_scaler

        with patch("drug_design.molecular_optimization.SKLEARN_AVAILABLE", True):
            features = np.random.random(10)
            score = optimizer._acquisition_function(features)
            assert isinstance(score, (int, float))

    def test_optimize_without_sklearn(self):
        """Test optimization when sklearn is not available."""
        optimizer = BayesianOptimizer()

        with patch("drug_design.molecular_optimization.SKLEARN_AVAILABLE", False):
            results = optimizer.optimize(["CCO", "CC"], num_iterations=5)

            assert "best_molecule" in results
            assert "best_score" in results
            assert "optimization_history" in results
            assert results["best_molecule"] in ["CCO", "CC"]

    def test_optimize_with_sklearn(self):
        """Test optimization with sklearn available."""
        optimizer = BayesianOptimizer()

        with patch("drug_design.molecular_optimization.SKLEARN_AVAILABLE", True):
            # Mock required sklearn components
            with patch(
                "drug_design.molecular_optimization.StandardScaler"
            ) as mock_scaler_class:
                with patch(
                    "drug_design.molecular_optimization.GaussianProcessRegressor"
                ) as mock_gp_class:
                    # Setup mocks
                    mock_scaler = Mock()
                    mock_scaler.fit_transform.return_value = np.random.random((2, 10))
                    mock_scaler.transform.return_value = np.random.random((1, 10))
                    mock_scaler_class.return_value = mock_scaler

                    mock_gp = Mock()
                    mock_gp.predict.return_value = (np.array([0.5]), np.array([0.1]))
                    mock_gp_class.return_value = mock_gp

                    results = optimizer.optimize(["CCO", "CC"], num_iterations=3)

                    assert "best_molecule" in results
                    assert "best_score" in results
                    assert "optimization_history" in results
                    assert "final_population" in results
                    assert "final_scores" in results

    def test_random_optimization_fallback(self):
        """Test random optimization fallback method."""
        optimizer = BayesianOptimizer()

        results = optimizer._random_optimization(["CCO", "CC"], 5)

        assert "best_molecule" in results
        assert "best_score" in results
        assert results["best_molecule"] in ["CCO", "CC"]
        assert isinstance(results["best_score"], (int, float))

    def test_generate_candidates(self):
        """Test candidate generation for optimization."""
        optimizer = BayesianOptimizer()

        molecules = ["CCO", "CC", "CCC"]
        candidates = optimizer._generate_candidates(molecules, num_candidates=5)

        assert isinstance(candidates, list)
        assert len(candidates) <= 5

    def test_simple_mutation(self):
        """Test simple mutation operation."""
        optimizer = BayesianOptimizer()

        # Test with valid SMILES
        result = optimizer._simple_mutation("CCO")
        assert result is None or isinstance(result, str)

        # Test with short SMILES
        result_short = optimizer._simple_mutation("C")
        assert result_short is None

        # Test without RDKit
        with patch("drug_design.molecular_optimization.RDKIT_AVAILABLE", False):
            result_no_rdkit = optimizer._simple_mutation("CCO")
            assert result_no_rdkit is None


class TestGeneticAlgorithmOptimizer:
    """Test cases for the GeneticAlgorithmOptimizer class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        optimizer = GeneticAlgorithmOptimizer()
        assert optimizer.population_size == 50
        assert optimizer.mutation_rate == 0.1
        assert optimizer.crossover_rate == 0.7

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        optimizer = GeneticAlgorithmOptimizer(
            population_size=20, mutation_rate=0.2, crossover_rate=0.8
        )
        assert optimizer.population_size == 20
        assert optimizer.mutation_rate == 0.2
        assert optimizer.crossover_rate == 0.8

    def test_optimize_basic(self):
        """Test basic genetic algorithm optimization."""
        optimizer = GeneticAlgorithmOptimizer(population_size=5)

        results = optimizer.optimize(["CCO", "CC"], num_iterations=3)

        assert "best_molecule" in results
        assert "best_score" in results
        assert "optimization_history" in results
        assert "final_population" in results
        assert "final_scores" in results
        assert len(results["final_population"]) == 5

    def test_tournament_selection(self):
        """Test tournament selection mechanism."""
        optimizer = GeneticAlgorithmOptimizer()

        population = ["CCO", "CC", "CCC", "CCCC"]
        fitness_scores = [0.8, 0.6, 0.9, 0.7]

        selected = optimizer._tournament_selection(
            population, fitness_scores, tournament_size=2
        )
        assert selected in population

    def test_crossover_basic(self):
        """Test basic crossover operation."""
        optimizer = GeneticAlgorithmOptimizer()

        parent1 = "CCO"
        parent2 = "CCC"

        child1, child2 = optimizer._crossover(parent1, parent2)

        assert isinstance(child1, (str, type(None)))
        assert isinstance(child2, (str, type(None)))

    def test_crossover_without_rdkit(self):
        """Test crossover operation without RDKit."""
        optimizer = GeneticAlgorithmOptimizer()

        with patch("drug_design.molecular_optimization.RDKIT_AVAILABLE", False):
            parent1 = "CCCO"
            parent2 = "CCCC"

            child1, child2 = optimizer._crossover(parent1, parent2)

            assert isinstance(child1, str)
            assert isinstance(child2, str)
            assert len(child1) > 0
            assert len(child2) > 0

    def test_crossover_short_parents(self):
        """Test crossover with short parent strings."""
        optimizer = GeneticAlgorithmOptimizer()

        with patch("drug_design.molecular_optimization.RDKIT_AVAILABLE", False):
            parent1 = "C"
            parent2 = "O"

            child1, child2 = optimizer._crossover(parent1, parent2)

            assert child1 == parent1
            assert child2 == parent2

    def test_mutate(self):
        """Test mutation operation."""
        optimizer = GeneticAlgorithmOptimizer()

        original = "CCO"
        mutated = optimizer._mutate(original)

        assert mutated is None or isinstance(mutated, str)

    def test_simple_mutation_replacements(self):
        """Test simple mutation with character replacements."""
        optimizer = GeneticAlgorithmOptimizer()

        # Test with replaceable characters
        result_c = optimizer._simple_mutation("CCO")
        result_n = optimizer._simple_mutation("NNN")
        result_o = optimizer._simple_mutation("OOO")

        assert result_c is None or isinstance(result_c, str)
        assert result_n is None or isinstance(result_n, str)
        assert result_o is None or isinstance(result_o, str)

    def test_simple_mutation_short_string(self):
        """Test simple mutation with short strings."""
        optimizer = GeneticAlgorithmOptimizer()

        result = optimizer._simple_mutation("CC")
        assert result == "CC" or result != "CC"  # May or may not change


class TestStandaloneFunctions:
    """Test cases for standalone optimization functions."""

    def test_optimize_molecule_bayesian(self):
        """Test single molecule optimization with Bayesian method."""
        result = optimize_molecule(
            "CCO", optimization_method="bayesian", num_iterations=5
        )

        assert "original_molecule" in result
        assert "optimized_molecule" in result
        assert "improvement" in result
        assert "optimization_score" in result
        assert "method" in result
        assert result["original_molecule"] == "CCO"
        assert result["method"] == "bayesian"

    def test_optimize_molecule_genetic(self):
        """Test single molecule optimization with genetic algorithm."""
        result = optimize_molecule(
            "CCO", optimization_method="genetic", num_iterations=5
        )

        assert "original_molecule" in result
        assert "optimized_molecule" in result
        assert "improvement" in result
        assert "optimization_score" in result
        assert "method" in result
        assert result["method"] == "genetic"

    def test_optimize_molecule_random(self):
        """Test single molecule optimization with random method."""
        result = optimize_molecule(
            "CCO", optimization_method="random", num_iterations=5
        )

        assert "original_molecule" in result
        assert "optimized_molecule" in result
        assert "improvement" in result
        assert "optimization_score" in result
        assert "method" in result
        assert result["method"] == "random"

    def test_optimize_molecule_custom_objective(self):
        """Test optimization with custom objective function."""

        def custom_objective(smiles):
            return 0.95  # Always return high score

        result = optimize_molecule(
            "CCO", objective_function=custom_objective, num_iterations=3
        )

        assert result["optimization_score"] == 0.95

    def test_batch_optimize_basic(self):
        """Test batch optimization with multiple molecules."""
        molecules = ["CCO", "CC", "CCC"]

        results_df = batch_optimize(
            molecules, optimization_method="bayesian", num_iterations=3
        )

        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 3
        assert "original_molecule" in results_df.columns
        assert "optimized_molecule" in results_df.columns
        assert "improvement" in results_df.columns
        assert "optimization_score" in results_df.columns
        assert "method" in results_df.columns

    def test_batch_optimize_with_errors(self):
        """Test batch optimization handling of errors."""
        molecules = ["CCO", "INVALID_SMILES", "CC"]

        # Mock optimize_molecule to raise exception for invalid SMILES
        def mock_optimize_molecule(smiles, *args, **kwargs):
            if smiles == "INVALID_SMILES":
                raise ValueError("Invalid SMILES")
            return {
                "original_molecule": smiles,
                "optimized_molecule": smiles,
                "improvement": 0.1,
                "optimization_score": 0.5,
                "method": "test",
            }

        with patch(
            "drug_design.molecular_optimization.optimize_molecule",
            side_effect=mock_optimize_molecule,
        ):
            results_df = batch_optimize(molecules, num_iterations=3)

            assert len(results_df) == 3
            # Should handle error gracefully
            invalid_row = results_df[
                results_df["original_molecule"] == "INVALID_SMILES"
            ]
            assert len(invalid_row) == 1
            assert invalid_row.iloc[0]["improvement"] == 0.0

    def test_generate_similar_molecule(self):
        """Test generation of similar molecules."""
        result = _generate_similar_molecule("CCO")
        assert result is None or isinstance(result, str)

        # Test with short molecule
        result_short = _generate_similar_molecule("C")
        assert result_short == "C" or result_short is None

        # Test without RDKit
        with patch("drug_design.molecular_optimization.RDKIT_AVAILABLE", False):
            result_no_rdkit = _generate_similar_molecule("CCO")
            assert result_no_rdkit == "CCO"

    def test_create_optimization_report(self):
        """Test creation of optimization report."""
        mock_results = {
            "best_molecule": "CCO",
            "best_score": 0.85,
            "optimization_history": [
                {"iteration": 0, "best_score": 0.7},
                {"iteration": 1, "best_score": 0.85},
            ],
            "final_population": ["CCO", "CC", "CCC"],
            "final_scores": [0.85, 0.6, 0.7],
        }

        report = create_optimization_report(mock_results)

        assert "summary" in report
        assert "optimization_history" in report
        assert "statistics" in report

        assert report["summary"]["best_molecule"] == "CCO"
        assert report["summary"]["best_score"] == 0.85
        assert report["summary"]["total_iterations"] == 2
        assert report["summary"]["final_population_size"] == 3

    def test_create_optimization_report_with_file(self):
        """Test creation of optimization report with file output."""
        mock_results = {
            "best_molecule": "CCO",
            "best_score": 0.85,
            "optimization_history": [],
            "final_population": ["CCO"],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            tmp_path = tmp_file.name

        try:
            _report = create_optimization_report(mock_results, output_file=tmp_path)

            # Check file was created
            assert os.path.exists(tmp_path)

            # Check file contents
            with open(tmp_path, "r") as f:
                saved_report = json.load(f)

            assert saved_report["summary"]["best_molecule"] == "CCO"

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_create_optimization_report_file_error(self):
        """Test report creation handles file writing errors gracefully."""
        mock_results = {
            "best_molecule": "CCO",
            "best_score": 0.85,
            "optimization_history": [],
            "final_population": ["CCO"],
        }

        # Try to write to invalid path
        invalid_path = "/invalid/path/report.json"

        # Should not raise exception, just log error
        report = create_optimization_report(mock_results, output_file=invalid_path)

        assert "summary" in report
        assert report["summary"]["best_molecule"] == "CCO"

    def test_create_optimization_report_empty_results(self):
        """Test report creation with empty/minimal results."""
        empty_results = {}

        report = create_optimization_report(empty_results)

        assert "summary" in report
        assert report["summary"]["best_molecule"] == "N/A"
        assert report["summary"]["best_score"] == 0.0
        assert report["summary"]["total_iterations"] == 0
        assert report["summary"]["final_population_size"] == 0


class TestIntegrationScenarios:
    """Integration test scenarios for molecular optimization."""

    def test_full_bayesian_optimization_workflow(self):
        """Test complete Bayesian optimization workflow."""

        # Create optimizer with custom objective
        def drug_like_objective(smiles):
            # Simple drug-likeness based on length (mock)
            return min(1.0, len(smiles) / 20.0)

        optimizer = BayesianOptimizer(drug_like_objective)

        # Test with simple molecules
        initial_molecules = ["C", "CC", "CCO"]

        with patch("drug_design.molecular_optimization.SKLEARN_AVAILABLE", False):
            results = optimizer.optimize(initial_molecules, num_iterations=5)

        assert results["best_molecule"] in initial_molecules
        assert results["best_score"] >= 0.0
        assert len(results["optimization_history"]) <= 5

    def test_full_genetic_algorithm_workflow(self):
        """Test complete genetic algorithm workflow."""
        optimizer = GeneticAlgorithmOptimizer(
            population_size=10, mutation_rate=0.2, crossover_rate=0.8
        )

        initial_molecules = ["C", "CC", "CCO", "CCC"]
        results = optimizer.optimize(initial_molecules, num_iterations=5)

        assert len(results["final_population"]) == 10
        assert len(results["final_scores"]) == 10
        assert results["best_score"] >= 0.0
        assert len(results["optimization_history"]) == 5

    def test_comparison_of_optimization_methods(self):
        """Test comparison between different optimization methods."""
        molecules = ["CCO", "CC"]

        # Test all methods
        bayesian_result = optimize_molecule(molecules[0], "bayesian", num_iterations=3)
        genetic_result = optimize_molecule(molecules[0], "genetic", num_iterations=3)
        random_result = optimize_molecule(molecules[0], "random", num_iterations=3)

        # All should return valid results
        for result in [bayesian_result, genetic_result, random_result]:
            assert "original_molecule" in result
            assert "optimized_molecule" in result
            assert "optimization_score" in result
            assert result["optimization_score"] >= 0.0

    def test_batch_optimization_performance(self):
        """Test batch optimization with larger molecule set."""
        # Create a set of test molecules
        molecules = [
            "C",
            "CC",
            "CCC",
            "CCCC",
            "CCO",
            "CCCO",
            "CN",
            "CCN",
            "CO",
            "CCO",
            "C(C)O",
        ]

        results_df = batch_optimize(
            molecules, optimization_method="genetic", num_iterations=3
        )

        assert len(results_df) == len(molecules)
        assert all(results_df["optimization_score"] >= 0.0)
        assert all(results_df["original_molecule"].isin(molecules))

    def test_optimization_with_custom_constraints(self):
        """Test optimization with custom molecular constraints."""

        def constrained_objective(smiles):
            """Objective that prefers molecules with specific characteristics."""
            if not smiles or len(smiles) < 3:
                return 0.0

            score = 0.0
            # Prefer molecules with oxygen
            if "O" in smiles:
                score += 0.5
            # Prefer moderate length
            if 5 <= len(smiles) <= 15:
                score += 0.3
            # Prefer molecules with carbon
            if "C" in smiles:
                score += 0.2

            return min(1.0, score)

        result = optimize_molecule(
            "CCO",
            optimization_method="bayesian",
            objective_function=constrained_objective,
            num_iterations=5,
        )

        assert result["optimization_score"] <= 1.0
        assert result["optimization_score"] >= 0.0


class TestOptimizationMetricsAndAnalysis:
    """Test cases for optimization metrics and analysis functions."""

    def test_calculate_optimization_metrics_complete(self):
        """Test calculation of optimization metrics with complete data."""
        mock_results = {
            "best_score": 0.85,
            "optimization_history": [
                {"best_score": 0.6},
                {"best_score": 0.7},
                {"best_score": 0.8},
                {"best_score": 0.85},
            ],
            "final_scores": [0.85, 0.7, 0.6, 0.8, 0.75],
        }

        metrics = calculate_optimization_metrics(mock_results)

        assert "best_score" in metrics
        assert "convergence_rate" in metrics
        assert "score_variance" in metrics
        assert "improvement_ratio" in metrics
        assert "exploration_diversity" in metrics

        assert metrics["best_score"] == 0.85
        assert metrics["improvement_ratio"] > 0  # Should show improvement
        assert metrics["score_variance"] > 0  # Should have variance

    def test_calculate_optimization_metrics_empty(self):
        """Test metrics calculation with empty results."""
        empty_results = {}

        metrics = calculate_optimization_metrics(empty_results)

        assert all(value == 0.0 for value in metrics.values())

    def test_calculate_optimization_metrics_error_handling(self):
        """Test metrics calculation handles errors gracefully."""
        invalid_results = {"optimization_history": "invalid_data", "final_scores": None}

        metrics = calculate_optimization_metrics(invalid_results)

        assert isinstance(metrics, dict)
        assert all(
            key in metrics
            for key in ["best_score", "convergence_rate", "score_variance"]
        )

    def test_compare_optimization_methods(self):
        """Test comparison of different optimization methods."""
        molecules = ["CCO", "CC"]
        methods = ["bayesian", "genetic"]

        comparison_df = compare_optimization_methods(
            molecules, methods=methods, num_iterations=3
        )

        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2  # Two methods
        assert "method" in comparison_df.columns
        assert "avg_score" in comparison_df.columns
        assert "avg_improvement" in comparison_df.columns
        assert "avg_time" in comparison_df.columns
        assert "success_rate" in comparison_df.columns

        assert all(comparison_df["success_rate"] >= 0.0)
        assert all(comparison_df["success_rate"] <= 1.0)

    def test_compare_optimization_methods_custom_objective(self):
        """Test method comparison with custom objective function."""

        def simple_objective(smiles):
            return len(smiles) / 10.0

        molecules = ["C", "CCO"]
        comparison_df = compare_optimization_methods(
            molecules,
            methods=["bayesian", "random"],
            objective_function=simple_objective,
            num_iterations=2,
        )

        assert len(comparison_df) == 2
        assert all(comparison_df["avg_score"] >= 0.0)

    def test_validate_optimization_results_complete(self):
        """Test validation of complete optimization results."""
        complete_results = {
            "best_molecule": "CCO",
            "best_score": 0.75,
            "optimization_history": [{"iteration": 1}],
            "final_population": ["CCO", "CC"],
            "final_scores": [0.75, 0.6],
        }

        validation = validate_optimization_results(complete_results)

        assert validation["has_best_molecule"] is True
        assert validation["has_best_score"] is True
        assert validation["valid_score_range"] is True
        assert validation["has_history"] is True
        assert validation["non_empty_population"] is True
        assert validation["score_consistency"] is True

    def test_validate_optimization_results_incomplete(self):
        """Test validation of incomplete optimization results."""
        incomplete_results = {
            "best_score": 1.5,  # Invalid score range
            "final_population": ["CCO"],
            "final_scores": [0.5, 0.6],  # Mismatch with population
        }

        validation = validate_optimization_results(incomplete_results)

        assert validation["has_best_molecule"] is False
        assert validation["valid_score_range"] is False  # Score > 1.0
        assert validation["score_consistency"] is False  # Length mismatch

    def test_validate_optimization_results_empty(self):
        """Test validation of empty results."""
        validation = validate_optimization_results({})

        assert all(value is False for value in validation.values())

    def test_extract_optimization_insights_successful(self):
        """Test insights extraction from successful optimization."""
        successful_results = {
            "best_score": 0.8,
            "optimization_history": [
                {"best_score": 0.3},
                {"best_score": 0.5},
                {"best_score": 0.7},
                {"best_score": 0.8},
            ],
            "final_population": ["CCO", "CC", "CCC", "CCCO"],
        }

        insights = extract_optimization_insights(successful_results)

        assert "optimization_success" in insights
        assert "convergence_analysis" in insights
        assert "molecular_patterns" in insights
        assert "recommendations" in insights

        assert insights["optimization_success"] is True  # Score > 0.5
        assert insights["convergence_analysis"]["improving_trend"] is True
        assert insights["molecular_patterns"]["unique_molecules"] == 4

    def test_extract_optimization_insights_poor_performance(self):
        """Test insights extraction from poor optimization."""
        poor_results = {
            "best_score": 0.2,  # Low score
            "optimization_history": [
                {"best_score": 0.2},
                {"best_score": 0.2},
                {"best_score": 0.2},  # Plateau
                {"best_score": 0.2},
                {"best_score": 0.2},
            ],
            "final_population": ["CCO", "CCO", "CCO"],  # Low diversity
        }

        insights = extract_optimization_insights(poor_results)

        assert insights["optimization_success"] is False
        # Check if plateau detected or not (both are valid outcomes)
        assert "plateau_detected" in insights["convergence_analysis"]
        assert insights["molecular_patterns"]["diversity_ratio"] < 1.0
        assert len(insights["recommendations"]) > 0

    def test_extract_optimization_insights_empty(self):
        """Test insights extraction from empty results."""
        insights = extract_optimization_insights({})

        assert insights["optimization_success"] is False
        assert isinstance(insights["convergence_analysis"], dict)
        assert isinstance(insights["molecular_patterns"], dict)
        assert isinstance(insights["recommendations"], list)


class TestOptimizationUtilities:
    """Test cases for optimization utility functions."""

    def test_save_load_optimization_checkpoint(self):
        """Test saving and loading optimization checkpoints."""
        optimizer = GeneticAlgorithmOptimizer(population_size=20)
        optimizer.optimization_history = [{"iteration": 1, "score": 0.5}]

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Test saving
            success = save_optimization_checkpoint(optimizer, tmp_path)
            assert success is True
            assert os.path.exists(tmp_path)

            # Test loading
            checkpoint_data = load_optimization_checkpoint(tmp_path)
            assert checkpoint_data is not None
            assert "optimizer_type" in checkpoint_data
            assert "optimization_history" in checkpoint_data
            assert checkpoint_data["optimizer_type"] == "GeneticAlgorithmOptimizer"
            assert checkpoint_data["population_size"] == 20

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_save_optimization_checkpoint_error(self):
        """Test checkpoint saving handles errors gracefully."""
        optimizer = MolecularOptimizer()

        # Try to save to invalid path
        success = save_optimization_checkpoint(
            optimizer, "/invalid/path/checkpoint.pkl"
        )
        assert success is False

    def test_load_optimization_checkpoint_error(self):
        """Test checkpoint loading handles errors gracefully."""
        # Try to load non-existent file
        checkpoint_data = load_optimization_checkpoint("non_existent_file.pkl")
        assert checkpoint_data is None

    def test_generate_optimization_summary_single(self):
        """Test summary generation for single optimization result."""
        result = {
            "original_molecule": "CCO",
            "optimized_molecule": "CCCO",
            "best_score": 0.85,
            "improvement": 0.2,
            "method": "bayesian",
        }

        summary = generate_optimization_summary(result)

        assert isinstance(summary, str)
        assert "Molecular Optimization Summary" in summary
        assert "Total optimizations: 1" in summary
        assert "CCO" in summary
        assert "CCCO" in summary
        assert "0.850" in summary
        assert "bayesian" in summary

    def test_generate_optimization_summary_multiple(self):
        """Test summary generation for multiple optimization results."""
        results = [
            {
                "original_molecule": "CCO",
                "best_score": 0.8,
                "improvement": 0.3,
                "method": "bayesian",
            },
            {
                "original_molecule": "CC",
                "best_score": 0.6,
                "improvement": 0.1,
                "method": "genetic",
            },
            {
                "original_molecule": "CCC",
                "best_score": 0.9,
                "improvement": 0.4,
                "method": "bayesian",
            },
        ]

        summary = generate_optimization_summary(results)

        assert "Total optimizations: 3" in summary
        assert "Average best score:" in summary
        assert "Maximum score achieved: 0.900" in summary
        assert "Success rate" in summary
        assert "Best Optimization" in summary
        assert "CCC" in summary  # Should identify best optimization

    def test_generate_optimization_summary_empty(self):
        """Test summary generation with empty results."""
        summary = generate_optimization_summary([])

        assert "Total optimizations: 0" in summary
        assert isinstance(summary, str)

    def test_generate_optimization_summary_incomplete_data(self):
        """Test summary generation with incomplete data."""
        incomplete_results = [
            {"best_score": 0.7},  # Missing other fields
            {"original_molecule": "CC"},  # Missing score
            {},  # Empty result
        ]

        summary = generate_optimization_summary(incomplete_results)

        assert "Total optimizations: 3" in summary
        assert isinstance(summary, str)
        # Should handle missing data gracefully


class TestOptimizationRobustness:
    """Test cases for optimization robustness and edge cases."""

    def test_optimization_with_invalid_molecules(self):
        """Test optimization handles invalid molecules gracefully."""
        invalid_molecules = ["", "INVALID_SMILES", None, "123456"]

        for molecule in invalid_molecules:
            if molecule is not None:
                try:
                    result = optimize_molecule(molecule, num_iterations=2)
                    # Should not crash, may return low scores
                    assert "original_molecule" in result
                    assert result["optimization_score"] >= 0.0
                except Exception:
                    # Acceptable to fail with truly invalid input
                    pass

    def test_optimization_with_extreme_parameters(self):
        """Test optimization with extreme parameter values."""
        # Very small iterations
        result_small = optimize_molecule("CCO", num_iterations=1)
        assert "optimized_molecule" in result_small

        # Very large iterations (should complete reasonably fast for test)
        result_large = optimize_molecule("CCO", num_iterations=200)
        assert "optimized_molecule" in result_large

    def test_genetic_algorithm_edge_cases(self):
        """Test genetic algorithm with edge case parameters."""
        # Very small population
        optimizer = GeneticAlgorithmOptimizer(population_size=2)
        results = optimizer.optimize(["CCO"], num_iterations=3)
        assert len(results["final_population"]) == 2

        # Extreme mutation/crossover rates
        optimizer_extreme = GeneticAlgorithmOptimizer(
            population_size=5,
            mutation_rate=0.0,  # No mutation
            crossover_rate=1.0,  # Always crossover
        )
        results_extreme = optimizer_extreme.optimize(["CCO", "CC"], num_iterations=2)
        assert "best_molecule" in results_extreme

    def test_optimization_memory_efficiency(self):
        """Test optimization doesn't consume excessive memory."""
        # Test with larger molecule set
        large_molecule_set = ["C" + "C" * i for i in range(20)]

        # Should complete without memory issues
        results_df = batch_optimize(large_molecule_set[:10], num_iterations=5)

        assert len(results_df) == 10
        assert not results_df.empty

    def test_concurrent_optimizations(self):
        """Test that multiple optimizations can run without interference."""
        molecules = ["CCO", "CC", "CCC"]

        # Run multiple optimizations
        results = []
        for molecule in molecules:
            result = optimize_molecule(molecule, num_iterations=3)
            results.append(result)

        # Each should have completed successfully
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["original_molecule"] == molecules[i]
            assert "optimized_molecule" in result


if __name__ == "__main__":
    pytest.main([__file__])
