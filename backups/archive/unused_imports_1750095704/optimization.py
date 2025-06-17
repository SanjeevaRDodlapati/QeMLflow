"""
QeMLflow Drug Discovery - Molecular Optimization
==============================================

Advanced molecular structure optimization algorithms for drug discovery.
approaches including Bayesian optimization, genetic algorithms, and gradient-based
methods to improve desired properties.
"""

import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    RDKIT_AVAILABLE = True
except ImportError:
    logging.warning("RDKit not available. Molecular optimization will be limited.")
    RDKIT_AVAILABLE = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning(
        "Scikit-learn not available. Some optimization methods will be limited."
    )
    SKLEARN_AVAILABLE = False


class MolecularOptimizer:
    """
    Base class for molecular optimization strategies.
    """

    def __init__(self, objective_function: Optional[Callable] = None) -> None:
        """
        Initialize the molecular optimizer.

        Args:
            objective_function: Function to optimize (takes SMILES, returns score)
        """
        self.objective_function = objective_function or self._default_objective
        self.optimization_history = []

    def _default_objective(self, smiles: str) -> float:
        """
        Default objective function - optimize for drug-likeness.

        Args:
            smiles: SMILES string

        Returns:
            Drug-likeness score (0-1)
        """
        if not RDKIT_AVAILABLE:
            return random.random()  # Random score as fallback

        # Handle empty or None SMILES
        if not smiles or smiles == "":
            return 0.0

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0

            # Calculate Lipinski descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            # Score based on Lipinski's rule of five
            score = 0.0
            score += 0.25 if mw <= 500 else 0.0
            score += 0.25 if logp <= 5 else 0.0
            score += 0.25 if hbd <= 5 else 0.0
            score += 0.25 if hba <= 10 else 0.0

            return score

        except Exception as e:
            logging.warning(f"Error calculating objective for {smiles}: {e}")
            return 0.0

    def optimize(
        self, initial_molecules: List[str], num_iterations: int = 100
    ) -> Dict[str, Union[str, float, List]]:
        """
        Abstract optimization method to be implemented by subclasses.

        Args:
            initial_molecules: Starting molecule pool
            num_iterations: Number of optimization iterations

        Returns:
            Optimization results dictionary
        """
        raise NotImplementedError("Subclasses must implement optimize method")


class BayesianOptimizer(MolecularOptimizer):
    """
    Bayesian optimization for molecular design.
    """

    def __init__(self, objective_function: Optional[Callable] = None):
        super().__init__(objective_function)
        self.gp_model = None
        self.scaler = None

    def _molecular_to_features(self, smiles: str) -> np.ndarray:
        """
        Convert SMILES to feature vector for Gaussian Process.

        Args:
            smiles: SMILES string

        Returns:
            Feature vector
        """
        if not RDKIT_AVAILABLE:
            # Simple hash-based features as fallback
            features = np.zeros(10)
            for i, char in enumerate(smiles[:10]):
                features[i] = hash(char) % 100
            return features

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(10)

            # Calculate molecular descriptors as features
            features = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.HeavyAtomCount(mol),
                # Use fallback for FractionCsp3 if not available
                getattr(Descriptors, "FractionCsp3", lambda x: 0.0)(mol),
            ]

            return np.array(features)

        except Exception as e:
            logging.warning(f"Error extracting features for {smiles}: {e}")
            return np.zeros(10)

    def _acquisition_function(self, features: np.ndarray) -> float:
        """
        Upper confidence bound acquisition function.

        Args:
            features: Molecular features

        Returns:
            Acquisition score
        """
        if self.gp_model is None or not SKLEARN_AVAILABLE:
            return random.random()

        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            mean, std = self.gp_model.predict(features_scaled, return_std=True)

            # Upper confidence bound
            kappa = 2.0  # Exploration parameter
            return mean[0] + kappa * std[0]

        except Exception as e:
            logging.warning(f"Error in acquisition function: {e}")
            return random.random()

    def optimize(
        self, initial_molecules: List[str], num_iterations: int = 100
    ) -> Dict[str, Union[str, float, List]]:
        """
        Perform Bayesian optimization on molecular structures.

        Args:
            initial_molecules: Initial molecule pool
            num_iterations: Number of optimization iterations

        Returns:
            Optimization results
        """
        if not SKLEARN_AVAILABLE:
            logging.warning("Scikit-learn not available. Using random optimization.")
            return self._random_optimization(initial_molecules, num_iterations)

        # Initialize data
        X = np.array(
            [self._molecular_to_features(smiles) for smiles in initial_molecules]
        )
        y = np.array([self.objective_function(smiles) for smiles in initial_molecules])

        # Initialize scaler and GP model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp_model = GaussianProcessRegressor(kernel=kernel, random_state=42)
        self.gp_model.fit(X_scaled, y)

        best_smiles = initial_molecules[np.argmax(y)]
        best_score = np.max(y)

        all_molecules = initial_molecules.copy()
        all_scores = y.tolist()

        for iteration in range(num_iterations):
            # Generate candidate molecules (simplified)
            candidate_molecules = self._generate_candidates(
                all_molecules, num_candidates=10
            )

            # Evaluate acquisition function for candidates
            best_candidate = None
            best_acquisition = -np.inf

            for candidate in candidate_molecules:
                if candidate not in all_molecules:  # Only consider new molecules
                    features = self._molecular_to_features(candidate)
                    acquisition_score = self._acquisition_function(features)

                    if acquisition_score > best_acquisition:
                        best_acquisition = acquisition_score
                        best_candidate = candidate

            if best_candidate is None:
                continue

            # Evaluate objective function for best candidate
            candidate_score = self.objective_function(best_candidate)

            # Update data
            all_molecules.append(best_candidate)
            all_scores.append(candidate_score)

            # Update GP model
            X_new = np.vstack([X, self._molecular_to_features(best_candidate)])
            y_new = np.append(y, candidate_score)

            X_scaled_new = self.scaler.fit_transform(X_new)
            self.gp_model.fit(X_scaled_new, y_new)

            X = X_new
            y = y_new

            # Update best if necessary
            if candidate_score > best_score:
                best_score = candidate_score
                best_smiles = best_candidate

            # Store iteration results
            self.optimization_history.append(
                {
                    "iteration": iteration,
                    "best_molecule": best_smiles,
                    "best_score": best_score,
                    "candidate_molecule": best_candidate,
                    "candidate_score": candidate_score,
                }
            )

        return {
            "best_molecule": best_smiles,
            "best_score": best_score,
            "optimization_history": self.optimization_history,
            "final_population": all_molecules,
            "final_scores": all_scores,
        }

    def _random_optimization(
        self, initial_molecules: List[str], num_iterations: int
    ) -> Dict[str, Union[str, float, List]]:
        """Fallback random optimization when sklearn is not available."""
        best_smiles = random.choice(initial_molecules)
        best_score = self.objective_function(best_smiles)

        for smiles in initial_molecules:
            score = self.objective_function(smiles)
            if score > best_score:
                best_score = score
                best_smiles = smiles

        # Random search
        all_molecules = initial_molecules.copy()
        for iteration in range(num_iterations):
            # Generate random candidate
            base_molecule = random.choice(all_molecules)
            candidate = self._simple_mutation(base_molecule)

            if candidate and candidate not in all_molecules:
                score = self.objective_function(candidate)
                all_molecules.append(candidate)

                if score > best_score:
                    best_score = score
                    best_smiles = candidate

        return {
            "best_molecule": best_smiles,
            "best_score": best_score,
            "optimization_history": [],
            "final_population": all_molecules,
            "final_scores": [self.objective_function(mol) for mol in all_molecules],
        }

    def _generate_candidates(
        self, molecules: List[str], num_candidates: int = 10
    ) -> List[str]:
        """Generate candidate molecules for optimization."""
        candidates = []

        for _ in range(num_candidates):
            base_molecule = random.choice(molecules)
            candidate = self._simple_mutation(base_molecule)
            if candidate:
                candidates.append(candidate)

        return candidates

    def _simple_mutation(self, smiles: str) -> Optional[str]:
        """Apply simple mutation to a SMILES string."""
        if not RDKIT_AVAILABLE or len(smiles) < 5:
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Simple mutation: randomly add/remove atoms (very simplified)
            # In practice, you'd want more sophisticated mutations
            return smiles  # Return original for now

        except Exception:
            return None


class GeneticAlgorithmOptimizer(MolecularOptimizer):
    """
    Genetic algorithm optimization for molecular design.
    """

    def __init__(
        self,
        objective_function: Optional[Callable] = None,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
    ):
        """
        Initialize genetic algorithm optimizer.

        Args:
            objective_function: Objective function to optimize
            population_size: Size of population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        super().__init__(objective_function)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def optimize(
        self, initial_molecules: List[str], num_iterations: int = 100
    ) -> Dict[str, Union[str, float, List]]:
        """
        Perform genetic algorithm optimization.

        Args:
            initial_molecules: Initial population
            num_iterations: Number of generations

        Returns:
            Optimization results
        """
        # Initialize population
        population = initial_molecules[: self.population_size]

        # Pad population if needed
        while len(population) < self.population_size:
            population.append(random.choice(initial_molecules))

        best_smiles = None
        best_score = -np.inf

        for generation in range(num_iterations):
            # Evaluate fitness
            fitness_scores = [self.objective_function(mol) for mol in population]

            # Update best
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_score:
                best_score = fitness_scores[max_idx]
                best_smiles = population[max_idx]

            # Selection (tournament selection)
            new_population = []
            for _ in range(self.population_size):
                parent = self._tournament_selection(population, fitness_scores)
                new_population.append(parent)

            # Crossover and mutation
            for i in range(0, len(new_population), 2):
                if i + 1 < len(new_population):
                    if random.random() < self.crossover_rate:
                        child1, child2 = self._crossover(
                            new_population[i], new_population[i + 1]
                        )
                        new_population[i] = child1 or new_population[i]
                        new_population[i + 1] = child2 or new_population[i + 1]

            # Mutation
            for i in range(len(new_population)):
                if random.random() < self.mutation_rate:
                    mutated = self._mutate(new_population[i])
                    if mutated:
                        new_population[i] = mutated

            population = new_population

            # Store generation results
            self.optimization_history.append(
                {
                    "generation": generation,
                    "best_molecule": best_smiles,
                    "best_score": best_score,
                    "avg_fitness": np.mean(fitness_scores),
                    "max_fitness": np.max(fitness_scores),
                }
            )

        return {
            "best_molecule": best_smiles,
            "best_score": best_score,
            "optimization_history": self.optimization_history,
            "final_population": population,
            "final_scores": [self.objective_function(mol) for mol in population],
        }

    def _tournament_selection(
        self,
        population: List[str],
        fitness_scores: List[float],
        tournament_size: int = 3,
    ) -> str:
        """Tournament selection for genetic algorithm."""
        tournament_indices = random.sample(
            range(len(population)), min(tournament_size, len(population))
        )
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]

    def _crossover(
        self, parent1: str, parent2: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Simple crossover operation for SMILES strings."""
        if not RDKIT_AVAILABLE:
            # Simple string crossover as fallback
            if len(parent1) > 2 and len(parent2) > 2:
                cut1 = random.randint(1, len(parent1) - 1)
                cut2 = random.randint(1, len(parent2) - 1)
                child1 = parent1[:cut1] + parent2[cut2:]
                child2 = parent2[:cut2] + parent1[cut1:]
                return child1, child2
            return parent1, parent2

        try:
            # In practice, you'd implement more sophisticated crossover
            # For now, return parents as simplified implementation
            return parent1, parent2

        except Exception:
            return parent1, parent2

    def _mutate(self, smiles: str) -> Optional[str]:
        """Simple mutation operation for SMILES strings."""
        # Use the same simple mutation as in Bayesian optimizer
        return self._simple_mutation(smiles)

    def _simple_mutation(self, smiles: str) -> Optional[str]:
        """Apply simple mutation to a SMILES string."""
        if len(smiles) < 3:
            return smiles

        # Very simple mutation - replace a random character
        try:
            char_list = list(smiles)
            idx = random.randint(0, len(char_list) - 1)
            # Simple character replacements
            replacements = {"C": "N", "N": "O", "O": "C", "c": "n", "n": "o", "o": "c"}
            if char_list[idx] in replacements:
                char_list[idx] = replacements[char_list[idx]]
                return "".join(char_list)
            return smiles
        except (IndexError, KeyError, ValueError):
            return smiles


# Module-level functions for molecular optimization


def optimize_molecule(
    smiles: str,
    optimization_method: str = "bayesian",
    objective_function: Optional[Callable] = None,
    num_iterations: int = 50,
) -> Dict[str, Union[str, float]]:
    """
    Optimize a single molecule using specified method.

    Args:
        smiles: Input SMILES string
        optimization_method: Method to use ("bayesian", "genetic", "random")
        objective_function: Custom objective function
        num_iterations: Number of optimization iterations

    Returns:
        Optimization results
    """
    if optimization_method.lower() == "bayesian":
        optimizer = BayesianOptimizer(objective_function)
    elif optimization_method.lower() == "genetic":
        optimizer = GeneticAlgorithmOptimizer(objective_function)
    elif optimization_method.lower() == "random":
        # For random method, use a simple random search approach
        optimizer = BayesianOptimizer(
            objective_function
        )  # Use Bayesian but force random mode
        return _random_optimize_single(smiles, objective_function, num_iterations)
    else:
        # Default to bayesian for any other method
        optimizer = BayesianOptimizer(objective_function)

    # Generate initial population around the input molecule
    initial_molecules = [smiles]

    # Add some variations (simplified)
    for _ in range(min(10, num_iterations // 5)):
        variant = _generate_similar_molecule(smiles)
        if variant and variant not in initial_molecules:
            initial_molecules.append(variant)

    results = optimizer.optimize(initial_molecules, num_iterations)

    original_score = optimizer.objective_function(smiles)

    return {
        "original_molecule": smiles,
        "optimized_molecule": results["best_molecule"],
        "improvement": results["best_score"] - original_score,
        "optimization_score": results["best_score"],
        "original_score": original_score,
    }


def _random_optimize_single(
    smiles: str, objective_function: Optional[Callable], num_iterations: int
) -> Dict[str, Union[str, float]]:
    """Random optimization for a single molecule."""
    optimizer = MolecularOptimizer(objective_function)

    best_smiles = smiles
    best_score = optimizer.objective_function(smiles)
    original_score = best_score

    # Simple random search
    for _ in range(num_iterations):
        # Generate a random variant
        variant = _generate_similar_molecule(smiles)
        if variant:
            score = optimizer.objective_function(variant)
            if score > best_score:
                best_score = score
                best_smiles = variant

    return {
        "original_molecule": smiles,
        "optimized_molecule": best_smiles,
        "improvement": best_score - original_score,
        "optimization_score": best_score,
        "original_score": original_score,
    }


def _generate_similar_molecule(smiles: str) -> Optional[str]:
    """
    Generate a molecule similar to the input SMILES.

    Args:
        smiles: Input SMILES string

    Returns:
        Similar SMILES string or None if generation fails
    """
    if not RDKIT_AVAILABLE or len(smiles) < 3:
        # Simple string-based mutation as fallback
        if len(smiles) > 3:
            # Random character replacement
            chars = list(smiles)
            idx = random.randint(0, len(chars) - 1)
            replacements = {"C": "N", "N": "O", "O": "C"}
            if chars[idx] in replacements:
                chars[idx] = replacements[chars[idx]]
                return "".join(chars)
        return smiles

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # For now, return the original molecule
        # In practice, you would implement more sophisticated molecular mutations
        return smiles

    except Exception:
        return None


# Additional utility functions for optimization workflows


def batch_optimize(
    molecules: List[str],
    optimization_method: str = "bayesian",
    objective_function: Optional[Callable] = None,
    num_iterations: int = 50,
    parallel: bool = False,
) -> List[Dict[str, Union[str, float]]]:
    """
    Optimize a batch of molecules.

    Args:
        molecules: List of SMILES strings
        optimization_method: Optimization method to use
        objective_function: Custom objective function
        num_iterations: Number of iterations per molecule
        parallel: Whether to use parallel processing

    Returns:
        List of optimization results
    """
    results = []

    for smiles in molecules:
        try:
            result = optimize_molecule(
                smiles, optimization_method, objective_function, num_iterations
            )
            results.append(result)
        except Exception as e:
            logging.warning(f"Failed to optimize {smiles}: {e}")
            # Add failed result
            results.append(
                {
                    "original_molecule": smiles,
                    "optimized_molecule": smiles,
                    "improvement": 0.0,
                    "optimization_score": 0.0,
                    "original_score": 0.0,
                    "error": str(e),
                }
            )

    return results
