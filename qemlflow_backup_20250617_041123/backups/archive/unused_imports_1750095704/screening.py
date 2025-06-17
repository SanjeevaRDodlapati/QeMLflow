"""
ChemML Drug Discovery - Virtual Screening
========================================

Comprehensive virtual screening capabilities for drug discovery pipelines.
- Similarity-based screening using molecular fingerprints
- Pharmacophore-based screening
- Screening performance metrics and validation
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.rdmolops import RDKFingerprint
    from rdkit.DataStructs import TanimotoSimilarity

    RDKIT_AVAILABLE = True
except ImportError:
    logging.warning(
        "RDKit not available. Virtual screening will use fallback implementations."
    )
    RDKIT_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning("Scikit-learn not available. Some features may be limited.")
    SKLEARN_AVAILABLE = False


class VirtualScreener:
    """
    Main virtual screening class for compound library filtering and ranking.

    This class provides a unified interface for different virtual screening
    approaches including similarity-based, pharmacophore-based, and ML-based screening.
    """

    def __init__(self, screening_method: str = "similarity", **kwargs) -> None:
        """
        Initialize the virtual screener.

        Args:
            screening_method: The screening method to use ('similarity', 'pharmacophore', 'ml')
            **kwargs: Additional parameters for specific screening methods
        """
        self.screening_method = screening_method
        self.is_configured = False
        self.reference_compounds = []
        self.screening_results = []
        self.config = kwargs

        # Initialize method-specific screeners
        if screening_method == "similarity":
            self.screener = SimilarityScreener(**kwargs)
        elif screening_method == "pharmacophore":
            self.screener = PharmacophoreScreener(**kwargs)
        else:
            # Default to similarity screening
            self.screener = SimilarityScreener(**kwargs)

        logging.info(f"VirtualScreener initialized with method: {screening_method}")

    def set_reference_compounds(self, compounds: List[Union[str, "Chem.Mol"]]) -> None:
        """
        Set reference compounds for screening.

        Args:
            compounds: List of SMILES strings or RDKit Mol objects
        """
        self.reference_compounds = compounds
        self.screener.set_reference_compounds(compounds)
        self.is_configured = True
        logging.info(f"Set {len(compounds)} reference compounds for screening")

    def screen_library(
        self,
        library: List[Union[str, "Chem.Mol"]],
        threshold: float = 0.7,
        max_compounds: int = 1000,
    ) -> pd.DataFrame:
        """
        Screen a compound library against reference compounds.

        Args:
            library: List of compounds to screen (SMILES or Mol objects)
            threshold: Similarity threshold for filtering
            max_compounds: Maximum number of compounds to return

        Returns:
            DataFrame with screening results including scores and rankings
        """
        if not self.is_configured:
            raise ValueError(
                "Virtual screener not configured. Set reference compounds first."
            )

        logging.info(
            f"Screening library of {len(library)} compounds with threshold {threshold}"
        )

        results = self.screener.screen_library(library, threshold, max_compounds)
        self.screening_results = results

        logging.info(
            f"Screening completed. Found {len(results)} compounds above threshold"
        )
        return results

    def get_top_hits(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N hits from screening results.

        Args:
            n: Number of top hits to return

        Returns:
            DataFrame with top hits
        """
        if self.screening_results is None:
            logging.warning("No screening results available")
            return pd.DataFrame()

        if isinstance(self.screening_results, list):
            if len(self.screening_results) == 0:
                return pd.DataFrame()
            # Convert to DataFrame if needed
            df = pd.DataFrame(self.screening_results)
        else:
            df = self.screening_results
            if df.empty:
                return pd.DataFrame()

        # Sort by score (assuming higher is better)
        if "score" in df.columns:
            top_hits = df.nlargest(n, "score")
        else:
            # Fallback: just return first n rows
            top_hits = df.head(n)

        logging.info(f"Retrieved top {len(top_hits)} hits")
        return top_hits

    def calculate_enrichment_factor(
        self, known_actives: List[str], fraction: float = 0.1
    ) -> float:
        """
        Calculate enrichment factor for screening validation.

        Args:
            known_actives: List of known active compound SMILES
            fraction: Fraction of top compounds to consider

        Returns:
            Enrichment factor
        """
        if self.screening_results is None or len(self.screening_results) == 0:
            return 0.0

        df = (
            self.screening_results
            if isinstance(self.screening_results, pd.DataFrame)
            else pd.DataFrame(self.screening_results)
        )

        n_selected = int(len(df) * fraction)
        top_compounds = df.nlargest(n_selected, "score")

        # Count actives in top fraction
        actives_in_top = 0
        for compound in top_compounds["smiles"]:
            if compound in known_actives:
                actives_in_top += 1

        # Calculate enrichment
        total_actives = len([c for c in df["smiles"] if c in known_actives])
        expected_actives = total_actives * fraction

        if expected_actives > 0:
            enrichment = actives_in_top / expected_actives
        else:
            enrichment = 0.0

        logging.info(f"Enrichment factor: {enrichment:.2f}")
        return enrichment


class SimilarityScreener:
    """
    Similarity-based virtual screening using molecular fingerprints.
    """

    def __init__(self, fingerprint_type: str = "morgan", **kwargs):
        """
        Initialize similarity screener.

        Args:
            fingerprint_type: Type of molecular fingerprint to use
            **kwargs: Additional parameters
        """
        self.fingerprint_type = fingerprint_type
        self.reference_fingerprints = []
        self.reference_compounds = []

    def set_reference_compounds(self, compounds: List[Union[str, "Chem.Mol"]]) -> None:
        """Set reference compounds and calculate their fingerprints."""
        self.reference_compounds = compounds
        self.reference_fingerprints = []

        for compound in compounds:
            fp = self._calculate_fingerprint(compound)
            if fp is not None:
                self.reference_fingerprints.append(fp)

        logging.info(
            f"Calculated fingerprints for {len(self.reference_fingerprints)} reference compounds"
        )

    def screen_library(
        self,
        library: List[Union[str, "Chem.Mol"]],
        threshold: float = 0.7,
        max_compounds: int = 1000,
    ) -> pd.DataFrame:
        """Screen library using similarity-based filtering."""
        results = []

        for i, compound in enumerate(library):
            try:
                # Calculate fingerprint for library compound
                lib_fp = self._calculate_fingerprint(compound)
                if lib_fp is None:
                    continue

                # Calculate maximum similarity to reference compounds
                max_similarity = 0.0
                best_reference_idx = -1

                for j, ref_fp in enumerate(self.reference_fingerprints):
                    similarity = self._calculate_similarity(lib_fp, ref_fp)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_reference_idx = j

                # Add to results if above threshold
                if max_similarity >= threshold:
                    smiles = (
                        compound
                        if isinstance(compound, str)
                        else Chem.MolToSmiles(compound)
                    )
                    results.append(
                        {
                            "compound_id": f"compound_{i}",
                            "smiles": smiles,
                            "score": max_similarity,
                            "best_reference": best_reference_idx,
                            "screening_method": "similarity",
                        }
                    )

            except Exception as e:
                logging.warning(f"Error processing compound {i}: {e}")
                continue

        # Sort by score and limit results
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:max_compounds]

        return pd.DataFrame(results)

    def _calculate_fingerprint(self, compound: Union[str, "Chem.Mol"]) -> Any:
        """Calculate molecular fingerprint."""
        if not RDKIT_AVAILABLE:
            # Simple hash-based fingerprint fallback
            smiles = compound if isinstance(compound, str) else str(compound)
            return hash(smiles) % 1000000

        try:
            if isinstance(compound, str):
                mol = Chem.MolFromSmiles(compound)
            else:
                mol = compound

            if mol is None:
                return None

            if self.fingerprint_type == "morgan":
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            else:
                fp = RDKFingerprint(mol)

            return fp

        except Exception as e:
            logging.warning(f"Error calculating fingerprint: {e}")
            return None

    def _calculate_similarity(self, fp1, fp2) -> float:
        """Calculate similarity between two fingerprints."""
        if not RDKIT_AVAILABLE:
            # Simple similarity for hash-based fingerprints
            return 1.0 if fp1 == fp2 else abs(fp1 - fp2) / 1000000

        try:
            return TanimotoSimilarity(fp1, fp2)
        except Exception:
            return 0.0


class PharmacophoreScreener:
    """
    Pharmacophore-based virtual screening.

    Note: This is a simplified implementation. Full pharmacophore screening
    would require specialized software like RDKit's pharmacophore features
    or external tools.
    """

    def __init__(self, pharmacophore_features: Optional[List[str]] = None):
        """
        Initialize pharmacophore screener.

        Args:
            pharmacophore_features: List of pharmacophore features to match
        """
        self.pharmacophore_features = pharmacophore_features or [
            "aromatic_ring",
            "hydrogen_bond_donor",
            "hydrogen_bond_acceptor",
        ]
        self.reference_compounds = []

    def set_reference_compounds(self, compounds: List[Union[str, "Chem.Mol"]]) -> None:
        """Set reference compounds for pharmacophore generation."""
        self.reference_compounds = compounds
        logging.info(f"Set {len(compounds)} compounds for pharmacophore analysis")

    def screen_library(
        self,
        library: List[Union[str, "Chem.Mol"]],
        threshold: float = 0.7,
        max_compounds: int = 1000,
    ) -> pd.DataFrame:
        """Screen library using pharmacophore-based filtering."""
        results = []

        for i, compound in enumerate(library):
            try:
                score = self._calculate_pharmacophore_score(compound)

                if score >= threshold:
                    smiles = (
                        compound
                        if isinstance(compound, str)
                        else Chem.MolToSmiles(compound)
                    )
                    results.append(
                        {
                            "compound_id": f"compound_{i}",
                            "smiles": smiles,
                            "score": score,
                            "screening_method": "pharmacophore",
                        }
                    )

            except Exception as e:
                logging.warning(f"Error processing compound {i}: {e}")
                continue

        # Sort by score and limit results
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:max_compounds]

        return pd.DataFrame(results)

    def _calculate_pharmacophore_score(self, compound: Union[str, "Chem.Mol"]) -> float:
        """
        Calculate pharmacophore matching score.

        This is a simplified implementation that counts basic molecular features.
        A full implementation would use sophisticated pharmacophore matching algorithms.
        """
        if not RDKIT_AVAILABLE:
            # Simple fallback based on molecular weight
            smiles = compound if isinstance(compound, str) else str(compound)
            return min(1.0, len(smiles) / 50.0)

        try:
            if isinstance(compound, str):
                mol = Chem.MolFromSmiles(compound)
            else:
                mol = compound

            if mol is None:
                return 0.0

            score = 0.0
            feature_count = 0

            # Count aromatic rings
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            if aromatic_rings > 0:
                score += 0.3
                feature_count += 1

            # Count hydrogen bond donors
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            if hbd > 0:
                score += 0.3
                feature_count += 1

            # Count hydrogen bond acceptors
            hba = rdMolDescriptors.CalcNumHBA(mol)
            if hba > 0:
                score += 0.3
                feature_count += 1

            # Add bonus for having multiple features
            if feature_count >= 2:
                score += 0.1

            return min(1.0, score)

        except Exception as e:
            logging.warning(f"Error calculating pharmacophore score: {e}")
            return 0.0


# Module-level convenience functions


def perform_virtual_screening(
    reference_smiles: List[str],
    library_smiles: List[str],
    method: str = "similarity",
    threshold: float = 0.7,
    max_hits: int = 100,
) -> Dict:
    """
    Convenience function to perform virtual screening.

    Args:
        reference_smiles: Reference compound SMILES
        library_smiles: Library compound SMILES to screen
        method: Screening method ('similarity' or 'pharmacophore')
        threshold: Similarity/score threshold
        max_hits: Maximum number of hits to return

    Returns:
        Dictionary with screening results and statistics
    """
    try:
        # Initialize screener
        screener = VirtualScreener(screening_method=method)
        screener.set_reference_compounds(reference_smiles)

        # Perform screening
        results = screener.screen_library(library_smiles, threshold, max_hits)

        # Get statistics
        stats = {
            "total_screened": len(library_smiles),
            "hits_found": len(results),
            "hit_rate": len(results) / len(library_smiles) if library_smiles else 0,
            "method": method,
            "threshold": threshold,
        }

        # Get top hits
        top_hits = screener.get_top_hits(min(10, len(results)))

        return {
            "results": results,
            "top_hits": top_hits,
            "statistics": stats,
            "screener": screener,
        }

    except Exception as e:
        logging.error(f"Virtual screening failed: {e}")
        return {
            "results": pd.DataFrame(),
            "top_hits": pd.DataFrame(),
            "statistics": {"error": str(e)},
            "screener": None,
        }


def calculate_screening_metrics(
    results: pd.DataFrame, known_actives: List[str]
) -> Dict[str, float]:
    """
    Calculate screening performance metrics.

    Args:
        results: Screening results DataFrame
        known_actives: List of known active compound SMILES

    Returns:
        Dictionary with performance metrics
    """
    if results.empty:
        return {
            "enrichment_factor": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    # Calculate metrics
    hits = set(results["smiles"].tolist())
    actives = set(known_actives)

    true_positives = len(hits.intersection(actives))
    false_positives = len(hits - actives)
    false_negatives = len(actives - hits)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Calculate enrichment factor (top 10%)
    n_top = max(1, len(results) // 10)
    top_hits = set(results.nlargest(n_top, "score")["smiles"].tolist())
    actives_in_top = len(top_hits.intersection(actives))
    expected_actives = len(actives) * 0.1
    enrichment_factor = actives_in_top / expected_actives if expected_actives > 0 else 0

    return {
        "enrichment_factor": enrichment_factor,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
