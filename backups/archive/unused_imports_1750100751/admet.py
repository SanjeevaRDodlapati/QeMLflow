"""
ChemML Drug Discovery - ADMET Prediction
=======================================

ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) prediction tools
for drug discovery workflows.

This module provides comprehensive tools for:
- ADMET property prediction
- Drug-likeness assessment
- Toxicity prediction
- Rule-based filtering
"""

import logging
import warnings
import numpy as np
import pandas as pd
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    RDKIT_AVAILABLE = True
except ImportError:
    logging.warning("RDKit not available. ADMET prediction will be limited.")
    RDKIT_AVAILABLE = False

try:
    from sklearn.metrics import accuracy_score, mean_squared_error

    SKLEARN_AVAILABLE = True
except ImportError:
    logging.warning("Scikit-learn not available. Evaluation metrics will be limited.")
    SKLEARN_AVAILABLE = False


class ADMETPredictor:
    """
    Predict ADMET properties using simple rule-based and ML approaches.
    """

    def __init__(self) -> None:
        self.models = {}
        self.scalers = {}

    def predict_admet_properties(self, smiles_list: List[str]) -> pd.DataFrame:
        """
        Predict ADMET properties for a list of SMILES.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            DataFrame with ADMET predictions
        """
        results = []

        for smiles in smiles_list:
            admet_props = {
                "SMILES": smiles,
                "absorption": self.predict_absorption(smiles),
                "bioavailability": self.predict_bioavailability(smiles),
                "bbb_permeability": self.predict_bbb_permeability(smiles),
                "cyp_inhibition": self.predict_cyp_inhibition(smiles),
                "hepatotoxicity": self.predict_hepatotoxicity(smiles),
                "mutagenicity": self.predict_mutagenicity(smiles),
                "drug_likeness": self.calculate_drug_likeness_score(smiles),
            }
            results.append(admet_props)

        return pd.DataFrame(results)

    def predict_absorption(self, smiles: str) -> float:
        """Predict absorption based on molecular properties."""
        if not RDKIT_AVAILABLE:
            return 0.5  # Default value

        if smiles is None or smiles == "":
            return 0.0

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        # Rule-based prediction using molecular properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        # Simple scoring based on known absorption rules
        score = 1.0

        # Molecular weight penalty
        if mw > 500:
            score *= 0.5
        elif mw > 700:
            score *= 0.2

        # LogP penalty
        if logp < -2 or logp > 5:
            score *= 0.7

        # TPSA penalty
        if tpsa > 140:
            score *= 0.6

        return min(score, 1.0)

    def predict_bioavailability(self, smiles: str) -> float:
        """Predict oral bioavailability."""
        if not RDKIT_AVAILABLE:
            return 0.5

        if smiles is None or smiles == "":
            return 0.0

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        # Lipinski's Rule of Five compliance
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1

        # Convert violations to bioavailability score
        bioavailability = max(0.0, 1.0 - violations * 0.25)
        return bioavailability

    def predict_bbb_permeability(self, smiles: str) -> float:
        """Predict blood-brain barrier permeability."""
        if not RDKIT_AVAILABLE:
            return 0.3

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        # Simple rule-based BBB prediction
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        # Rules for BBB permeability
        if mw > 450 or tpsa > 90:
            return 0.1  # Low permeability

        if 1 <= logp <= 3 and mw < 400 and tpsa < 70:
            return 0.8  # Good permeability

        return 0.4  # Moderate permeability

    def predict_cyp_inhibition(self, smiles: str) -> Dict[str, float]:
        """Predict CYP enzyme inhibition."""
        if not RDKIT_AVAILABLE:
            return {
                "CYP1A2": 0.2,
                "CYP2C9": 0.2,
                "CYP2C19": 0.2,
                "CYP2D6": 0.2,
                "CYP3A4": 0.2,
            }

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                cyp: 0.0 for cyp in ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]
            }

        # Simple rule-based CYP inhibition prediction
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)

        # Base inhibition probability
        base_prob = 0.1

        # Increase probability for larger, more lipophilic molecules
        if mw > 400:
            base_prob += 0.2
        if logp > 3:
            base_prob += 0.2

        return {
            "CYP1A2": min(base_prob + 0.1, 0.8),
            "CYP2C9": min(base_prob, 0.8),
            "CYP2C19": min(base_prob, 0.8),
            "CYP2D6": min(base_prob + 0.05, 0.8),
            "CYP3A4": min(base_prob + 0.15, 0.8),  # Most promiscuous
        }

    def predict_hepatotoxicity(self, smiles: str) -> float:
        """Predict hepatotoxicity risk."""
        if not RDKIT_AVAILABLE:
            return 0.3

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.5

        # Rule-based hepatotoxicity prediction
        risk_score = 0.1  # Base risk

        # Check for toxic substructures (simplified)
        smiles_lower = smiles.lower()
        toxic_patterns = ["[nH+]", "n(=o)=o", "c(=o)c", "cc(c)c"]

        for pattern in toxic_patterns:
            if pattern in smiles_lower:
                risk_score += 0.2

        # Molecular weight factor
        mw = Descriptors.MolWt(mol)
        if mw > 600:
            risk_score += 0.2

        return min(risk_score, 0.9)

    def predict_mutagenicity(self, smiles: str) -> float:
        """Predict mutagenicity using Ames test surrogates."""
        if not RDKIT_AVAILABLE:
            return 0.2

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.5

        # Rule-based mutagenicity prediction
        risk_score = 0.05  # Base risk

        # Check for mutagenic substructures (simplified)
        mutagenic_patterns = [
            "n=c=s",  # Isothiocyanate
            "n(=o)=o",  # Nitro
            "nn",  # Azo
            "c=c-c=c",  # Conjugated system
        ]

        smiles_lower = smiles.lower()
        for pattern in mutagenic_patterns:
            if pattern in smiles_lower:
                risk_score += 0.3

        # Aromatic ring factor
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        if aromatic_rings > 2:
            risk_score += 0.1

        return min(risk_score, 0.8)

    def calculate_drug_likeness_score(self, smiles: str) -> float:
        """Calculate overall drug-likeness score."""
        if not RDKIT_AVAILABLE:
            return 0.5

        if smiles is None or smiles == "":
            return 0.0

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        # Combined score based on multiple factors
        absorption = self.predict_absorption(smiles)
        bioavailability = self.predict_bioavailability(smiles)
        bbb_perm = self.predict_bbb_permeability(smiles)
        hepatotox = self.predict_hepatotoxicity(smiles)
        mutagenic = self.predict_mutagenicity(smiles)

        # Weighted combination
        drug_likeness = (
            absorption * 0.3
            + bioavailability * 0.3
            + bbb_perm * 0.1
            + (1 - hepatotox) * 0.2
            + (1 - mutagenic)  # Lower hepatotoxicity is better
            * 0.1  # Lower mutagenicity is better
        )

        return min(drug_likeness, 1.0)


class DrugLikenessAssessor:
    """
    Assess drug-likeness of molecules using various filters and metrics.
    """

    def __init__(self):
        self.filters = {
            "lipinski": True,
            "ghose": True,
            "veber": True,
            "egan": True,
            "muegge": True,
        }

    def assess_drug_likeness(self, smiles: str) -> Dict[str, Union[bool, float]]:
        """
        Assess drug-likeness using multiple filters.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary with filter results and overall score
        """
        if not RDKIT_AVAILABLE:
            return {
                "lipinski_violations": 0,
                "ghose_violations": 0,
                "veber_violations": 0,
                "egan_violations": 0,
                "muegge_violations": 0,
                "overall_score": 0.5,
            }

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._get_default_drug_likeness_result(failed=True)

            # Calculate descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)

            # Apply filters
            lipinski_violations = self._assess_lipinski_filter(mw, logp, hbd, hba)
            ghose_violations = self._assess_ghose_filter(mw, logp, tpsa)
            veber_violations = self._assess_veber_filter(rotatable_bonds, tpsa)
            egan_violations = self._assess_egan_filter(logp, tpsa)
            muegge_violations = self._assess_muegge_filter(
                mw, logp, tpsa, rotatable_bonds
            )

            # Calculate overall score
            total_violations = (
                lipinski_violations
                + ghose_violations
                + veber_violations
                + egan_violations
                + muegge_violations
            )
            max_violations = 17  # Maximum possible violations
            overall_score = max(0.0, 1.0 - (total_violations / max_violations))

            return {
                "lipinski_violations": lipinski_violations,
                "ghose_violations": ghose_violations,
                "veber_violations": veber_violations,
                "egan_violations": egan_violations,
                "muegge_violations": muegge_violations,
                "overall_score": overall_score,
            }

        except Exception as e:
            logging.warning(f"Error assessing drug-likeness for SMILES {smiles}: {e}")
            return self._get_default_drug_likeness_result(failed=True)

    def _assess_lipinski_filter(
        self, mw: float, logp: float, hbd: int, hba: int
    ) -> int:
        """Assess Lipinski's Rule of Five violations."""
        violations = 0
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1
        return violations

    def _assess_ghose_filter(self, mw: float, logp: float, tpsa: float) -> int:
        """Assess Ghose filter violations."""
        violations = 0
        if mw < 160 or mw > 480:
            violations += 1
        if logp < -0.4 or logp > 5.6:
            violations += 1
        if tpsa > 140:
            violations += 1
        return violations

    def _assess_veber_filter(self, rotatable_bonds: int, tpsa: float) -> int:
        """Assess Veber filter violations."""
        violations = 0
        if rotatable_bonds > 10:
            violations += 1
        if tpsa > 140:
            violations += 1
        return violations

    def _assess_egan_filter(self, logp: float, tpsa: float) -> int:
        """Assess Egan filter violations."""
        violations = 0
        if logp < -1 or logp > 6:
            violations += 1
        if tpsa > 150:
            violations += 1
        return violations

    def _assess_muegge_filter(
        self, mw: float, logp: float, tpsa: float, rotatable_bonds: int
    ) -> int:
        """Assess Muegge filter violations."""
        violations = 0
        if mw < 200 or mw > 600:
            violations += 1
        if logp < -2 or logp > 5:
            violations += 1
        if tpsa > 150:
            violations += 1
        if rotatable_bonds > 15:
            violations += 1
        return violations

    def _get_default_drug_likeness_result(
        self, failed: bool = False
    ) -> Dict[str, Union[bool, float]]:
        """Get default drug-likeness result for error cases."""
        if failed:
            return {
                "lipinski_violations": 4,
                "ghose_violations": 4,
                "veber_violations": 2,
                "egan_violations": 2,
                "muegge_violations": 4,
                "overall_score": 0.0,
            }
        else:
            return {
                "lipinski_violations": 0,
                "ghose_violations": 0,
                "veber_violations": 0,
                "egan_violations": 0,
                "muegge_violations": 0,
                "overall_score": 0.5,
            }


class ToxicityPredictor:
    """
    Predict various toxicity endpoints for molecules.
    """

    def __init__(self):
        self.endpoints = [
            "mutagenicity",
            "carcinogenicity",
            "acute_toxicity",
            "skin_sensitization",
            "eye_irritation",
        ]

    def predict_toxicity(
        self, smiles: Union[str, List[str]]
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Predict multiple toxicity endpoints.

        Args:
            smiles: SMILES string or list of SMILES strings

        Returns:
            Dictionary with toxicity predictions or list of dictionaries
        """
        if isinstance(smiles, list):
            # Handle multiple SMILES
            return [self._predict_single_toxicity(s) for s in smiles]
        else:
            # Handle single SMILES
            return self._predict_single_toxicity(smiles)

    def _predict_single_toxicity(self, smiles: str) -> Dict[str, float]:
        """Predict toxicity for a single SMILES string."""
        if not RDKIT_AVAILABLE:
            return {endpoint: 0.5 for endpoint in self.endpoints}

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {
                    endpoint: 1.0 for endpoint in self.endpoints
                }  # High toxicity for invalid SMILES

            # Simple rule-based toxicity prediction
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)

            # Basic toxicity risk assessment
            base_risk = 0.1

            # Higher risk for larger, more lipophilic molecules
            if mw > 800:
                base_risk += 0.3
            elif mw > 600:
                base_risk += 0.2
            elif mw > 400:
                base_risk += 0.1

            if logp > 6:
                base_risk += 0.3
            elif logp > 4:
                base_risk += 0.2
            elif logp > 2:
                base_risk += 0.1

            # Check for known toxic substructures (simplified)
            smiles_lower = smiles.lower()
            toxic_patterns = ["nitro", "azide", "halogen", "aldehyde"]
            for pattern in toxic_patterns:
                if pattern in smiles_lower:
                    base_risk += 0.2
                    break

            toxicity_predictions = {}
            for endpoint in self.endpoints:
                # Add some variation for different endpoints based on endpoint type
                endpoint_modifier = {
                    "mutagenicity": 0.0,
                    "carcinogenicity": 0.1,
                    "acute_toxicity": -0.1,
                    "skin_sensitization": 0.05,
                    "eye_irritation": 0.05,
                }.get(endpoint, 0.0)

                prediction = min(1.0, max(0.0, base_risk + endpoint_modifier))
                toxicity_predictions[endpoint] = prediction

            return toxicity_predictions

        except Exception as e:
            logging.warning(f"Error predicting toxicity for SMILES {smiles}: {e}")
            return {endpoint: 0.5 for endpoint in self.endpoints}


# Module-level convenience functions


def predict_admet_profile(molecules: Union[str, List[str]]) -> pd.DataFrame:
    """
    Convenience function to predict ADMET profile for molecules.

    Args:
        molecules: Single SMILES string or list of SMILES

    Returns:
        DataFrame with ADMET predictions
    """
    if isinstance(molecules, str):
        molecules = [molecules]

    predictor = ADMETPredictor()
    return predictor.predict_admet_properties(molecules)


def evaluate_admet_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, prediction_type: str = "classification"
) -> Dict[str, float]:
    """
    Evaluate ADMET prediction performance.

    Args:
        y_true: True values
        y_pred: Predicted values
        prediction_type: 'classification' or 'regression'

    Returns:
        Dictionary of evaluation metrics
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "sklearn not available"}

    if prediction_type == "classification":
        accuracy = accuracy_score(y_true, y_pred > 0.5)
        return {"accuracy": accuracy}
    else:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return {"mse": mse, "rmse": rmse}


def apply_admet_filters(
    molecules_df: pd.DataFrame, strict: bool = False
) -> pd.DataFrame:
    """
    Apply ADMET-based filters to remove problematic molecules.

    Args:
        molecules_df: DataFrame with SMILES column
        strict: Whether to apply strict filtering criteria

    Returns:
        Filtered DataFrame
    """
    if "SMILES" not in molecules_df.columns:
        return molecules_df

    predictor = ADMETPredictor()
    admet_props = predictor.predict_admet_properties(molecules_df["SMILES"].tolist())

    # Merge ADMET properties
    if len(admet_props) > 0:
        result_df = pd.concat(
            [molecules_df.reset_index(drop=True), admet_props.drop("SMILES", axis=1)],
            axis=1,
        )
    else:
        return molecules_df

    # Apply filters
    filters = result_df["bioavailability"] > 0.5
    filters &= result_df["hepatotoxicity"] < 0.7
    filters &= result_df["mutagenicity"] < 0.6

    if strict:
        filters &= result_df["absorption"] > 0.6
        filters &= result_df["drug_likeness"] > 0.6

    return result_df[filters].reset_index(drop=True)


def predict_admet_properties(
    molecules: Union[str, List[str]],
) -> Union[Dict, List[Dict]]:
    """
    Standalone function to predict ADMET properties.

    Args:
        molecules: Single SMILES string or list of SMILES

    Returns:
        Dictionary or list of dictionaries with ADMET predictions
    """
    if isinstance(molecules, str):
        # Single molecule
        predictor = ADMETPredictor()
        result_df = predictor.predict_admet_properties([molecules])
        result_dict = result_df.iloc[0].to_dict()
        result_dict.pop("SMILES", None)  # Remove SMILES from result
        return result_dict
    elif isinstance(molecules, list):
        if len(molecules) == 0:
            return []
        # Multiple molecules
        predictor = ADMETPredictor()
        result_df = predictor.predict_admet_properties(molecules)
        results = []
        for i, row in result_df.iterrows():
            result_dict = row.to_dict()
            result_dict.pop("SMILES", None)  # Remove SMILES from result
            results.append(result_dict)
        return results
    else:
        raise ValueError("Input must be a string or list of strings")


def assess_drug_likeness(molecules: Union[str, List[str]]) -> pd.DataFrame:
    """
    Standalone function to assess drug-likeness of molecules.

    Args:
        molecules: Single SMILES string or list of SMILES

    Returns:
        DataFrame with drug-likeness assessment
    """
    if isinstance(molecules, str):
        molecules = [molecules]

    if len(molecules) == 0:
        return pd.DataFrame()

    assessor = DrugLikenessAssessor()
    results = []

    for smiles in molecules:
        assessment = assessor.assess_drug_likeness(smiles)
        result = {"SMILES": smiles}
        result.update(assessment)
        # Rename 'overall_score' to 'drug_like_score' for consistency
        if "overall_score" in result:
            result["drug_like_score"] = result.pop("overall_score")
        results.append(result)

    return pd.DataFrame(results)
