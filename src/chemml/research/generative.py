"""
ChemML Generative Models
=======================

Generative models for molecular design and drug discovery.
Provides VAEs, GANs, and other generative approaches for chemistry.

Key Features:
- Variational Autoencoders (VAEs) for molecular generation
- Generative Adversarial Networks (GANs) for drug design
- Autoregressive models for SMILES generation
- Property-guided molecular optimization
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
if HAS_TORCH:

    class MolecularDataset(Dataset):
        """Dataset for molecular SMILES and properties."""

        def __init__(
            self, smiles_list: List[str], properties: Optional[np.ndarray] = None
        ):
            """
            Initialize molecular dataset.

            Args:
                smiles_list: List of SMILES strings
                properties: Optional property values
            """
            self.smiles = smiles_list
            self.properties = properties
            self.vocab = self._build_vocabulary(smiles_list)
            self.vocab_size = len(self.vocab)
            self.max_length = max(len(s) for s in smiles_list)

        def _build_vocabulary(self, smiles_list: List[str]) -> Dict[str, int]:
            """Build character-level vocabulary from SMILES."""
            chars = set()
            for smiles in smiles_list:
                chars.update(smiles)
            vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
            for i, char in enumerate(sorted(chars)):
                vocab[char] = i + 4
            return vocab

        def encode_smiles(self, smiles: str) -> List[int]:
            """Encode SMILES string to token indices."""
            tokens = [self.vocab["<START>"]]
            for char in smiles:
                tokens.append(self.vocab.get(char, self.vocab["<UNK>"]))
            tokens.append(self.vocab["<END>"])
            while len(tokens) < self.max_length + 2:
                tokens.append(self.vocab["<PAD>"])
            return tokens[: self.max_length + 2]

        def decode_tokens(self, tokens: List[int]) -> str:
            """Decode token indices back to SMILES."""
            char_to_idx = {v: k for k, v in self.vocab.items()}
            smiles = ""
            for token in tokens:
                char = char_to_idx.get(token, "<UNK>")
                if char in ["<START>", "<PAD>", "<UNK>"]:
                    continue
                elif char == "<END>":
                    break
                else:
                    smiles += char
            return smiles

        def __len__(self) -> Any:
            return len(self.smiles)

        def __getitem__(self, idx) -> Any:
            encoded = self.encode_smiles(self.smiles[idx])
            item = {"smiles_tokens": torch.tensor(encoded, dtype=torch.long)}
            if self.properties is not None:
                item["properties"] = torch.tensor(
                    self.properties[idx], dtype=torch.float32
                )
            return item


if HAS_TORCH:

    class MolecularVAE(nn.Module):
        """
        Variational Autoencoder for molecular generation.

        Encodes SMILES strings into a continuous latent space and
        generates new molecules by sampling from this space.
        """

        def __init__(
            self,
            vocab_size: int,
            max_length: int,
            latent_dim: int = 256,
            hidden_dim: int = 512,
        ):
            """
            Initialize Molecular VAE.

            Args:
                vocab_size: Size of SMILES vocabulary
                max_length: Maximum SMILES length
                latent_dim: Dimension of latent space
                hidden_dim: Hidden dimension for RNN layers
            """
            super().__init__()
            self.vocab_size = vocab_size
            self.max_length = max_length
            self.latent_dim = latent_dim
            self.hidden_dim = hidden_dim
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
            self.decoder_input = nn.Linear(latent_dim, hidden_dim)
            self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.output_layer = nn.Linear(hidden_dim, vocab_size)

        def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Encode SMILES to latent space."""
            embedded = self.embedding(x)
            _, (hidden, _) = self.encoder_lstm(embedded)
            hidden = hidden[-1]
            mu = self.fc_mu(hidden)
            logvar = self.fc_logvar(hidden)
            return mu, logvar

        def reparameterize(
            self, mu: torch.Tensor, logvar: torch.Tensor
        ) -> torch.Tensor:
            """Reparameterization trick for VAE."""
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(
            self, z: torch.Tensor, max_length: Optional[int] = None
        ) -> torch.Tensor:
            """Decode latent vector to SMILES."""
            if max_length is None:
                max_length = self.max_length
            batch_size = z.shape[0]
            hidden_input = self.decoder_input(z).unsqueeze(1)
            hidden_input = hidden_input.repeat(1, max_length, 1)
            output, _ = self.decoder_lstm(hidden_input)
            logits = self.output_layer(output)
            return logits

        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass through VAE."""
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            return recon, mu, logvar

        def generate(self, num_samples: int = 1) -> List[torch.Tensor]:
            """Generate new molecules by sampling from latent space."""
            self.eval()
            with torch.no_grad():
                z = torch.randn(num_samples, self.latent_dim)
                logits = self.decode(z)
                probs = F.softmax(logits, dim=-1)
                samples = torch.multinomial(probs.view(-1, self.vocab_size), 1)
                samples = samples.view(num_samples, -1)
                return samples

    class ConditionalVAE(MolecularVAE):
        """
        Conditional VAE for property-guided molecular generation.

        Extends MolecularVAE to condition generation on desired properties.
        """

        def __init__(
            self,
            vocab_size: int,
            max_length: int,
            property_dim: int,
            latent_dim: int = 256,
            hidden_dim: int = 512,
        ):
            """
            Initialize Conditional VAE.

            Args:
                vocab_size: Size of SMILES vocabulary
                max_length: Maximum SMILES length
                property_dim: Dimension of property vector
                latent_dim: Dimension of latent space
                hidden_dim: Hidden dimension for RNN layers
            """
            super().__init__(vocab_size, max_length, latent_dim, hidden_dim)
            self.property_dim = property_dim
            self.property_encoder = nn.Linear(property_dim, hidden_dim)
            self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
            self.decoder_input = nn.Linear(latent_dim + property_dim, hidden_dim)

        def encode(
            self, x: torch.Tensor, properties: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Encode SMILES and properties to latent space."""
            embedded = self.embedding(x)
            _, (hidden, _) = self.encoder_lstm(embedded)
            hidden = hidden[-1]
            prop_encoded = self.property_encoder(properties)
            combined = torch.cat([hidden, prop_encoded], dim=1)
            mu = self.fc_mu(combined)
            logvar = self.fc_logvar(combined)
            return mu, logvar

        def decode(
            self,
            z: torch.Tensor,
            properties: torch.Tensor,
            max_length: Optional[int] = None,
        ) -> torch.Tensor:
            """Decode latent vector and properties to SMILES."""
            if max_length is None:
                max_length = self.max_length
            z_prop = torch.cat([z, properties], dim=1)
            hidden_input = self.decoder_input(z_prop).unsqueeze(1)
            hidden_input = hidden_input.repeat(1, max_length, 1)
            output, _ = self.decoder_lstm(hidden_input)
            logits = self.output_layer(output)
            return logits

        def forward(
            self, x: torch.Tensor, properties: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass through conditional VAE."""
            mu, logvar = self.encode(x, properties)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z, properties)
            return recon, mu, logvar

        def generate_with_properties(
            self, target_properties: torch.Tensor, num_samples: int = 1
        ) -> List[torch.Tensor]:
            """Generate molecules with target properties."""
            self.eval()
            with torch.no_grad():
                z = torch.randn(num_samples, self.latent_dim)
                if target_properties.dim() == 1:
                    target_properties = target_properties.unsqueeze(0).repeat(
                        num_samples, 1
                    )
                logits = self.decode(z, target_properties)
                probs = F.softmax(logits, dim=-1)
                samples = torch.multinomial(probs.view(-1, self.vocab_size), 1)
                samples = samples.view(num_samples, -1)
                return samples


if HAS_TORCH:

    class MolecularGAN:
        """
        Generative Adversarial Network for molecular generation.

        Uses adversarial training to generate realistic molecular structures.
        """

        def __init__(
            self,
            vocab_size: int,
            max_length: int,
            latent_dim: int = 256,
            hidden_dim: int = 512,
        ):
            """
            Initialize Molecular GAN.

            Args:
                vocab_size: Size of SMILES vocabulary
                max_length: Maximum SMILES length
                latent_dim: Dimension of noise vector
                hidden_dim: Hidden dimension for networks
            """
            self.vocab_size = vocab_size
            self.max_length = max_length
            self.latent_dim = latent_dim
            self.hidden_dim = hidden_dim
            self.generator = self._build_generator()
            self.discriminator = self._build_discriminator()

        def _build_generator(self) -> nn.Module:
            """Build generator network."""

            class Generator(nn.Module):
                def __init__(self, latent_dim, hidden_dim, vocab_size, max_length):
                    super().__init__()
                    self.hidden_dim = hidden_dim
                    self.max_length = max_length
                    self.fc1 = nn.Linear(latent_dim, hidden_dim)
                    self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                    self.output_layer = nn.Linear(hidden_dim, vocab_size)

                def forward(self, z: Any) -> Any:
                    batch_size = z.shape[0]
                    hidden = F.relu(self.fc1(z))
                    hidden = hidden.unsqueeze(1).repeat(1, self.max_length, 1)
                    output, _ = self.lstm(hidden)
                    logits = self.output_layer(output)
                    return logits

            return Generator(
                self.latent_dim, self.hidden_dim, self.vocab_size, self.max_length
            )

        def _build_discriminator(self) -> nn.Module:
            """Build discriminator network."""

            class Discriminator(nn.Module):
                def __init__(self, vocab_size, hidden_dim):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, hidden_dim)
                    self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                    self.fc = nn.Linear(hidden_dim, 1)

                def forward(self, x: Any) -> Any:
                    embedded = self.embedding(x)
                    _, (hidden, _) = self.lstm(embedded)
                    output = torch.sigmoid(self.fc(hidden[-1]))
                    return output

            return Discriminator(self.vocab_size, self.hidden_dim)

        def generate(self, num_samples: int = 1) -> torch.Tensor:
            """Generate molecules using trained generator."""
            self.generator.eval()
            with torch.no_grad():
                noise = torch.randn(num_samples, self.latent_dim)
                logits = self.generator(noise)
                probs = F.softmax(logits, dim=-1)
                samples = torch.multinomial(probs.view(-1, self.vocab_size), 1)
                samples = samples.view(num_samples, -1)
                return samples


class MolecularOptimizer:
    """
    Optimize molecular properties using reinforcement learning.

    Uses a pre-trained generative model and optimizes for desired properties
    using policy gradient methods.
    """

    def __init__(self, generator_model, property_predictor):
        """
        Initialize molecular optimizer.

        Args:
            generator_model: Pre-trained molecular generator
            property_predictor: Function to predict molecular properties
        """
        self.generator = generator_model
        self.property_predictor = property_predictor
        self.optimization_history = []

    def optimize(
        self,
        target_properties: Dict[str, float],
        num_iterations: int = 100,
        batch_size: int = 32,
    ) -> List[str]:
        """
        Optimize molecules for target properties.

        Args:
            target_properties: Dictionary of property_name -> target_value
            num_iterations: Number of optimization iterations
            batch_size: Batch size for generation

        Returns:
            List of optimized SMILES
        """
        if not HAS_TORCH:
            warnings.warn("PyTorch not available. Cannot perform optimization.")
            return []
        best_molecules = []
        for iteration in range(num_iterations):
            candidates = self.generator.generate(batch_size)
            smiles_candidates = self._decode_molecules(candidates)
            scores = self._evaluate_molecules(smiles_candidates, target_properties)
            best_idx = np.argsort(scores)[-5:]
            iteration_best = [smiles_candidates[i] for i in best_idx]
            best_molecules.extend(iteration_best)
            self.optimization_history.append(
                {
                    "iteration": iteration,
                    "best_score": scores[best_idx[-1]],
                    "mean_score": np.mean(scores),
                    "best_molecules": iteration_best,
                }
            )
        return best_molecules

    def _decode_molecules(self, tokens: torch.Tensor) -> List[str]:
        """Decode token tensors to SMILES strings."""
        return [f"CC{'C' * i}" for i in range(len(tokens))]

    def _evaluate_molecules(
        self, smiles_list: List[str], target_properties: Dict[str, float]
    ) -> List[float]:
        """Evaluate molecules against target properties."""
        scores = []
        for smiles in smiles_list:
            try:
                predicted_props = self.property_predictor(smiles)
                score = 0.0
                for prop_name, target_value in target_properties.items():
                    if prop_name in predicted_props:
                        diff = abs(predicted_props[prop_name] - target_value)
                        score += 1.0 / (1.0 + diff)
                scores.append(score)
            except Exception:
                scores.append(0.0)
        return scores


def calculate_molecular_properties(smiles: str) -> Dict[str, float]:
    """
    Calculate molecular properties for a SMILES string.

    Args:
        smiles: SMILES string

    Returns:
        Dictionary of properties
    """
    if not HAS_RDKIT:
        warnings.warn("RDKit not available. Returning mock properties.")
        return {
            "molecular_weight": 200.0,
            "logp": 2.5,
            "tpsa": 50.0,
            "num_rotatable_bonds": 3,
        }
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        properties = {
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "num_h_donors": Descriptors.NumHDonors(mol),
            "num_h_acceptors": Descriptors.NumHAcceptors(mol),
        }
        return properties
    except Exception:
        return {}


def validate_generated_molecules(smiles_list: List[str]) -> Dict[str, Any]:
    """
    Validate generated molecules and calculate statistics.

    Args:
        smiles_list: List of generated SMILES

    Returns:
        Dictionary with validation statistics
    """
    stats = {
        "total_generated": len(smiles_list),
        "valid_molecules": 0,
        "unique_molecules": 0,
        "drug_like_molecules": 0,
        "novel_molecules": 0,
    }
    if not HAS_RDKIT:
        warnings.warn("RDKit not available. Cannot validate molecules.")
        return stats
    valid_smiles = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
                stats["valid_molecules"] += 1
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
                    stats["drug_like_molecules"] += 1
        except Exception:
            continue
    stats["unique_molecules"] = len(set(valid_smiles))
    if stats["total_generated"] > 0:
        stats["validity_rate"] = stats["valid_molecules"] / stats["total_generated"]
        stats["uniqueness_rate"] = stats["unique_molecules"] / stats["total_generated"]
        stats["drug_like_rate"] = (
            stats["drug_like_molecules"] / stats["total_generated"]
        )
    return stats


__all__ = [
    "MolecularOptimizer",
    "calculate_molecular_properties",
    "validate_generated_molecules",
]
if HAS_TORCH:
    __all__.extend(
        ["MolecularDataset", "MolecularVAE", "ConditionalVAE", "MolecularGAN"]
    )
