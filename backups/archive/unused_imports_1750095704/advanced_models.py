"""
QeMLflow Advanced Models
=====================

Provides cutting-edge architectures and novel approaches for molecular ML.

Key Features:
- Graph Neural Networks for molecular representation
- Transformer models for SMILES and molecular sequences
- Meta-learning approaches for few-shot property prediction
- Multi-task learning architectures
- Attention mechanisms for molecular interpretation
"""

import warnings

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

    HAS_TORCH_GEOMETRIC = True
    HAS_TORCH = True
except ImportError:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        HAS_TORCH = True
        HAS_TORCH_GEOMETRIC = False
    except ImportError:
        HAS_TORCH = False
        HAS_TORCH_GEOMETRIC = False
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def smiles_to_graph(smiles: str) -> Optional[Dict]:
    """
    Convert SMILES string to graph representation.

    Args:
        smiles: SMILES string

    Returns:
        Dictionary with node features, edge indices, and edge features
    """
    if not HAS_RDKIT:
        warnings.warn("RDKit not available. Cannot create molecular graphs.")
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetNumRadicalElectrons(),
            ]
            node_features.append(features)
        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])
            bond_features = [
                int(bond.GetBondType()),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
            ]
            edge_features.extend([bond_features, bond_features])
        graph_data = {
            "node_features": np.array(node_features),
            "edge_indices": (
                np.array(edge_indices).T if edge_indices else np.empty((2, 0))
            ),
            "edge_features": (
                np.array(edge_features) if edge_features else np.empty((0, 3))
            ),
            "num_nodes": len(node_features),
        }
        return graph_data
    except Exception as e:
        warnings.warn(f"Error creating graph from SMILES: {e}")
        return None


if HAS_TORCH and HAS_TORCH_GEOMETRIC:

    class MolecularGCN(nn.Module):
        """
        Graph Convolutional Network for molecular property prediction.

        Uses graph convolutions to learn from molecular structure.
        """

        def __init__(
            self,
            node_features: int,
            hidden_dim: int = 128,
            num_layers: int = 3,
            output_dim: int = 1,
            dropout: float = 0.2,
        ):
            """
            Initialize Molecular GCN.

            Args:
                node_features: Number of node feature dimensions
                hidden_dim: Hidden dimension size
                num_layers: Number of GCN layers
                output_dim: Output dimension
                dropout: Dropout probability
            """
            super().__init__()
            self.num_layers = num_layers
            self.dropout = dropout
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(node_features, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, data) -> Any:
            """Forward pass through GCN."""
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            x = global_mean_pool(x, batch)
            output = self.classifier(x)
            return output

    class MolecularGAT(nn.Module):
        """
        Graph Attention Network for molecular property prediction.

        Uses attention mechanisms to focus on important molecular substructures.
        """

        def __init__(
            self,
            node_features: int,
            hidden_dim: int = 128,
            num_heads: int = 4,
            num_layers: int = 3,
            output_dim: int = 1,
            dropout: float = 0.2,
        ):
            """
            Initialize Molecular GAT.

            Args:
                node_features: Number of node feature dimensions
                hidden_dim: Hidden dimension size
                num_heads: Number of attention heads
                num_layers: Number of GAT layers
                output_dim: Output dimension
                dropout: Dropout probability
            """
            super().__init__()
            self.num_layers = num_layers
            self.dropout = dropout
            self.convs = nn.ModuleList()
            self.convs.append(
                GATConv(node_features, hidden_dim, heads=num_heads, dropout=dropout)
            )
            for _ in range(num_layers - 1):
                self.convs.append(
                    GATConv(
                        hidden_dim * num_heads,
                        hidden_dim,
                        heads=num_heads,
                        dropout=dropout,
                    )
                )
            final_dim = hidden_dim * num_heads
            self.classifier = nn.Sequential(
                nn.Linear(final_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, data):
            """Forward pass through GAT."""
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = F.elu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
            x = global_mean_pool(x, batch)
            output = self.classifier(x)
            return output


if HAS_TORCH:

    class MolecularTransformer(nn.Module):
        """
        Transformer model for SMILES sequence processing.

        Uses self-attention to capture long-range dependencies in molecular sequences.
        """

        def __init__(
            self,
            vocab_size: int,
            d_model: int = 256,
            num_heads: int = 8,
            num_layers: int = 6,
            max_length: int = 512,
            output_dim: int = 1,
            dropout: float = 0.1,
        ):
            """
            Initialize Molecular Transformer.

            Args:
                vocab_size: Size of SMILES vocabulary
                d_model: Model dimension
                num_heads: Number of attention heads
                num_layers: Number of transformer layers
                max_length: Maximum sequence length
                output_dim: Output dimension
                dropout: Dropout probability
            """
            super().__init__()
            self.d_model = d_model
            self.max_length = max_length
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_length, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, output_dim),
            )

        def forward(self, x, attention_mask=None):
            """Forward pass through transformer."""
            batch_size, seq_len = x.shape
            positions = (
                torch.arange(seq_len, device=x.device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )
            token_emb = self.token_embedding(x)
            pos_emb = self.position_embedding(positions)
            embeddings = token_emb + pos_emb
            if attention_mask is None:
                attention_mask = x != 0
            transformer_output = self.transformer(
                embeddings, src_key_padding_mask=~attention_mask
            )
            pooled = (transformer_output * attention_mask.unsqueeze(-1)).sum(
                dim=1
            ) / attention_mask.sum(dim=1, keepdim=True)
            output = self.classifier(pooled)
            return output

    class MultiTaskMolecularModel(nn.Module):
        """
        Multi-task learning model for molecular property prediction.

        Predicts multiple molecular properties simultaneously with shared representations.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int] = [512, 256, 128],
            task_configs: Dict[str, Dict] = None,
            dropout: float = 0.2,
        ):
            """
            Initialize multi-task model.

            Args:
                input_dim: Input feature dimension
                hidden_dims: List of hidden layer dimensions
                task_configs: Dictionary with task configurations
                dropout: Dropout probability
            """
            super().__init__()
            if task_configs is None:
                task_configs = {
                    "solubility": {"type": "regression", "output_dim": 1},
                    "toxicity": {"type": "classification", "output_dim": 2},
                    "bioactivity": {"type": "regression", "output_dim": 1},
                }
            self.task_configs = task_configs
            backbone_layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                backbone_layers.extend(
                    [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
                )
                prev_dim = hidden_dim
            self.backbone = nn.Sequential(*backbone_layers)
            self.task_heads = nn.ModuleDict()
            for task_name, config in task_configs.items():
                head_layers = [
                    nn.Linear(prev_dim, prev_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(prev_dim // 2, config["output_dim"]),
                ]
                self.task_heads[task_name] = nn.Sequential(*head_layers)

        def forward(self, x, tasks=None):
            """Forward pass through multi-task model."""
            shared_features = self.backbone(x)
            outputs = {}
            if tasks is None:
                tasks = list(self.task_configs.keys())
            for task in tasks:
                if task in self.task_heads:
                    outputs[task] = self.task_heads[task](shared_features)
            return outputs

    class MetaLearningModel(nn.Module):
        """
        Meta-learning model for few-shot molecular property prediction.

        Uses Model-Agnostic Meta-Learning (MAML) for rapid adaptation to new properties.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            output_dim: int = 1,
            num_layers: int = 3,
        ):
            """
            Initialize meta-learning model.

            Args:
                input_dim: Input feature dimension
                hidden_dim: Hidden dimension
                output_dim: Output dimension
                num_layers: Number of layers
            """
            super().__init__()
            layers = []
            prev_dim = input_dim
            for i in range(num_layers):
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if i < num_layers - 1:
                    layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            """Forward pass."""
            return self.network(x)

        def adapt(
            self,
            support_x: torch.Tensor,
            support_y: torch.Tensor,
            learning_rate: float = 0.01,
            num_steps: int = 5,
        ) -> nn.Module:
            """
            Adapt model to new task using support set.

            Args:
                support_x: Support set features
                support_y: Support set targets
                learning_rate: Learning rate for adaptation
                num_steps: Number of adaptation steps

            Returns:
                Adapted model
            """
            adapted_model = type(self)(
                input_dim=support_x.shape[1],
                hidden_dim=256,
                output_dim=support_y.shape[1] if support_y.dim() > 1 else 1,
            )
            adapted_model.load_state_dict(self.state_dict())
            optimizer = torch.optim.SGD(adapted_model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            for step in range(num_steps):
                optimizer.zero_grad()
                predictions = adapted_model(support_x)
                loss = criterion(predictions, support_y)
                loss.backward()
                optimizer.step()
            return adapted_model

    class AttentionMolecularModel(nn.Module):
        """
        Molecular model with attention mechanisms for interpretability.

        Provides attention weights to understand which molecular features are important.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_heads: int = 8,
            output_dim: int = 1,
        ):
            """
            Initialize attention-based model.

            Args:
                input_dim: Input feature dimension
                hidden_dim: Hidden dimension
                num_heads: Number of attention heads
                output_dim: Output dimension
            """
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.feature_projection = nn.Linear(input_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, output_dim),
            )

        def forward(self, x, return_attention=False):
            """Forward pass with optional attention weights."""
            _batch_size = x.shape[0]
            features = self.feature_projection(x)
            features = features.unsqueeze(1)
            attended_features, attention_weights = self.attention(
                features, features, features
            )
            pooled_features = attended_features.squeeze(1)
            output = self.classifier(pooled_features)
            if return_attention:
                return output, attention_weights
            else:
                return output


def create_molecular_graph_dataset(
    smiles_list: List[str], labels: Optional[np.ndarray] = None
) -> List[Dict]:
    """
    Create dataset of molecular graphs from SMILES.

    Args:
        smiles_list: List of SMILES strings
        labels: Optional labels for supervised learning

    Returns:
        List of graph data dictionaries
    """
    dataset = []
    for i, smiles in enumerate(smiles_list):
        graph_data = smiles_to_graph(smiles)
        if graph_data is not None:
            if labels is not None:
                graph_data["label"] = labels[i]
            dataset.append(graph_data)
    return dataset


def analyze_attention_weights(
    model: AttentionMolecularModel,
    features: torch.Tensor,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Analyze attention weights for model interpretability.

    Args:
        model: Trained attention model
        features: Input features
        feature_names: Names of features

    Returns:
        Dictionary with attention analysis
    """
    model.eval()
    with torch.no_grad():
        _, attention_weights = model(features, return_attention=True)
    avg_attention = attention_weights.mean(dim=1).mean(dim=0).cpu().numpy()
    analysis = {
        "attention_weights": avg_attention,
        "top_features": np.argsort(avg_attention)[::-1][:10],
        "attention_distribution": {
            "mean": float(avg_attention.mean()),
            "std": float(avg_attention.std()),
            "max": float(avg_attention.max()),
            "min": float(avg_attention.min()),
        },
    }
    if feature_names is not None:
        analysis["top_feature_names"] = [
            feature_names[i] for i in analysis["top_features"]
        ]
    return analysis


def ensemble_predictions(
    models: List[nn.Module], features: torch.Tensor, method_type: str = "mean"
) -> torch.Tensor:
    """
    Combine predictions from multiple models.

    Args:
        models: List of trained models
        features: Input features
        method: Ensemble method ('mean', 'median', 'weighted')

    Returns:
        Ensemble predictions
    """
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(features)
            predictions.append(pred)
    predictions = torch.stack(predictions)
    if method_type == "mean":
        return predictions.mean(dim=0)
    elif method_type == "median":
        return predictions.median(dim=0)[0]
    elif method_type == "weighted":
        weights = torch.ones(len(models)) / len(models)
        return (predictions * weights.view(-1, 1, 1)).sum(dim=0)
    else:
        raise ValueError(f"Unknown ensemble method: {method_type}")


__all__ = [
    "smiles_to_graph",
    "create_molecular_graph_dataset",
    "analyze_attention_weights",
    "ensemble_predictions",
]
if HAS_TORCH:
    __all__.extend(
        [
            "MolecularTransformer",
            "MultiTaskMolecularModel",
            "MetaLearningModel",
            "AttentionMolecularModel",
        ]
    )
if HAS_TORCH_GEOMETRIC:
    __all__.extend(["MolecularGCN", "MolecularGAT"])
