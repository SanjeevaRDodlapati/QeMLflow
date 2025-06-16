#!/usr/bin/env python3
"""
Test script to verify VAE decode tensor dimension fix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MolecularVAE(nn.Module):
    """Variational Autoencoder for SMILES generation - Test Version"""

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        latent_dim=128,
        max_length=128,
    ):
        super(MolecularVAE, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_length = max_length

        # Encoder
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            embedding_dim + latent_dim, hidden_dim, batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

        self.dropout = nn.Dropout(0.2)

    def encode(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        output, (hidden, _) = self.encoder_lstm(embedded)
        # Take the last hidden state from both directions
        hidden = torch.cat(
            [hidden[-2], hidden[-1]], dim=1
        )  # [batch_size, hidden_dim * 2]

        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, target_seq=None):
        batch_size = z.size(0)

        # Initialize decoder
        hidden = self.decoder_input(z).unsqueeze(0)  # [1, batch_size, hidden_dim]
        cell = torch.zeros_like(hidden)

        outputs = []

        if target_seq is not None:
            # Training mode - teacher forcing
            target_embedded = self.embedding(
                target_seq
            )  # [batch_size, seq_len, embedding_dim]

            for i in range(target_seq.size(1)):
                # Concatenate latent vector with current input
                z_expanded = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
                decoder_input = torch.cat(
                    [target_embedded[:, i : i + 1, :], z_expanded], dim=-1
                )

                output, (hidden, cell) = self.decoder_lstm(
                    decoder_input, (hidden, cell)
                )
                output = self.output_layer(output.squeeze(1))
                outputs.append(output)

            return torch.stack(outputs, dim=1)  # [batch_size, seq_len, vocab_size]
        else:
            # Inference mode - FIXED VERSION
            current_input = torch.zeros(batch_size, 1, self.embedding_dim).to(z.device)

            for i in range(self.max_length):
                z_expanded = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
                decoder_input = torch.cat([current_input, z_expanded], dim=-1)

                output, (hidden, cell) = self.decoder_lstm(
                    decoder_input, (hidden, cell)
                )
                output = self.output_layer(output.squeeze(1))
                outputs.append(output)

                # Use output as next input - FIXED: Remove extra .unsqueeze(1)
                next_token = torch.argmax(output, dim=-1, keepdim=True)
                current_input = self.embedding(
                    next_token
                )  # Shape: [batch_size, 1, embedding_dim]

            return torch.stack(outputs, dim=1)

    def forward(self, x, target_seq=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, target_seq)
        return recon, mu, logvar


def test_vae_decode_fix():
    """Test the fixed VAE decode method for tensor dimension compatibility"""

    print("🧪 Testing VAE Decode Tensor Dimension Fix")
    print("=" * 50)

    # Test parameters
    batch_size = 4
    seq_len = 32
    vocab_size = 50
    embedding_dim = 128
    hidden_dim = 256
    latent_dim = 64
    max_length = 32

    print("Test Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Max length: {max_length}")

    # Create test VAE model
    vae_model = MolecularVAE(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        max_length=max_length,
    )

    print(
        f"\n📊 Model created with {sum(p.numel() for p in vae_model.parameters()):,} parameters"
    )

    # Test Case 1: Training mode (teacher forcing)
    print("\n📋 Test Case 1: Training mode (teacher forcing)")
    try:
        input_seq = torch.randint(0, vocab_size, (batch_size, seq_len))
        target_seq = torch.randint(0, vocab_size, (batch_size, seq_len - 1))

        with torch.no_grad():
            recon, mu, logvar = vae_model(input_seq, target_seq)

        print("✅ SUCCESS")
        print(f"   Input shape: {input_seq.shape}")
        print(f"   Target shape: {target_seq.shape}")
        print(f"   Reconstruction shape: {recon.shape}")
        print(f"   Mu shape: {mu.shape}")
        print(f"   Logvar shape: {logvar.shape}")

    except Exception as e:
        print(f"❌ FAILED - {type(e).__name__}: {e}")
        return False

    # Test Case 2: Inference mode (no teacher forcing)
    print("\n📋 Test Case 2: Inference mode (generation)")
    try:
        input_seq = torch.randint(0, vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            recon, mu, logvar = vae_model(
                input_seq, None
            )  # No target_seq = inference mode

        print("✅ SUCCESS")
        print(f"   Input shape: {input_seq.shape}")
        print(f"   Generated shape: {recon.shape}")
        print(f"   Expected shape: [{batch_size}, {max_length}, {vocab_size}]")
        print(f"   Mu shape: {mu.shape}")
        print(f"   Logvar shape: {logvar.shape}")

        # Verify correct dimensions
        assert recon.shape == (
            batch_size,
            max_length,
            vocab_size,
        ), f"Wrong output shape: {recon.shape}"
        print("   ✓ Output dimensions correct")

    except Exception as e:
        print(f"❌ FAILED - {type(e).__name__}: {e}")
        return False

    # Test Case 3: Different batch sizes
    print("\n📋 Test Case 3: Varying batch sizes")
    for test_batch_size in [1, 2, 8, 16]:
        try:
            input_seq = torch.randint(0, vocab_size, (test_batch_size, seq_len))

            with torch.no_grad():
                recon, mu, logvar = vae_model(input_seq, None)

            expected_shape = (test_batch_size, max_length, vocab_size)
            assert (
                recon.shape == expected_shape
            ), f"Wrong shape for batch {test_batch_size}"

            print(f"   Batch {test_batch_size:2d}: ✅ Shape {recon.shape}")

        except Exception as e:
            print(f"   Batch {test_batch_size:2d}: ❌ {type(e).__name__}: {e}")
            return False

    # Test Case 4: Direct decode method test
    print("\n📋 Test Case 4: Direct decode method test")
    try:
        z = torch.randn(batch_size, latent_dim)

        with torch.no_grad():
            generated = vae_model.decode(z, None)

        print("✅ SUCCESS")
        print(f"   Latent shape: {z.shape}")
        print(f"   Generated shape: {generated.shape}")
        print(f"   Expected shape: [{batch_size}, {max_length}, {vocab_size}]")

        assert generated.shape == (batch_size, max_length, vocab_size)
        print("   ✓ Direct decode dimensions correct")

    except Exception as e:
        print(f"❌ FAILED - {type(e).__name__}: {e}")
        return False

    print("\n🎉 All VAE decode tests passed!")
    print("✅ Tensor dimension mismatch issue has been resolved")
    return True


if __name__ == "__main__":
    success = test_vae_decode_fix()
    if success:
        print("\n" + "=" * 60)
        print("🏆 VAE DECODE FIX VERIFIED SUCCESSFULLY")
        print("🚀 VAE molecule generation should now work correctly!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️  VAE DECODE FIX VERIFICATION FAILED")
        print("🔧 Additional debugging required")
        print("=" * 60)
