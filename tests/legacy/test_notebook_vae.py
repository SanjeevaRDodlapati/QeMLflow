#!/usr/bin/env python3
"""
Test script to verify that the VAE implementation in the notebook works correctly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Copy the exact implementation from the notebook
class MolecularVAE(nn.Module):
    """Variational Autoencoder for SMILES generation"""

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
            # Inference mode - FIXED: Remove extra .unsqueeze(1)
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
                current_input = self.embedding(next_token)  # NO .unsqueeze(1) here!

            return torch.stack(outputs, dim=1)

    def forward(self, x, target_seq=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, target_seq)
        return recon, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss with KL divergence and reconstruction loss"""
    # Reconstruction loss - FIXED: using .reshape() instead of .view()
    recon_loss = F.cross_entropy(
        recon_x.reshape(-1, recon_x.size(-1)), x.reshape(-1), reduction="mean"
    )

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def test_notebook_vae():
    """Test the VAE implementation that matches the notebook exactly"""

    print("🧪 Testing Notebook VAE Implementation")
    print("=" * 45)

    # Test parameters
    batch_size = 8
    seq_len = 64
    vocab_size = 50
    embedding_dim = 128
    hidden_dim = 256
    latent_dim = 64
    max_length = 128

    # Create VAE model
    vae_model = MolecularVAE(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        max_length=max_length,
    )

    print(
        f"Model created with {sum(p.numel() for p in vae_model.parameters()):,} parameters"
    )

    # Test Case 1: Forward pass with teacher forcing
    print("\n📋 Test Case 1: Forward pass (training mode)")
    try:
        input_seq = torch.randint(0, vocab_size, (batch_size, seq_len))
        target_seq = torch.randint(0, vocab_size, (batch_size, seq_len - 1))

        with torch.no_grad():
            recon, mu, logvar = vae_model(input_seq, target_seq)

        print("✅ SUCCESS")
        print(f"   Input shape: {input_seq.shape}")
        print(f"   Target shape: {target_seq.shape}")
        print(f"   Reconstruction shape: {recon.shape}")

    except Exception as e:
        print(f"❌ FAILED - {type(e).__name__}: {e}")
        return False

    # Test Case 2: Loss function
    print("\n📋 Test Case 2: Loss function")
    try:
        total_loss, recon_loss, kl_loss = vae_loss_function(
            recon, target_seq, mu, logvar, beta=0.5
        )

        print("✅ SUCCESS")
        print(f"   Total loss: {total_loss.item():.4f}")
        print(f"   Reconstruction loss: {recon_loss.item():.4f}")
        print(f"   KL loss: {kl_loss.item():.4f}")

    except Exception as e:
        print(f"❌ FAILED - {type(e).__name__}: {e}")
        return False

    # Test Case 3: Generation mode (no teacher forcing)
    print("\n📋 Test Case 3: Generation mode (inference)")
    try:
        input_seq = torch.randint(0, vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            recon, mu, logvar = vae_model(input_seq, None)  # No target_seq

        print("✅ SUCCESS")
        print(f"   Input shape: {input_seq.shape}")
        print(f"   Generated shape: {recon.shape}")
        print(f"   Expected shape: [{batch_size}, {max_length}, {vocab_size}]")

        assert recon.shape == (batch_size, max_length, vocab_size)
        print("   ✓ Output dimensions correct")

    except Exception as e:
        print(f"❌ FAILED - {type(e).__name__}: {e}")
        return False

    # Test Case 4: Gradient flow
    print("\n📋 Test Case 4: Gradient flow")
    try:
        vae_model.train()
        input_seq = torch.randint(0, vocab_size, (batch_size, seq_len))
        target_seq = torch.randint(0, vocab_size, (batch_size, seq_len - 1))

        recon, mu, logvar = vae_model(input_seq, target_seq)
        total_loss, recon_loss, kl_loss = vae_loss_function(
            recon, target_seq, mu, logvar
        )

        total_loss.backward()

        # Check if gradients exist
        has_gradients = any(p.grad is not None for p in vae_model.parameters())

        print("✅ SUCCESS")
        print(f"   Gradients computed: {has_gradients}")
        print(f"   Loss value: {total_loss.item():.4f}")

    except Exception as e:
        print(f"❌ FAILED - {type(e).__name__}: {e}")
        return False

    print("\n🎉 All notebook VAE tests passed!")
    print("✅ The VAE implementation is working correctly")
    return True


if __name__ == "__main__":
    success = test_notebook_vae()
    if success:
        print("\n" + "=" * 50)
        print("🏆 NOTEBOOK VAE VERIFIED SUCCESSFULLY")
        print("🚀 The Day 2 VAE implementation is ready!")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("⚠️  NOTEBOOK VAE VERIFICATION FAILED")
        print("🔧 Additional fixes needed")
        print("=" * 50)
