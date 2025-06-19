import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_perm.view(-1, self.embedding_dim)

        distances = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(1)
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = encodings @ self.embeddings.weight
        quantized = quantized.view(z_perm.shape)

        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), z_perm)
        codebook_loss = F.mse_loss(quantized, z_perm.detach())
        loss = commitment_loss + codebook_loss

        quantized = z_perm + (quantized - z_perm).detach()
        return quantized.permute(0, 3, 1, 2).contiguous(), loss


class VQVAE2512(nn.Module):
    def __init__(self, config):
        super().__init__()
        C = config["image_channels"]
        self.top_dim = config["top_latent_dim"]
        self.bottom_dim = config["bottom_latent_dim"]
        self.top_codebook_size = config["top_num_embeddings"]
        self.bottom_codebook_size = config["bottom_num_embeddings"]
        self.commitment_cost = config.get("commitment_cost", 0.25)

        # Shared pre-encoder
        self.pre_encoder = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Bottom Encoder (updated)
        self.enc_b1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 112
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.enc_b2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.enc_b3 = nn.Sequential(
            nn.Conv2d(256, self.bottom_dim, kernel_size=3, stride=2, padding=1),  # 28
            nn.BatchNorm2d(self.bottom_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Top Encoder (unchanged)
        self.enc_t1 = nn.Sequential(
            nn.Conv2d(self.bottom_dim, 448, kernel_size=3, stride=2, padding=1),  # 14
            nn.BatchNorm2d(448),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.enc_t2 = nn.Sequential(
            nn.Conv2d(448, self.top_dim, kernel_size=3, stride=1, padding=1),  # 7
            nn.BatchNorm2d(self.top_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.vq_top = VectorQuantizer(self.top_codebook_size, self.top_dim, self.commitment_cost)
        self.vq_bottom = VectorQuantizer(self.bottom_codebook_size, self.bottom_dim, self.commitment_cost)

        # Decoder
        self.upsample_top = nn.Sequential(
            nn.ConvTranspose2d(self.top_dim, self.bottom_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.bottom_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(self.bottom_dim * 2, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 224
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(64, C, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.pre_encoder(x)
        z_b = self.enc_b1(x)
        z_b = self.enc_b2(z_b)
        z_b = self.enc_b3(z_b)
        z_t = self.enc_t1(z_b)
        z_t = self.enc_t2(z_t)

        q_t, self.vq_loss_top = self.vq_top(z_t)
        q_b, self.vq_loss_bottom = self.vq_bottom(z_b)

        return q_t, q_b

    def decode(self, q_t, q_b):
        up_t = self.upsample_top(q_t)
        z = torch.cat([up_t, q_b], dim=1)
        z = self.dec1(z)
        z = self.dec2(z)
        z = self.dec3(z)
        return self.final(z)

    def forward(self, x):
        q_t, q_b = self.encode(x)
        x_recon = self.decode(q_t, q_b)
        self.vq_loss = self.vq_loss_top + self.vq_loss_bottom
        return x_recon


# Integracja z main.py
model_class = VQVAE2512
config_path = "configs/vqv2/vq_v_ae2_512.json"
