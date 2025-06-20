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


class VQVAE1024(nn.Module):
    def __init__(self, config):
        super().__init__()
        image_channels = config["image_channels"]
        latent_dim = config["latent_dim"]
        assert latent_dim == 1024, "This model is designed for latent_dim=1024"

        num_embeddings = config.get("num_embeddings", 4096)

        self.pre_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 112x112
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 56x56
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),  # 28x28
            nn.InstanceNorm2d(384),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, latent_dim, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.InstanceNorm2d(latent_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2)
        )

        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dim)

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(512 , 384, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.InstanceNorm2d(384),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(384 , 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 56x56
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256 , 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 112x112
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(128 , 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 224x224
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.final = nn.Conv2d(64, image_channels, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

    def encode(self, x):
        x = self.pre_encoder(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        z_e = self.enc5(x)
        z_q, self.vq_loss = self.quantizer(z_e)
        return z_q

    def decode(self, z):
        z = self.dec1(z)
        z = self.dec2(z)
        z = self.dec3(z)
        z = self.dec4(z)
        z = self.dec5(z)
        return self.activation(self.final(z))

    def forward(self, x):
        z_q = self.encode(x)
        return self.decode(z_q)

    def get_vq_losses(self):
        result = {}
        if hasattr(self, "vq_loss"):
            result["vq_loss"] = self.vq_loss.item()
        return result


# Integracja z main.py
model_class = VQVAE1024
config_path = "configs/vqv/vq_v_ae_1024.json"