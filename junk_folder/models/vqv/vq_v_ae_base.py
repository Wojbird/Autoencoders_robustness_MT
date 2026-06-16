import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    for g in (8, 4, 2):
        if channels % g == 0:
            return g
    return 1


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=_num_groups(channels), num_channels=channels)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: torch.Tensor):
        z_perm = z.permute(0, 2, 3, 1).contiguous()
        flat_z = z_perm.view(-1, self.embedding_dim)

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_z @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(dim=1)
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        encodings = torch.zeros(
            encoding_indices.size(0),
            self.num_embeddings,
            device=z.device,
            dtype=z.dtype,
        )
        encodings.scatter_(1, encoding_indices, 1)

        quantized = encodings @ self.embeddings.weight
        quantized = quantized.view(z_perm.shape)

        commitment_loss = self.commitment_cost * F.mse_loss(
            quantized.detach(),
            z_perm,
        )
        codebook_loss = F.mse_loss(
            quantized,
            z_perm.detach(),
        )
        loss = commitment_loss + codebook_loss

        quantized = z_perm + (quantized - z_perm).detach()

        return quantized.permute(0, 3, 1, 2).contiguous(), loss


def make_stage(
    in_channels: int,
    out_channels: int,
    *,
    transpose: bool = False,
    use_dropout: bool = False,
    dropout_p: float = 0.0,
) -> nn.Sequential:
    if not transpose:
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            _norm(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        ]
    else:
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            _norm(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        ]

    if use_dropout and dropout_p > 0.0:
        layers.append(nn.Dropout2d(dropout_p))

    return nn.Sequential(*layers)


class VQVAEBase(nn.Module):
    """
    Lightweight VQ-VAE.

    Latent before quantization:
        [B, latent_dim, image_size / 32, image_size / 32]

    For image_size=224:
        [B, latent_dim, 7, 7]
    """

    def __init__(self, config: dict):
        super().__init__()

        image_channels = int(config["image_channels"])
        image_size = int(config.get("image_size", 224))

        latent_dim = int(config["latent_dim"])
        num_embeddings = int(config["num_embeddings"])
        commitment_cost = float(config.get("commitment_cost", 0.25))

        stem_channels = list(config["stem_channels"])
        encoder_channels = list(config["encoder_channels"])
        bottleneck_channels = int(config["bottleneck_channels"])
        dropout = float(config.get("dropout", 0.2))

        if len(stem_channels) != 2:
            raise ValueError("config['stem_channels'] must contain exactly 2 values.")
        if len(encoder_channels) != 4:
            raise ValueError("config['encoder_channels'] must contain exactly 4 values.")
        if image_size % 32 != 0:
            raise ValueError("image_size must be divisible by 32.")

        s1, s2 = map(int, stem_channels)
        c1, c2, c3, c4 = map(int, encoder_channels)

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.latent_channels = latent_dim
        self.latent_hw = image_size // 32
        self.latent_size = latent_dim * self.latent_hw * self.latent_hw

        self.pre_encoder = nn.Sequential(
            nn.Conv2d(image_channels, s1, kernel_size=3, padding=1, bias=False),
            _norm(s1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(s1, s2, kernel_size=3, padding=1, bias=False),
            _norm(s2),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.enc1 = make_stage(s2, c1, use_dropout=False, dropout_p=dropout)
        self.enc2 = make_stage(c1, c2, use_dropout=True, dropout_p=dropout)
        self.enc3 = make_stage(c2, c3, use_dropout=True, dropout_p=dropout)
        self.enc4 = make_stage(c3, c4, use_dropout=True, dropout_p=dropout)
        self.enc5 = make_stage(c4, bottleneck_channels, use_dropout=True, dropout_p=dropout)

        self.to_quant = nn.Conv2d(bottleneck_channels, latent_dim, kernel_size=1)
        self.from_quant = nn.Conv2d(latent_dim, bottleneck_channels, kernel_size=1)

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
        )

        self.dec1 = make_stage(bottleneck_channels, c4, transpose=True, use_dropout=True, dropout_p=dropout)
        self.dec2 = make_stage(c4, c3, transpose=True, use_dropout=True, dropout_p=dropout)
        self.dec3 = make_stage(c3, c2, transpose=True, use_dropout=True, dropout_p=dropout)
        self.dec4 = make_stage(c2, c1, transpose=True, use_dropout=False, dropout_p=dropout)
        self.dec5 = make_stage(c1, s2, transpose=True, use_dropout=False, dropout_p=dropout)

        self.final = nn.Sequential(
            nn.Conv2d(s2, image_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.vq_loss = None

    def _encode_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_encoder(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self._encode_map(x)
        z_e = self.to_quant(z_e)
        z_q, self.vq_loss = self.quantizer(z_e)
        return z_q

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        z = self.from_quant(z_q)
        z = self.dec1(z)
        z = self.dec2(z)
        z = self.dec3(z)
        z = self.dec4(z)
        z = self.dec5(z)
        return self.final(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_q = self.encode(x)
        return self.decode(z_q)

    def get_vq_losses(self):
        result = {}
        if self.vq_loss is not None:
            result["vq_loss"] = float(self.vq_loss.item())
        return result