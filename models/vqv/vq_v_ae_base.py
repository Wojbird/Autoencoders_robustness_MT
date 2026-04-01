import torch
import torch.nn as nn
import torch.nn.functional as F


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
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(1)
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.size(0),
            self.num_embeddings,
            device=z.device,
            dtype=z.dtype
        )
        encodings.scatter_(1, encoding_indices, 1)

        quantized = encodings @ self.embeddings.weight
        quantized = quantized.view(z_perm.shape)

        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), z_perm)
        codebook_loss = F.mse_loss(quantized, z_perm.detach())
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
            nn.BatchNorm2d(out_channels),
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
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        ]

    if use_dropout and dropout_p > 0.0:
        layers.append(nn.Dropout2d(dropout_p))

    return nn.Sequential(*layers)


class VQVAEBase(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        image_channels = int(config["image_channels"])
        image_size = int(config.get("image_size", 224))

        embedding_dim = int(config["embedding_dim"])
        num_embeddings = int(config["num_embeddings"])
        commitment_cost = float(config.get("commitment_cost", 0.25))

        pre_channels = list(config["pre_channels"])
        enc_channels = list(config["enc_channels"])
        dropout = float(config.get("dropout", 0.2))

        if len(pre_channels) != 2:
            raise ValueError("config['pre_channels'] must contain exactly 2 values.")
        if len(enc_channels) != 5:
            raise ValueError("config['enc_channels'] must contain exactly 5 values.")
        if image_size % 32 != 0:
            raise ValueError("image_size must be divisible by 32.")
        if enc_channels[-1] != embedding_dim:
            raise ValueError("embedding_dim must be equal to enc_channels[-1].")

        self.pre_encoder = nn.Sequential(
            nn.Conv2d(image_channels, pre_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(pre_channels[0]),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(pre_channels[0], pre_channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(pre_channels[1]),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Encoder: enc1 bez dropout, enc2-enc5 z dropout
        self.enc1 = make_stage(pre_channels[1], enc_channels[0], use_dropout=False, dropout_p=dropout)
        self.enc2 = make_stage(enc_channels[0], enc_channels[1], use_dropout=True, dropout_p=dropout)
        self.enc3 = make_stage(enc_channels[1], enc_channels[2], use_dropout=True, dropout_p=dropout)
        self.enc4 = make_stage(enc_channels[2], enc_channels[3], use_dropout=True, dropout_p=dropout)
        self.enc5 = make_stage(enc_channels[3], enc_channels[4], use_dropout=True, dropout_p=dropout)

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )

        dec_channels = list(reversed(enc_channels[:-1])) + [pre_channels[1]]

        # Decoder: dec1 bez dropout, dec2-dec3 z dropout, dec4-dec5 bez dropout
        self.dec1 = make_stage(enc_channels[4], dec_channels[0], transpose=True, use_dropout=False, dropout_p=dropout)
        self.dec2 = make_stage(dec_channels[0], dec_channels[1], transpose=True, use_dropout=True, dropout_p=dropout)
        self.dec3 = make_stage(dec_channels[1], dec_channels[2], transpose=True, use_dropout=True, dropout_p=dropout)
        self.dec4 = make_stage(dec_channels[2], dec_channels[3], transpose=True, use_dropout=False, dropout_p=dropout)
        self.dec5 = make_stage(dec_channels[3], dec_channels[4], transpose=True, use_dropout=False, dropout_p=dropout)

        self.final = nn.Conv2d(pre_channels[1], image_channels, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()

        self.vq_loss = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_encoder(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        z_e = self.enc5(x)

        z_q, self.vq_loss = self.quantizer(z_e)
        return z_q

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.dec1(z)
        z = self.dec2(z)
        z = self.dec3(z)
        z = self.dec4(z)
        z = self.dec5(z)
        return self.activation(self.final(z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_q = self.encode(x)
        return self.decode(z_q)

    def get_vq_losses(self):
        result = {}
        if self.vq_loss is not None:
            result["vq_loss"] = float(self.vq_loss.item())
        return result