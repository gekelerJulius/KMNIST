import torch
from torch import nn

from CONFIG import MODEL


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(MODEL.encoder_dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.activation(self.blocks(images) + self.shortcut(images))


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = MODEL.encoder_channels
        blocks_per_stage = MODEL.encoder_blocks_per_stage
        if len(blocks_per_stage) != len(channels):
            raise ValueError(
                "MODEL.encoder_blocks_per_stage must have one entry per encoder stage; "
                f"got {len(blocks_per_stage)} block counts for {len(channels)} stages."
            )
        layers: list[nn.Module] = [
            nn.Conv2d(MODEL.image_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        ]
        in_channels = channels[0]
        for index, (out_channels, block_count) in enumerate(zip(channels, blocks_per_stage)):
            stride = 2 if index in {1, 2} else 1
            layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
            in_channels = out_channels
            for _ in range(block_count - 1):
                layers.append(ResidualBlock(in_channels, out_channels))

        self.blocks = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        pooled_embedding_size = MODEL.encoder_channels[-1] * 2
        if pooled_embedding_size != MODEL.embedding_size:
            raise ValueError(
                f"MODEL.embedding_size must equal final encoder channels * 2; "
                f"got {MODEL.embedding_size} and {MODEL.encoder_channels[-1]} channels."
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.blocks(images)
        avg_features = self.avg_pool(features).flatten(1)
        max_features = self.max_pool(features).flatten(1)
        return torch.cat([avg_features, max_features], dim=1)


class ConvDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        seed_features = MODEL.decoder_seed_channels * MODEL.decoder_seed_size * MODEL.decoder_seed_size
        self.seed = nn.Sequential(
            nn.Linear(MODEL.embedding_size, seed_features),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            nn.Upsample(size=(MODEL.image_size, MODEL.image_size), mode="bilinear", align_corners=False),
            nn.Conv2d(MODEL.decoder_seed_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(MODEL.decoder_dropout),
            nn.Conv2d(32, MODEL.image_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = embeddings.size(0)
        seed = self.seed(embeddings)
        seed = seed.view(batch_size, MODEL.decoder_seed_channels, MODEL.decoder_seed_size, MODEL.decoder_seed_size)
        return self.blocks(seed)


def build_encoder() -> ConvEncoder:
    return ConvEncoder()


def build_decoder() -> ConvDecoder:
    return ConvDecoder()
