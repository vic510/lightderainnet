"""Lightweight deraining model built for this project.

External models (DRSformer, SSDRNet, etc.) are not reimplemented here; they are loaded from their official repos via common.load_model.
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class LightDerainNet(nn.Module):
    """Lightweight U-shaped residual CNN.

    - Downsamples to 1/2 resolution with PixelUnshuffle before convolving, greatly reducing compute for high-resolution video.
    - The network predicts the rain layer R and outputs B = O - R (residual learning, as in most deraining CNNs).
    - width / blocks are adjustable for structural-compression experiments (e.g. 32 -> 16 channels).
    Input height and width must be multiples of 4; common.forward_padded pads automatically at inference.
    """

    def __init__(self, width: int = 32, blocks: int = 6):
        super().__init__()
        w = width
        self.unshuffle = nn.PixelUnshuffle(2)          # 3ch, H   -> 12ch, H/2
        self.head = nn.Conv2d(12, w, 3, padding=1)
        self.enc = nn.Sequential(ResBlock(w), ResBlock(w))
        self.down = nn.Conv2d(w, 2 * w, 3, stride=2, padding=1)  # H/2 -> H/4
        self.mid = nn.Sequential(*[ResBlock(2 * w) for _ in range(blocks)])
        self.up = nn.Sequential(nn.Conv2d(2 * w, 4 * w, 1), nn.PixelShuffle(2))  # H/4 -> H/2
        self.fuse = nn.Conv2d(2 * w, w, 1)
        self.dec = nn.Sequential(ResBlock(w), ResBlock(w))
        self.tail = nn.Conv2d(w, 12, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)              # 12ch, H/2 -> 3ch, H
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        e = self.enc(self.act(self.head(self.unshuffle(x))))
        m = self.mid(self.act(self.down(e)))
        d = self.dec(self.fuse(torch.cat([self.up(m), e], dim=1)))
        rain = self.shuffle(self.tail(d))
        return x - rain


ARCHS = {"light": LightDerainNet}


def build_arch(arch: str, **kwargs) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"Unknown architecture {arch}; available: {list(ARCHS)}")
    return ARCHS[arch](**kwargs)
