"""本專案自建的輕量去雨模型。

外部模型（DRSformer、SSDRNet 等）不在此重寫，而是透過 common.load_model 從其官方程式庫載入。
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
    """輕量 U 形殘差 CNN。

    - 先以 PixelUnshuffle 降到 1/2 解析度再做卷積，大幅降低高解析度影片的運算量。
    - 網路預測雨層 R，輸出 B = O - R（殘差學習，與多數去雨 CNN 相同）。
    - width / blocks 可調，用來做「結構壓縮」實驗（例如 32→16 通道）。
    輸入長寬需為 4 的倍數，推論時由 common.forward_padded 自動補齊。
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
        raise ValueError(f"未知架構 {arch}，可用：{list(ARCHS)}")
    return ARCHS[arch](**kwargs)
