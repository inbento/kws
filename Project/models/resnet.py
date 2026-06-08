import torch.nn as nn
from models.base_model import BaseModel

"""
Остаточная свёрточная сеть.
Работает с мел-спектрограммами, подаваемыми как (B, 1, n_mels, T).
Архитектура: начальный stem-блок (Conv + BN + ReLU + MaxPool),
четыре стадии остаточных блоков с увеличением каналов и уменьшением
пространственного разрешения, адаптивный средний пулинг, dropout и линейный классификатор.
"""

class ResBlock(nn.Module):
    """
    Базовый остаточный блок: две свёртки 3×3 с пропуском.
    in_ch : int - число входных каналов.
    out_ch: int - число выходных каналов.
    stride: int - шаг первой свёртки (по умолчанию 1).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


def _make_stage(in_ch: int, out_ch: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
    layers = [ResBlock(in_ch, out_ch, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(ResBlock(out_ch, out_ch))
    return nn.Sequential(*layers)


class ResNetModel(BaseModel):
    """
    Остаточная CNN.
    """

    """
    @width_mult Множитель ширины сети; 1.0 — базовые каналы [64, 128, 256, 512], уменьшение (например, 0.5) даёт более лёгкую модель
    @blocks_per_stage Количество остаточных блоков на каждой из 4 стадий
    @dropout Вероятность dropout перед финальным классификатором
    """
    HYPERPARAMS = {
        **BaseModel.HYPERPARAMS,
        "width_mult": 1.0,
        "blocks_per_stage": [4, 2, 4, 2],
        "dropout": 0.25,
        "batch_size": 128,
        "epochs": 50,
    }

    def create_model(self, num_classes: int, **kwargs) -> nn.Module:
        """
        Создаёт и возвращает нейронную сеть ResNet.
        """
        w = self.hparams["width_mult"]
        b = self.hparams["blocks_per_stage"]
        d = self.hparams["dropout"]
        c = [int(ch * w) for ch in [64, 128, 256, 512]]

        class _ResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(1, c[0], kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(c[0]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                self.stage1 = _make_stage(c[0], c[0], b[0], stride=1)
                self.stage2 = _make_stage(c[0], c[1], b[1], stride=2)
                self.stage3 = _make_stage(c[1], c[2], b[2], stride=2)
                self.stage4 = _make_stage(c[2], c[3], b[3], stride=2)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.dropout = nn.Dropout(d)
                self.classifier = nn.Linear(c[3], num_classes)

            def forward(self, x):
                x = self.stem(x)
                x = self.stage1(x)
                x = self.stage2(x)
                x = self.stage3(x)
                x = self.stage4(x)
                x = self.pool(x)
                x = x.flatten(1)
                x = self.dropout(x)
                return self.classifier(x)

        return _ResNet()


MODEL_CLASS = ResNetModel