import torch.nn as nn
from models.base_model import BaseModel
"""
Работает с мел-спектрограммами, подаваемыми как (B, 1, n_mels, T).
Архитектура — VGG-подобные блоки, адаптивный пулинг, dropout и линейный слой.
"""

class ConvBlock(nn.Module):
    """
    Один свёрточный блок: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class CNNModel(BaseModel):
    """
    @channels Список из 4 чисел, задающих количество каналов в каждом из 4 свёрточных блоков
    @dropout Вероятность dropout перед финальным линейным слоем
    """
    HYPERPARAMS = {
        **BaseModel.HYPERPARAMS,
        "channels": [32, 64, 128, 256],
        "dropout": 0.30,
        "batch_size": 256,
        "epochs": 60,
    }

    def create_model(self, num_classes: int, **kwargs) -> nn.Module:
        """
        Создаёт и возвращает нейронную сеть CNN.
        """
        channels = self.hparams["channels"]
        dropout = self.hparams["dropout"]

        class _CNN(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                in_ch = 1
                for out_ch in channels:
                    layers.append(ConvBlock(in_ch, out_ch))
                    in_ch = out_ch
                self.features = nn.Sequential(*layers)
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(channels[-1], num_classes)

            def forward(self, x):
                x = self.features(x)
                x = self.pool(x)
                x = x.flatten(1)
                x = self.dropout(x)
                return self.classifier(x)

        return _CNN()


MODEL_CLASS = CNNModel