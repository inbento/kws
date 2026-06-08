import math
import torch
import torch.nn as nn
from models.base_model import BaseModel


"""
Принимает тензор (B, 1, n_mels, T), преобразует в (B, T, n_mels),
проецирует в d_model, добавляет синусоидальные позиционные кодировки,
пропускает через несколько слоёв TransformerEncoder (Pre-LN),
усредняет выходы по времени (mean-pool), применяет dropout и линейный слой для классификации.
"""

class SinusoidalPositionalEncoding(nn.Module):
    """
    Фиксированное синусоидальное позиционное кодирование.

    d_model : int - размерность модели.
    max_len : int - максимальная длина последовательности (по умолчанию 500).
    dropout : float - вероятность dropout после сложения с позиционным кодом.
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerClassifier(BaseModel):
    """
    Transformer-классификатор для речевых команд.
    """

    """
    @d_model Размерность эмбеддинга / модели (должна делиться на nhead)
    @nhead Количество голов в многоголовом внимании
    @dim_feedforward Размерность скрытого слоя FFN (обычно 4 * d_model)
    @num_layers Количество слоёв TransformerEncoder
    """
    HYPERPARAMS = {
        **BaseModel.HYPERPARAMS,
        "d_model": 128,
        "nhead": 4,
        "dim_feedforward": 512,
        "num_layers": 8,
        "dropout": 0.15,
        "batch_size": 128,
        "epochs": 70,
    }

    def create_model(self, num_classes: int, n_mels: int = 64, **kwargs) -> nn.Module:
        """
        Создаёт и возвращает нейронную сеть Transformer.
        """
        d_model = self.hparams["d_model"]
        nhead = self.hparams["nhead"]
        dim_feedforward = self.hparams["dim_feedforward"]
        num_layers = self.hparams["num_layers"]
        dropout = self.hparams["dropout"]

        class _Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(n_mels, d_model)
                self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(d_model, num_classes)

            def forward(self, x):
                B, C, M, T = x.shape
                x = x.squeeze(1)
                x = x.permute(0, 2, 1)
                x = self.input_proj(x)
                x = self.pos_enc(x)
                x = self.encoder(x)
                x = x.mean(dim=1)
                x = self.dropout(x)
                return self.classifier(x)

        return _Transformer()


MODEL_CLASS = TransformerClassifier