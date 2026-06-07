import torch.nn as nn
from models.base_model import BaseModel

"""
Принимает тензор (B, 1, n_mels, T), преобразует в (B, T, n_mels),
пропускает через многослойную двунаправленную LSTM, усредняет
скрытые состояния по времени (mean-pool), применяет dropout и линейный слой для предсказания класса.
"""

class BiLSTMClassifier(BaseModel):

    """
    @hidden_size Размер скрытого состояния LSTM (в одном направлении; полный выход = 2 * hidden_size)
    @num_layers Количество слоёв LSTM
    @dropout Вероятность dropout между слоями LSTM (если num_layers > 1) и перед линейным слоем
    """
    HYPERPARAMS = {
        **BaseModel.HYPERPARAMS,
        "hidden_size": 256,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 128,
        "epochs": 100,
    }

    def create_model(self, num_classes: int, n_mels: int = 64, **kwargs) -> nn.Module:
        """
        Создаёт и возвращает нейронную сеть BiLSTM.
        """
        hidden_size = self.hparams["hidden_size"]
        num_layers = self.hparams["num_layers"]
        dropout = self.hparams["dropout"]
        lstm_dropout = dropout if num_layers > 1 else 0.0

        class _BiLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_mels,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=lstm_dropout,
                )
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(hidden_size * 2, num_classes)

            def forward(self, x):
                B, C, M, T = x.shape
                x = x.squeeze(1)          # (B, n_mels, T)
                x = x.permute(0, 2, 1)    # (B, T, n_mels)
                out, _ = self.lstm(x)     # (B, T, 2*hidden_size)
                out = out.mean(dim=1)     # (B, 2*hidden_size)
                out = self.dropout(out)
                return self.classifier(out)

        return _BiLSTM()


MODEL_CLASS = BiLSTMClassifier