import torch
import torch.nn as nn


class BaseModel:

    """
    Абстрактный базовый класс модели.
    Каждая модель-наследник задаёт собственные гиперпараметры через
    атрибут класса HYPERPARAMS и реализует метод create_model(),
    возвращающий готовый nn.Module.

    hparams : dict или None - словарь с переопределениями гиперпараметров.
    """
    
    """
    Общие гиперпараметры, наследуемые всеми моделями.
    @param learning_rate Скорость обучения для оптимизатора Adam
    @param weight_decay Коэффициент L2‑регуляризации для оптимизатора Adam
    @param batch_size Размер мини-батча при обучении
    @param epochs Общее количество эпох обучения
    @param scheduler Тип планировщика скорости обучения: "cosine", "step" или "none"
    @param step_size Период уменьшения lr для StepLR (используется только если scheduler == "step")
    @param gamma Коэффициент уменьшения lr для StepLR (используется только если scheduler == "step")
    """
    HYPERPARAMS = {
        "feature_type": "mel",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "epochs": 50,
        "scheduler": "cosine",
        "step_size": 20,
        "gamma": 0.1,
    }

    def __init__(self, hparams: dict = None):
        self.hparams = {**self.HYPERPARAMS, **(hparams or {})}

    def create_model(self, num_classes: int, **kwargs) -> nn.Module:
        """
        Абстрактный метод. Должен быть переопределён в подклассе.
        Возвращает экземпляр nn.Module - готовую модель.
        """
        raise NotImplementedError

    def get_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Возвращает оптимизатор Adam для модели, используя learning_rate
        и weight_decay из гиперпараметров.
        """
        return torch.optim.Adam(
            model.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )

    def get_scheduler(self, optimizer, num_epochs: int = None):
        """
        Возвращает планировщик скорости обучения или None.
        Поддерживает 'cosine' (CosineAnnealingLR) и 'step' (StepLR).
        Если scheduler == 'none' или не задан, возвращает None.
        """
        sched = self.hparams["scheduler"]
        epochs = num_epochs or self.hparams["epochs"]

        if sched == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if sched == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams["step_size"],
                gamma=self.hparams["gamma"],
            )
        return None

    def get_transform(self):
        """
        Возвращает тип признаков, указанный в гиперпараметрах.
        Фактическое преобразование создаётся в train.py через audio_features.py.
        """
        return self.hparams.get("feature_type", "mel")

    def get_batch_size(self) -> int:
        """Возвращает размер батча из гиперпараметров."""
        return self.hparams["batch_size"]

    def get_epochs(self) -> int:
        """Возвращает количество эпох обучения из гиперпараметров."""
        return self.hparams["epochs"]