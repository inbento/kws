import os
import sys
import time
import json
import gc
import torch
import torch.nn as nn
import numpy as np
import optuna
import dataset as ds
from models.audio_features import get_feature_transform
from models.cnn import CNNModel
from models.lstm import BiLSTMClassifier
from models.resnet import ResNetModel
from models.transformer import TransformerClassifier

"""
Скрипт автоматического подбора гиперпараметров с помощью Optuna.
Использует байесовский метод для поиска оптимальных параметров.
"""

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

"""
@param MODEL_NAME- название архитектуры модели для оптимизации.
@param MODEL_HPARAMS или None - базовые гиперпараметры модели. Если None, используются дефолтные.
@param N_TRIALS- количество испытаний для Optuna.
@param TIMEOUT_HOURS - максимальное время оптимизации в часах.
@param N_EPOCHS_PER_TRIAL - количество эпох обучения для каждой попытки.
"""

MODEL_NAME = "cnn"
N_TRIALS = 30
TIMEOUT_HOURS = 24
N_EPOCHS_PER_TRIAL = 20

MODEL_CLASSES = {
    "cnn": CNNModel,
    "lstm": BiLSTMClassifier,
    "resnet": ResNetModel,
    "transformer": TransformerClassifier,
}

DATASET_CONFIG = {
    "data_dir": "data_set",
    "classes": None,
    "use_official_split": False,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "seed": 42,
    "num_workers": 4,
    "save_test_files": False,
    "copy_test_files": False,
    "test_files_output": "test_files.txt",
}

_LOADERS_CACHE = None
_AUDIO_CFG = None
_LABEL_MAP = None


def get_cached_data(batch_size):
    """
    Загружает данные один раз и кэширует. При изменении batch_size пересоздаёт загрузчики.
    """
    global _LOADERS_CACHE, _AUDIO_CFG, _LABEL_MAP

    audio_cfg = ds.AUDIO_CONFIG.copy()
    data_dir = os.path.join(_SCRIPT_DIR, DATASET_CONFIG["data_dir"])

    if _LOADERS_CACHE is None or _LOADERS_CACHE[0] != batch_size:
        cfg_with_batch = {
            **DATASET_CONFIG,
            "batch_size": batch_size,
            "data_dir": data_dir,
        }
        cfg_with_batch["test_files_output"] = os.path.join(
            _SCRIPT_DIR, DATASET_CONFIG["test_files_output"]
        )

        train_loader, val_loader, _, label_map = ds.get_dataloaders(
            data_dir=data_dir,
            transform=None,
            dataset_config=cfg_with_batch,
        )
        _LOADERS_CACHE = (batch_size, train_loader, val_loader)
        _AUDIO_CFG = audio_cfg
        _LABEL_MAP = label_map
        print(f"[cache] Data loaded for batch_size={batch_size}")

    return _LOADERS_CACHE[1], _LOADERS_CACHE[2], _LABEL_MAP, _AUDIO_CFG


@torch.no_grad()
def evaluate(model, loader, criterion, device, gpu_transform):
    """
    Оценивает модель на валидационном наборе.

    Возвращает: float - точность в процентах.
    """
    model.eval()
    correct, total = 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = gpu_transform(inputs)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def objective(trial):
    """
    Целевая функция для Optuna.
    Возвращает точность на валидации (максимизируем).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    feature_type = trial.suggest_categorical(
        "feature_type", ["mel", "fbank", "stft"]
    )

    learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 1e-2, log=True
    )
    weight_decay = trial.suggest_float(
        "weight_decay", 1e-6, 1e-2, log=True
    )
    batch_size = trial.suggest_categorical(
        "batch_size", [64, 128, 256]
    )
    dropout = trial.suggest_float(
        "dropout", 0.0, 0.5
    )

    scheduler_type = trial.suggest_categorical(
        "scheduler", ["cosine", "step", "none"]
    )

    if MODEL_NAME == "cnn":
        channels_choice = trial.suggest_categorical(
            "channels_config", ["small", "medium", "large"]
        )
        channels_map = {
            "small": [16, 32, 64, 128],
            "medium": [32, 64, 128, 256],
            "large": [64, 128, 256, 512],
        }
        channels = channels_map[channels_choice]
        extra_hparams = {"channels": channels, "dropout": dropout}
        n_mels = 64

    elif MODEL_NAME == "lstm":
        hidden_size = trial.suggest_categorical(
            "hidden_size", [128, 256, 512]
        )
        num_layers = trial.suggest_int("num_layers", 1, 4)
        extra_hparams = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        n_mels = 64

    elif MODEL_NAME == "resnet":
        width_mult = trial.suggest_float(
            "width_mult", 0.25, 2.0, step=0.25
        )
        extra_hparams = {"width_mult": width_mult, "dropout": dropout}
        n_mels = 64

    elif MODEL_NAME == "transformer":
        d_model = trial.suggest_categorical(
            "d_model", [64, 128, 256]
        )
        nhead = trial.suggest_categorical(
            "nhead", [2, 4, 8]
        )
        if d_model % nhead != 0:
            nhead = max([h for h in [2, 4, 8] if d_model % h == 0])
        num_layers = trial.suggest_int("num_layers", 2, 8)
        extra_hparams = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        n_mels = 64

    else:
        extra_hparams = {"dropout": dropout}
        n_mels = 64

    hparams = {
        "feature_type": feature_type,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "epochs": N_EPOCHS_PER_TRIAL,
        "scheduler": scheduler_type,
        "step_size": 10,
        "gamma": 0.1,
        **extra_hparams,
    }

    train_loader, val_loader, label_map, audio_cfg = get_cached_data(batch_size)
    num_classes = len(label_map)

    gpu_transform = get_feature_transform(feature_type, audio_cfg).to(device)

    model_cls = MODEL_CLASSES[MODEL_NAME]
    model_manager = model_cls(hparams=hparams)
    model = model_manager.create_model(num_classes=num_classes, n_mels=n_mels)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[trial {trial.number}] params={total_params:,}  "
          f"lr={learning_rate:.2e}  bs={batch_size}  "
          f"feat={feature_type}  drop={dropout:.2f}")

    optimizer = model_manager.get_optimizer(model)
    scheduler = model_manager.get_scheduler(optimizer, num_epochs=N_EPOCHS_PER_TRIAL)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0.0

    for epoch in range(1, N_EPOCHS_PER_TRIAL + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            inputs = gpu_transform(inputs)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        val_acc = evaluate(model, val_loader, criterion, device, gpu_transform)
        avg_loss = running_loss / len(train_loader)
        epoch_time = time.time() - t0

        if scheduler is not None:
            scheduler.step()

        best_val_acc = max(best_val_acc, val_acc)

        print(f"  epoch {epoch:2d}/{N_EPOCHS_PER_TRIAL}  "
              f"loss={avg_loss:.4f}  val_acc={val_acc:.2f}%  "
              f"best={best_val_acc:.2f}%  [{epoch_time:.1f}s]")

        trial.report(val_acc, epoch)
        if trial.should_prune():
            print(f"  ↳ pruned at epoch {epoch}")
            raise optuna.exceptions.TrialPruned()

        if val_acc > 95.0:
            print(f"  ↳ early stop: accuracy > 95%")
            break

    return best_val_acc


def main():
    """
    Запускает процесс подбора гиперпараметров с помощью Optuna.
    """
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{MODEL_NAME}_hpo",
        storage=f"sqlite:///hpo_{MODEL_NAME}.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,
            interval_steps=3,
        ),
    )

    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            timeout=TIMEOUT_HOURS * 3600,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем. Результаты сохранены.")

    if len(study.trials) == 0:
        print("Нет успешных испытаний.")
        return

    best = study.best_trial
    print("\n" + "=" * 60)
    print("Лучшие гиперпараметры:")
    print("=" * 60)
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    print(f"\nЛучшая валидационная точность: {best.value:.2f}%")

    best_params_path = f"best_params_{MODEL_NAME}.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump({
            "params": best.params,
            "val_acc": best.value,
        }, f, indent=2)
    print(f"Параметры сохранены в {best_params_path}")

    print("\nИспользование в train.py:")
    print("MODEL_HPARAMS = {")
    for key, value in best.params.items():
        if isinstance(value, str):
            print(f'    "{key}": "{value}",')
        else:
            print(f'    "{key}": {value},')
    print("}")


if __name__ == "__main__":
    main()