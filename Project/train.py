import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import dataset as ds
from models.audio_features import get_feature_transform
from models.cnn import CNNModel
from models.lstm import BiLSTMClassifier
from models.resnet import ResNetModel
from models.transformer import TransformerClassifier

"""
Поддерживаемые модели: "cnn", "lstm", "resnet", "transformer".
Поддерживаемые признаки: "mel", "fbank", "stft", "wavelet".
"""

MODEL_NAME = "transformer"
MODEL_HPARAMS = {
    "feature_type": "stft",
}

DATASET_CONFIG = {
    "data_dir": "data_set",
    "classes": None,
    "use_official_split": False,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "seed": 67,
    "num_workers": 4,
    "save_test_files": True,
    "copy_test_files": False,
    "test_files_output": "test_files.txt",
}

TRAINING_CONFIG = {
    "checkpoint_dir": "checkpoints",
    "save_best": True,
    "save_every_n_epochs": 0,
    "log_every_batches": 50,
}

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


MODEL_CLASSES = {
    "cnn": CNNModel,
    "lstm": BiLSTMClassifier,
    "resnet": ResNetModel,
    "transformer": TransformerClassifier,
}


def _fmt_time(seconds: float) -> str:
    """
    Форматирует время в строку ЧЧ:ММ:СС.

    @param seconds : float - количество секунд.

    Возвращает: str - строка формата "чч:мм:сс".
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@torch.no_grad()
def evaluate(model, loader, criterion, device, gpu_transform=None):
    """
    Оценивает модель на заданном наборе данных.

    @param model : nn.Module - модель.
    @param loader : DataLoader - загрузчик данных.
    @param criterion : функция потерь.
    @param device : torch.device - устройство для вычислений.
    @param gpu_transform : Callable или None - преобразование данных на GPU.

    Возвращает: tuple - (средняя потеря, точность в процентах).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if gpu_transform is not None:
            inputs = gpu_transform(inputs)
        outputs = model(inputs)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate_with_cm(model, loader, criterion, device, num_classes: int, gpu_transform=None):
    """
    Оценивает модель и строит матрицу ошибок.

    @param model : nn.Module - модель.
    @param loader : DataLoader - загрузчик данных.
    @param criterion : функция потерь.
    @param device : torch.device - устройство для вычислений.
    @param num_classes : int - количество классов.
    @param gpu_transform : Callable или None - преобразование данных на GPU.

    Возвращает: tuple - (средняя потеря, точность в процентах, матрица ошибок numpy).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if gpu_transform is not None:
            inputs = gpu_transform(inputs)
        outputs = model(inputs)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            cm[t, p] += 1
    return total_loss / total, 100.0 * correct / total, cm


def save_checkpoint(path, model, optimizer, epoch, val_acc, model_name,
                    label_map, hparams, audio_config):
    """
    Сохраняет чекпойнт модели и метаданные для инференса.

    @param path : str - путь для сохранения.
    @param model : nn.Module - модель.
    @param optimizer : Optimizer - оптимизатор.
    @param epoch : int - номер эпохи.
    @param val_acc : float - точность на валидации.
    @param model_name : str - название архитектуры.
    @param label_map : dict - словарь меток.
    @param hparams : dict - гиперпараметры модели.
    @param audio_config : dict - конфигурация аудио.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "model_name": model_name,
            "num_classes": len(label_map),
            "label_map": label_map,
            "hparams": hparams,
            "audio_config": audio_config,
        },
        path,
    )


def plot_training_curves(history: dict, ckpt_dir: str, model_name: str) -> str:
    """
    Рисует и сохраняет графики обучения: точность, потери, FP на валидации.

    @param history : dict - история метрик по эпохам.
    @param ckpt_dir : str - папка для сохранения графика.
    @param model_name : str - название модели для заголовка.

    Возвращает: str - путь к сохранённому PNG.
    """
    epochs = list(range(1, len(history["train_acc"]) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training history — {model_name.upper()}", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, history["train_acc"], label="Train", linewidth=1.8)
    ax.plot(epochs, history["val_acc"], label="Val", linewidth=1.8, linestyle="--")
    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_acc = max(history["val_acc"])
    ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1, label=f"Best epoch ({best_epoch})")
    ax.annotate(f"{best_acc:.1f}%",
                xy=(best_epoch, best_acc),
                xytext=(best_epoch + max(1, len(epochs) * 0.03), best_acc - 3),
                fontsize=8, color="gray")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy, %")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax = axes[1]
    ax.plot(epochs, history["train_loss"], label="Train", linewidth=1.8)
    ax.plot(epochs, history["val_loss"], label="Val", linewidth=1.8, linestyle="--")
    ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1, label=f"Best epoch ({best_epoch})")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax = axes[2]
    ax.plot(epochs, history["val_fp"], color="tomato", linewidth=1.8, label="Val FP (total)")
    ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1, label=f"Best epoch ({best_epoch})")
    ax.set_title("False Positives (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FP count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    out_path = os.path.join(ckpt_dir, "training_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrix(cm: np.ndarray, label_map: dict, ckpt_dir: str,
                          title: str = "Confusion Matrix (test set)") -> str:
    """
    Рисует и сохраняет матрицу ошибок.

    @param cm : np.ndarray - матрица ошибок (counts).
    @param label_map : dict - словарь меток.
    @param ckpt_dir : str - папка для сохранения.
    @param title : str - заголовок графика.

    Возвращает: str - путь к сохранённому PNG.
    """
    n = cm.shape[0]
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums * 100, 0)

    fig_size = max(8, n * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall, %")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=12, fontweight="bold")

    if n <= 40:
        for i in range(n):
            for j in range(n):
                if cm[i, j] > 0:
                    color = "white" if cm_norm[i, j] > 60 else "black"
                    ax.text(j, i, str(cm[i, j]),
                            ha="center", va="center", fontsize=5, color=color)

    fig.tight_layout()
    out_path = os.path.join(ckpt_dir, "confusion_matrix.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def train():
    """
    Запускает полный цикл обучения модели на речевых командах.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train] Using device: {device}")
    if device.type == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    model_cls = MODEL_CLASSES[MODEL_NAME]
    model_manager = model_cls(hparams=MODEL_HPARAMS)

    feature_type = model_manager.hparams.get("feature_type", "mel")
    batch_size = model_manager.get_batch_size()
    epochs = model_manager.get_epochs()

    audio_cfg = ds.AUDIO_CONFIG.copy()
    gpu_transform = get_feature_transform(feature_type, audio_cfg).to(device)
    print(f"[train] Feature type: {feature_type}")

    data_dir = os.path.join(_SCRIPT_DIR, DATASET_CONFIG["data_dir"])
    cfg_with_batch = {**DATASET_CONFIG, "batch_size": batch_size, "data_dir": data_dir}
    cfg_with_batch["test_files_output"] = os.path.join(
        _SCRIPT_DIR, DATASET_CONFIG["test_files_output"]
    )

    train_loader, val_loader, test_loader, label_map = ds.get_dataloaders(
        data_dir=data_dir,
        transform=None,
        dataset_config=cfg_with_batch,
    )
    num_classes = len(label_map)
    n_mels = audio_cfg["n_mels"]

    model = model_manager.create_model(num_classes=num_classes, n_mels=n_mels)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[train] Model      : {MODEL_NAME.upper()}")
    print(f"[train] Parameters : {total_params:,}")

    optimizer = model_manager.get_optimizer(model)
    scheduler = model_manager.get_scheduler(optimizer, num_epochs=epochs)

    criterion = nn.CrossEntropyLoss()

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    _model_dir = os.path.join(_SCRIPT_DIR, TRAINING_CONFIG["checkpoint_dir"], MODEL_NAME)
    os.makedirs(_model_dir, exist_ok=True)
    _trial_idx = 1
    while os.path.exists(os.path.join(_model_dir, f"trial_{_trial_idx}")):
        _trial_idx += 1
    ckpt_dir = os.path.join(_model_dir, f"trial_{_trial_idx}", f"{feature_type}")
    os.makedirs(ckpt_dir)
    print(f"[train] Checkpoint dir : {ckpt_dir}")
    best_ckpt = os.path.join(ckpt_dir, "best.pt")
    last_ckpt = os.path.join(ckpt_dir, "last.pt")

    best_val_acc = 0.0
    log_every = TRAINING_CONFIG["log_every_batches"]
    save_every = TRAINING_CONFIG["save_every_n_epochs"]

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_fp": [],
    }

    print(f"\n[train] Training for {epochs} epochs …\n")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        t_epoch = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = gpu_transform(inputs)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            if log_every and batch_idx % log_every == 0:
                avg_loss = running_loss / running_total
                avg_acc = 100.0 * running_correct / running_total
                print(
                    f"  Epoch {epoch:3d}/{epochs}  "
                    f"Batch {batch_idx:4d}/{len(train_loader):4d}  "
                    f"Loss {avg_loss:.4f}  Acc {avg_acc:.2f}%"
                )

        train_loss = running_loss / running_total
        train_acc = 100.0 * running_correct / running_total
        val_loss, val_acc, val_cm = evaluate_with_cm(
            model, val_loader, criterion, device, num_classes, gpu_transform
        )

        val_fp = int(val_cm.sum() - np.trace(val_cm))

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t_epoch
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"Train Loss {train_loss:.4f}  Train Acc {train_acc:.2f}%  "
            f"Val Loss {val_loss:.4f}  Val Acc {val_acc:.2f}%  "
            f"Val FP {val_fp:5d}  [{elapsed:.1f}s]"
        )

        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 4))
        history["val_fp"].append(val_fp)

        if TRAINING_CONFIG["save_best"] and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                best_ckpt, model, optimizer, epoch, val_acc,
                MODEL_NAME, label_map, model_manager.hparams, audio_cfg,
            )
            print(f"  ↳ New best model saved  (val_acc={val_acc:.2f}%)")

        if save_every and epoch % save_every == 0:
            periodic = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
            save_checkpoint(
                periodic, model, optimizer, epoch, val_acc,
                MODEL_NAME, label_map, model_manager.hparams, audio_cfg,
            )

    save_checkpoint(
        last_ckpt, model, optimizer, epochs, val_acc,
        MODEL_NAME, label_map, model_manager.hparams, audio_cfg,
    )

    total_time = time.time() - t_start
    print(f"\n[train] Training complete in {_fmt_time(total_time)}")
    print(f"[train] Best val accuracy : {best_val_acc:.2f}%")

    curves_path = plot_training_curves(history, ckpt_dir, MODEL_NAME)
    print(f"\n[train] Training curves saved → {curves_path}")

    print("\n[train] Evaluating on test set (using best checkpoint) …")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc, test_cm = evaluate_with_cm(
        model, test_loader, criterion, device, num_classes, gpu_transform
    )
    test_fp = int(test_cm.sum() - np.trace(test_cm))
    print(f"[train] Test Loss : {test_loss:.4f}")
    print(f"[train] Test Acc  : {test_acc:.2f}%")
    print(f"[train] Test FP   : {test_fp}")

    cm_path = plot_confusion_matrix(test_cm, label_map, ckpt_dir)
    print(f"[train] Confusion matrix saved → {cm_path}")

    results = {
        "model": MODEL_NAME,
        "feature_type": feature_type,
        "hparams": model_manager.hparams,
        "best_val_acc": round(best_val_acc, 4),
        "test_acc": round(test_acc, 4),
        "test_loss": round(test_loss, 6),
        "test_fp": test_fp,
        "num_classes": num_classes,
        "total_params": total_params,
        "train_time_s": round(total_time, 1),
        "history": history,
    }
    results_path = os.path.join(ckpt_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[train] Results saved → {results_path}")

    ckpt["history"] = history
    ckpt["curves_plot"] = curves_path
    ckpt["cm_plot"] = cm_path
    ckpt["test_acc"] = round(test_acc, 4)
    ckpt["test_fp"] = test_fp
    torch.save(ckpt, best_ckpt)
    print(f"[train] Best checkpoint updated with history → {best_ckpt}")


if __name__ == "__main__":
    train()