import os
import shutil
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

"""
Загрузка и разбиение датасета Google Speech Commands.
Поддерживает официальное разбиение (testing_list.txt/validation_list.txt),
случайное разбиение по соотношениям, сохранение списка тестовых файлов и копирование тестовых аудио в отдельную папку.
"""

"""
@sample_rate : int - частота дискретизации.
@target_length : int - фиксированная длина аудио в сэмплах.
@n_fft : int - размер окна для STFT.
@hop_length : int - шаг между окнами для STFT.
@n_mels : int - количество мел-фильтров для спектрограммы.
"""
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "target_length": 16000,
    "n_fft": 400,
    "hop_length": 160,
    "n_mels": 64,
}


def get_mel_transform(audio_config: Optional[dict] = None) -> torch.nn.Sequential:
    """
    Возвращает конвейер MelSpectrogram → AmplitudeToDB.

    @audio_config : dict или None - параметры спектрограммы (по умолчанию AUDIO_CONFIG).

    Возвращает: nn.Sequential - последовательность преобразований.
    """
    cfg = audio_config or AUDIO_CONFIG
    return torch.nn.Sequential(
        T.MelSpectrogram(
            sample_rate=cfg["sample_rate"],
            n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"],
            n_mels=cfg["n_mels"],
        ),
        T.AmplitudeToDB(top_db=80),
    )


class SpeechCommandsDataset(Dataset):
    """
    Torch Dataset для Google Speech Commands.

    Возвращает (features, label_idx), где features - либо исходный сигнал (1, target_length)
    при transform=None, либо мел-спектрограмма (1, n_mels, time_frames) при
    transform=get_mel_transform().
    """

    def __init__(
        self,
        root_dir: str,
        file_list: List[Tuple[str, int]],
        transform=None,
        target_length: int = 16000,
    ):
        self.root_dir = root_dir
        self.file_list = file_list
        self.transform = transform
        self.target_length = target_length

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        rel_path, label = self.file_list[idx]
        wav_path = os.path.join(self.root_dir, rel_path)

        waveform, sr = torchaudio.load(wav_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != 16000:
            waveform = T.Resample(sr, 16000)(waveform)

        n = waveform.shape[1]
        if n < self.target_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.target_length - n))
        else:
            waveform = waveform[:, : self.target_length]

        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform, label


def get_classes(data_dir: str, include_classes: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Строит словарь {название_класса: индекс} по подпапкам в data_dir.
    Папки, начинающиеся с '_' (например, _background_noise_), исключаются.
    @include_classes ограничивает выборку указанным списком.
    """
    all_classes = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith("_")
    )

    if include_classes is not None:
        missing = set(include_classes) - set(all_classes)
        if missing:
            raise ValueError(f"Запрошенные классы отсутствуют в датасете: {missing}")
        all_classes = [c for c in all_classes if c in include_classes]

    return {cls: idx for idx, cls in enumerate(all_classes)}


def build_splits(
    data_dir: str,
    label_map: Dict[str, int],
    use_official_split: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Формирует списки (train_files, val_files, test_files) как (относительный_путь, метка).

    @data_dir : str - путь к датасету.
    @label_map : dict - словарь меток.
    @use_official_split : bool - использовать официальное разбиение или случайное.
    @train_ratio, val_ratio : float - доли при случайном разбиении.
    @seed : int - зерно генератора.

    Возвращает: tuple из трёх списков файлов.
    """
    if use_official_split:
        return _official_split(data_dir, label_map)
    return _random_split(data_dir, label_map, train_ratio, val_ratio, seed)


def _official_split(
    data_dir: str,
    label_map: Dict[str, int],
) -> Tuple[List, List, List]:
    """Читает официальные testing_list.txt и validation_list.txt и распределяет файлы."""
    test_path = os.path.join(data_dir, "testing_list.txt")
    val_path = os.path.join(data_dir, "validation_list.txt")

    with open(test_path, encoding="utf-8") as f:
        test_set = {line.strip() for line in f if line.strip()}
    with open(val_path, encoding="utf-8") as f:
        val_set = {line.strip() for line in f if line.strip()}

    train_files, val_files, test_files = [], [], []

    for cls_name, label_idx in label_map.items():
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.endswith(".wav"):
                continue
            rel_fwd = f"{cls_name}/{fname}"
            rel_os = os.path.join(cls_name, fname)

            if rel_fwd in test_set:
                test_files.append((rel_os, label_idx))
            elif rel_fwd in val_set:
                val_files.append((rel_os, label_idx))
            else:
                train_files.append((rel_os, label_idx))

    return train_files, val_files, test_files


def _random_split(
    data_dir: str,
    label_map: Dict[str, int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List, List, List]:
    """Cлучайное разбиение по заданным долям."""
    rng = np.random.default_rng(seed)

    train_files, val_files, test_files = [], [], []

    for cls_name, label_idx in label_map.items():
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        files = [
            (os.path.join(cls_name, f), label_idx)
            for f in sorted(os.listdir(cls_dir))
            if f.endswith(".wav")
        ]
        rng.shuffle(files)

        n = len(files)
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, int(n * (1.0 - train_ratio - val_ratio)))
        n_train = n - n_val - n_test

        train_files.extend(files[:n_train])
        val_files.extend(files[n_train: n_train + n_val])
        test_files.extend(files[n_train + n_val:])

    return train_files, val_files, test_files


def save_test_files_list(
    test_files: List[Tuple[str, int]],
    output_path: str,
    label_map: Optional[Dict[str, int]] = None,
) -> None:
    """
    Сохраняет пути тестовых файлов в .txt (по одному на строку, прямые слэши).
    Опционально добавляет метку класса через табуляцию.
    """
    idx_to_class: Dict[int, str] = {}
    if label_map is not None:
        idx_to_class = {v: k for k, v in label_map.items()}

    with open(output_path, "w", encoding="utf-8") as f:
        for rel_path, label_idx in test_files:
            rel_fwd = rel_path.replace(os.sep, "/")
            if idx_to_class:
                f.write(f"{rel_fwd}\t{idx_to_class[label_idx]}\n")
            else:
                f.write(rel_fwd + "\n")

    print(f"[dataset] Saved {len(test_files)} test file paths → {output_path}")


def copy_test_files(
    test_files: List[Tuple[str, int]],
    data_dir: str,
    output_dir: str,
) -> None:
    """Копирует тестовые аудиофайлы в output_dir с сохранением структуры подпапок классов."""
    os.makedirs(output_dir, exist_ok=True)
    for rel_path, _ in test_files:
        src = os.path.join(data_dir, rel_path)
        dst = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    print(f"[dataset] Copied {len(test_files)} test files → {output_dir}")


def get_dataloaders(
    data_dir: str,
    transform,
    dataset_config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Создаёт и возвращает (train_loader, val_loader, test_loader, label_map).

    Ожидаемые ключи в dataset_config:
        classes            : list[str] или None
        use_official_split : bool
        train_ratio        : float
        val_ratio          : float
        seed               : int
        target_length      : int
        batch_size         : int
        num_workers        : int
        save_test_files    : bool
        test_files_output  : str - путь для сохранения списка/копирования файлов
        copy_test_files    : bool - True → копировать файлы, False → сохранить список путей
    """
    label_map = get_classes(data_dir, dataset_config.get("classes"))

    train_files, val_files, test_files = build_splits(
        data_dir=data_dir,
        label_map=label_map,
        use_official_split=dataset_config.get("use_official_split", True),
        train_ratio=dataset_config.get("train_ratio", 0.8),
        val_ratio=dataset_config.get("val_ratio", 0.1),
        seed=dataset_config.get("seed", 42),
    )

    if dataset_config.get("save_test_files", True):
        output = dataset_config.get("test_files_output", "test_files.txt")
        if dataset_config.get("copy_test_files", False):
            copy_test_files(test_files, data_dir, output)
        else:
            save_test_files_list(test_files, output, label_map)

    target_length = dataset_config.get("target_length", AUDIO_CONFIG["target_length"])
    batch_size = dataset_config.get("batch_size", 64)
    num_workers = dataset_config.get("num_workers", 0)

    train_ds = SpeechCommandsDataset(data_dir, train_files, transform, target_length)
    val_ds = SpeechCommandsDataset(data_dir, val_files, transform, target_length)
    test_ds = SpeechCommandsDataset(data_dir, test_files, transform, target_length)

    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent,
    )

    print(f"[dataset] Classes : {len(label_map)}")
    print(f"[dataset] Train   : {len(train_files):,}")
    print(f"[dataset] Val     : {len(val_files):,}")
    print(f"[dataset] Test    : {len(test_files):,}")

    return train_loader, val_loader, test_loader, label_map