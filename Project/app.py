import os
import sys
import time
import threading
import tkinter as tk
from tkinter import filedialog
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)
from models.cnn import CNNModel
from models.lstm import BiLSTMClassifier
from models.resnet import ResNetModel
from models.transformer import TransformerClassifier

"""
Приложение с графическим интерфейсом для распознавания речевых команд.
Позволяет загружать чекпойнт обученной модели, выбирать аудиофайл
или записывать звук с микрофона, воспроизводить аудио и выполнять
инференс с отображением топ-предсказания и полного списка классов.
"""

MODEL_CLASSES = {
    "cnn": CNNModel,
    "lstm": BiLSTMClassifier,
    "resnet": ResNetModel,
    "transformer": TransformerClassifier,
}


class InferenceEngine:
    """
    Движок инференса: загружает чекпойнт, создаёт модель и выполняет предсказание.
    """

    def __init__(self):
        self.model = None
        self.label_map = None
        self.idx_to_cls = None
        self.audio_cfg = None
        self.transform = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = None

    def load_checkpoint(self, ckpt_path: str) -> str:
        """
        Загружает чекпойнт, восстанавливает модель и аудиопреобразование.
        Возвращает информационную строку о модели.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        model_name = ckpt["model_name"]
        num_classes = ckpt["num_classes"]
        label_map = ckpt["label_map"]
        hparams = ckpt.get("hparams", {})
        audio_cfg = ckpt.get("audio_config", {
            "sample_rate": 16000, "target_length": 16000,
            "n_fft": 400, "hop_length": 160, "n_mels": 64,
        })

        model_cls = MODEL_CLASSES[model_name]
        model_manager = model_cls(hparams=hparams)

        n_mels = audio_cfg.get("n_mels", 64)
        model = model_manager.create_model(num_classes=num_classes, n_mels=n_mels)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()

        self.model = model
        self.label_map = label_map
        self.idx_to_cls = {v: k for k, v in label_map.items()}
        self.audio_cfg = audio_cfg
        self.ckpt_path = ckpt_path

        self.transform = torch.nn.Sequential(
            T.MelSpectrogram(
                sample_rate=audio_cfg["sample_rate"],
                n_fft=audio_cfg["n_fft"],
                hop_length=audio_cfg["hop_length"],
                n_mels=n_mels,
            ),
            T.AmplitudeToDB(top_db=80),
        ).to(self.device)

        test_acc = ckpt.get("test_acc", None)
        test_acc_str = f"{test_acc:.2f}%" if test_acc is not None else "?"
        return (
            f"Model: {model_name.upper()}  |  "
            f"Classes: {num_classes}  |  "
            f"Test acc: {test_acc_str}  |  "
            f"Device: {self.device}"
        )

    @torch.no_grad()
    def predict(self, wav_path: str):
        """
        Возвращает список всех классов с вероятностями, отсортированный по убыванию.
        Каждый элемент — кортеж (название_класса, вероятность).
        """
        if self.model is None:
            raise RuntimeError("Модель не загружена.")

        sr_target = self.audio_cfg["sample_rate"]
        tgt_len = self.audio_cfg["target_length"]

        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sr_target:
            waveform = T.Resample(sr, sr_target)(waveform)

        n = waveform.shape[1]
        if n < tgt_len:
            waveform = torch.nn.functional.pad(waveform, (0, tgt_len - n))
        else:
            waveform = waveform[:, :tgt_len]

        waveform = waveform.unsqueeze(0).to(self.device)
        mel = self.transform(waveform)
        logits = self.model(mel)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        order = np.argsort(probs)[::-1]
        return [(self.idx_to_cls[i], float(probs[i])) for i in order]



"""
@_ACCENT, _BG, _BG2, _FG, _FG_DIM, _GREEN, _RED, _YELLOW - цветовая палитра интерфейса.
@_ROW_H - высота строки в списке классов.
@_LPAD, _RPAD - отступы слева и справа для текста в строке класса.
"""
_ACCENT = "#4A90D9"
_BG = "#1E1E2E"
_BG2 = "#2A2A3E"
_FG = "#CDD6F4"
_FG_DIM = "#6C7086"
_GREEN = "#A6E3A1"
_RED = "#F38BA8"
_YELLOW = "#F9E2AF"

_ROW_H = 34
_LPAD = 130
_RPAD = 54


class App:
    """
    Основной класс GUI-приложения.
    Создаёт интерфейс, управляет загрузкой файлов, воспроизведением,
    записью с микрофона и запуском инференса.
    """

    def __init__(self, root):
        self.root = root
        self.engine = InferenceEngine()
        self.wav_path = None
        self.audio_data = None
        self.audio_sr = None
        self._results = []
        self._resize_job = None
        self._recording = False

        root.title("Распознаватель речевых команд")
        root.geometry("820x700")
        root.minsize(620, 520)
        root.configure(bg=_BG)
        root.resizable(True, True)

        self._build_ui()

    def _build_ui(self):
        """Собирает все элементы интерфейса."""
        top = tk.Frame(self.root, bg=_BG)
        top.pack(side="top", fill="x")

        tk.Label(top, text="Распознаватель речевых команд",
                 font=("Segoe UI", 15, "bold"), bg=_BG, fg=_ACCENT,
                 anchor="w").pack(fill="x", padx=16, pady=(14, 4))

        row = tk.Frame(top, bg=_BG)
        row.pack(fill="x", padx=16, pady=3)
        tk.Label(row, text="Checkpoint:", bg=_BG, fg=_FG,
                 font=("Segoe UI", 10)).pack(side="left")
        self.ckpt_var = tk.StringVar(value="(not loaded)")
        self._make_entry(row, self.ckpt_var).pack(side="left", padx=(8, 6), fill="x", expand=True)
        self._make_btn(row, "Browse…", self._browse_ckpt).pack(side="left")

        self.model_info = tk.Label(top, text="", bg=_BG, fg=_FG_DIM,
                                   font=("Segoe UI", 9), anchor="w")
        self.model_info.pack(fill="x", padx=16, pady=(0, 2))

        rec_frame = tk.Frame(top, bg=_BG2,
                             highlightbackground=_ACCENT, highlightthickness=1)
        rec_frame.pack(fill="x", padx=16, pady=(6, 4))
        tk.Label(rec_frame, text="Запись с микрофона", bg=_BG2, fg=_FG,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8, pady=(6, 2))
        rec_row = tk.Frame(rec_frame, bg=_BG2)
        rec_row.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(rec_row, text="Имя файла:", bg=_BG2, fg=_FG,
                 font=("Segoe UI", 10)).pack(side="left")
        self.rec_name_var = tk.StringVar(value="my_recording")
        self._make_entry(rec_row, self.rec_name_var, width=20).pack(side="left", padx=(6, 2))
        tk.Label(rec_row, text=".wav", bg=_BG2, fg=_FG_DIM,
                 font=("Segoe UI", 10)).pack(side="left", padx=(0, 10))
        self.btn_rec = self._make_btn(rec_row, "● Записать (1с)",
                                      self._record_audio, w=18)
        self.btn_rec.pack(side="left", padx=(0, 10))
        self.rec_status_var = tk.StringVar(value="")
        tk.Label(rec_row, textvariable=self.rec_status_var, bg=_BG2, fg=_FG_DIM,
                 font=("Segoe UI", 9)).pack(side="left")

        row2 = tk.Frame(top, bg=_BG)
        row2.pack(fill="x", padx=16, pady=3)
        tk.Label(row2, text="Путь к файлу:", bg=_BG, fg=_FG,
                 font=("Segoe UI", 10)).pack(side="left")
        self.path_var = tk.StringVar()
        pe = self._make_entry(row2, self.path_var)
        pe.pack(side="left", padx=(8, 6), fill="x", expand=True)
        pe.bind("<Return>", lambda _: self._load_wav_from_entry())
        self._make_btn(row2, "Open…", self._browse_wav).pack(side="left")

        btns = tk.Frame(top, bg=_BG)
        btns.pack(pady=(4, 0))
        self.btn_play = self._make_btn(btns, "▶  Прослушать", self._play_audio,
                                       state="disabled", w=14)
        self.btn_play.pack(side="left", padx=8)
        self.btn_infer = self._make_btn(btns, "🔍  Распознать", self._run_infer,
                                        state="disabled", w=16, accent=True)
        self.btn_infer.pack(side="left", padx=8)
        self.btn_stop = self._make_btn(btns, "⏹  Остановить", self._stop_audio,
                                       state="disabled", w=10)
        self.btn_stop.pack(side="left", padx=8)

        self.status_var = tk.StringVar(value="Загружайте чекпойнт модели и аудиофайла для распознавания.")
        tk.Label(top, textvariable=self.status_var, bg=_BG, fg=_FG_DIM,
                 font=("Segoe UI", 9), anchor="w").pack(fill="x", padx=16, pady=(4, 0))

        tk.Frame(top, bg=_ACCENT, height=1).pack(fill="x", padx=16, pady=(8, 0))

        res_row = tk.Frame(top, bg=_BG)
        res_row.pack(fill="x", padx=16, pady=(8, 0))
        self.result_word = tk.Label(res_row, text="—", bg=_BG, fg=_ACCENT,
                                    font=("Segoe UI", 38, "bold"))
        self.result_word.pack(side="left")
        self.result_conf = tk.Label(res_row, text="", bg=_BG, fg=_FG_DIM,
                                    font=("Segoe UI", 14))
        self.result_conf.pack(side="left", padx=(16, 0))

        tk.Label(top, text="Все классы",
                 bg=_BG, fg=_FG_DIM, font=("Segoe UI", 9), anchor="w",
                 ).pack(fill="x", padx=16, pady=(6, 2))

        lf = tk.Frame(self.root, bg=_BG)
        lf.pack(side="bottom", fill="both", expand=True, padx=16, pady=(0, 10))
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)

        self.list_canvas = tk.Canvas(lf, bg=_BG, highlightthickness=0)
        self.list_canvas.grid(row=0, column=0, sticky="nsew")

        self.vscroll = tk.Scrollbar(lf, orient="vertical",
                                    command=self.list_canvas.yview,
                                    bg=_BG2, troughcolor=_BG)
        self.vscroll.grid(row=0, column=1, sticky="ns")
        self.list_canvas.configure(yscrollcommand=self.vscroll.set)

        self.list_canvas.bind("<MouseWheel>", self._on_wheel)
        self.list_canvas.bind("<Button-4>", self._on_wheel)
        self.list_canvas.bind("<Button-5>", self._on_wheel)
        self.list_canvas.bind("<Configure>", self._on_list_resize)

    def _make_btn(self, parent, text, cmd, state="normal", w=12, accent=False):
        """Создаёт кнопку в стиле приложения."""
        return tk.Button(
            parent, text=text, command=cmd,
            bg=_ACCENT if accent else _BG2,
            fg="white" if accent else _FG,
            activebackground=_ACCENT, activeforeground="white",
            relief="flat", font=("Segoe UI", 10),
            width=w, cursor="hand2", state=state,
        )

    def _make_entry(self, parent, var, width=40):
        """Создаёт поле ввода в стиле приложения."""
        return tk.Entry(
            parent, textvariable=var, width=width,
            bg=_BG2, fg=_FG, insertbackground=_FG,
            relief="flat", font=("Consolas", 9),
        )

    def _browse_ckpt(self):
        """Открывает диалог выбора файла чекпойнта и загружает его."""
        path = filedialog.askopenfilename(
            title="Select checkpoint (best.pt)",
            initialdir=os.path.join(_DIR, "checkpoints"),
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self._load_ckpt(path)

    def _load_ckpt(self, path: str):
        """Загружает чекпойнт и обновляет интерфейс."""
        self.ckpt_var.set(path)
        self._set_status("Loading checkpoint…")
        try:
            info = self.engine.load_checkpoint(path)
            self.model_info.config(text=info, fg=_GREEN)
            self._set_status("Checkpoint loaded. Drop a .wav or use Open.")
            if self.wav_path:
                self.btn_infer.config(state="normal")
        except Exception as exc:
            self.model_info.config(text=str(exc), fg=_RED)
            self._set_status("Failed to load checkpoint.")

    def _browse_wav(self):
        """Открывает диалог выбора WAV-файла и загружает его."""
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if path:
            self.path_var.set(path)
            self._load_wav(path)

    def _load_wav_from_entry(self):
        """Загружает WAV по пути, введённому в текстовое поле."""
        path = self.path_var.get().strip().strip("{}")
        if path:
            self._load_wav(path)

    def _load_wav(self, path: str):
        """Читает WAV-файл и сохраняет аудиоданные."""
        if not os.path.isfile(path):
            self._set_status(f"File not found: {path}")
            return
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=False)
            self.audio_data = data
            self.audio_sr = sr
            self.wav_path = path
            fname = os.path.basename(path)
            dur = (data.shape[0] if data.ndim > 1 else len(data)) / sr
            self._set_status(f"Loaded: {fname}  ({sr} Hz,  {dur:.2f} s)")
            self.btn_play.config(state="normal")
            if self.engine.model is not None:
                self.btn_infer.config(state="normal")
            self.result_word.config(text="—", fg=_ACCENT)
            self.result_conf.config(text="")
            self._results = []
            self.list_canvas.delete("all")
            self.list_canvas.config(scrollregion=(0, 0, 0, 0))
        except Exception as exc:
            self._set_status(f"Error loading file: {exc}")

    def _play_audio(self):
        """Воспроизводит загруженный аудиофайл в отдельном потоке."""
        if self.audio_data is None:
            return
        self._stop_audio()
        self.btn_stop.config(state="normal")
        self._set_status("Прослушивание")

        def _worker():
            try:
                sd.play(self.audio_data, self.audio_sr)
                sd.wait()
            finally:
                self.root.after(0, lambda: self.btn_stop.config(state="disabled"))
                self.root.after(0, lambda: self._set_status("Прослушивание завершено."))

        threading.Thread(target=_worker, daemon=True).start()

    def _stop_audio(self):
        """Останавливает текущее воспроизведение."""
        sd.stop()
        self.btn_stop.config(state="disabled")

    def _record_audio(self):
        """Записывает 1 секунду с микрофона, сохраняет в папку app_test и загружает."""
        if self._recording:
            return
        name = self.rec_name_var.get().strip()
        name = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
        if not name:
            name = "recording"
        app_test_dir = os.path.join(_DIR, "app_test")
        os.makedirs(app_test_dir, exist_ok=True)
        save_path = os.path.join(app_test_dir, f"{name}.wav")

        sr = 16000
        n_frames = 16000
        self._recording = True
        self.btn_rec.config(state="disabled")

        def _worker():
            try:
                for i in (3, 2, 1):
                    self.root.after(0, lambda i=i: self.rec_status_var.set(f"Запись через {i}…"))
                    time.sleep(1)
                self.root.after(0, lambda: self.rec_status_var.set("🔴 Идёт запись…"))
                audio = sd.rec(n_frames, samplerate=sr, channels=1, dtype="float32")
                sd.wait()
                sf.write(save_path, audio.flatten(), sr)

                def _done():
                    self.rec_status_var.set(f"✔ Сохранено: {os.path.basename(save_path)}")
                    self.path_var.set(save_path)
                    self._load_wav(save_path)

                self.root.after(0, _done)
            except Exception as exc:
                msg = str(exc)
                self.root.after(0, lambda: self.rec_status_var.set(f"Ошибка: {msg}"))
            finally:
                self._recording = False
                self.root.after(0, lambda: self.btn_rec.config(state="normal"))

        threading.Thread(target=_worker, daemon=True).start()

    def _run_infer(self):
        """Запускает инференс в фоновом потоке и отображает результаты."""
        if not self.wav_path or not self.engine.model:
            return
        self._set_status("Running inference…")
        self.btn_infer.config(state="disabled")

        def _worker():
            try:
                results = self.engine.predict(self.wav_path)
                self.root.after(0, lambda: self._show_results(results))
            except Exception as exc:
                msg = str(exc)
                self.root.after(0, lambda: self._set_status(f"Ошибка: {msg}"))
            finally:
                self.root.after(0, lambda: self.btn_infer.config(state="normal"))

        threading.Thread(target=_worker, daemon=True).start()

    def _show_results(self, results):
        """Обновляет интерфейс с результатами предсказания."""
        if not results:
            return
        self._results = results
        top_word, top_prob = results[0]
        color = _GREEN if top_prob >= 0.7 else (_YELLOW if top_prob >= 0.4 else _RED)
        self.result_word.config(text=top_word.upper(), fg=color)
        self.result_conf.config(text=f"Уверенность: {top_prob * 100:.1f}%", fg=color)
        self._set_status(f"Предсказание: {top_word}  ({top_prob * 100:.1f}%)")
        self._redraw_list()

    def _on_list_resize(self, _=None):
        """Планирует перерисовку списка при изменении размера с задержкой."""
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(60, self._redraw_list)

    def _on_wheel(self, event):
        """Обрабатывает прокрутку колёсиком мыши в списке классов."""
        if event.num == 4:
            self.list_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.list_canvas.yview_scroll(1, "units")
        else:
            self.list_canvas.yview_scroll(-int(event.delta / 120), "units")

    def _redraw_list(self):
        """Перерисовывает скроллируемый список всех классов с прогресс-барами."""
        if not self._results:
            return

        c = self.list_canvas
        c.delete("all")

        W = c.winfo_width()
        if W <= 1:
            W = 750

        results = self._results
        n = len(results)
        total_h = n * _ROW_H + 8

        bar_x0 = _LPAD
        bar_x1 = W - _RPAD - 8
        bar_max_w = max(1, bar_x1 - bar_x0)

        c.config(scrollregion=(0, 0, W, total_h))

        for i, (cls, prob) in enumerate(results):
            y_mid = 4 + i * _ROW_H + _ROW_H // 2
            y_top = y_mid - 9
            y_bot = y_mid + 9

            if i == 0:
                text_color = _GREEN
                name_font = ("Segoe UI", 10, "bold")
            elif prob >= 0.05:
                text_color = _ACCENT
                name_font = ("Segoe UI", 10)
            else:
                text_color = _FG_DIM
                name_font = ("Segoe UI", 9)

            row_bg = _BG2 if i % 2 == 0 else _BG
            c.create_rectangle(0, y_top - 4, W, y_bot + 4,
                               fill=row_bg, outline="")

            c.create_text(20, y_mid, text=str(i + 1), anchor="center",
                          fill=_FG_DIM, font=("Segoe UI", 8))

            c.create_text(34, y_mid, text=cls, anchor="w",
                          fill=text_color, font=name_font)

            track_bg = _BG if i % 2 == 0 else _BG2
            c.create_rectangle(bar_x0, y_top, bar_x1, y_bot,
                               fill=track_bg, outline="")

            bw = max(2, int(bar_max_w * prob))
            c.create_rectangle(bar_x0, y_top, bar_x0 + bw, y_bot,
                               fill=text_color, outline="")

            c.create_text(bar_x1 + 6, y_mid,
                          text=f"{prob * 100:.1f}%", anchor="w",
                          fill=_FG_DIM, font=("Segoe UI", 9))

    def _set_status(self, msg: str):
        """Устанавливает текст в статусной строке."""
        self.status_var.set(msg)
        self.root.update_idletasks()


def main():
    root = tk.Tk()
    app = App(root)

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        root.after(200, lambda: app._load_ckpt(sys.argv[1]))

    root.mainloop()


if __name__ == "__main__":
    main()