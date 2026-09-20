#!/usr/bin/env python3
"""True streaming Vietnamese microphone ASR with hynt Zipformer + sherpa-onnx.

Designed for:
  hynt/Zipformer-30M-RNNT-Streaming-6000h

The recognizer is genuinely streaming (OnlineRecognizer). Silero VAD is used only
for speech start/end decisions. A raw-audio pre-roll buffer is fed to Zipformer
when speech starts so the beginning of short Vietnamese words is not clipped.
"""

from __future__ import annotations

import argparse
import queue
import re
import sys
import time
import unicodedata
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple


SAMPLE_RATE = 16_000
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_DIR / "models" / "Zipformer-30M-RNNT-Streaming-6000h"
DEFAULT_VAD_MODEL = PROJECT_DIR / "models" / "silero_vad.onnx"


# ---------------------------------------------------------------------------
# Vietnamese command-focused post-processing
# ---------------------------------------------------------------------------
# Intentionally small vocabulary: this is not a general Vietnamese spell
# corrector. It only normalizes words used by the smart-home command grammar.
COMMAND_WORDS = (
    "bật", "mở", "tắt", "đóng",
    "quạt", "đèn",
    "hết", "cả", "hai", "và", "đi",
)

COMMAND_FILLERS = {"đi"}

# Frequent ASR confusions observed for our very small command vocabulary.
# Keep this list explicit instead of making fuzzy matching more aggressive,
# otherwise ordinary Vietnamese words could be changed unexpectedly.
COMMAND_CONFUSIONS = {
    "bực": "bật",
    "bậc": "bật",
    # Common ASR variants around the OFF command "đóng".
    "đống": "đóng",
    "đông": "đóng",
}


def _strip_vietnamese_marks(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(
        ch
        for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    ).lower()


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


_COMMAND_ASCII = {}
for _word in COMMAND_WORDS:
    _COMMAND_ASCII.setdefault(_strip_vietnamese_marks(_word), []).append(_word)


def _normalize_command_token(token: str) -> str:
    if token == "2":
        return "hai"
    if token in COMMAND_WORDS:
        return token
    if token in COMMAND_CONFUSIONS:
        return COMMAND_CONFUSIONS[token]

    plain = _strip_vietnamese_marks(token)

    # Missing Vietnamese accents is common in ASR/token post-processing.
    exact = _COMMAND_ASCII.get(plain, [])
    if len(exact) == 1:
        return exact[0]

    # Conservative fuzzy correction: at most one edit and only against command
    # words with the same initial letter. This catches forms such as bậc->bật,
    # tắc->tắt, while avoiding a general-purpose autocorrect.
    if len(plain) >= 2:
        candidates = []
        for word in COMMAND_WORDS:
            target = _strip_vietnamese_marks(word)
            if not target or not plain or target[0] != plain[0]:
                continue
            distance = _edit_distance(plain, target)
            if distance <= 1:
                candidates.append((distance, abs(len(target) - len(plain)), word))

        if candidates:
            candidates.sort()
            best = candidates[0]
            tied = [c for c in candidates if c[:2] == best[:2]]
            if len(tied) == 1:
                return best[2]

    return token


def postprocess_command_text(text: str) -> str:
    """Normalize ASR text only for the Vietnamese smart-home command domain.

    Supported command vocabulary is deliberately small: bật/mở, tắt/đóng,
    quạt/đèn, cả hai/hết cả hai, và, đi. The filler ``đi`` is removed.
    """
    text = unicodedata.normalize("NFC", text).lower().strip()
    if not text:
        return ""

    # Keep Unicode word characters and digits, turn punctuation into spaces.
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    tokens = [_normalize_command_token(tok) for tok in text.split()]
    tokens = [tok for tok in tokens if tok not in COMMAND_FILLERS]
    text = " ".join(tokens)

    # Phrase-level canonicalization used by the command parser.
    text = re.sub(r"\bhết\s+hai\b", "hết cả hai", text)
    text = re.sub(r"\bcả\s+hai\s+cái\b", "cả hai", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ModelFiles(NamedTuple):
    tokens: Path
    encoder: Path
    decoder: Path
    joiner: Path


def parse_device(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "True streaming Vietnamese ASR using "
            "hynt/Zipformer-30M-RNNT-Streaming-6000h"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--vad-model", type=Path, default=DEFAULT_VAD_MODEL)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        choices=(16, 32, 64),
        help="Zipformer encoder chunk size. Larger chunks trade latency for more context.",
    )
    parser.add_argument(
        "--device",
        type=parse_device,
        default=None,
        help="Microphone device number or a unique part of its name",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List audio devices and exit"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Load the ASR and VAD models, then exit without opening the microphone",
    )
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument(
        "--provider",
        default="cpu",
        choices=("cpu", "cuda", "coreml"),
        help="ONNX execution provider",
    )
    parser.add_argument(
        "--decoding-method",
        default="modified_beam_search",
        choices=("greedy_search", "modified_beam_search"),
    )
    parser.add_argument("--max-active-paths", type=int, default=15)
    parser.add_argument(
        "--blank-penalty",
        type=float,
        default=0.25,
        help="Blank-token penalty used by the Zipformer transducer decoder",
    )
    parser.add_argument(
        "--partial-interval",
        type=float,
        default=0.25,
        help="Seconds between partial transcript displays; 0 disables partial text",
    )
    parser.add_argument("--vad-threshold", type=float, default=0.35)
    parser.add_argument("--min-silence", type=float, default=0.8)
    parser.add_argument("--min-speech", type=float, default=0.12)
    parser.add_argument("--max-speech", type=float, default=20.0)
    parser.add_argument(
        "--pre-roll",
        type=float,
        default=0.5,
        help="Raw microphone audio kept before VAD fires, in seconds",
    )
    parser.add_argument(
        "--tail-padding",
        type=float,
        default=0.30,
        help="Extra zeros decoded before finalizing an utterance, in seconds",
    )
    return parser.parse_args()


def import_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Microphone support is unavailable. Install Python dependencies and "
            "on Ubuntu/Debian run: sudo apt install libportaudio2. Original error: "
            f"{exc}"
        ) from exc
    return sd


def import_runtime() -> tuple[Any, Any]:
    try:
        import numpy as np
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError(
            "Missing Python dependencies. Install numpy and sherpa-onnx first."
        ) from exc
    return np, sherpa_onnx


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            return file.read(80).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            )
    except OSError:
        return False


def require_real_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if is_lfs_pointer(path):
        raise RuntimeError(
            f"{label} is only a Git LFS/Xet pointer, not the downloaded model: {path}"
        )


def find_model_files(model_dir: Path, chunk_size: int = 32) -> ModelFiles:
    tokens = model_dir / "tokens.txt"
    if not tokens.is_file():
        # hynt's repository currently ships the token table as config.json.
        tokens = model_dir / "config.json"

    suffix = f"epoch-31-avg-11-chunk-{chunk_size}-left-128.fp16.onnx"
    files = ModelFiles(
        tokens=tokens,
        encoder=model_dir / f"encoder-{suffix}",
        decoder=model_dir / f"decoder-{suffix}",
        joiner=model_dir / f"joiner-{suffix}",
    )

    for label, path in zip(ModelFiles._fields, files):
        require_real_file(path, label)
    return files


def create_recognizer(
    sherpa_onnx: Any, files: ModelFiles, args: argparse.Namespace
) -> Any:
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(files.tokens),
        encoder=str(files.encoder),
        decoder=str(files.decoder),
        joiner=str(files.joiner),
        num_threads=args.num_threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method=args.decoding_method,
        max_active_paths=args.max_active_paths,
        blank_penalty=args.blank_penalty,
        provider=args.provider,
        enable_endpoint_detection=False,
    )


def create_vad(sherpa_onnx: Any, args: argparse.Namespace) -> tuple[Any, int]:
    require_real_file(args.vad_model, "VAD model")
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(args.vad_model)
    config.silero_vad.threshold = args.vad_threshold
    config.silero_vad.min_silence_duration = args.min_silence
    config.silero_vad.min_speech_duration = args.min_speech
    config.silero_vad.max_speech_duration = args.max_speech
    config.sample_rate = SAMPLE_RATE
    window_size = int(config.silero_vad.window_size)
    detector = sherpa_onnx.VoiceActivityDetector(
        config,
        buffer_size_in_seconds=max(30, int(args.max_speech) + 5),
    )
    return detector, window_size


def decode_ready(recognizer: Any, stream: Any) -> None:
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)


def finalize_stream(np: Any, recognizer: Any, stream: Any, tail_padding: float) -> str:
    if tail_padding > 0:
        tail = np.zeros(int(tail_padding * SAMPLE_RATE), dtype=np.float32)
        stream.accept_waveform(SAMPLE_RATE, tail)
        decode_ready(recognizer, stream)

    # Match the hynt demo behavior: signal that no more audio will arrive,
    # then drain any remaining frames before reading the final result.
    stream.input_finished()
    decode_ready(recognizer, stream)
    return get_result_text(recognizer, stream)


def get_result_text(recognizer: Any, stream: Any) -> str:
    result = recognizer.get_result(stream)
    if hasattr(result, "text"):
        return result.text.strip()
    return str(result).strip()


class TranscriptDisplay:
    def __init__(self) -> None:
        self.is_terminal = sys.stdout.isatty()
        self.last_partial = ""
        self.sentence_number = 0

    def speech_started(self) -> None:
        if self.is_terminal:
            print("\r\033[2KListening...", end="", flush=True)
        else:
            print("Listening...", flush=True)

    def partial(self, text: str) -> None:
        if not text or text == self.last_partial:
            return
        self.last_partial = text
        if self.is_terminal:
            print(f"\r\033[2K[partial] {text}", end="", flush=True)
        else:
            print(f"[partial] {text}", flush=True)

    def final(self, text: str) -> None:
        self.sentence_number += 1
        prefix = "\r\033[2K" if self.is_terminal else ""
        shown = text or "(speech detected, but no text recognized)"
        print(f"{prefix}[final {self.sentence_number}] {shown}", flush=True)
        self.last_partial = ""


def run_microphone(
    np: Any,
    sd: Any,
    recognizer: Any,
    vad: Any,
    window_size: int,
    args: argparse.Namespace,
    on_final: Callable[[str], None] | None = None,
) -> None:
    audio_queue: queue.Queue[Any] = queue.Queue(maxsize=250)
    status_queue: queue.Queue[str] = queue.Queue()
    dropped_blocks = [0]

    def audio_callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info
        if status:
            try:
                status_queue.put_nowait(str(status))
            except queue.Full:
                pass
        try:
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            dropped_blocks[0] += 1

    block_seconds = window_size / SAMPLE_RATE
    pre_roll_blocks = max(1, int(args.pre_roll / block_seconds + 0.999))
    pre_roll: deque[Any] = deque(maxlen=pre_roll_blocks)

    active = False
    stream = None
    last_partial_time = 0.0
    display = TranscriptDisplay()

    device_info = sd.query_devices(args.device, "input")
    print(f"Microphone: {device_info['name']}")
    print(
        f"Streaming Zipformer chunk={args.chunk_size}, pre-roll={args.pre_roll:.2f}s, "
        f"VAD threshold={args.vad_threshold:.2f}, "
        f"beam_paths={args.max_active_paths}, blank_penalty={args.blank_penalty:.2f}"
    )
    print("Ready. Speak Vietnamese; pause to finalize. Ctrl+C exits.")

    try:
        with sd.InputStream(
            device=args.device,
            channels=1,
            samplerate=SAMPLE_RATE,
            dtype="float32",
            blocksize=window_size,
            latency="low",
            callback=audio_callback,
        ):
            while True:
                try:
                    samples = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                while not status_queue.empty():
                    print(
                        f"\nAudio warning: {status_queue.get_nowait()}",
                        file=sys.stderr,
                    )

                samples = np.asarray(samples, dtype=np.float32)
                pre_roll.append(samples.copy())

                # VAD only decides when speech starts/ends. ASR receives raw samples.
                vad.accept_waveform(samples)
                speech_now = vad.is_speech_detected()

                if not active and speech_now:
                    active = True
                    stream = recognizer.create_stream()
                    # Critical fix: feed raw audio from BEFORE the VAD trigger.
                    buffered = np.concatenate(list(pre_roll))
                    stream.accept_waveform(SAMPLE_RATE, buffered)
                    decode_ready(recognizer, stream)
                    last_partial_time = 0.0
                    display.speech_started()

                elif active and stream is not None:
                    # Genuine streaming: each new block is accepted exactly once.
                    stream.accept_waveform(SAMPLE_RATE, samples)
                    decode_ready(recognizer, stream)

                if active and stream is not None and args.partial_interval > 0:
                    now = time.monotonic()
                    if now - last_partial_time >= args.partial_interval:
                        display.partial(postprocess_command_text(get_result_text(recognizer, stream)))
                        last_partial_time = now

                completed_segment = False
                while not vad.empty():
                    # Do not decode vad.front.samples. It can be trimmed at the start.
                    vad.pop()
                    completed_segment = True

                if completed_segment and active and stream is not None:
                    raw_text = finalize_stream(
                        np, recognizer, stream, args.tail_padding
                    )
                    text = postprocess_command_text(raw_text)
                    display.final(text)
                    if text and on_final is not None:
                        on_final(text)

                    active = False
                    stream = None
                    last_partial_time = 0.0
                    pre_roll.clear()

    except KeyboardInterrupt:
        if active and stream is not None:
            raw_text = finalize_stream(
                np, recognizer, stream, args.tail_padding
            )
            text = postprocess_command_text(raw_text)
            display.final(text)
            if text and on_final is not None:
                on_final(text)
        print("Stopped.")
    finally:
        if dropped_blocks[0]:
            print(
                f"Warning: dropped {dropped_blocks[0]} microphone blocks because "
                "processing could not keep up.",
                file=sys.stderr,
            )


def main() -> int:
    args = parse_args()

    if args.num_threads < 1:
        raise ValueError("--num-threads must be at least 1")
    if args.partial_interval < 0:
        raise ValueError("--partial-interval cannot be negative")
    if args.pre_roll < 0:
        raise ValueError("--pre-roll cannot be negative")
    if args.tail_padding < 0:
        raise ValueError("--tail-padding cannot be negative")

    if args.list_devices:
        print(import_sounddevice().query_devices())
        return 0

    np, sherpa_onnx = import_runtime()
    files = find_model_files(args.model_dir, args.chunk_size)
    print("Loading streaming Zipformer ASR model...")
    recognizer = create_recognizer(sherpa_onnx, files, args)
    vad, window_size = create_vad(sherpa_onnx, args)

    if args.check:
        print("OK: streaming ASR and VAD models loaded successfully.")
        print(f"Encoder: {files.encoder.name}")
        return 0

    run_microphone(np, import_sounddevice(), recognizer, vad, window_size, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
