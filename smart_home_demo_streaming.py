#!/usr/bin/env python3
"""Smart-home demo with a colorful console UI and Vietnamese streaming Zipformer ASR."""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import realtime_asr_streaming as asr


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"


USE_COLOR = True


def color(text: str, *codes: str) -> str:
    if not USE_COLOR:
        return text
    return "".join(codes) + text + C.RESET


def hr(width: int = 64) -> str:
    return color("─" * width, C.BRIGHT_BLACK)


def print_banner(args: argparse.Namespace) -> None:
    title = " SMART HOME • VIETNAMESE STREAMING ASR "
    print()
    print(color("╭" + "─" * 62 + "╮", C.BRIGHT_CYAN))
    print(color("│", C.BRIGHT_CYAN) + color(title.center(62), C.BOLD, C.BRIGHT_CYAN) + color("│", C.BRIGHT_CYAN))
    print(color("├" + "─" * 62 + "┤", C.BRIGHT_CYAN))
    print(
        color("│ ", C.BRIGHT_CYAN)
        + color("🎙  Model  ", C.BOLD, C.WHITE)
        + color("hynt/Zipformer-30M-RNNT-Streaming-6000h", C.BRIGHT_GREEN)
        + " " * 5
        + color("│", C.BRIGHT_CYAN)
    )
    settings = (
        f"chunk={args.chunk_size}  beam={args.max_active_paths}  "
        f"blank={args.blank_penalty:.2f}  VAD={args.vad_threshold:.2f}  "
        f"silence={args.min_silence:.2f}s"
    )
    print(
        color("│ ", C.BRIGHT_CYAN)
        + color("⚙  ", C.BRIGHT_YELLOW)
        + color(settings.ljust(59), C.DIM, C.WHITE)
        + color("│", C.BRIGHT_CYAN)
    )
    print(color("╰" + "─" * 62 + "╯", C.BRIGHT_CYAN))
    print()


ACTION_ALIASES = {
    "on": ("bật", "mở"),
    "off": ("tắt", "đóng"),
}

DEVICE_ALIASES = {
    "light": ("đèn",),
    "fan": ("quạt",),
}

BOTH_ALIASES = ("cả hai", "hết cả hai")

ACTION_LABELS = {"on": "BẬT", "off": "TẮT"}
DEVICE_LABELS = {"light": "ĐÈN", "fan": "QUẠT"}
DEVICE_ICONS = {"light": "💡", "fan": "🌀"}


@dataclass(frozen=True)
class Command:
    device: str
    action: str


def normalize(text: str) -> str:
    # Reuse the command-domain ASR post-processor so keyboard input and
    # microphone transcripts follow exactly the same normalization rules.
    return asr.postprocess_command_text(text)


def find_alias(text: str, aliases: dict[str, tuple[str, ...]]) -> str | None:
    for canonical, words in aliases.items():
        if any(re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text) for word in words):
            return canonical
    return None


def find_devices(text: str) -> list[str]:
    devices: list[str] = []
    for device, words in DEVICE_ALIASES.items():
        if any(re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text) for word in words):
            devices.append(device)
    return devices


def contains_both(text: str) -> bool:
    return any(phrase in text for phrase in BOTH_ALIASES)


def parse_commands(text: str) -> list[Command]:
    """Parse the small smart-home Vietnamese command grammar in word order.

    Important: actions are processed sequentially, so commands such as
    ``tắt đèn bật quạt`` become LIGHT=OFF then FAN=ON instead of applying the
    first action to every device in the sentence.

    Supported examples:
      bật quạt / mở quạt
      tắt quạt / đóng quạt
      bật đèn và quạt
      tắt đèn bật quạt
      bật đèn và tắt quạt
      bật cả hai
      tắt hết cả hai
      đóng cả hai đi
    """
    normalized = normalize(text)
    if not normalized:
        return []

    action_by_word = {
        word: canonical
        for canonical, words in ACTION_ALIASES.items()
        for word in words
    }
    device_by_word = {
        word: canonical
        for canonical, words in DEVICE_ALIASES.items()
        for word in words
    }

    tokens = normalized.split()
    commands: list[Command] = []
    current_action: str | None = None
    i = 0

    while i < len(tokens):
        token = tokens[i]

        # A new action immediately replaces the previous carried action.
        # Example: "tắt đèn bật quạt" -> OFF light, then ON fan.
        if token in action_by_word:
            current_action = action_by_word[token]
            i += 1
            continue

        # "hết cả hai" means OFF for both devices, even without an explicit
        # tắt/đóng. If an action was explicitly provided just before it, keep
        # that action so "tắt hết cả hai" behaves naturally.
        if (
            token == "hết"
            and i + 2 < len(tokens)
            and tokens[i + 1] == "cả"
            and tokens[i + 2] == "hai"
        ):
            both_action = current_action or "off"
            commands.extend(
                [Command("light", both_action), Command("fan", both_action)]
            )
            i += 3
            continue

        # "cả hai" applies the current action to both devices.
        if token == "cả" and i + 1 < len(tokens) and tokens[i + 1] == "hai":
            if current_action:
                commands.extend(
                    [Command("light", current_action), Command("fan", current_action)]
                )
            i += 2
            continue

        # Device words inherit only the most recent action seen before them.
        if token in device_by_word and current_action:
            commands.append(Command(device_by_word[token], current_action))
            i += 1
            continue

        # Conjunctions/fillers/unknown words do not reset the current action.
        # This preserves "bật đèn và quạt" -> ON for both devices.
        i += 1

    # Preserve execution order while removing exact duplicate commands.
    return list(dict.fromkeys(commands))


class ColorTranscriptDisplay:
    """Drop-in replacement for realtime_asr_streaming.TranscriptDisplay."""

    def __init__(self) -> None:
        self.is_terminal = sys.stdout.isatty()
        self.last_partial = ""
        self.sentence_number = 0

    def speech_started(self) -> None:
        message = color("● LISTENING", C.BOLD, C.BRIGHT_GREEN) + color("  đang nghe...", C.DIM, C.WHITE)
        if self.is_terminal:
            print("\r\033[2K" + message, end="", flush=True)
        else:
            print(message, flush=True)

    def partial(self, text: str) -> None:
        if not text or text == self.last_partial:
            return
        self.last_partial = text
        message = color("≈ PARTIAL", C.BRIGHT_YELLOW) + "  " + color(text, C.WHITE)
        if self.is_terminal:
            print("\r\033[2K" + message, end="", flush=True)
        else:
            print(message, flush=True)

    def final(self, text: str) -> None:
        self.sentence_number += 1
        shown = text or "(đã phát hiện giọng nói nhưng chưa nhận dạng được)"
        prefix = "\r\033[2K" if self.is_terminal else ""
        label = color(f"✓ FINAL {self.sentence_number:02d}", C.BOLD, C.BRIGHT_CYAN)
        print(prefix + label + "  " + color(shown, C.BOLD, C.WHITE), flush=True)
        self.last_partial = ""


class SmartHomeController:
    def __init__(self) -> None:
        self.state = {"light": False, "fan": False}

    def print_state(self) -> None:
        light_on = self.state["light"]
        fan_on = self.state["fan"]
        light_state = color(" ON ", C.BOLD, C.BRIGHT_GREEN) if light_on else color(" OFF ", C.BOLD, C.BRIGHT_RED)
        fan_state = color(" ON ", C.BOLD, C.BRIGHT_GREEN) if fan_on else color(" OFF ", C.BOLD, C.BRIGHT_RED)

        print(color("┌─ DEVICE STATUS ─────────────────────────────┐", C.BRIGHT_BLUE))
        print(
            color("│ ", C.BRIGHT_BLUE)
            + "💡 Đèn  : " + light_state
            + " " * 8
            + "🌀 Quạt : " + fan_state
            + " " * 4
            + color("│", C.BRIGHT_BLUE)
        )
        print(color("└─────────────────────────────────────────────┘", C.BRIGHT_BLUE))

    def handle_text(self, text: str) -> None:
        normalized = normalize(text)
        print(color("  🧠 ASR", C.BOLD, C.BRIGHT_MAGENTA) + "  " + color(f'“{text}”', C.WHITE))
        if normalized and normalized != text.strip().lower():
            print(
                color("  ✨ POST", C.BOLD, C.BRIGHT_YELLOW)
                + "  "
                + color(f'“{normalized}”', C.WHITE)
            )
        commands = parse_commands(normalized)

        if not commands:
            print(
                color("  ⚠ SYSTEM", C.BOLD, C.BRIGHT_YELLOW)
                + "  Không tìm thấy lệnh điều khiển đèn hoặc quạt."
            )
            print(hr(48))
            return

        for command in commands:
            desired_state = command.action == "on"
            self.state[command.device] = desired_state
            action_color = C.BRIGHT_GREEN if desired_state else C.BRIGHT_RED
            print(
                color("  ⚡ ACTION", C.BOLD, C.BRIGHT_CYAN)
                + "  "
                + DEVICE_ICONS[command.device]
                + " "
                + color(
                    f"{ACTION_LABELS[command.action]} {DEVICE_LABELS[command.device]}",
                    C.BOLD,
                    action_color,
                )
            )

        self.print_state()
        print()


def parse_device(value: str) -> int | str:
    return asr.parse_device(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo đèn/quạt với Vietnamese streaming Zipformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--text", help="Thử trực tiếp một câu và thoát")
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="Nhập nhiều câu từ bàn phím thay vì microphone",
    )
    parser.add_argument("--no-color", action="store_true", help="Tắt màu ANSI trong console")
    parser.add_argument("--device", type=parse_device, default=None)
    parser.add_argument("--model-dir", type=Path, default=asr.DEFAULT_MODEL_DIR)
    parser.add_argument("--vad-model", type=Path, default=asr.DEFAULT_VAD_MODEL)
    parser.add_argument("--chunk-size", type=int, default=32, choices=(16, 32, 64))
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--partial-interval", type=float, default=0.25)
    parser.add_argument("--min-silence", type=float, default=0.8)
    parser.add_argument("--min-speech", type=float, default=0.12)
    parser.add_argument("--max-speech", type=float, default=20.0)
    parser.add_argument("--vad-threshold", type=float, default=0.35)
    parser.add_argument("--pre-roll", type=float, default=0.5)
    parser.add_argument("--tail-padding", type=float, default=0.30)
    parser.add_argument("--provider", default="cpu", choices=("cpu", "cuda", "coreml"))
    parser.add_argument(
        "--decoding-method",
        default="modified_beam_search",
        choices=("greedy_search", "modified_beam_search"),
    )
    parser.add_argument("--max-active-paths", type=int, default=15)
    parser.add_argument("--blank-penalty", type=float, default=0.25)
    return parser.parse_args()


def keyboard_loop(controller: SmartHomeController) -> None:
    print(color("⌨  KEYBOARD MODE", C.BOLD, C.BRIGHT_CYAN))
    print(color("Nhập câu điều khiển; nhập q để thoát.\n", C.DIM, C.WHITE))
    while True:
        try:
            prompt = color("smart-home ❯ ", C.BOLD, C.BRIGHT_GREEN)
            text = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.lower() in {"q", "quit", "exit"}:
            return
        if text:
            controller.handle_text(text)


def main() -> int:
    global USE_COLOR

    args = parse_args()
    USE_COLOR = not args.no_color

    print_banner(args)
    controller = SmartHomeController()
    controller.print_state()
    print()

    if args.text:
        controller.handle_text(args.text)
        return 0

    if args.keyboard:
        keyboard_loop(controller)
        return 0

    np, sherpa_onnx = asr.import_runtime()
    files = asr.find_model_files(args.model_dir, args.chunk_size)

    print(color("⏳ MODEL", C.BOLD, C.BRIGHT_YELLOW) + "  Đang tải Streaming Vietnamese Zipformer...")
    recognizer = asr.create_recognizer(sherpa_onnx, files, args)
    vad, window_size = asr.create_vad(sherpa_onnx, args)
    print(color("✓ MODEL", C.BOLD, C.BRIGHT_GREEN) + "  Zipformer + Silero VAD đã sẵn sàng.\n")

    # run_microphone() resolves TranscriptDisplay from the ASR module at runtime,
    # so replacing it here also colors partial/final ASR output without changing
    # realtime_asr_streaming.py.
    asr.TranscriptDisplay = ColorTranscriptDisplay

    asr.run_microphone(
        np,
        asr.import_sounddevice(),
        recognizer,
        vad,
        window_size,
        args,
        on_final=controller.handle_text,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
