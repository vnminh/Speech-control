#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from gpiozero import LED, OutputDevice, PWMOutputDevice

import realtime_asr_streaming as asr


# ============================================================
# GPIO CONFIGURATION
# ============================================================
# INMP441 I2S on Raspberry Pi 4:
#   GPIO18 = BCLK
#   GPIO19 = LRCLK / WS
#   GPIO20 = DATA IN
#
# Therefore DO NOT use GPIO18 for the motor PWM/ENA pin.
LIGHT_PIN = 17

MOTOR_IN1_PIN = 23
MOTOR_IN2_PIN = 24
MOTOR_ENA_PIN = 13


# ============================================================
# CONSOLE UI
# ============================================================
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


def hr(width: int = 48) -> str:
    return color("─" * width, C.BRIGHT_BLACK)


# ============================================================
# REAL HARDWARE CONTROLLER
# ============================================================
class HardwareController:
    def __init__(self) -> None:
        self.light = LED(
            LIGHT_PIN,
            active_high=True,
            initial_value=False,
        )

        self.motor_in1 = OutputDevice(
            MOTOR_IN1_PIN,
            active_high=True,
            initial_value=False,
        )
        self.motor_in2 = OutputDevice(
            MOTOR_IN2_PIN,
            active_high=True,
            initial_value=False,
        )
        self.motor_ena = PWMOutputDevice(
            MOTOR_ENA_PIN,
            active_high=True,
            initial_value=0,
            frequency=1000,
        )

        self._closed = False

    def turn_light_on(self) -> None:
        self.light.on()

    def turn_light_off(self) -> None:
        self.light.off()

    def turn_fan_on(self, speed: float = 1.0) -> None:
        speed = max(0.0, min(1.0, float(speed)))

        self.motor_in1.on()
        self.motor_in2.off()
        self.motor_ena.value = speed

    def turn_fan_off(self) -> None:
        self.motor_ena.value = 0
        self.motor_in1.off()
        self.motor_in2.off()

    def get_light_state(self) -> bool:
        return self.light.is_lit

    def get_fan_state(self) -> bool:
        return self.motor_ena.value > 0

    def cleanup(self) -> None:
        if self._closed:
            return

        self._closed = True

        self.turn_light_off()
        self.turn_fan_off()

        self.light.close()
        self.motor_in1.close()
        self.motor_in2.close()
        self.motor_ena.close()


# ============================================================
# COMMAND PARSER
# ============================================================
ACTION_ALIASES = {
    "on": ("bật", "mở"),
    "off": ("tắt", "đóng"),
}

DEVICE_ALIASES = {
    "light": ("đèn",),
    "fan": ("quạt",),
}

ACTION_LABELS = {"on": "BẬT", "off": "TẮT"}
DEVICE_LABELS = {"light": "ĐÈN", "fan": "QUẠT"}
DEVICE_ICONS = {"light": "💡", "fan": "🌀"}


@dataclass(frozen=True)
class Command:
    device: str
    action: str


def normalize(text: str) -> str:
    return asr.postprocess_command_text(text)


def parse_commands(text: str) -> list[Command]:
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

        if token in action_by_word:
            current_action = action_by_word[token]
            i += 1
            continue

        if (
            token == "hết"
            and i + 2 < len(tokens)
            and tokens[i + 1] == "cả"
            and tokens[i + 2] == "hai"
        ):
            action = current_action or "off"
            commands.extend(
                [
                    Command("light", action),
                    Command("fan", action),
                ]
            )
            i += 3
            continue

        if (
            token == "cả"
            and i + 1 < len(tokens)
            and tokens[i + 1] == "hai"
        ):
            if current_action:
                commands.extend(
                    [
                        Command("light", current_action),
                        Command("fan", current_action),
                    ]
                )
            i += 2
            continue

        if token in device_by_word and current_action:
            commands.append(Command(device_by_word[token], current_action))

        i += 1

    return list(dict.fromkeys(commands))


# ============================================================
# SMART-HOME CONTROLLER
# ============================================================
class SmartHomeController:
    def __init__(
        self,
        hardware: HardwareController,
        fan_speed: float = 1.0,
    ) -> None:
        self.hardware = hardware
        self.fan_speed = fan_speed

    def print_state(self) -> None:
        light_on = self.hardware.get_light_state()
        fan_on = self.hardware.get_fan_state()

        light_state = (
            color("ON", C.BOLD, C.BRIGHT_GREEN)
            if light_on
            else color("OFF", C.BOLD, C.BRIGHT_RED)
        )
        fan_state = (
            color("ON", C.BOLD, C.BRIGHT_GREEN)
            if fan_on
            else color("OFF", C.BOLD, C.BRIGHT_RED)
        )

        print(
            f"💡 Đèn: {light_state}    "
            f"🌀 Quạt: {fan_state}"
        )

    def _execute(self, command: Command) -> None:
        if command.device == "light":
            if command.action == "on":
                self.hardware.turn_light_on()
            else:
                self.hardware.turn_light_off()

        elif command.device == "fan":
            if command.action == "on":
                self.hardware.turn_fan_on(self.fan_speed)
            else:
                self.hardware.turn_fan_off()

    def handle_text(self, text: str) -> None:
        normalized = normalize(text)

        print(
            color("🧠 ASR", C.BOLD, C.BRIGHT_MAGENTA)
            + "  "
            + color(f"“{text}”", C.WHITE)
        )

        commands = parse_commands(normalized)

        if not commands:
            print(
                color("⚠ SYSTEM", C.BOLD, C.BRIGHT_YELLOW)
                + "  Không tìm thấy lệnh đèn/quạt."
            )
            print(hr())
            return

        for command in commands:
            self._execute(command)

            desired_on = command.action == "on"
            action_color = (
                C.BRIGHT_GREEN if desired_on else C.BRIGHT_RED
            )

            print(
                color("⚡ ACTION", C.BOLD, C.BRIGHT_CYAN)
                + "  "
                + DEVICE_ICONS[command.device]
                + " "
                + color(
                    f"{ACTION_LABELS[command.action]} "
                    f"{DEVICE_LABELS[command.device]}",
                    C.BOLD,
                    action_color,
                )
            )

        self.print_state()
        print()


# ============================================================
# ASR DISPLAY
# ============================================================
class ColorTranscriptDisplay:
    def __init__(self) -> None:
        self.is_terminal = sys.stdout.isatty()
        self.last_partial = ""
        self.sentence_number = 0

    def speech_started(self) -> None:
        message = (
            color("● LISTENING", C.BOLD, C.BRIGHT_GREEN)
            + color("  đang nghe...", C.DIM, C.WHITE)
        )

        if self.is_terminal:
            print("\r\033[2K" + message, end="", flush=True)
        else:
            print(message, flush=True)

    def partial(self, text: str) -> None:
        if not text or text == self.last_partial:
            return

        self.last_partial = text
        message = (
            color("≈ PARTIAL", C.BRIGHT_YELLOW)
            + "  "
            + color(text, C.WHITE)
        )

        if self.is_terminal:
            print("\r\033[2K" + message, end="", flush=True)
        else:
            print(message, flush=True)

    def final(self, text: str) -> None:
        self.sentence_number += 1
        shown = text or "(không nhận dạng được)"
        prefix = "\r\033[2K" if self.is_terminal else ""

        print(
            prefix
            + color(
                f"✓ FINAL {self.sentence_number:02d}",
                C.BOLD,
                C.BRIGHT_CYAN,
            )
            + "  "
            + color(shown, C.BOLD, C.WHITE),
            flush=True,
        )

        self.last_partial = ""


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real Raspberry Pi smart-home voice controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--text")
    parser.add_argument("--keyboard", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument(
        "--fan-speed",
        type=float,
        default=1.0,
        help="Fan PWM duty cycle from 0.0 to 1.0",
    )

    asr.add_microphone_arguments(parser)

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=asr.DEFAULT_MODEL_DIR,
    )
    parser.add_argument(
        "--vad-model",
        type=Path,
        default=asr.DEFAULT_VAD_MODEL,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        choices=(16, 32, 64),
    )
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--partial-interval", type=float, default=0.25)
    parser.add_argument("--min-silence", type=float, default=0.8)
    parser.add_argument("--min-speech", type=float, default=0.12)
    parser.add_argument("--max-speech", type=float, default=20.0)
    parser.add_argument("--vad-threshold", type=float, default=0.35)
    parser.add_argument("--pre-roll", type=float, default=0.5)
    parser.add_argument("--tail-padding", type=float, default=0.30)
    parser.add_argument(
        "--provider",
        default="cpu",
        choices=("cpu", "cuda", "coreml"),
    )
    parser.add_argument(
        "--decoding-method",
        default="modified_beam_search",
        choices=("greedy_search", "modified_beam_search"),
    )
    parser.add_argument("--max-active-paths", type=int, default=15)
    parser.add_argument("--blank-penalty", type=float, default=0.25)

    return parser.parse_args()


def keyboard_loop(controller: SmartHomeController) -> None:
    print("Keyboard mode. q = quit.")

    while True:
        try:
            text = input("smart-home > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if text.lower() in {"q", "quit", "exit"}:
            return

        if text:
            controller.handle_text(text)


# ============================================================
# MAIN
# ============================================================
def main() -> int:
    global USE_COLOR

    args = parse_args()
    USE_COLOR = not args.no_color

    if not 0.0 <= args.fan_speed <= 1.0:
        raise ValueError("--fan-speed must be between 0.0 and 1.0")

    hardware = HardwareController()
    controller = SmartHomeController(
        hardware=hardware,
        fan_speed=args.fan_speed,
    )

    try:
        print()
        print(
            color(
                "SMART HOME • REAL GPIO • VIETNAMESE ASR",
                C.BOLD,
                C.BRIGHT_CYAN,
            )
        )
        print(
            f"Light GPIO={LIGHT_PIN} | "
            f"Motor IN1={MOTOR_IN1_PIN} IN2={MOTOR_IN2_PIN} "
            f"ENA={MOTOR_ENA_PIN}"
        )
        controller.print_state()
        print()

        if args.text:
            controller.handle_text(args.text)
            return 0

        if args.keyboard:
            keyboard_loop(controller)
            return 0

        np, sherpa_onnx = asr.import_runtime()
        files = asr.find_model_files(
            args.model_dir,
            args.chunk_size,
        )

        print("Loading Zipformer + Silero VAD...")
        recognizer = asr.create_recognizer(
            sherpa_onnx,
            files,
            args,
        )
        vad, window_size = asr.create_vad(
            sherpa_onnx,
            args,
        )
        print("ASR ready.\n")

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

    finally:
        print("\nCleaning up GPIO...")
        hardware.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())