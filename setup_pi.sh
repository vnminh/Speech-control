#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
MODEL_NAME="Zipformer-30M-RNNT-Streaming-6000h"
MODEL_DIR="${ROOT_DIR}/models/${MODEL_NAME}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"

case "${CHUNK_SIZE}" in
  16|32|64) ;;
  *)
    echo "ERROR: CHUNK_SIZE must be 16, 32, or 64"
    exit 1
    ;;
esac

echo "==> Installing Raspberry Pi system packages"
sudo apt update
sudo apt install -y \
  python3 \
  python3-pip \
  python3-venv \
  python3-gpiozero \
  python3-lgpio \
  alsa-utils \
  libportaudio2 \
  sox \
  curl

echo "==> Configuring I2S / Google VoiceHAT sound card overlay"
if [[ -f /boot/firmware/config.txt ]]; then
  BOOT_CONFIG="/boot/firmware/config.txt"
elif [[ -f /boot/config.txt ]]; then
  BOOT_CONFIG="/boot/config.txt"
else
  echo "ERROR: Could not find Raspberry Pi config.txt"
  exit 1
fi

REBOOT_REQUIRED=0

if ! grep -Eq '^[[:space:]]*dtparam=i2s=on([[:space:]]*)$' "${BOOT_CONFIG}"; then
  echo "dtparam=i2s=on" | sudo tee -a "${BOOT_CONFIG}" >/dev/null
  REBOOT_REQUIRED=1
fi

if ! grep -Eq '^[[:space:]]*dtoverlay=googlevoicehat-soundcard([[:space:]]*)$' "${BOOT_CONFIG}"; then
  echo "dtoverlay=googlevoicehat-soundcard" | sudo tee -a "${BOOT_CONFIG}" >/dev/null
  REBOOT_REQUIRED=1
fi

echo "==> Creating Python virtual environment"
if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv --system-site-packages "${VENV_DIR}"
fi

echo "==> Installing Python requirements"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements.txt"

echo "==> Downloading models"
mkdir -p "${MODEL_DIR}"

HF_BASE="https://huggingface.co/hynt/${MODEL_NAME}/resolve/main"
SUFFIX="epoch-31-avg-11-chunk-${CHUNK_SIZE}-left-128.fp16.onnx"

download() {
  local url="$1"
  local output="$2"

  if [[ -s "${output}" ]]; then
    echo "    exists: $(basename "${output}")"
    return
  fi

  echo "    downloading: $(basename "${output}")"
  curl -fL --retry 3 --retry-delay 2 \
    -o "${output}.part" \
    "${url}"
  mv "${output}.part" "${output}"
}

download "${HF_BASE}/config.json?download=true" \
  "${MODEL_DIR}/config.json"

download "${HF_BASE}/encoder-${SUFFIX}?download=true" \
  "${MODEL_DIR}/encoder-${SUFFIX}"

download "${HF_BASE}/decoder-${SUFFIX}?download=true" \
  "${MODEL_DIR}/decoder-${SUFFIX}"

download "${HF_BASE}/joiner-${SUFFIX}?download=true" \
  "${MODEL_DIR}/joiner-${SUFFIX}"

download \
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" \
  "${ROOT_DIR}/models/silero_vad.onnx"

echo
echo "============================================================"
echo "Setup complete"
echo "============================================================"
echo "Virtual environment:"
echo "  source .venv/bin/activate"
echo
echo "Downloaded Zipformer chunk:"
echo "  ${CHUNK_SIZE}"
echo
echo "Check microphone after reboot:"
echo "  arecord -l"
echo
echo "Raw INMP441 test (replace hw:3,0 if your ALSA card differs):"
echo "  arecord -D hw:3,0 -c 2 -r 48000 -f S32_LE -d 5 test.wav"
echo "  aplay test.wav"
echo

if [[ "${REBOOT_REQUIRED}" -eq 1 ]]; then
  echo "IMPORTANT: I2S configuration changed."
  echo "Reboot before testing the microphone:"
  echo "  sudo reboot"
else
  echo "I2S configuration was already present; reboot is not required by this script."
fi
