# IoT Vietnamese Streaming ASR on Raspberry Pi 4

Streaming Vietnamese speech recognition and smart-home control on Raspberry Pi 4 using:

- INMP441 I2S microphone
- Silero Voice Activity Detection
- Zipformer streaming encoder
- Recurrent Neural Network Transducer decoding
- `sherpa-onnx`
- GPIO light and L298N fan control

The ASR code expects the Vietnamese streaming model
`hynt/Zipformer-30M-RNNT-Streaming-6000h`.

## Repository structure

```text
asr/
├── models/
│   ├── silero_vad.onnx
│   └── Zipformer-30M-RNNT-Streaming-6000h/
│       ├── config.json
│       ├── encoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx
│       ├── decoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx
│       └── joiner-epoch-31-avg-11-chunk-32-left-128.fp16.onnx
├── realtime_asr_streaming.py
├── smart_home_real.py
├── requirements.txt
├── setup_pi.sh
└── README.md
```

By default, `setup_pi.sh` downloads the chunk-32 Zipformer files because the
Python programs use chunk size 32 by default.

---

## 1. Hardware

### INMP441 → Raspberry Pi 4

| INMP441 | Raspberry Pi 4 |
|---|---|
| VDD | 3.3 V |
| GND | GND |
| SCK / BCLK | GPIO18, physical pin 12 |
| WS / LRCLK | GPIO19, physical pin 35 |
| SD | GPIO20, physical pin 38 |
| L/R | GND for left channel |

Do not power the INMP441 from 5 V.

### Smart-home GPIO

| Device | GPIO |
|---|---:|
| Light / LED | GPIO17 |
| L298N IN1 | GPIO23 |
| L298N IN2 | GPIO24 |
| L298N ENA / PWM | GPIO13 |

GPIO18 is reserved for the I2S microphone clock, so it must not be reused as
the motor PWM pin.

---

## 2. One-command Raspberry Pi setup

Make the setup script executable:

```bash
chmod +x setup_pi.sh
```

Run:

```bash
./setup_pi.sh
```

The script performs the following steps:

1. Installs Raspberry Pi system packages for ALSA, PortAudio, GPIO, Python,
   `curl`, and SoX.
2. Enables I2S.
3. Enables the `googlevoicehat-soundcard` Device Tree overlay.
4. Creates `.venv`.
5. Runs:

   ```bash
   pip install -r requirements.txt
   ```

6. Downloads the Zipformer encoder, decoder, joiner, and `config.json`.
7. Downloads `silero_vad.onnx`.

If the script changes the Raspberry Pi boot configuration, reboot:

```bash
sudo reboot
```

---

## 3. Raspberry Pi I2S configuration

On Raspberry Pi OS Bookworm the boot configuration is normally:

```text
/boot/firmware/config.txt
```

On older Raspberry Pi OS versions it may be:

```text
/boot/config.txt
```

The following lines are required:

```ini
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
```

`setup_pi.sh` adds them automatically if they are missing.

After reboot:

```bash
arecord -l
```

A working setup should show a Google VoiceHAT capture card, commonly similar to:

```text
card 3: sndrpigooglevoi [snd_rpi_googlevoicehat_soundcar]
```

The ALSA card number may differ between systems.

---

## 4. Test the INMP441 before running ASR

First test raw ALSA capture:

```bash
arecord -D hw:3,0 \
  -c 2 \
  -r 48000 \
  -f S32_LE \
  -d 5 \
  test.wav
```

Play the recording:

```bash
aplay test.wav
```

If `L/R` on the INMP441 is connected to GND, the microphone should normally be
on the left channel.

Inspect the left channel:

```bash
sox test.wav -n remix 1 stat
```

Inspect the right channel:

```bash
sox test.wav -n remix 2 stat
```

The active microphone channel should have a clearly larger RMS amplitude.

---

## 5. Python environment

Activate the environment created by `setup_pi.sh`:

```bash
source .venv/bin/activate
```

The Python dependencies are defined in `requirements.txt`:

```text
numpy
sounddevice
sherpa-onnx
gpiozero
```

Check imports:

```bash
python -c "import numpy, sounddevice, sherpa_onnx, gpiozero; print('OK')"
```

---

## 6. Download model weights manually

You normally do not need this section because `setup_pi.sh` downloads the
models automatically.

Create the directories:

```bash
mkdir -p models/Zipformer-30M-RNNT-Streaming-6000h
```

Set variables:

```bash
MODEL_DIR="models/Zipformer-30M-RNNT-Streaming-6000h"
BASE="https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h/resolve/main"
SUFFIX="epoch-31-avg-11-chunk-32-left-128.fp16.onnx"
```

Download the Zipformer files:

```bash
curl -fL \
  -o "${MODEL_DIR}/config.json" \
  "${BASE}/config.json?download=true"

curl -fL \
  -o "${MODEL_DIR}/encoder-${SUFFIX}" \
  "${BASE}/encoder-${SUFFIX}?download=true"

curl -fL \
  -o "${MODEL_DIR}/decoder-${SUFFIX}" \
  "${BASE}/decoder-${SUFFIX}?download=true"

curl -fL \
  -o "${MODEL_DIR}/joiner-${SUFFIX}" \
  "${BASE}/joiner-${SUFFIX}?download=true"
```

Download Silero Voice Activity Detection:

```bash
curl -fL \
  -o models/silero_vad.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
```

---

## 7. Other streaming chunk sizes

The model repository provides chunk sizes:

```text
16
32
64
```

The setup script downloads chunk 32 by default.

To download chunk 16:

```bash
CHUNK_SIZE=16 ./setup_pi.sh
```

To download chunk 64:

```bash
CHUNK_SIZE=64 ./setup_pi.sh
```

The files for previously downloaded chunk sizes are preserved.

When running the recognizer, select the matching value:

```bash
python realtime_asr_streaming.py --chunk-size 16
```

or:

```bash
python realtime_asr_streaming.py --chunk-size 64
```

---

## 8. Check the models

Activate the environment:

```bash
source .venv/bin/activate
```

Load the default chunk-32 ASR model and Silero model without opening the
microphone:

```bash
python realtime_asr_streaming.py --check
```

A successful check should print that the streaming ASR and Voice Activity
Detection models loaded successfully.

---

## 9. List Python audio devices

ALSA card numbers and `sounddevice` device indices are not necessarily the
same.

List the devices visible to Python:

```bash
python realtime_asr_streaming.py --list-devices
```

Use the `sounddevice` input index/name with `--device`.

---

## 10. Run realtime ASR

Activate the environment:

```bash
source .venv/bin/activate
```

Then:

```bash
python realtime_asr_streaming.py
```

If you need to select a specific Python input device:

```bash
python realtime_asr_streaming.py --device <DEVICE_INDEX>
```

The default recognition settings include:

```text
sample rate       : 16000 Hz
Zipformer chunk   : 32
decoding method   : modified_beam_search
max active paths  : 15
VAD threshold     : 0.35
minimum silence   : 0.8 s
pre-roll          : 0.5 s
```

---

## 11. Run the smart-home controller

First test the GPIO controller without speech recognition:

```bash
python smart_home_real.py --keyboard
```

Example commands:

```text
bật đèn
tắt đèn
bật quạt
tắt quạt
bật cả hai
tắt cả hai
```

Then run voice control:

```bash
python smart_home_real.py
```

You can reduce fan speed with:

```bash
python smart_home_real.py --fan-speed 0.7
```

The light uses GPIO17. `bật đèn` sets GPIO17 HIGH and `tắt đèn` sets GPIO17 LOW.

---

## 12. Troubleshooting

### No Google VoiceHAT card

Check:

```bash
grep -E 'i2s|googlevoicehat' /boot/firmware/config.txt
arecord -l
```

Expected configuration:

```ini
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
```

Then reboot.

### `arecord` works but Python does not

Check Python input devices:

```bash
python realtime_asr_streaming.py --list-devices
```

Do not assume ALSA `card 3` is Python device index `3`.

### Model file missing

Check:

```bash
find models -maxdepth 2 -type f -printf '%p\n'
```

For chunk 32 you need:

```text
models/silero_vad.onnx
models/Zipformer-30M-RNNT-Streaming-6000h/config.json
models/Zipformer-30M-RNNT-Streaming-6000h/encoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx
models/Zipformer-30M-RNNT-Streaming-6000h/decoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx
models/Zipformer-30M-RNNT-Streaming-6000h/joiner-epoch-31-avg-11-chunk-32-left-128.fp16.onnx
```

### GPIO permission/backend problems

The setup script installs Raspberry Pi GPIO packages and creates the virtual
environment with system site packages enabled.

If needed, verify:

```bash
python -c "from gpiozero import LED; print('gpiozero OK')"
```

---

## Model sources

Zipformer:

```text
https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h
```

Silero Voice Activity Detection model used by sherpa-onnx:

```text
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
```

The Zipformer Hugging Face repository currently declares the
`cc-by-nc-nd-4.0` license. Check the upstream model card before redistribution
or commercial use.
