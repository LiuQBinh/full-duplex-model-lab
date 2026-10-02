# full-duplex-model-lab

Lab for full-duplex speech models, starting with [Moshi](https://github.com/kyutai-labs/moshi) (Kyutai) on a **Mac with an M4 chip**.

Moshi on Apple Silicon uses the MLX package `moshi-mlx`, not the PyTorch server. PyTorch Moshi expects an NVIDIA GPU with about 24 GB of VRAM. MLX runs the same Moshiko dialogue model from quantized weights in unified memory.

## Which weights

| Flag | Hugging Face repo | Weight file | Use when |
| --- | --- | --- | --- |
| `--q4` (default) | `kyutai/moshiko-mlx-q4` | `model.q4.safetensors`, 4.8 GB | 16 GB unified memory |
| `--q8` | `kyutai/moshiko-mlx-q8` | `model.q8.safetensors`, 8.2 GB | 24 GB unified memory or more |

Both repos also ship the Mimi codec (`tokenizer-e351c8d8-checkpoint125.safetensors`, 385 MB) and the SentencePiece tokenizer. The first launch downloads them into the Hugging Face cache.

`-q` must match the repo. `run_moshi_mac.py` sets both.

## Setup (on the Mac)

Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Install this environment on the Mac. The launcher exits on other platforms so the checkpoint is not downloaded there.

## Talk to Moshi

Web UI (microphone, browser):

```bash
python scripts/run_moshi_mac.py
```

Opens [http://127.0.0.1:8998](http://127.0.0.1:8998). Allow the microphone for that origin.

8-bit, if the Mac has 24 GB or more:

```bash
python scripts/run_moshi_mac.py --q8
```

Terminal client (no echo cancellation):

```bash
python scripts/run_moshi_mac.py --cli
```

Ask from a wav instead of the microphone. The server above must already be running. The file is resampled to 24 kHz, then a few seconds of silence are appended so Moshi can answer.

```bash
python scripts/ask_from_wav.py question.wav --reply reply.wav
```

On a machine that is not Apple Silicon the script prints this setup and exits with code 2. It does not download the checkpoint.
