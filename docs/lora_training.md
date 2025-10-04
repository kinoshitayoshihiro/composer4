# LoRA Training Guide

This repository uses [PEFT](https://github.com/huggingface/peft) to inject LoRA adapters into GPT-2 models for piano and saxophone token generation.  When `rank > 0`, the transformer is wrapped with `get_peft_model` using a `LoraConfig` targeting the `c_attn` modules.

Command line options allow controlling the adapter rank and scaling factor:

- `--rank` – adapter rank (default `4`)
- `--lora_alpha` – LoRA scaling factor, defaults to `rank * 2`

Adapters are saved with `model.save_pretrained()` after training.  Use `--auto-hparam` to scale rank and training steps based on dataset size.  Pass `--safe` to serialize adapters as `.safetensors` files and `--eval` to run a quick evaluation after training.

Example usage for the saxophone model:

```bash
python train_sax_lora.py --data sax.jsonl --out sax_model --safe --eval
```

Install optional dependencies with:

```bash
pip install -r requirements/base.txt -r requirements/extra-ml.txt -r requirements/extra-audio.txt
```

Using `--eval` requires optional packages from `requirements/extra-ml.txt`.
The *scipy* library is mandatory when `--eval` is enabled and will be installed
with that extras file.
