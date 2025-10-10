#!/usr/bin/env python3
# pyright: reportGeneralTypeIssues=false
# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Evaluate MERT alignment with multi-crop audio windows."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, cast

import numpy as np
import torch
import torchaudio  # type: ignore[import-not-found]
import soundfile as sf  # type: ignore[import-not-found]
from transformers import AutoModel, AutoProcessor

TARGET_SR = 24_000


FloatArray = Any
TensorDict = Dict[str, torch.Tensor]


def _to_tensor_dict(batch: Mapping[str, Any]) -> TensorDict:
    tensor_dict: TensorDict = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor_dict[str(key)] = value
        else:
            tensor_dict[str(key)] = torch.as_tensor(value)
    return tensor_dict


def load_audio(
    path: Path,
    target_sr: int = TARGET_SR,
) -> Tuple[Any, int]:
    raw, sr = sf.read(path, always_2d=False)  # type: ignore[assignment]
    data = np.asarray(raw)
    if data.ndim > 1:
        data = data.mean(axis=1)
    waveform = np.asarray(data, dtype=np.float32)
    if sr != target_sr:
        tensor = torch.from_numpy(waveform).unsqueeze(0)
        resampled = torchaudio.functional.resample(tensor, sr, target_sr)
        waveform = resampled.squeeze(0).numpy()
        sr = target_sr
    return waveform, sr


def crop_indices(num_samples: int, crop_len: int) -> Sequence[Tuple[int, int]]:
    if num_samples <= crop_len:
        return [(0, num_samples)]

    mid = max(0, (num_samples // 2) - (crop_len // 2))
    indices = [
        (0, crop_len),
        (mid, crop_len),
        (max(0, num_samples - crop_len), crop_len),
    ]
    start = random.randint(0, max(0, num_samples - crop_len))
    indices.append((start, crop_len))

    dedup: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for item in indices:
        if item not in seen:
            dedup.append(item)
            seen.add(item)
    return dedup


def _normalize(features: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(features, dim=-1)


def _to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _prepare_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list",
        required=True,
        help="Text file with one audio path per line",
    )
    parser.add_argument(
        "--captions",
        required=True,
        help="JSON mapping basename to caption",
    )
    parser.add_argument("--out", default="artifacts/mert_multicrops.jsonl")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--crop-sec", type=float, default=8.0)
    parser.add_argument("--max-files", type=int, default=64)
    parser.add_argument("--model", default="m-a-p/MERT-v1-95M")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for crop sampling",
    )
    return parser.parse_args()


def _select_files(list_path: Path, limit: int) -> List[str]:
    with list_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if limit > 0:
        return lines[:limit]
    return lines


def _load_captions(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(key): str(value) for key, value in data.items()}


def _write_row(destination: Path, payload: dict[str, object]) -> None:
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> int:
    args = _prepare_args()
    random.seed(args.seed)

    output_path = Path(args.out)
    if output_path.parent:
        os.makedirs(output_path.parent, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    device = torch.device(args.device)
    processor: Any = AutoProcessor.from_pretrained(args.model)  # type: ignore[call-arg]
    model: Any = (
        AutoModel.from_pretrained(args.model).to(device).eval()
    )  # type: ignore[attr-defined]

    captions = _load_captions(Path(args.captions))
    audio_files = _select_files(Path(args.list), args.max_files)

    crop_len = int(args.crop_sec * TARGET_SR)

    for audio_path in audio_files:
        file_path = Path(audio_path)
        basename = file_path.stem
        caption = captions.get(basename) or captions.get(audio_path) or ""
        if not caption:
            continue

        waveform, _ = load_audio(file_path)
        segments = crop_indices(len(waveform), crop_len)
        if not segments:
            continue

        with torch.no_grad():
            text_inputs_raw = cast(
                Mapping[str, Any],
                processor(
                    text=[caption],
                    return_tensors="pt",
                    padding=True,
                ),
            )
            text_inputs = _to_tensor_dict(text_inputs_raw)
            # type: ignore[attr-defined]
            text_features = model.get_text_features(**_to_device(text_inputs, device))
            text_features = cast(torch.Tensor, text_features)
            text_features = _normalize(text_features)

        cosines: List[float] = []
        for start, length in segments:
            crop = np.asarray(waveform[start : start + length], dtype=np.float32)
            if crop.shape[0] < crop_len:
                pad = crop_len - crop.shape[0]
                crop = np.pad(crop, (0, pad)).astype(np.float32)
            audio_inputs_raw = cast(
                Mapping[str, Any],
                processor(
                    audios=[crop],
                    sampling_rate=TARGET_SR,
                    return_tensors="pt",
                ),
            )
            audio_inputs = _to_tensor_dict(audio_inputs_raw)
            with torch.no_grad():
                # type: ignore[attr-defined]
                audio_features = model.get_audio_features(**_to_device(audio_inputs, device))
                audio_features = cast(torch.Tensor, audio_features)
                audio_features = _normalize(audio_features)
            cosine = float(torch.matmul(audio_features, text_features.T).cpu().numpy().squeeze())
            cosines.append(cosine)

        if not cosines:
            continue

        array = np.asarray(cosines, dtype=np.float32)
        iqr = float(np.percentile(array, 75) - np.percentile(array, 25)) if array.size else None
        payload: Dict[str, object] = {
            "file": audio_path,
            "basename": basename,
            "crops": len(segments),
            "cos_mean": float(array.mean()) if array.size else None,
            "cos_std": float(array.std(ddof=0)) if array.size else None,
            "cos_iqr": iqr,
        }
        _write_row(output_path, payload)
        print("[mert_crops]", payload)

    print(f"[mert_crops] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
