#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token utilities for REMI v1.1 tokenizer and model inference.
Shared by train_prepare, train, eval_generate, and piano_adapter.
"""

import torch
from typing import List, Union
import pretty_midi


def load_remi_tokenizer(remi_enabled=True):
    """Load REMI v1.1 tokenizer from tokenizer_remi.py."""
    try:
        from tokenizer_remi import REMITokenizer
        return REMITokenizer(remi_enabled=remi_enabled)
    except ImportError:
        raise ImportError(
            "tokenizer_remi.py not found. Ensure it exists in project root."
        )


def encode_pm(tokenizer, pm: pretty_midi.PrettyMIDI) -> List[int]:
    """
    Encode PrettyMIDI to token IDs.
    Handles API differences (encode_pm, encode, encode_path).
    """
    if hasattr(tokenizer, "encode_pm"):
        return tokenizer.encode_pm(pm)
    if hasattr(tokenizer, "encode"):
        result = tokenizer.encode(pm)
        if isinstance(result, (list, tuple)):
            return result
    raise RuntimeError(
        "REMITokenizer does not have encode_pm/encode methods. "
        "Check tokenizer_remi.py API."
    )


def decode_ids_to_pm(tokenizer, ids: List[int]) -> pretty_midi.PrettyMIDI:
    """
    Decode token IDs to PrettyMIDI.
    Handles API differences (decode_ids_to_pm, decode).
    """
    if hasattr(tokenizer, "decode_ids_to_pm"):
        return tokenizer.decode_ids_to_pm(ids)
    if hasattr(tokenizer, "decode"):
        result = tokenizer.decode(ids)
        if isinstance(result, pretty_midi.PrettyMIDI):
            return result
    raise RuntimeError(
        "REMITokenizer does not have decode_ids_to_pm/decode methods. "
        "Check tokenizer_remi.py API."
    )


def sample_model(
    model,
    prompt_ids: List[int],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    eos_token_id: int = 1
) -> List[int]:
    """
    Nucleus sampling from causal language model.
    
    Args:
        model: HuggingFace AutoModelForCausalLM
        prompt_ids: Initial token sequence
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold
        eos_token_id: Stop generation when this token is sampled
    
    Returns:
        Full token sequence (prompt + generated)
    """
    model.eval()
    device = next(model.parameters()).device
    
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x).logits[:, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            
            # Nucleus (top-p) sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum <= top_p
            mask[..., 0] = True  # Always keep top-1
            
            filt_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
            filt_probs = filt_probs / torch.sum(filt_probs, dim=-1, keepdim=True)
            
            next_id = torch.multinomial(filt_probs, num_samples=1)
            next_token = sorted_idx.gather(-1, next_id)
            
            x = torch.cat([x, next_token], dim=1)
            
            # Stop on EOS
            if next_token.item() == eos_token_id:
                break
    
    return x[0].tolist()


def build_prefix_ids_from_conditions(tokenizer, conditions: dict) -> List[int]:
    """
    Convert generation conditions to prompt token IDs.
    
    Args:
        tokenizer: REMI tokenizer
        conditions: Dict with keys like tempo, time_sig, length_bars, style, density
    
    Returns:
        List of token IDs representing the condition prefix
    
    Note:
        This is a placeholder. Implement based on your tokenizer's condition encoding.
        For now, returns BOS token or empty list.
    """
    # TODO: Implement condition → token encoding
    # Example: [BOS, TEMPO_110, TIME_SIG_4_4, STYLE_BLOCK, ...]
    bos_id = getattr(tokenizer, "bos_id", None)
    if bos_id is not None:
        return [bos_id]
    return []
