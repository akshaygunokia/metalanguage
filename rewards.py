# train_grpo_limr_zero3.py
# pip install "evalchemy @ git+https://github.com/mlfoundations/evalchemy.git"
import os, re, json, argparse, sys, subprocess, glob, time, datetime
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
import hashlib

# -------------------------------
# Reward functions
# -------------------------------
def reward_exact(prompts, completions, completion_ids, gold=None, answer=None, **kwargs):
    # choose labels from either 'gold' or 'answer'
    labels = gold if gold is not None else answer
    if labels is None:
        raise ValueError("Reward needs a 'gold' or 'answer' column in the dataset.")

    B = len(prompts)
    C = len(completions)
    if B == 0 or C % B != 0:
        raise ValueError(f"Shape mismatch: len(completions)={C}, len(prompts)={B}. Check num_generations.")
    G = C // B

    # tile labels to align with flattened completions
    labels_tiled = []
    for lab in labels:
        labels_tiled.extend([lab] * G)

    rewards = []
    for y, target in zip(completions, labels_tiled):
        # Try to read the first token after "Answer:", fallback to first token
        m = re.search(r"Answer:\s*([^\n ,\.]+)", y, flags=re.I) or re.search(r"^\s*([^\n ,\.]+)", y)
        pred = m.group(1) if m else ""
        rewards.append(1.0 if pred == target else -1.0)
    return rewards

_BOXED = re.compile(r"\\boxed\{([^}]*)\}", re.I)

def _extract_final(text: str) -> str:
    m = _BOXED.search(text)
    if m: return m.group(1).strip()
    tail = text[-400:]
    m = re.search(r"(?i)(?:final answer|answer|ans)\s*[:\-]?\s*([^\n]+)", tail)
    if m: return m.group(1).strip()
    # last non-empty line
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""

def _equiv(a: str, b: str) -> bool:
    a = re.sub(r"\s+|,", "", a).replace("−","-").strip(".,").lower()
    b = re.sub(r"\s+|,", "", b).replace("−","-").strip(".,").lower()
    if a == b: return True
    try:
        from sympy import nsimplify, simplify
        return simplify(nsimplify(a) - nsimplify(b)) == 0
    except Exception:
        return False

def reward_rlvr_oneshot(prompts, completions, completion_ids, reward_model=None, **_):
    """
    Binary 0/1 reward: matches One-Shot RLVR. Expects a column named `reward_model`
    (each row a dict with key 'ground_truth').
    """
    if reward_model is None:
        raise ValueError("Dataset must include a `reward_model` column with 'ground_truth'.")

    labels = [(rm or {}).get("ground_truth", "") for rm in reward_model]

    B, C = len(prompts), len(completions)
    if B == 0 or C % B != 0:
        raise ValueError(f"Shape mismatch: len(completions)={C}, len(prompts)={B}. Check num_generations.")
    G = C // B

    tiled = []
    for lab in labels:
        tiled.extend([lab] * G)

    rewards = []
    for y, target in zip(completions, tiled):
        pred = _extract_final(y)
        rewards.append(1.0 if _equiv(pred, target) else 0.0)  # use -1.0 instead of 0.0 if you prefer ±1
    return rewards

def small_eval_oneshot(trainer, eval_ds, tok, max_new_tokens: int = 64):
    if eval_ds is None or len(eval_ds) == 0:
        return {}

    prompts = [r["prompt"] for r in eval_ds]
    # RLVR ground truth lives inside reward_model dict
    golds = [(r["reward_model"] or {}).get("ground_truth", "") for r in eval_ds]

    texts = []

    # If using vLLM (you passed --use_vllm), prefer it here:
    if getattr(trainer, "llm", None) is not None:
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
        outs = trainer.llm.generate(prompts, sp)
        texts = [o.outputs[0].text for o in outs]
    else:
        # HF fallback (works on main process). Unwrap for ZeRO-3.
        model = trainer.accelerator.unwrap_model(trainer.model)
        model.eval()
        bs = 8
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        with torch.no_grad():
            for i in range(0, len(prompts), bs):
                batch = prompts[i:i+bs]
                inputs = tok(
                    batch, return_tensors="pt", padding=True, truncation=True
                ).to(trainer.accelerator.device)
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=pad_id,
                )
                texts.extend(tok.batch_decode(out_ids, skip_special_tokens=True))

    # exact-match scoring using your robust extractor/equivalence
    corr = 0
    for y, target in zip(texts, golds):
        pred = _extract_final(y)
        corr += int(_equiv(pred, target))

    return {"eval_exact_acc": corr / len(golds)}


def small_eval(trainer, eval_ds, tok, max_new_tokens: int = 64):
    if eval_ds is None or len(eval_ds) == 0:
        return {}

    prompts = [r["prompt"] for r in eval_ds]
    golds   = [r["answer"] for r in eval_ds]

    texts = []

    # If using vLLM (you passed --use_vllm), prefer it here:
    if getattr(trainer, "llm", None) is not None:
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
        outs = trainer.llm.generate(prompts, sp)
        texts = [o.outputs[0].text for o in outs]
    else:
        # HF fallback (works on main process). Unwrap for ZeRO-3.
        model = trainer.accelerator.unwrap_model(trainer.model)
        model.eval()
        bs = 8
        with torch.no_grad():
            for i in range(0, len(prompts), bs):
                batch = prompts[i:i+bs]
                inputs = tok(batch, return_tensors="pt", padding=True, truncation=True).to(trainer.accelerator.device)
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=tok.eos_token_id,
                )
                texts.extend(tok.batch_decode(out_ids, skip_special_tokens=True))

    # exact-match scoring
    corr = 0
    for y, target in zip(texts, golds):
        m = re.search(r"Answer:\s*([^\n ,\.]+)", y, flags=re.I) or re.search(r"^\s*([^\n ,\.]+)", y)
        pred = m.group(1) if m else ""
        corr += int(pred == target)
    return {"eval_exact_acc": corr / len(golds)}

