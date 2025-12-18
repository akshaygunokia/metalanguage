# train_grpo_limr_zero3.py
# pip install "evalchemy @ git+https://github.com/mlfoundations/evalchemy.git"
import os, re, json, argparse, sys, subprocess, glob, time, datetime
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
import hashlib
from sympy import nsimplify, simplify, Eq
import math
from typing import Optional, List, Any, Dict
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig
from canvas_wrapper_tool import update_score
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

def reward_bigmath_accuracy(
    prompts: List[Any],
    completions: List[Any],
    completion_ids: List[List[int]],
    solution: Optional[List[str]] = None,
    **_
) -> List[Optional[float]]:
    """
    Accuracy reward adapted to the Big-Math GRPO interface.

    Parameters
    ----------
    prompts : list[str]
        Problem statements (unused except for shape checks).
    completions : list[Any]
        Model outputs (one or more per prompt).
    completion_ids : list[list[int]]
        Token ids of completions (unused here).
    solution : list[str]
        Ground-truth worked solutions from the dataset.

    Returns
    -------
    list[Optional[float]]
        1.0 if the parsed completion matches the gold parse,
        0.0 if it parses but is wrong,
        None if parsing fails (trainer will skip).
    """
    if solution is None:
        raise ValueError("Dataset must provide the `solution` column.")

    B, C = len(prompts), len(completions)
    if B == 0 or C % B != 0:
        raise ValueError(
            f"Shape mismatch: len(completions)={C}, len(prompts)={B}. "
            "Check num_generations."
        )
    G = C // B

    # Repeat each gold solution to align with flattened completions
    tiled_gold = []
    for sol in solution:
        tiled_gold.extend([sol] * G)

    rewards: List[Optional[float]] = []
    for completion, sol in zip(completions, tiled_gold):
        gold_parsed = parse(sol, extraction_mode="first_match")
        if len(gold_parsed) == 0:
            # Cannot parse gold → skip
            rewards.append(0)
            continue

        # Parse the model output with strong normalization
        answer_parsed = parse(
            completion[-1]["content"],
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        read_versions = extract_read_versions(completion)
        if len(answer_parsed) == 0:
            rewards.append(0)
            continue

        # Verify symbolic equivalence
        try:
            reward = float(verify(gold_parsed, answer_parsed))
            if reward > 0:
                seen = set()
                for v in read_versions:
                    key = (v["sig"], v["blob"])
                    if key in seen:
                        continue
                    seen.add(key)
                    update_score(sig=v["sig"], blob=v["blob"], delta=reward)
            rewards.append(reward)
        except Exception as e:
            print(f"verify failed: {e}")
            rewards.append(0)

    return rewards


# --- Canon/Parsing helpers ---

def extract_read_versions(completion: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Extract READ tool responses from a chat-style completion.

    Returns a list of dicts:
      {
        "sig": str,
        "blob": str,
        "ver_id": str,   # hash of blob
      }
    """
    results: List[Dict[str, str]] = []

    for msg in completion:
        if msg.get("role") != "tool":
            continue
        if msg.get("name") != "READ":
            continue

        content = msg.get("content")
        if not isinstance(content, str):
            continue

        try:
            payload = json.loads(content)
        except Exception:
            continue

        data = payload.get("data")
        if not isinstance(data, dict):
            continue

        sig = data.get("sig")
        blob = data.get("blob")
        if not isinstance(sig, str) or not isinstance(blob, str):
            continue

        results.append({"sig": sig, "blob": blob})

    return results


def _latex_to_ascii(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = s.replace("−", "-").replace("·", "*").replace("π", "pi")
    s = s.replace("∞", "oo").replace("°", "deg")
    # remove common wrappers
    s = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", s)
    # basic LaTeX spacing/formatting commands to drop
    s = re.sub(r"\\(?:left|right|quad|qquad|,|;|!|:)", "", s)
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    # unbox/fbox
    s = re.sub(r"\\(?:boxed|fbox)\s*\{([^}]*)\}", r"\1", s)
    # convert LaTeX-style exponent x^{2} or x^2 -> x**2
    s = re.sub(r"\^\s*\{([^}]+)\}", r"**(\1)", s)  # x^{y} -> x**(y)
    s = re.sub(r"\^\s*([A-Za-z0-9\)\]])", r"**\1", s)  # x^2, (x+1)^3, etc.
    # ops
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    s = re.sub(r"\\sqrt\s*\{([^}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\overline\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\%","%", s)
    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_final(text: str) -> str:
    if not text:
        return ""
    # prefer last \boxed / \fbox
    boxed = list(re.finditer(r"\\(?:boxed|fbox)\s*\{([^}]*)\}", text, flags=re.S))
    if boxed:
        return boxed[-1].group(1).strip()

    tail = text[-1200:]  # search near the end
    # explicit "answer:" markers
    m = re.search(r"(?i)(?:final\s*answer|answer|ans)\s*[:\-]?\s*\$?([^\n]+)", tail)
    if m:
        return m.group(1).strip(" .")

    # last display math $$...$$
    m = re.search(r"\$\$([^$]+)\$\$(?!.*\$\$)", tail, flags=re.S)
    if m:
        return m.group(1).strip()

    # last inline math \( ... \) or \[ ... \]
    m = re.search(r"\\\(([^)]*)\\\)(?!.*\\\()", tail, flags=re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"\\\[((?:.|\n)*?)\\\](?!.*\\\[)", tail)
    if m:
        return m.group(1).strip()

    # last inline $...$
    m = re.search(r"\$([^$]+)\$(?!.*\$)", tail, flags=re.S)
    if m:
        return m.group(1).strip()

    # fallback: last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""

_INTERVAL_RE = re.compile(r"^\s*([\(\[])\s*(.+?)\s*,\s*(.+?)\s*([\)\]])\s*$")

def _looks_like_interval(s: str) -> bool:
    return bool(_INTERVAL_RE.match(s.strip()))

def _parse_interval(s: str):
    # returns ((left_open: bool, right_open: bool), (left_val, right_val))
    from sympy import nsimplify
    m = _INTERVAL_RE.match(s.strip())
    if not m:
        raise ValueError("not an interval")
    lbr, a, b, rbr = m.groups()
    left_open  = (lbr == "(")
    right_open = (rbr == ")")
    a1, b1 = _latex_to_ascii(a), _latex_to_ascii(b)
    try:
        A, B = nsimplify(a1), nsimplify(b1)
    except Exception:
        A, B = a1.lower(), b1.lower()
    return (left_open, right_open), (A, B)

def _looks_like_set(s: str) -> bool:
    t = s.strip()
    return (t.startswith("{") and t.endswith("}")) or t.lower().startswith("set")

def _looks_like_tuple(s: str) -> bool:
    t = s.strip()
    return t.startswith("(") and t.endswith(")")

def _parse_ordered_tuple(s: str):
    core = s.strip().strip("()")
    parts = [p.strip() for p in re.split(r"\s*,\s*", core) if p.strip()]
    out = []
    for p in parts:
        p_ascii = _latex_to_ascii(p)
        try:
            out.append(nsimplify(p_ascii))
        except Exception:
            out.append(p_ascii.lower())
    return tuple(out)

def _parse_list_as_set(s: str):
    core = s.strip().strip("{}[]()")
    parts = [p.strip() for p in re.split(r"\s*,\s*", core) if p.strip()]
    out = []
    for p in parts:
        p_ascii = _latex_to_ascii(p)
        try:
            out.append(nsimplify(p_ascii))
        except Exception:
            out.append(p_ascii.lower())
    return set(out)

def _to_number_with_percent(s: str):
    t = s.strip().replace(" %", "%")
    if t.endswith("%"):
        return float(t[:-1].strip()) / 100.0
    return float(t)

_THOUSANDS = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
def _maybe_desep_number(s: str) -> str:
    t = s.strip()
    return t.replace(",", "") if _THOUSANDS.match(t) else t

def _equiv(a: str, b: str) -> bool:
    def canon(s: str) -> str:
        s = s.strip().strip(".,;")
        s = s.replace("−", "-")
        s = re.sub(r"\s+", " ", s)
        return s

    a0, b0 = canon(a), canon(b)

    # quick literal ignoring spaces/case
    if re.sub(r"\s+", "", a0).lower() == re.sub(r"\s+", "", b0).lower():
        return True

    a1, b1 = _latex_to_ascii(a0), _latex_to_ascii(b0)

    # sets (unordered)
    if _looks_like_set(a1) and _looks_like_set(b1):
        try:
            return _parse_list_as_set(a1) == _parse_list_as_set(b1)
        except Exception:
            pass

    # intervals (order matters + bracket semantics matter)
    if _looks_like_interval(a1) and _looks_like_interval(b1):
        try:
            (alo, aro), (A, B) = _parse_interval(a1)
            (blo, bro), (C, D) = _parse_interval(b1)
            if alo != blo or aro != bro:
                return False
            try:
                from sympy import simplify
                return simplify(A - C) == 0 and simplify(B - D) == 0
            except Exception:
                return str(A) == str(C) and str(B) == str(D)
        except Exception:
            pass

    # ordered tuples
    if _looks_like_tuple(a1) and _looks_like_tuple(b1):
        try:
            return _parse_ordered_tuple(a1) == _parse_ordered_tuple(b1)
        except Exception:
            pass

    # algebraic/numeric equivalence
    try:
        A, B = nsimplify(a1), nsimplify(b1)
        try:
            return simplify(A - B) == 0
        except Exception:
            try:
                return bool(Eq(A, B))
            except Exception:
                pass
    except Exception:
        pass

    # numeric fallback (handles thousands separators & percents)
    a2, b2 = _maybe_desep_number(a1), _maybe_desep_number(b1)
    try:
        af = _to_number_with_percent(a2)
        bf = _to_number_with_percent(b2)
        return math.isclose(af, bf, rel_tol=1e-6, abs_tol=1e-8)
    except Exception:
        return False



def reward_bigmath(prompts, completions, completion_ids,
                   solution=None, **_):
    """
    Binary 0/1 reward for the open-r1/Big-Math-RL-Verified-Processed dataset.

    Parameters
    ----------
    prompts : list[str]
        Problem statements (ignored except for shape checks).
    completions : list[str]
        Model-generated outputs (one or more per prompt).
    completion_ids : list[list[int]]
        Token ids of completions (unused here but required by TRL).
    solution : list[str]
        Ground-truth worked solutions from the dataset.
        Must contain the correct *final* answer somewhere inside.
    **_ : dict
        Catch-all for extra arguments passed by the trainer.

    Returns
    -------
    list[float]
        Reward per completion: 1.0 if the extracted final answer
        matches the ground truth, else 0.0.
    """
    if solution is None:
        raise ValueError("Dataset must provide the `solution` column.")

    B, C = len(prompts), len(completions)
    if B == 0 or C % B != 0:
        raise ValueError(
            f"Shape mismatch: len(completions)={C}, len(prompts)={B}. "
            "Check num_generations."
        )
    G = C // B

    # Tile the gold solutions to match the G completions per prompt
    tiled = []
    for sol in solution:
        tiled.extend([sol] * G)

    rewards = []
    for pred_text, gold_text in zip(completions, tiled):
        pred = _extract_final(pred_text)     # <-- implement to pull last boxed/number
        gold = _extract_final(gold_text)     # same extraction for consistency
        rewards.append(1.0 if _equiv(pred, gold) else 0.0)
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


