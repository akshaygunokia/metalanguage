# train_grpo_limr_zero3.py
# pip install "evalchemy @ git+https://github.com/mlfoundations/evalchemy.git"
import os, re, json, argparse, sys, subprocess, glob, time, datetime
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
import hashlib

# -------------------------------
# Data
# -------------------------------
def build_dataset(dataset_slug: str, add_answer_tag: bool, eval_holdout: int):
    ds = load_dataset(dataset_slug, split="train")
    if add_answer_tag:
        def _map(ex):
            return {"prompt": ex["prompt"].rstrip() + "\nAnswer:", "answer": ex["answer"].strip()}
    else:
        def _map(ex):
            return {"prompt": ex["prompt"].rstrip(), "answer": ex["answer"].strip()}
    ds = ds.map(_map, remove_columns=ds.column_names)

    eval_ds = None
    if eval_holdout and eval_holdout > 0 and len(ds) > eval_holdout:
        split = ds.train_test_split(test_size=eval_holdout, shuffle=True, seed=42)
        ds, eval_ds = split["train"], split["test"]
    return ds, eval_ds

def build_dataset_openr1_bigmath_oneshot(
    subsets=("level_5","quintile_5"),
    allow_sources: set | None = None,          # e.g., {"olympiads","aops_forum"}
    allow_domains: set | None = None,          # e.g., {"Number Theory","Algebra","Geometry","Combinatorics"}
    solve_rate_min: float | None = None,
    solve_rate_max: float | None = None,
    max_chars_prompt: int = 4000,
    max_chars_solution: int = 80,              # short final forms preferred for auto-check
    keep_regex: str = r"^\\s*([0-9\\-\\.]+|\\\\frac\\{[^}]+\\}\\{[^}]+\\}|\\\\sqrt\\{[^}]+\\}|[0-9]+\\^[0-9]+|\\\\pi|\\\\boxed\\{.*\\})\\s*$",
    max_train_examples: int = 126,            # “few good ones”
    eval_holdout: int = 10,
    add_answer_tag: bool = True,
    seed: int = 42,
):
    """
    Build a tiny one-shot RLVR set from open-r1/Big-Math-RL-Verified-Processed.
    - merges selected subsets, filters to short, parseable final answers
    - dedups identical problems across subsets/sources
    - outputs columns: prompt, reward_model={'ground_truth': solution}
    """
    import random, re
    random.seed(seed)

    # 1) Load and merge chosen subsets
    all_rows = []
    for sb in subsets:
        ds = load_dataset("open-r1/Big-Math-RL-Verified-Processed", sb, split="train")
        if allow_sources:
            ds = ds.filter(lambda x: x.get("source") in allow_sources)
        if allow_domains:
            def _dom_ok(x):
                dom = x.get("domain")
                s = " ".join(dom) if isinstance(dom, list) else (dom or "")
                return any(d.lower() in s.lower() for d in allow_domains)
            ds = ds.filter(_dom_ok)
        if (solve_rate_min is not None) or (solve_rate_max is not None):
            def _rate_ok(x):
                r = x.get("llama8b_solve_rate", None)
                if r is None: return False
                if (solve_rate_min is not None) and (r < solve_rate_min): return False
                if (solve_rate_max is not None) and (r > solve_rate_max): return False
                return True
            ds = ds.filter(_rate_ok)
        all_rows.append(ds)

    if not all_rows:
        raise ValueError("No data loaded; check subsets/sources/domains filters.")

    from datasets import concatenate_datasets
    merged = concatenate_datasets(all_rows)

    # 2) Light “good one-shot” filtering: lengths + regex-able final answers
    keep_re = re.compile(keep_regex)
    def _good(x):
        p = x.get("prompt") or ""
        s = str(x.get("solution") or "").strip()
        if not p or not s: return False
        if len(p) > max_chars_prompt: return False
        if len(s) > max_chars_solution: return False
        return bool(keep_re.match(s))
    merged = merged.filter(_good)

    # 3) Deduplicate by content hash
    def _hash(ex):
        h = hashlib.sha256()
        h.update((ex["prompt"] + "\n<<ANS>>" + str(ex["solution"])).encode("utf-8"))
        return {"_h": h.hexdigest()}
    merged = merged.map(_hash)
    try:
        merged = merged.drop_duplicates(column_names=["_h"])
    except AttributeError:
        seen = set()
        def _mark(ex):
            h = ex["_h"]; keep = h not in seen; 
            if keep: seen.add(h)
            return {"__keep": keep}
        merged = merged.map(_mark).filter(lambda x: x["__keep"]).remove_columns(["__keep"])

    merged = merged.remove_columns(["_h"])

    # 4) Shuffle, cap size, split
    merged = merged.shuffle(seed=seed)
    if max_train_examples and len(merged) > (max_train_examples + eval_holdout):
        merged = merged.select(range(max_train_examples + eval_holdout))

    # 5) Map to RLVR one-shot format
    def _map_rlvr(ex):
        p = ex["prompt"].rstrip()
        if add_answer_tag:
            p += "\nAnswer:"
        return {
            "prompt": p,
            "reward_model": {"ground_truth": str(ex["solution"]).strip()},
            "answer": str(ex["solution"]).strip(),
        }
    merged = merged.map(_map_rlvr, remove_columns=merged.column_names)

    # 6) Holdout for quick eval
    eval_ds = None
    if eval_holdout and len(merged) > eval_holdout:
        split = merged.train_test_split(test_size=eval_holdout, seed=seed, shuffle=True)
        return split["train"], split["test"]
    return merged, eval_ds


def build_dataset_oneshot(dataset_slug: str, tok, add_answer_tag: bool, eval_holdout: int):
    raw = load_dataset("ypwang61/One-Shot-RLVR-Datasets", split="merge_pi1_pi2_pi13_pi1209_r128")

    def _map(ex):
        msgs = ex["prompt"]                      # list of {"role","content"}
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        if add_answer_tag:
            text += " Answer:"                   # optional, matches your old parsing
        return {
            "prompt": text,
            "reward_model": ex["reward_model"],  # KEEP the dict column for the reward
        }

    ds = raw.map(_map, remove_columns=raw.column_names)

    eval_ds = None
    if eval_holdout and len(ds) > eval_holdout:
        split = ds.train_test_split(test_size=eval_holdout, seed=42, shuffle=True)
        ds, eval_ds = split["train"], split["test"]
    return ds, eval_ds

