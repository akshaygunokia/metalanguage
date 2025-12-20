# train_grpo_limr_zero3.py
# pip install "evalchemy @ git+https://github.com/mlfoundations/evalchemy.git"
import os, re, json, argparse, sys, subprocess, glob, time, datetime
import torch
import hashlib
from datasets import load_dataset, concatenate_datasets

system_prompt = '''You are an assistant that can use external tools and a persistent module canvas.

Goal
- Produce correct final answers.
- When helpful, build a reusable library of small, general modules that can speed up future solutions.

Canvas tools (persistent across tasks)
- LIST(top_k): show existing module signatures and short docs.
- READ(sig): read the current best version of a module (doc + blob).
- PROPOSE(sig, doc, blob): propose a new reusable module or new version of an existing one.

How to use the canvas
- Treat the canvas as a shared library, not a scratchpad.
- Prefer reusing existing modules over re-deriving the same idea.
- If a module seems relevant, you may LIST then READ it and apply it.
- If you discover a method/pattern/lemma/procedure that is likely to recur across problems, you may PROPOSE a module capturing it.
- It is acceptable to propose a module even if it is only a partial helper, as long as it is broadly reusable.
- Do not propose modules that are trivial, purely problem-specific, or duplicate an existing module with no meaningful improvement.
- Keep proposed modules concise, with clear documentation of what they do and when to use them.

Quality bar for proposals
- The signature should be stable and descriptive.
- The doc should explain purpose, inputs/outputs (or assumptions/conclusions), and a short usage note.
- The blob should be a compact, general artifact (e.g., a function, lemma sketch, algorithm outline, transformation rule, checklist).

General behavior
- Use tools only when they provide clear value.
- Do not mention internal scoring or training details.
- If you do not use the canvas, solve normally and provide the best final answer you can.

Output
- Provide the final solution clearly and directly.
'''

# -------------------------------
# Data
# -------------------------------
def build_verl_parquet_openr1_bigmath_oneshot(
    local_save_dir,
    subset="level_5",
    test_holdout=0,              # int (count)
    seed=42,
    instruction_suffix=None,     # optional extra instruction appended to user content
):
    os.makedirs(local_save_dir, exist_ok=True)
    merged = load_dataset("open-r1/Big-Math-RL-Verified-Processed", subset, split="train")

    # 3) Split holdout (count-based)
    if test_holdout and len(merged) > test_holdout:
        split = merged.train_test_split(test_size=test_holdout, seed=seed, shuffle=True)
        train_ds, test_ds = split["train"], split["test"]
    else:
        train_ds, test_ds = merged, None

    # 4) Map into VeRL schema (the important part)
    data_source = "open-r1/Big-Math-RL-Verified-Processed"
    ability = "math"

    def make_map_fn(split_name):
        def process_fn(example, idx):
            user_prompt = example.get("prompt", "")
            if instruction_suffix:
                user_prompt = f"{user_prompt}\n\n{instruction_suffix}"

            return {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "ability": ability,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example.get("solution"),
                },
                "extra_info": {
                    "split": split_name,
                    "index": idx,
                    "source": example.get("source"),
                    "domain": example.get("domain"),
                    "difficulty": example.get("llama8b_solve_rate")
                },
            }
        return process_fn

    train_ds = train_ds.map(make_map_fn("train"), with_indices=True, remove_columns=train_ds.column_names)
    if test_ds is not None:
        test_ds = test_ds.map(make_map_fn("test"), with_indices=True, remove_columns=test_ds.column_names)

    # 5) Write parquet files VeRL expects
    train_path = os.path.join(local_save_dir, "train.parquet")
    train_ds.to_parquet(train_path)

    test_path = None
    if test_ds is not None:
        test_path = os.path.join(local_save_dir, "test.parquet")
        test_ds.to_parquet(test_path)

    return train_path, test_path


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
    subsets=["level_5"],
    allow_sources: set | None = None,          # e.g., {"olympiads","aops_forum"}
    allow_domains: set | None = None,          # e.g., {"Number Theory","Algebra","Geometry","Combinatorics"}
    max_train_examples: int  | None = None,            # “few good ones”
    batch_size: int | None = None,
    eval_holdout: int = 0,
    seed: int = 42,
    tok=None,
):
    """
    Build a small one-shot dataset from open-r1/Big-Math-RL-Verified-Processed.
    - merges selected subsets
    - optional filtering by source or domain
    - shuffles, caps size, splits train/eval
    - optionally pads/truncates to a batch-size multiple
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
        all_rows.append(ds)

    if not all_rows:
        raise ValueError("No data loaded; check subsets/sources/domains filters.")

    from datasets import concatenate_datasets
    merged = concatenate_datasets(all_rows)

    # 4) Shuffle, cap size, split
    merged = merged.shuffle(seed=seed)
    if max_train_examples and len(merged) > (max_train_examples + eval_holdout):
        merged = merged.select(range(max_train_examples + eval_holdout))

    # 6) Holdout for quick eval
    eval_ds = None
    if eval_holdout and len(merged) > eval_holdout:
        split = merged.train_test_split(test_size=eval_holdout, seed=seed, shuffle=True)

        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = merged, None
    # Apply chat template if tokenizer provided
    if tok is not None:
        def _apply_chat(ex):
            ex["prompt"] = [{"role": "system", "content": system_prompt},{"role": "user", "content": ex["prompt"]}]
            return ex
        train_ds = train_ds.map(_apply_chat)
        if eval_ds is not None:
            eval_ds = eval_ds.map(_apply_chat)
    if batch_size is not None:
        from random import choices
        n = len(train_ds)
        if n % batch_size == 0:
            pass  # already fine
        elif n < batch_size:
            # Compute full repeats and remainder for balanced upsampling
            repeats, remainder = divmod(batch_size, n)
            # Full repeats of entire dataset
            idx = list(range(n)) * repeats
            # Evenly distribute remainder (take first `remainder` examples)
            idx += list(range(remainder))
            train_ds = train_ds.select(idx)
        else:
            # more examples than batch size but not divisible
            # truncate down to largest multiple of batch_size
            target = (n // batch_size) * batch_size
            train_ds = train_ds.select(range(target))

    return train_ds, eval_ds



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



