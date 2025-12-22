# train_grpo_limr_zero3.py
# pip install "evalchemy @ git+https://github.com/mlfoundations/evalchemy.git"
import os, re, json, argparse, sys, subprocess, glob, time, datetime
import torch
import hashlib
from datasets import load_dataset, concatenate_datasets
from functools import partial

strong_system_prompt = '''You are an assistant with access to a persistent Canvas tool that acts like a shared library of reusable modules.

Canvas operations:
- LIST(top_k): see available module signatures (sig) and short docs
- READ(sig): fetch the current winning version (doc + blob)
- PROPOSE(sig, doc, blob): add a new version for a signature

Behavior:
- First try LIST then READ to reuse something relevant to solve the problem.
- If nothing fits, PROPOSE a small, reusable thought or module with a clear signature and short doc.
- Do not spam PROPOSE; only add modules that will help across many problems.
'''

soft_system_prompt = '''You are a solver with access to a shared Canvas — a living library 
that grows across many solvers and many tasks.

Operations:
- LIST(top_k) → see available modules
- READ(sig) → fetch a module's blob
- PROPOSE(sig, doc, blob) → contribute a new module

How to work:
1. LIST to see what exists
2. READ anything that might help
3. Solve the task
4. If your reasoning could help future solvers, PROPOSE it

The canvas is shared:
- What you READ was left by others
- What you PROPOSE may help others
- Modules that help survive. Modules that don't fade.

Solve the task. Use what helps. Leave what's useful.'''

def example_map_fn(example, idx, process_fn, data_source, ability, split):
    question, solution = process_fn(example)
    data = {
        "data_source": data_source,
        "prompt": [{"role":"system", "content": soft_system_prompt},{"role": "user", "content": question}],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": split, "index": idx},
        "agent_name": "tool_agent",
    }
    return data

def build_verl_parquet_openr1_bigmath_oneshot(subset="level_5", max_unique_prompts=1024, max_train_size=1024, seed=42):
    data_source = "open-r1/Big-Math-RL-Verified-Processed"
    ability = "math"

    train_ds = load_dataset(data_source, subset, split="train").shuffle(seed=seed)

    if max_unique_prompts and len(train_ds) > max_unique_prompts:
        train_ds = train_ds.select(range(max_unique_prompts))

    if max_train_size is not None:
        n = len(train_ds)
        if n < max_train_size:
            repeats, remainder = divmod(max_train_size, n)
            idx = list(range(n)) * repeats
            idx += list(range(remainder))
            train_ds = train_ds.select(idx)
        else:
            train_ds = train_ds.select(range(max_train_size))

    def process_openr1_bigmath(example):
        user_prompt = example.get("prompt", "")
        return user_prompt, example.get("solution")

    train_map_fn = partial(
        example_map_fn, process_fn=process_openr1_bigmath, data_source=data_source, ability=ability, split="train"
    )
    train_ds = train_ds.map(train_map_fn, with_indices=True, remove_columns=train_ds.column_names)
    return train_ds

def build_aime2024_dataset():
    def process_aime2024(example):
        problem = example["Problem"]
        # Force parseable final line
        problem = (
            problem
            + "\n\nGive ONLY the final answer on the last line in exactly one of these formats:\n"
            r"Answer: \boxed{<integer>}"
            "\n"
        )
        return problem, str(example["Answer"])

    data_source = "Maxwell-Jia/AIME_2024"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    map_fn = partial(
        example_map_fn, process_fn=process_aime2024, data_source=data_source, ability="Math", split="test"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_gpqa_diamond_dataset():
    import random

    GPQA_QUERY_TEMPLATE = (
        "Answer the following multiple choice question. The last line of your response should be of the following "
        "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before "
        "answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    )

    def process_gpqa_diamond(example):
        choices = [example["Incorrect Answer 1"], example["Incorrect Answer 2"], example["Incorrect Answer 3"]]
        random.shuffle(choices)
        gold_index = random.randint(0, 3)
        choices.insert(gold_index, example["Correct Answer"])
        query_prompt = GPQA_QUERY_TEMPLATE.format(
            A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=example["Question"]
        )
        gold_choice = "ABCD"[gold_index]
        return query_prompt, gold_choice

    data_source = "Idavidrein/gpqa"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    dataset = load_dataset(data_source, "gpqa_diamond", split="train")
    map_fn = partial(
        example_map_fn, process_fn=process_gpqa_diamond, data_source=data_source, ability="Science", split="test"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_cnmo2024_dataset():
    def process_cnmo2024(example):
        return example["question"], example["answer"]

    data_source = "opencompass/LiveMathBench"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    dataset_en = load_dataset(data_source, "v202412_CNMO_en", split="test")
    map_fn_en = partial(
        example_map_fn, process_fn=process_cnmo2024, data_source="opencompass/cnmo2024_en", ability="Math", split="test"
    )
    dataset_en = dataset_en.map(map_fn_en, with_indices=True, remove_columns=dataset_en.column_names)

    dataset_zh = load_dataset(data_source, "v202412_CNMO_cn", split="test")
    map_fn_zh = partial(
        example_map_fn, process_fn=process_cnmo2024, data_source="opencompass/cnmo2024_zh", ability="Math", split="test"
    )
    dataset_zh = dataset_zh.map(map_fn_zh, with_indices=True, remove_columns=dataset_zh.column_names)

    dataset = concatenate_datasets([dataset_en, dataset_zh])
    return dataset


def build_livecodebench_dataset():
    import base64
    import json
    import pickle
    import zlib

    def process_livecodebench(example):
        # Construct Query Prompt
        # From https://github.com/LiveCodeBench/LiveCodeBench/blob/998c52d394b836f15fff3b9a29866191108ff81b/lcb_runner/prompts/code_generation.py#L140
        query_prompt = (
            f"You will be given a question (problem specification) and will generate a correct Python program "
            f"that matches the specification and passes all tests.\n\nQuestion: {example['question_content']}\n\n"
        )
        if example["starter_code"]:
            query_prompt += (
                f"You will use the following starter code to write the solution to the problem and enclose your "
                f"code within delimiters.\n```python\n{example['starter_code']}\n```"
            )
        else:
            query_prompt += (
                "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test "
                "on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
                "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
                "```python\n# YOUR CODE HERE\n```"
            )

        # Construct test cases
        public_test_cases = json.loads(example["public_test_cases"])
        try:
            private_test_cases = json.loads(example["private_test_cases"])
        except Exception as e:
            print(f"Error loading private test cases: {e}")
            private_test_cases = json.loads(
                pickle.loads(zlib.decompress(base64.b64decode(example["private_test_cases"].encode("utf-8"))))
            )
        full_test_cases = public_test_cases + private_test_cases

        metadata = json.loads(example["metadata"])
        test_cases = {
            "inputs": [t["input"] for t in full_test_cases],
            "outputs": [t["output"] for t in full_test_cases],
            "fn_name": metadata.get("func_name", None),
        }
        text_cases_compressed = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(test_cases)))).decode("utf-8")
        return query_prompt, text_cases_compressed

    data_source = "livecodebench/code_generation_lite"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="test")
    # R1 Evaluation use LiveCodeBench 24.08-25.01
    dataset = dataset.filter(lambda line: "2024-08-00T00:00:00" <= line["contest_date"] < "2025-01-00T00:00:00")
    map_fn = partial(
        example_map_fn, process_fn=process_livecodebench, data_source=data_source, ability="Code", split="test"
    )

    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names, num_proc=8)
    return dataset

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
