# train_grpo_limr_zero3.py
# pip install "evalchemy @ git+https://github.com/mlfoundations/evalchemy.git"
import os, re, json, argparse, sys, subprocess, glob, time, datetime
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
import hashlib

# -------------------------------
# Helpers: Evalchemy one-shot runner & latest checkpoint
# -------------------------------
def latest_checkpoint_dir(root: str):
    """
    Return the newest 'checkpoint-XXXX' subdir in `root`, or None.
    """
    if not os.path.isdir(root):
        return None
    ckpts = [p for p in glob.glob(os.path.join(root, "checkpoint-*")) if os.path.isdir(p)]
    if not ckpts:
        return None
    # sort by global step (suffix) primarily, then mtime
    def _score(p):
        base = os.path.basename(p)
        try:
            step = int(base.split("-")[-1])
        except Exception:
            step = -1
        return (step, os.path.getmtime(p))
    return sorted(ckpts, key=_score, reverse=True)[0]

def run_evalchemy(model_backend: str,
                  model_args: str,
                  tasks: str,
                  batch_size: int,
                  output_path: str,
                  cuda_visible_devices: str | None = None):
    """
    Run Evalchemy CLI: python -m eval.eval --model <backend> --model_args <...>
    """
    os.makedirs(output_path, exist_ok=True)
    cmd = [
        sys.executable, "-m", "eval.eval",
        "--model", model_backend,
        "--tasks", tasks,
        "--model_args", model_args,
        "--batch_size", str(batch_size),
        "--output_path", output_path,
    ]
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    print(f"[Evalchemy] Running: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, check=False)
    dt = time.time() - t0
    print(f"[Evalchemy] Exit code={proc.returncode} in {dt:.1f}s -> {output_path}")
    return proc.returncode == 0

# -------------------------------
# Periodic benchmark callback (runs on every save or every N saves)
# -------------------------------
class EvalchemyCallback(TrainerCallback):
    """
    Runs Evalchemy on AIME24/AMC23/MATH500 each time a checkpoint is saved,
    then logs metrics back to the trainer (TensorBoard/W&B/etc.).
    """
    def __init__(
        self,
        trainer,                                    # pass GRPOTrainer instance
        tasks="AIME24,AMC23,MATH500",
        batch_size=2,
        every_n_saves=1,                            # run on every save
        model_backend="hf",                         # or "vllm"
        extra_model_args="",                        # e.g. "dtype=bfloat16"
        output_subdir="benchmarks",
        bench_cuda=None,                            # e.g. "1" to use GPU 1
    ):
        self.trainer = trainer
        self.tasks = tasks
        self.batch_size = batch_size
        self.every_n_saves = every_n_saves
        self.model_backend = model_backend
        self.extra_model_args = extra_model_args
        self.output_subdir = output_subdir
        self._save_count = 0
        self.bench_cuda = bench_cuda

    def _latest_results_json(self, outdir):
        files = glob.glob(f"{outdir}/**/*.json", recursive=True)
        return max(files, key=lambda p: (os.path.getmtime(p), p)) if files else None

    def _log_results(self, state, results_path):
        try:
            with open(results_path, "r") as f:
                data = json.load(f)
            results = data.get("results", {})
            to_log = {}
            for task, metrics in results.items():
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        to_log[f"bench/{task}/{k}"] = float(v)
            if to_log:
                to_log["global_step"] = state.global_step
                self.trainer.log(to_log)
                print(f"[Evalchemy] Logged: {to_log}")
        except Exception as e:
            print(f"[Evalchemy] Could not parse results: {e}")

    def on_save(self, args, state, control, **kwargs):
        # only run on main process
        if not self.trainer.accelerator.is_main_process:
            return

        self._save_count += 1
        if self._save_count % self.every_n_saves != 0:
            return

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            return

        outdir = os.path.join(args.output_dir, self.output_subdir, f"step_{state.global_step}")
        os.makedirs(outdir, exist_ok=True)

        model_args = f"pretrained={ckpt_dir}"
        if self.extra_model_args:
            model_args += f",{self.extra_model_args}"

        # block training until eval finishes (safer on single-GPU).
        env_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        if self.bench_cuda is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.bench_cuda)
        try:
            ok = run_evalchemy(
                model_backend=self.model_backend,
                model_args=model_args,
                tasks=self.tasks,
                batch_size=self.batch_size,
                output_path=outdir,
                cuda_visible_devices=None,  # already set via env override above
            )
            if ok:
                rp = self._latest_results_json(outdir)
                if rp:
                    self._log_results(state, rp)
        finally:
            # restore
            if self.bench_cuda is not None:
                if env_cuda is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = env_cuda

# -------------------------------
# Reward function
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

# -------------------------------
# Training (per rank)
# -------------------------------
def real_main(args):
    # Helpful alloc settings
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    set_seed(args.seed)

    # Write (or overwrite) DeepSpeed ZeRO-3 JSON so it’s self-contained
    ds_conf = {
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": True,
            "stage3_param_persistence_threshold": 1000000,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            # Flip these to "cpu" if you still OOM:
            "offload_param":     {"device": "none", "pin_memory": False},
            "offload_optimizer": {"device": "none", "pin_memory": False}
        },
        "bf16": {"enabled": args.bf16},
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size
    }
    os.makedirs(os.path.dirname(args.ds_config) or ".", exist_ok=True)
    with open(args.ds_config, "w") as f:
        json.dump(ds_conf, f, indent=2)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Dataset  (pass tok, and keep RLVR builder)
    HARD_SUBSETS = ("level_5", "quintile_5")
    HARD_SOURCES = {"olympiads", "aops_forum"}              # None to keep all
    HARD_DOMAINS = {"Number Theory","Algebra","Geometry","Combinatorics"}  # None to keep all
    MAX_TRAIN_EX = 126
    EVAL_HOLDOUT = 7
    ADD_ANSWER_TAG = True
    train_ds, eval_ds = build_dataset_openr1_bigmath_oneshot(
        subsets=HARD_SUBSETS,
        allow_sources=HARD_SOURCES,
        allow_domains=HARD_DOMAINS,
        solve_rate_min=None,
        solve_rate_max=None,
        max_chars_prompt=4000,
        max_chars_solution=80,
        keep_regex=r"^\s*([0-9\-\.]+|\\frac\{[^}]+\}\{[^}]+\}|\\sqrt\{[^}]+\}|[0-9]+\^[0-9]+|\\pi|\\boxed\{.*\})\s*$",
        max_train_examples=MAX_TRAIN_EX,
        eval_holdout=EVAL_HOLDOUT,
        add_answer_tag=ADD_ANSWER_TAG,
        seed=args.seed,
    )

    # GRPO config
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        deepspeed=args.ds_config,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.grad_ckpt,

        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,

        scale_rewards=False,
        loss_type="dr_grpo",
        importance_sampling_level="sequence",
        beta=0.0,

        logging_steps=1,
        save_steps=1,
        save_total_limit=1,
        report_to=["tensorboard"],

        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        generation_batch_size=126,  # slightly lower than default 256 to reduce spikes
    )

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=dtype, attn_implementation="flash_attention_2")
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        reward_funcs=reward_rlvr_oneshot,
        processing_class=tok,
    )

    # Attach periodic benchmark callback only if requested
    if args.do_bench and args.bench_periodic:
        bench_cb = EvalchemyCallback(
            trainer=trainer,
            tasks=args.bench_tasks,
            batch_size=args.bench_batch_size,
            every_n_saves=args.bench_every_n_saves,
            model_backend=args.bench_backend,              # "hf" or "vllm"
            extra_model_args=args.bench_extra_model_args,  # e.g., "dtype=bfloat16"
            bench_cuda=args.bench_cuda,
        )
        trainer.add_callback(bench_cb)

    trainer.train()

    # Small eval + save on rank 0 (use trainer.accelerator)
    if trainer.accelerator.is_main_process:
        metrics = small_eval_oneshot(trainer, eval_ds, tok, max_new_tokens=min(64, args.max_completion_length))
        if metrics:
            print("Eval metrics:", metrics)
        trainer.save_model()
        tok.save_pretrained(args.output_dir)

    # Optional final one-shot benchmark after training
    if args.do_bench and args.bench_final:
        ckpt_dir = latest_checkpoint_dir(args.output_dir) or args.output_dir
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join(args.output_dir, "benchmarks", f"final_{ts}")
        model_args = f"pretrained={ckpt_dir}"
        if args.bench_extra_model_args:
            model_args += f",{args.bench_extra_model_args}"
        run_evalchemy(
            model_backend=args.bench_backend,
            model_args=model_args,
            tasks=args.bench_tasks,
            batch_size=args.bench_batch_size,
            output_path=outdir,
            cuda_visible_devices=args.bench_cuda,
        )

# -------------------------------
# CLI
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # What to do
    p.add_argument("--do_train", action="store_true", help="Run training")
    p.add_argument("--do_bench", action="store_true", help="Run benchmark (periodic and/or final)")
    p.add_argument("--bench_only", action="store_true", help="Run benchmark only (no training)")

    # Model / data / training
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--dataset_slug", type=str, default="GAIR/LIMR")
    p.add_argument("--ds_config", type=str, default="ds_zero3.json")
    p.add_argument("--output_dir", type=str, default="limr-grpo-qwen3-8b-zero3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--num_epochs", type=float, default=100.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--num_generations", type=int, default=2)
    p.add_argument("--max_prompt_length", type=int, default=768)
    p.add_argument("--max_completion_length", type=int, default=128)
    p.add_argument("--add_answer_tag", action="store_true")
    p.add_argument("--eval_holdout", type=int, default=64)
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--vllm_mode", type=str, default="server", choices=["server", "colocate"])
    p.add_argument("--num_processes", type=int, default=1, help="World size (GPUs) to launch.")

    # Benchmark config (shared by callback and one-shot)
    p.add_argument("--bench_tasks", type=str, default="AIME25,MATH500")
    p.add_argument("--bench_batch_size", type=int, default=2)
    p.add_argument("--bench_every_n_saves", type=int, default=100)
    p.add_argument("--bench_backend", type=str, choices=["hf","vllm"], default="hf")
    p.add_argument("--bench_extra_model_args", type=str, default="", help='e.g. "dtype=bfloat16"')
    p.add_argument("--bench_periodic", action="store_true", help="Run Evalchemy on every save via callback")
    p.add_argument("--bench_final", action="store_true", help="Run a final one-shot Evalchemy after training")
    p.add_argument("--bench_cuda", type=str, default=None, help='e.g. "1" to eval on GPU 1')

    return p.parse_args()

def launch():
    args = parse_args()

    # Default behavior: if nothing specified, train only (backward-compatible)
    if not args.do_train and not args.do_bench and not args.bench_only:
        args.do_train = True

    # If user didn’t set bf16/fp16, enable bf16 if supported
    if not args.bf16 and not args.fp16:
        try:
            if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
                args.bf16 = True
        except Exception:
            pass

    # --- BENCHMARK ONLY path (no Accelerate workers needed) ---
    if args.bench_only:
        # Choose target
        target_path = latest_checkpoint_dir(args.output_dir) or args.output_dir

        # Build model_args for evalchemy
        if os.path.isdir(target_path):
            model_args = f"pretrained={target_path}"
        else:
            # assume HF Hub id
            model_args = f"pretrained={target_path}"
        if args.bench_extra_model_args:
            model_args += f",{args.bench_extra_model_args}"

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join(args.output_dir, "benchmarks", f"adhoc_{ts}")
        ok = run_evalchemy(
            model_backend=args.bench_backend,
            model_args=model_args,
            tasks=args.bench_tasks,
            batch_size=args.bench_batch_size,
            output_path=outdir,
            cuda_visible_devices=args.bench_cuda,
        )
        sys.exit(0 if ok else 1)

    # --- TRAIN (with optional periodic + final benchmark) ---
    if args.num_processes == 1:
        real_main(args)
    else:
        from accelerate import notebook_launcher
        # Ensure the same CUDA_VISIBLE_DEVICES order is used
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(args.num_processes)))
        notebook_launcher(real_main, (args,), num_processes=args.num_processes)

if __name__ == "__main__":
    launch()

