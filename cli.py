# train_grpo_limr_zero3.py
# pip install "evalchemy @ git+https://github.com/mlfoundations/evalchemy.git"
import os, re, json, argparse, sys, subprocess, glob, time, datetime
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
import hashlib
from fs_utils import latest_checkpoint_dir
from bench_eval import run_evalchemy
from train import train


# -------------------------------
# CLI
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # What to do
    p.add_argument("--do_train", action="store_true", default=True, help="Run training")
    p.add_argument("--do_bench", action="store_true", help="Run benchmark (periodic and/or final)")
    p.add_argument("--bench_only", action="store_true", help="Run benchmark only (no training)")

    # Model / data / training
    p.add_argument("--model_id", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--dataset_slug", type=str, default="GAIR/LIMR")
    p.add_argument("--ds_config", type=str, default="ds_zero3.json")
    p.add_argument("--output_dir", type=str, default="limr-grpo-qwen3-8b-zero3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--grad_ckpt", action="store_true", default=True)
    p.add_argument("--num_epochs", type=float, default=100.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--max_prompt_length", type=int, default=768)
    p.add_argument("--generation_batch_size", type=int, default=128*8)
    p.add_argument("--max_completion_length", type=int, default=3072)
    p.add_argument("--use_vllm", action="store_true", default=True)
    p.add_argument("--vllm_mode", type=str, default="server", choices=["server", "colocate"])
    p.add_argument("--num_processes", type=int, default=4, help="World size (GPUs) to launch.")

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

    target_path = latest_checkpoint_dir(args.output_dir) or args.output_dir
    args.resume_from_checkpoint = latest_checkpoint_dir(args.output_dir)
    # --- BENCHMARK ONLY path (no Accelerate workers needed) ---
    if args.bench_only:
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
    train(args)

if __name__ == "__main__":
    launch()


