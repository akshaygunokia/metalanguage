# train_grpo_limr_zero3.py
# pip install "evalchemy @ git+https://github.com/mlfoundations/evalchemy.git"
import os, re, json, argparse, sys, subprocess, glob, time, datetime
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
import hashlib
from data_utils import build_dataset_openr1_bigmath_oneshot
from rewards import small_eval_oneshot, reward_bigmath_accuracy
from bench_eval import EvalchemyCallback
from bench_eval import run_evalchemy
# -------------------------------
# Training (per rank)
# -------------------------------
def train(args):
    # Helpful alloc settings
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    set_seed(args.seed)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Dataset  (pass tok, and keep RLVR builder)
    HARD_SUBSETS = ["level_4"]
    MAX_TRAIN_EX = 4
    EVAL_HOLDOUT = 2
    train_ds, eval_ds = build_dataset_openr1_bigmath_oneshot(
        subsets=HARD_SUBSETS,
        max_train_examples=MAX_TRAIN_EX,
        eval_holdout=EVAL_HOLDOUT,
        batch_size=args.batch_size,
        seed=args.seed,
        tok=tok
    )

    # GRPO config
    cfg = GRPOConfig(
        output_dir=args.output_dir,
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
        report_to="wandb",
        logging_steps=1,
        save_steps=1,
        save_total_limit=1,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_server_host=args.vllm_host,
        vllm_server_port=args.vllm_port,
        generation_batch_size=args.generation_batch_size,
    )

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=dtype, attn_implementation="flash_attention_2")
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        reward_funcs=reward_bigmath_accuracy,
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
        ckpt_dir = args.resume_from_checkpoint
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

