# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from data_utils import build_dataset_openr1_bigmath_oneshot
from rewards import reward_rlvr_oneshot, small_eval_oneshot
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TrainerCallback
import torch

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
    seed=42,
)

cfg = GRPOConfig(
    output_dir="Qwen3-0.6B-Base",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=126,
    num_train_epochs=100,
    bf16=True,
    num_generations=7,
    max_prompt_length=782,
    max_completion_length=3072,
    scale_rewards=False,
    loss_type="dr_grpo",
    importance_sampling_level="sequence",
    beta=0.0,
    logging_steps=1,
    save_steps=10,
    save_total_limit=1,
    use_vllm=True,
    vllm_mode="server",
    generation_batch_size=882

)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_rlvr_oneshot,
    args=cfg,
    train_dataset=train_ds,
)
trainer.train()