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
