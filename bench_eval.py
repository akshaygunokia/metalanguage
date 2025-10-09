# train_grpo_limr_zero3.py
# pip install "evalchemy @ git+https://github.com/mlfoundations/evalchemy.git"
import os, re, json, argparse, sys, subprocess, glob, time, datetime
from transformers import TrainerCallback
import hashlib
from typing import Optional

def run_lm_eval(
    model_backend: str,
    model_args: str,
    tasks: str,
    batch_size: int,
    output_path: str,
    cuda_visible_devices: str | None = None,
):
    """
    Run EleutherAI/lm-evaluation-harness:
      lm_eval --model <backend> --tasks <tasks> --model_args <...>
    """
    os.makedirs(output_path, exist_ok=True)
    cmd = [
        "lm_eval",
        "--model", model_backend,
        "--tasks", tasks,
        "--model_args", model_args,
        "--batch_size", str(batch_size),
        "--output_path", output_path,
        "--device", "cuda",
        "--verbosity", "INFO",
    ]

    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    print(f"[lm-eval] Running: {' '.join(cmd)}")
    t0 = time.time()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    proc = subprocess.run(cmd, env=env, check=False)
    dt = time.time() - t0
    print(f"[lm-eval] Exit code={proc.returncode} in {dt:.1f}s -> {output_path}")
    return proc.returncode == 0


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
    proc = subprocess.run(cmd, env=env, check=False, cwd="/home/evalchemy")
    dt = time.time() - t0
    print(f"[Evalchemy] Exit code={proc.returncode} in {dt:.1f}s -> {output_path}")
    return proc.returncode == 0

class LmEvalCallback(TrainerCallback):
    """
    Runs EleutherAI/lm-evaluation-harness on AIME25/MATH500/etc. each time a checkpoint is saved.
    Logs benchmark metrics back to the trainer.
    """
    def __init__(
        self,
        trainer,
        tasks="aime25,math500",
        batch_size=2,
        every_n_saves=1,
        model_backend="vllm",  # or "hf"
        extra_model_args="",    # e.g. "dtype=bfloat16"
        output_subdir="benchmarks",
        bench_cuda=None,
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
                        to_log[f"lm-eval/{task}/{k}"] = float(v)
            if to_log:
                to_log["global_step"] = state.global_step
                self.trainer.log(to_log)
                print(f"[lm-eval] Logged: {to_log}")
        except Exception as e:
            print(f"[lm-eval] Could not parse results: {e}")

    def on_save(self, args, state, control, **kwargs):
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

        env_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        if self.bench_cuda is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.bench_cuda)

        try:
            ok = run_lm_eval(
                model_backend=self.model_backend,
                model_args=model_args,
                tasks=self.tasks,
                batch_size=self.batch_size,
                output_path=outdir,
                cuda_visible_devices=None,
            )
            if ok:
                rp = self._latest_results_json(outdir)
                if rp:
                    self._log_results(state, rp)
        finally:
            if self.bench_cuda is not None:
                if env_cuda is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = env_cuda


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




