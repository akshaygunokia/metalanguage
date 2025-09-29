#!/usr/bin/env python3
"""
Programmatic launcher for TRL vLLM server with data parallelism.

Equivalent to:
CUDA_VISIBLE_DEVICES=0,1,2,3 trl vllm-serve \
    --model Qwen/Qwen3-4B-Thinking-2507 \
    --max_model_len 10240 \
    --tensor_parallel_size 1 \
    --data_parallel_size 4
"""

import os
from trl.scripts import vllm_serve

def main():
    # ---- configure these ----
    model_id = "Qwen/Qwen3-4B-Instruct-2507"
    max_model_len = 10240
    tensor_parallel = 1
    data_parallel = 5
    host = "0.0.0.0"
    port = 8191
    # -------------------------

    # Build the same dataclass used by the CLI
    args = vllm_serve.ScriptArguments(
        model=model_id,
        tensor_parallel_size=tensor_parallel,
        data_parallel_size=data_parallel,
        max_model_len=max_model_len,
        host=host,
        port=port,
        # optional extras
        dtype="auto",
        gpu_memory_utilization=0.9,
        trust_remote_code=False,
    )

    # Make sure we expose all GPUs you want to use
    # e.g. "0,1,2,3"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4")

    # Launch the multi-process FastAPI server
    vllm_serve.main(args)

if __name__ == "__main__":
    main()


