#!/bin/bash
set -e

# Launch vLLM server on GPUs 0-3
echo "[INFO] Starting vLLM server on GPUs 0-3..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./vllm_server.py > vllm_server.log 2>&1 &

# Capture server PID in case you want to kill it later
VLLM_PID=$!

# Wait for vLLM server to be ready (health check on port 8191)
echo "[INFO] Waiting for vLLM server to come online..."
until curl -s http://0.0.0.0:8191/health > /dev/null; do
    sleep 2
done
echo "[INFO] vLLM server is up."

# Launch training on GPUs 4-7
echo "[INFO] Launching training with accelerate on GPUs 4-7..."
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
    --config_file ./ds_zero3.yaml \
    cli.py

# Cleanup: kill vLLM server if training ends
kill $VLLM_PID

