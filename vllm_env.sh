#!/bin/bash
# 1) Use only the good set (your GPU1–GPU4). This remaps them to local IDs 0..3.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# 2) Keep NCCL off all GPU↔GPU P2P paths and SHM (CUDA IPC) paths.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

# 3) You don't have libibverbs, so make the transport explicit:
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket

# 4) Bind to the NIC NCCL already detected in your logs:
export NCCL_SOCKET_IFNAME=eno1

# 5) (Optional) Silence the spammy “disabled P2P” info lines:
export NCCL_IGNORE_DISABLED_P2P=1

# 6) (Optional) Easier debugging if anything else goes wrong:
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1

