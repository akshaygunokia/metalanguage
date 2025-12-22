# run on 8xH100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

RUN_NAME="grpo-$(date +%Y%m%d-%H%M%S)"
echo "Run name: $RUN_NAME"

PROJECT_DIR="$(pwd)"
mkdir -p logs
python -c "
import os
from data_utils import build_verl_parquet_openr1_bigmath_oneshot, build_aime2024_dataset

local_save_dir='data'
os.makedirs(local_save_dir, exist_ok=True)

train_ds = build_verl_parquet_openr1_bigmath_oneshot(
    subset='level_5',
    max_unique_prompts=16,
    max_train_size=128,
    seed=4,
)

test_ds = build_aime2024_dataset()

train_ds.to_parquet(os.path.join(local_save_dir, 'train.parquet'))
test_ds.to_parquet(os.path.join(local_save_dir, 'test.parquet'))
print('Wrote:', os.path.join(local_save_dir,'train.parquet'), 'and', os.path.join(local_save_dir,'test.parquet'))
"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$PROJECT_DIR/data/train.parquet \
    data.val_files=$PROJECT_DIR/data/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.multi_turn=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/tool_config/canvas_tool_config.yaml" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=$PROJECT_DIR/rewards.py \
    custom_reward_function.name=verl_reward_func \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='huggingface' \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1500 $@ \
    2>&1 | tee logs/${RUN_NAME}.log