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
    --config-path="$PROJECT_DIR" \
    --config-name='verl_trainer_config' \
    algorithm.adv_estimator=grpo \
    data.train_files=$PROJECT_DIR/data/train.parquet \
    data.val_files=$PROJECT_DIR/data/test.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$PROJECT_DIR/canvas_tool_config.yaml \
    actor_rollout_ref.rollout.n=8 \
    custom_reward_function.path=$PROJECT_DIR/rewards.py \
    custom_reward_function.name=verl_reward_func \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='huggingface' \
    trainer.experiment_name=$RUN_NAME $@ \
    2>&1 | tee logs/${RUN_NAME}.log