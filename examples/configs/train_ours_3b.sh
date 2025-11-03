#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98

CUDA_IDS=4,5,6,7
N_GPU=4

MODEL_PATH=/data3/cyn/models/Qwen2.5-VL-3B-Instruct/

TOTAL_EPOCHES=2
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=384
VAL_BATCH_SIZE=512
MINI_ROLLOUT_BATCH_SIZE=384
MAX_PROMPT_LENGTH=4096
rollout=5


top_p_perception_tokens=0.4
advantage_scaling_min=0.9
entropy_penalty_coef=0.06

top_p_entropy_tokens=0.2


EXP_NAME="entropy_tokens_${top_p_entropy_tokens}_ep${TOTAL_EPOCHES}_rollout${rollout}_MINI_ROLLOUT_BATCH_SIZE${MINI_ROLLOUT_BATCH_SIZE}"

CONGI_FILE="examples/configs/config.yaml"
TRAIN_FILE="/data3/cyn/data/ViRL39K_train/"
VAL_FILE="/data3/cyn/data/MMK12_test/"

FORMAT_PROMPT="examples/format_prompt/math_format_perception.jinja"
REWARD_FUNCTION="examples/reward_function/math.py:compute_score_wo_format"

CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=${CONGI_FILE} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.mini_rollout_batch_size=${MINI_ROLLOUT_BATCH_SIZE} \
    data.format_prompt=${FORMAT_PROMPT} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.total_epochs=${TOTAL_EPOCHES} \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    trainer.project_name="3b_vppo" \
    trainer.logger=['console','wandb'] \
    algorithm.use_vppo_on_entropy=True \
    algorithm.use_vppo_on_perception=False \
    algorithm.use_advantage_shaping=False \
    algorithm.use_entropy_penalty=False \
    algorithm.top_p_perception_tokens=${top_p_perception_tokens} \
    algorithm.entropy_penalty_coef=${entropy_penalty_coef} \
    algorithm.advantage_scaling_min=${advantage_scaling_min} \
    worker.rollout.n=${rollout} \
    worker.actor.micro_batch_size_per_device_for_experience=8 \
    worker.actor.micro_batch_size_per_device_for_update=2