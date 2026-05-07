#!/bin/bash
# SDPG training script for Qwen3-1.7B on math reasoning tasks.
#
# SDPG (Self-Distillation Policy Gradient) extends GRPO/DAPO with an
# exact full-vocabulary forward KL distillation term computed against the
# current actor with privileged context c:
#   D_KL(π_θ(·|x) ‖ π_teacher(·|c,x))  per response token
# The KL is computed on-the-fly inside update_policy (no separate
# teacher_log_prob pre-computation step).
#
# Key SDPG hyperparameters:
#   BETA     – full-vocab KL distillation coefficient (default 0.001)
#   ALPHA    – KL regularization coefficient against frozen π_old (default 0.001)
#   KL_MODE  – alpha-term mode: fkl, rkl, ufkl, urkl (default urkl)
#
# Memory note: SDPG materializes (B, T, V) logit tensors inside update_policy.
# Use entropy_checkpointing=True to reduce peak GPU memory.
#
# Usage:
#   BETA=0.001 ALPHA=0.001 KL_MODE=urkl bash run_qwen3_1.7b_sdpg.sh
#
# Data format: each sample's prompt[0].content must contain
#   "<actor question>[TEACHER_CONTEXT_TOKEN][teacher context]"
# The actor sees only the part before [TEACHER_CONTEXT_TOKEN];
# the teacher (current π_θ with context c) sees the full content.

set -xeuo pipefail

export HF_ENDPOINT='https://hf-mirror.com'

if [ -f /usr/local/gib/scripts/set_nccl_env.sh ]; then
    source /usr/local/gib/scripts/set_nccl_env.sh
fi
unset NCCL_CUMEM_ENABLE
export NCCL_NET=Socket
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_SKIP_FLASHINFER_AUTOTUNE=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_DEEP_GEMM=0

BETA=${BETA:-0.001}
ALPHA=${ALPHA:-0.001}
BETA_WARMUP_STEPS=${BETA_WARMUP_STEPS:-0}
BETA_DECAY_STEPS=${BETA_DECAY_STEPS:-0}
BETA_POSITIVE_ADV_ONLY=${BETA_POSITIVE_ADV_ONLY:-False}
project_name='verl-sdpg'
KL_MODE=${KL_MODE:-urkl}
exp_name="Qwen3-1.7B-SDPG-${KL_MODE}-BETA${BETA}-ALPHA${ALPHA}-cliphigh"
NNODES=${NNODES:-1}

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 2))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 / 2))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

clip_ratio_low=0.2
clip_ratio_high=0.28

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=256
gen_prompt_bsz=$((train_prompt_bsz * 1))
train_prompt_mini_bsz=32
n_resp_per_prompt=8

RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${PWD}"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-1.7B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/math-dapo-teacher-shuffled.parquet"}
TEST_FILE=${TEST_FILE:-"[${RAY_DATA_HOME}/data/amc-23.parquet,${RAY_DATA_HOME}/data/aime-2024.parquet,${RAY_DATA_HOME}/data/aime25.parquet]"}

rollout_engine=vllm
rollout_mode=async
gpu_memory_utilization=0.75
shuffle_dataset=True

test_freq=10
save_freq=20
total_epochs=2
total_training_steps=400

temperature=1.0
top_p=1.0
top_k=-1
val_top_p=1.0

sp_size=1
use_dynamic_bsz=True
infer_micro_batch_size=null
train_micro_batch_size=null
offload=False
gen_tp=1
entropy_checkpointing=True

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=sdpg \
    data.train_files="${TRAIN_FILE}" \
    data.val_files=${TEST_FILE} \
    data.prompt_key=prompt \
    data.shuffle=$shuffle_dataset \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    +data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.truncation='left' \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    +algorithm.filter_groups.enable=${enable_filter_groups} \
    +algorithm.filter_groups.metric=${filter_groups_metric} \
    +algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.beta=${BETA} \
    actor_rollout_ref.actor.policy_loss.alpha=${ALPHA} \
    actor_rollout_ref.actor.policy_loss.kl_mode=${KL_MODE} \
    actor_rollout_ref.actor.policy_loss.beta_warmup_steps=${BETA_WARMUP_STEPS} \
    actor_rollout_ref.actor.policy_loss.beta_decay_steps=${BETA_DECAY_STEPS} \
    actor_rollout_ref.actor.policy_loss.beta_distill_positive_advantage_only=${BETA_POSITIVE_ADV_ONLY} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    +reward_model.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    reward_model.reward_manager=dapo \
    +reward_model.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=1 \
     $@
