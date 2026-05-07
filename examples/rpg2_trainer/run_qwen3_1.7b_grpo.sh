#!/bin/bash
# GRPO baseline training script for Qwen3-1.7B on math reasoning tasks.
# Used as a GRPO baseline.
#
# Usage:
#   bash run_qwen3_1.7b_grpo.sh
#
# Data format: standard prompt/response parquet (no [TEACHER_CONTEXT_TOKEN] required).

set -xeuo pipefail

export HF_ENDPOINT='https://hf-mirror.com'

# On GCP A3/A3+ instances (H100/H200) the Google Infrastructure Base (GIB) provides a
# custom NCCL network backend.  Without these env vars NCCL cannot find the gIB plugin
# and crashes immediately during ncclCommInitRank on rank 0.
# The script is a no-op on non-GCP machines where the file doesn't exist.
if [ -f /usr/local/gib/scripts/set_nccl_env.sh ]; then
    source /usr/local/gib/scripts/set_nccl_env.sh
fi
# Do NOT set NCCL_CUMEM_ENABLE — the GIB NCCL shim requires it to be completely unset.
unset NCCL_CUMEM_ENABLE
# Override NCCL_NET to Socket: the gIB RDMA plugin fails to initialize on this instance
# (gIB userspace devices are not accessible), causing an ncclInvalidUsage crash.
# NVLink handles all intra-node GPU↔GPU transfers; Socket is only used for rendezvous.
export NCCL_NET=Socket
# flashinfer-cubin and flashinfer-python may be different patch versions.
# Bypass the strict version check to avoid a RuntimeError/segfault in vLLM workers.
export FLASHINFER_DISABLE_VERSION_CHECK=1
# FlashInfer autotune crashes with std::length_error: vector::reserve on H100/H200
# with Qwen3 model and vLLM 0.12.0. Skip it (performance optimization only, not required).
export VLLM_SKIP_FLASHINFER_AUTOTUNE=1
# Force FlashAttention-2 backend: flashinfer-cubin 0.6.7.post3 has a binary ABI
# mismatch with PyTorch 2.9.0 that causes a segfault when the cubin is dlopen'd.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# DeepGEMM JIT compiler crashes with std::length_error: vector::reserve on H100
# when imported (in deep_gemm_warmup via _lazy_init).  Qwen3-1.7B is dense bf16,
# so deep_gemm (FP8 block-sparse) provides no benefit.  Disable it entirely.
export VLLM_USE_DEEP_GEMM=0

project_name='verl'
exp_name="Qwen3-1.7B-GRPO"
NNODES=${NNODES:-1}

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
# use_iterative_ref_model=False
kl_loss_coef=1e-3
entropy_coeff=0

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 2))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 / 2))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True # True, debug=False
filter_groups_metric=acc
max_num_gen_batches=10 # 10, debug=10
train_prompt_bsz=256 # 512, debug=32
gen_prompt_bsz=$((train_prompt_bsz * 1)) # $((train_prompt_bsz * 1))
train_prompt_mini_bsz=32 # 32, debug=16
n_resp_per_prompt=8 # 16, debug=2
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${PWD}"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-1.7B"}
# Use local HF cache path if available (avoids 429 rate-limit from hf-mirror when
# vLLM re-validates the model config at startup).  Falls back to the HF model ID.
# _LOCAL_QWEN3_1p7B="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/$(ls "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/" 2>/dev/null | tail -1)"
# if [ -f "${_LOCAL_QWEN3_1p7B}/config.json" ]; then
#     MODEL_PATH=${MODEL_PATH:-"${_LOCAL_QWEN3_1p7B}"}
# else
#     MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-1.7B"}
# fi
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/math-dapo-noteacher-shuffled.parquet"}
# TEST_FILE as a comma-separated list without inner quotes (Hydra parses [a,b,c] as a list)
TEST_FILE=${TEST_FILE:-"[${RAY_DATA_HOME}/data/amc-23.parquet,${RAY_DATA_HOME}/data/aime-2024.parquet,${RAY_DATA_HOME}/data/aime25.parquet]"}

rollout_engine=vllm
rollout_mode=async
gpu_memory_utilization=0.75
shuffle_dataset=True

test_freq=10
save_freq=20
total_epochs=2
total_training_steps=400

# Sampling params at rollouts
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=1.0

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
infer_micro_batch_size=null
train_micro_batch_size=null
offload=False # small model, offloading slows training
gen_tp=1
entropy_checkpointing=True
# to run on H100 with Python 3.13, actor_rollout_ref.rollout.enforce_eager=True
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
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
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    +algorithm.filter_groups.enable=${enable_filter_groups} \
    +algorithm.filter_groups.metric=${filter_groups_metric} \
    +algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
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
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.entropy_checkpointing=${entropy_checkpointing} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
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
