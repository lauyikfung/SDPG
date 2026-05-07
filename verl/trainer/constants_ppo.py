# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # NOTE: NCCL_CUMEM_ENABLE is intentionally NOT set here.
        # On GCP (H100/H200) the NCCL shim requires this variable to be completely *unset*;
        # setting it to 0 (even as a precaution for vllm disaggregated weight-sync) causes the
        # guest_config_checker to abort the NCCL init and kill the worker.
        # vllm's own workers set NCCL_CUMEM_ENABLE=0 internally when they need it.
        # If you run in disaggregated mode on non-GCP hardware and experience weight-sync hangs,
        # export NCCL_CUMEM_ENABLE=0 in your launch script instead.
        # NOTE: NCCL_NET=Socket is set when the GIB setup script is present but the gIB net
        # plugin fails to initialize (because gIB RDMA is not available as a userspace device).
        # For single-node (intra-node) training, NVLink handles all GPU-GPU transfers; Socket
        # is only needed for the initial rendezvous.  This avoids the "Failed to initialize
        # any NET plugin" ncclInvalidUsage crash that occurs when NCCL_NET=gIB is set but
        # libibverbs / gIB devices are not accessible.
        # TODO: disable compile cache due to cache corruption issue
        # https://github.com/vllm-project/vllm/issues/31199
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        # Needed for multi-processes colocated on same NPU device
        # https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0143.html
        "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
        # flashinfer-cubin and flashinfer-python may be different patch versions.
        # The strict version check causes a RuntimeError that segfaults the vLLM
        # worker process on this machine (cubin 0.6.7.post3 vs python 0.5.3).
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        # FlashInfer autotune crashes with std::length_error: vector::reserve on
        # certain hardware/model combinations (Qwen3 on H100/H200, vLLM 0.12.0).
        # This env var bypasses the autotune in kernel_warmup.py (patched inline).
        # Autotune is a performance optimization only — skipping it is safe.
        "VLLM_SKIP_FLASHINFER_AUTOTUNE": "1",
        # Force FlashAttention-2 backend instead of FlashInfer.
        # The installed flashinfer-cubin (0.6.7.post3) has a binary ABI mismatch
        # with PyTorch 2.9.0 that causes a segfault when its CUDA extension is
        # loaded via dlopen (during c10::OperatorEntry::registerKernel).
        # FLASH_ATTN is fully supported on H100 and avoids loading the cubin.
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        # DeepGEMM JIT compiler crashes with std::length_error: vector::reserve
        # on H100 (SM 9.0) when `import deep_gemm` is executed inside
        # VllmWorker-0 (spawned by EngineCore_DP0).  Qwen3-1.7B is dense bf16 —
        # no FP8 block-quantization, no MoE — so deep_gemm provides no benefit.
        "VLLM_USE_DEEP_GEMM": "0",
    },
}

# GCP A3/A3+ (H100/H200) GIB NCCL environment.  These are read by the GIB NCCL plugin
# (/usr/local/gib/lib64/libnccl-net.so) to enable high-bandwidth GPU interconnects.
# They are added to PPO_RAY_RUNTIME_ENV only when the GIB setup script is present so
# that non-GCP machines are not affected.
import os as _os
_gib_set_env = "/usr/local/gib/scripts/set_nccl_env.sh"
if _os.path.exists(_gib_set_env):
    import subprocess as _subprocess
    _result = _subprocess.run(
        ["bash", "-c", f"source {_gib_set_env} && env"],
        capture_output=True, text=True
    )
    _gib_env_keys = {
        # NOTE: NCCL_NET is intentionally excluded here.
        # On this GCP instance the gIB RDMA userspace devices are not accessible
        # (libibverbs is present but gIB hardware init fails), so NCCL_NET=gIB
        # causes "Failed to initialize any NET plugin" → ncclInvalidUsage crash.
        # NVLink handles all intra-node GPU↔GPU transfers; Socket is only needed
        # for the initial rendezvous.  NCCL_NET=Socket is set below explicitly.
        "NCCL_CROSS_NIC", "NCCL_NET_GDR_LEVEL",
        "NCCL_P2P_NET_CHUNKSIZE", "NCCL_NVLS_CHUNKSIZE",
        "NCCL_IB_ADAPTIVE_ROUTING", "NCCL_IB_QPS_PER_CONNECTION",
        "NCCL_IB_TC", "NCCL_IB_FIFO_TC", "NCCL_TUNER_CONFIG_PATH",
    }
    for _line in _result.stdout.splitlines():
        if "=" in _line:
            _k, _, _v = _line.partition("=")
            if _k in _gib_env_keys:
                PPO_RAY_RUNTIME_ENV["env_vars"][_k] = _v
    # Force Socket transport so gIB plugin failure doesn't block NCCL init.
    # This overrides NCCL_NET=gIB that set_nccl_env.sh would otherwise inject.
    PPO_RAY_RUNTIME_ENV["env_vars"]["NCCL_NET"] = "Socket"


_PPO_RUNTIME_ENV_FORCE_OVERRIDE = {
    "NCCL_NET",
    # Ray workers may be spawned (not forked), so they don't automatically inherit the
    # parent process's exported env vars.  Any var that must reach spawned workers
    # (vLLM subprocesses, FSDP workers) needs to be in env_vars explicitly.
    "FLASHINFER_DISABLE_VERSION_CHECK",
    # Must be force-propagated so vLLM worker subprocesses see it (they are spawned,
    # not forked, and won't inherit from the parent unless it's in env_vars).
    "VLLM_SKIP_FLASHINFER_AUTOTUNE",
    "VLLM_ATTENTION_BACKEND",
    # DeepGEMM must be disabled in all spawned subprocesses (VllmWorker-0 etc.)
    "VLLM_USE_DEEP_GEMM",
}
"""Keys that should always be set in the Ray runtime env, even if already in os.environ.

These are vars where the value in PPO_RAY_RUNTIME_ENV is intentionally different from
(i.e. corrects) what the parent process has set, OR vars that must reach spawned
(non-forked) worker processes that don't inherit the parent env automatically."""


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    for key in list(runtime_env["env_vars"].keys()):
        if key in _PPO_RUNTIME_ENV_FORCE_OVERRIDE:
            continue  # always propagate these, even if the parent env has them
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    return runtime_env
