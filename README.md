# SDPG: Self-Distilled Policy Gradient

[![arXiv](https://img.shields.io/badge/arXiv-2606.04036-b31b1b.svg)](https://arxiv.org/abs/2606.04036)
[![Website](https://img.shields.io/badge/Project-Website-blue)](https://complex-reasoning.github.io/SDPG) 
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange.svg) 

On-policy self-distillation, where a language model conditions on privileged context to supervise its own generations, is a promising source of dense supervision for sparse-reward reinforcement learning. Actually, it can be instantiated as an auxiliary full-vocabulary student-to-teacher reverse Kullback-Leibler divergence loss. We therefore propose **SDPG**, a self-distilled policy-gradient framework that combines group-relative verifier advantages with normalized standard deviation, exact full-vocabulary on-policy self-distillation, as well as reference-policy KL regularization. Empirically, SDPG improves stability and performance over RLVR and self-distillation baselines. This repository implements the paper "[Self-Distilled Policy Gradient](https://arxiv.org/abs/2606.04036)" and related privileged-context training methods on top of the [verl](https://github.com/volcengine/verl) RLHF framework.

- Authors: [Yifeng Liu](https://lauyikfung.github.io)\*, [Shiyuan Zhang](https://scholar.google.com/citations?user=MpeEZ0IAAAAJ)\*, [Yifan Zhang](https://yifzhang.com)\*, [Quanquan Gu](https://web.cs.ucla.edu/~qgu/)

[[Webpage](https://complex-reasoning.github.io/SDPG)] [[Huggingface](https://huggingface.co/papers/2606.04036)]

## Methods

| Method | Loss | Teacher | Ref model |
|--------|------|---------|-----------|
| GRPO | DAPO dual-clip PPO | None | Optional |
| **SDPG** | DAPO clip + full-vocab KL distillation + α-reg | Current π_θ(·\|c,x) | Yes (frozen, for α term) |
| OPSD | PPO-REINFORCE with per-token weight | Frozen π_ref(·\|c,x) | Yes |
| RLSD | DAPO clip with evidence-reweighted advantage | Current π_θ(·\|c,x) | No |

**SDPG** is the main contribution. It extends GRPO with an exact per-token forward KL between the actor (without privileged context) and itself conditioned on privileged context c:

$$\mathcal{L} = \underbrace{\ell^{\text{clip}}}_{\text{GRPO}} + \beta \cdot \underbrace{D_{\mathrm{KL}}(\pi_\theta(\cdot|x) \| \pi_\theta(\cdot|c,x))}_{\text{self-distillation}} + \alpha \cdot \underbrace{f(\pi_\theta, \pi_{\text{ref}})}_{\text{KL reg}}$$

The KL is computed on-the-fly inside `update_policy` — no separate teacher log-prob pre-computation. The α-regularization supports four modes (`fkl`, `rkl`, `ufkl`, `urkl`).

## Data Format

All methods that use privileged context share the same data format. The first message content encodes both the actor question and teacher context, separated by a special token:

```
prompt[0].content = "<actor question>[TEACHER_CONTEXT_TOKEN]<teacher context>"
```

At training time `rl_dataset.py` splits on this sentinel:
- **Actor** receives everything **before** `[TEACHER_CONTEXT_TOKEN]` (plain question)
- **Teacher** receives the full content including the privileged context after the token

Two datasets are provided:

| File | Use |
|------|-----|
| `math-dapo-noteacher-shuffled-boxed.parquet` | GRPO baseline (no teacher context) |
| `math-dapo-teacher-shuffled-boxed.parquet` | SDPG / OPSD / RLSD (includes teacher context) |

The teacher context format is:
```
The correct answer to this problem is: {answer}
Use this to verify your reasoning, but show your full solution process.
```

## Requirements

- 8× A100/H100/H200 GPUs (scripts default to 1 node, 8 GPUs)
- [verl](https://github.com/volcengine/verl) dependencies installed
- Ray cluster running locally (`ray start --head --num-gpus=8 --num-cpus=104`)
- Model: `Qwen/Qwen3-4B` (or set `MODEL_PATH` to a local cache)
- Data placed under `$RAY_DATA_HOME/data/`

## Reproducing Qwen3-4B Experiments

All scripts are in `examples/rpg2_trainer/`. Set environment variables to override defaults.

### GRPO Baseline

```bash
bash examples/rpg2_trainer/run_qwen3_4b_grpo_original_boxed.sh
```

Key settings: `lr=1e-6`, `n=8`, `train_batch_size=128`, `gpu_memory_utilization=0.6`.  
Uses noteacher data. No ref model, no teacher.

---

### SDPG

```bash
# Default: BETA=0.001, ALPHA=0.001, KL_MODE=urkl
bash examples/rpg2_trainer/run_qwen3_4b_sdpg_boxed.sh

# Custom hyperparameters
BETA=0.001 ALPHA=0.001 KL_MODE=urkl bash examples/rpg2_trainer/run_qwen3_4b_sdpg_boxed.sh
```

Key settings: `lr=1e-6`, `n=8`, `train_batch_size=128`, `gpu_memory_utilization=0.75`, `entropy_checkpointing=True`.  
Uses teacher data. Spawns a frozen ref model worker for the α-regularization term.

**KL mode options** (`KL_MODE`):

| Mode | α term |
|------|--------|
| `fkl` | $\pi_{\text{ref}} / \pi_\theta$ |
| `rkl` | $\tfrac{1}{2}(\log w + 1)^2$ |
| `ufkl` | $\pi_{\text{ref}}/\pi_\theta + \log w$ |
| `urkl` | $\tfrac{1}{2}(\log w)^2$ **(default)** |

**Beta schedule** (optional):
```bash
BETA_WARMUP_STEPS=50 BETA_DECAY_STEPS=100 bash examples/rpg2_trainer/run_qwen3_4b_sdpg_boxed.sh
```

**Distillation gating** — restrict β term to positively-advantaged responses:
```bash
BETA_POSITIVE_ADV_ONLY=True bash examples/rpg2_trainer/run_qwen3_4b_sdpg_boxed.sh
```

> **Memory note:** SDPG materializes `(B, T, V)` actor+teacher logits simultaneously during `update_policy`. If `generate_sequences` is slow (vLLM KV-cache preemptions), lower `gpu_memory_utilization` to `0.6`.

---

### OPSD

```bash
bash examples/rpg2_trainer/run_qwen3_4b_opsd_boxed.sh
```

Key settings: `lr=5e-6`, `n=8`, `entropy_coeff=0.01`, `gpu_memory_utilization=0.6`.  
Uses teacher data. Teacher = **frozen** π_ref (initial weights, not the current actor).

---

### RLSD

```bash
# Default: lambda=0.5, lambda_decay_steps=50, epsilon_w=0.2
bash examples/rpg2_trainer/run_qwen3_4b_rlsd_boxed.sh

RLSD_LAMBDA=0.5 RLSD_LAMBDA_DECAY_STEPS=50 bash examples/rpg2_trainer/run_qwen3_4b_rlsd_boxed.sh
```

Key settings: `lr=1e-6`, `n=8`, `gpu_memory_utilization=0.75`.  
Uses teacher data. No frozen ref model. Teacher signal only reweights advantage magnitude.

---

## Evaluation

All scripts evaluate on three benchmarks every `test_freq=10` steps:

| Dataset | File |
|---------|------|
| AMC 2023 | `amc-23-boxed.parquet` |
| AIME 2024 | `aime-2024-boxed.parquet` |
| AIME 2025 | `aime25-boxed.parquet` |

Validation uses `n=32` samples per problem at `temperature=1.0`.

## Key Files

| File | Description |
|------|-------------|
| `verl/trainer/ppo/core_algos.py` | All loss functions (`compute_sdpg_loss`, GRPO, OPSD, RLSD) |
| `verl/workers/actor/dp_actor.py` | `update_policy`: dispatches loss modes, runs on-the-fly KL for SDPG |
| `verl/trainer/ppo/ray_trainer.py` | Training loop: teacher log-prob computation for RLSD/OPSD |
| `verl/utils/dataset/rl_dataset.py` | `[TEACHER_CONTEXT_TOKEN]` splitting, `teacher_input_ids` tokenization |
| `verl/trainer/ppo/utils.py` | `need_reference_policy()`: spawns frozen ref worker for SDPG/OPSD |
| `verl/workers/config/actor.py` | `PolicyLossConfig`: β, α, `kl_mode`, beta schedule fields |

## Acknowledgements

  - [volcengine/verl: verl: Volcano Engine Reinforcement Learning for LLMs](https://github.com/volcengine/verl) for providing coding base

## Citation

If you use SDPG in your research or application, please consider citing it!

```bibtex
@article{liu2026self,
      title={Self-Distilled Policy Gradient}, 
      author={Liu, Yifeng and Zhang, Shiyuan and Zhang, Yifan and Gu, Quanquan},
      journal={arXiv preprint arXiv:2606.04036}, 
}
```
