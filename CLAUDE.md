
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl (Volcano Engine Reinforcement Learning for LLMs) is a flexible, efficient RL post-training library for large language models. It supports multiple training backends (FSDP/FSDP2, Megatron-LM, TorchTitan, Veomni), rollout engines (vLLM, SGLang, HF Transformers, TensorRT-LLM), and RL algorithms (PPO, GRPO, GSPO, ReMax, REINFORCE++, RLOO, DAPO, PRIME, etc.). Scales to 671B models with expert parallelism.

## Common Commands

### Installation
```bash
pip install -e .[test,vllm]    # Development install with vLLM
pip install -e .[test,sglang]  # Development install with SGLang
```

### Linting and Formatting
```bash
pip install pre-commit && pre-commit install
pre-commit run                              # Staged changes only
pre-commit run --all-files                  # All files
pre-commit run --all-files ruff             # Ruff only
pre-commit run --all-files autogen-trainer-cfg  # Verify generated YAML configs
```

Ruff is the linter/formatter (line-length=120). MyPy is used for type checking. Pre-commit also runs license header checks, docstring coverage checks, and Python compilation validation.

### Running Tests
```bash
# CPU tests (file name pattern: test_*_on_cpu.py)
pytest tests/**/test_*_on_cpu.py

# GPU unit tests (all test_*.py files excluding on_cpu.py suffix)
pytest tests/path/to/test_file.py

# Single test
pytest tests/path/to/test_file.py::test_function_name
```

Test dependencies: `pytest`, `pytest-asyncio`, `pytest-rerunfailures`

### Building Docs
```bash
cd docs && pip install -r requirements-docs.txt && make clean && make html
```

### Running Training (PPO example)
```bash
python -m verl.trainer.main_ppo  # Uses Hydra config at verl/trainer/config/ppo_trainer.yaml
```

## Architecture

### Core Abstractions

**DataProto** (`verl/protocol.py`): The universal data transfer protocol between all components. Built on TensorDict, it supports batching, padding, distributed operations, and serialization. All worker communication flows through DataProto.

**Single Controller Pattern** (`verl/single_controller/`): The programming model where one controller process orchestrates distributed workers via Ray. The `@register` decorator on worker methods defines dispatch/execute semantics (how data is distributed to and collected from workers). Key dispatch modes: `DP_COMPUTE` (data-parallel split), `ONE_TO_ALL` (broadcast), `ALL_TO_ALL`.

**HybridFlow**: Combines single-controller coordination with multi-controller execution. The trainer (e.g., `RayPPOTrainer`) acts as the single controller, dispatching work to worker groups that internally use data-parallel or tensor-parallel execution.

### Worker System

Workers are the execution units, each corresponding to a model role:

- **Actor Workers** (`verl/workers/actor/`): Policy model training. Implementations per backend: `dp_actor.py` (FSDP), `megatron_actor.py`, `torchtitan_actor.py`, `veomni_actor.py`
- **Critic Workers** (`verl/workers/critic/`): Value function computation, same backend pattern
- **Rollout Workers** (`verl/workers/rollout/`): Generation via inference engines — `vllm_rollout/`, `sglang_rollout/`, `hf_rollout.py`, `trtllm_rollout/`
- **Reward Manager** (`verl/workers/reward_manager/`): Computes rewards from rule-based functions or reward models
- **Engine Workers** (`verl/workers/engine_workers.py`): New unified worker implementation (activated with `use_legacy_worker_impl: disable`) that fuses actor+rollout+ref into `ActorRolloutRefWorker`

The legacy worker path (`verl/workers/fsdp_workers.py`, `megatron_workers.py`) and new engine-based path coexist, controlled by `trainer.use_legacy_worker_impl` config.

### Training Engine Backends (`verl/workers/engine/`)

Each backend (`fsdp/`, `megatron/`, `torchtitan/`, `veomni/`) implements a common interface for: model initialization, forward/backward passes, optimizer steps, weight syncing between training and inference, and checkpointing. The engine abstraction is defined in `base.py`.

### Configuration System

Uses **Hydra** with composable YAML configs under `verl/trainer/config/`. The main trainer configs (e.g., `ppo_trainer.yaml`) use Hydra defaults to compose per-component configs from subdirectories:
- `actor/`, `critic/`, `ref/` — per-backend actor/critic/ref configs
- `rollout/` — rollout engine configs
- `engine/` — model engine configs (dp, megatron, torchtitan, veomni)
- `algorithm/`, `data/`, `reward/`, `model/`, `optim/` — other component configs

The `model_engine` default (e.g., `dp`, `megatron`) drives which backend-specific configs are composed. Config overrides use standard Hydra CLI syntax.

Generated configs (`_generated_*.yaml`) are auto-produced by `scripts/generate_trainer_config.sh` and verified by the `autogen-trainer-cfg` pre-commit hook. Do not edit generated configs manually.

### Trainer Orchestration

`verl/trainer/ppo/ray_trainer.py` contains `RayPPOTrainer`, the main training loop for PPO/GRPO-family algorithms. The entry point (`verl/trainer/main_ppo.py`) initializes Ray, creates a `TaskRunner` that wires up worker roles to resource pools, then calls `trainer.fit()`.

The `Role` enum maps logical roles (ActorRollout, Critic, RefPolicy, RewardModel) to Ray worker groups. Resource pools define GPU allocations per role.

### Model Registry

`verl/models/registry.py` and `verl/models/transformers/weight_loader_registry.py` handle model registration and weight loading. Model implementations exist for specific architectures (Llama, Qwen2) and for Megatron-Core (`verl/models/mcore/`).

### Experimental Features (`verl/experimental/`)

Active development areas: fully async policy training, one-step off-policy, transfer queues for inter-worker data transfer, reward loops, agent loops for multi-turn RL, and VLA (Vision-Language Action) models. These may have unstable APIs.

## Test Layout

Each folder under `tests/` mirrors a `verl/` subpackage. Special prefixed folders:
- `special_distributed/` — multi-GPU unit tests
- `special_e2e/` — end-to-end training/generation scripts (shell scripts, not pytest)
- `special_npu/` — Ascend NPU tests
- `special_sanity/` — quick sanity checks (docstrings, license headers, dataproto usage)

**Naming convention**: Files ending in `_on_cpu.py` run in CPU-only CI; all other `test_*.py` files require GPU.

## Code Style

- Line length: 120 characters
- Linter: Ruff (rules: E, F, UP, B, I, G; key ignores: F403/F405 star imports, E731 lambda assignment)
- All source files require Apache 2.0 license headers
- Type checking with MyPy is enforced (strict only for `verl.trainer.config.algorithm`, `verl.trainer.ppo.core_algos`, `verl.trainer.ppo.reward`, `verl.workers.reward_manager`)
