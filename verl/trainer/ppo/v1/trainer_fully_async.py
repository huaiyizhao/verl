# Copyright 2026 Tencent Inc. and/or its affiliates
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
"""Streaming (``fully_async``) PPO trainer for V1.

Builds on ``separate_async`` (standalone rollout + checkpoint-engine weight sync). The only
behavioral difference is *who drives prompt feeding*: instead of ``step()`` feeding exactly one
batch per training step, an autonomous background feeder thread continuously streams prompts into
TransferQueue (bounded by a staleness / in-flight budget), while ``step()`` only samples + trains.
This decouples production rate from consumption rate so rollout overlaps training.

The whole feeder (thread loop, throttling, abnormal-rollout discard, weight-sync pause/resume)
lives here and touches the base trainer only through its public state (``train_dataloader``,
``agent_loop_manager``, ``global_steps``, ``replay_buffer``). ``trainer_base`` is left exactly as
upstream: the base ``step()`` still calls ``self._add_batch_to_generate()`` unconditionally, and we
make that a no-op once the feeder owns generation (see :meth:`_add_batch_to_generate`).
"""

import logging
import math
import os
import threading
import uuid
from collections import defaultdict

import numpy as np
import torch
import transfer_queue as tq
from omegaconf import DictConfig
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.utils import tensordict_utils as tu
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# Per-step timing breakdown to stdout (off by default). Enabled by the master VERL_PROFILE switch
# or the dedicated VERL_STEP_PROFILE, so you can get just the step profile without the verbose
# postprocess/resolve/rowcheck diagnostics.
_STEP_PROFILE = os.getenv("VERL_PROFILE", "0") not in ("0", "false", "False", "") or os.getenv(
    "VERL_STEP_PROFILE", "0"
) not in ("0", "false", "False", "")

# Leaf phases recorded in self.timing_raw during a step, in pipeline order (some are optional and
# only present when the corresponding component is enabled).
_STEP_PHASES = (
    "gen",
    "reward",
    "old_log_prob",
    "ref",
    "values",
    "adv",
    "update_critic",
    "update_actor",
    "update_weights",
    "save_checkpoint",
)


def compute_max_inflight_prompts(staleness_threshold: float, parameter_sync_step: int, train_batch_size: int) -> int:
    """Compute the in-flight prompt budget for the streaming feeder.

    The budget caps how many un-consumed prompts may exist in TransferQueue at once, bounding how
    far the rollouter runs ahead of training. Mirrors the ``fully_async_policy`` reference formula
    with ``require_batches=1``.
    """
    budget = int((1 + staleness_threshold) * parameter_sync_step * train_batch_size)
    assert budget >= train_batch_size, (
        f"in-flight budget ({budget}) must be >= train_batch_size ({train_batch_size}); "
        f"check staleness_threshold/parameter_sync_step"
    )
    return budget


@register_trainer("fully_async")
class PPOTrainerFullyAsync(PPOTrainerSeparateAsync):
    """Streaming asynchronous PPO trainer (autonomous background feeder + standalone rollout)."""

    def __init__(self, config: DictConfig):
        super().__init__(config)  # inherits separate_async asserts + bypass_mode
        fa = self.config.trainer.v1.fully_async
        self._poll_interval = fa.feeder_poll_interval
        self._budget = compute_max_inflight_prompts(
            fa.staleness_threshold, fa.parameter_sync_step, self.config.data.train_batch_size
        )
        # parameter version used to tag fed prompts; tracks self.global_steps (see on_step_end).
        self._param_version = 0
        self._param_version_lock = threading.Lock()
        # Serializes the (non-thread-safe) train dataloader iterator between the feeder thread and
        # the main thread's checkpoint save. Owned here so trainer_base needs no change.
        self._dataloader_lock = threading.Lock()
        # Feeder thread + signals. self._feeder is None until on_train_begin starts it; that None
        # check is also what makes warmup feed but step()'s feed call a no-op (see below).
        self._feeder: threading.Thread | None = None
        self._feeder_stop = threading.Event()
        self._feeder_paused = threading.Event()  # set -> loop does not dispatch new prompts
        self._feeder_error = False

    # ------------------------------------------------------------------ feeding

    def _add_batch_to_generate(self):
        """Override the base per-step feed.

        Before the feeder starts (``on_train_begin`` warmup) this feeds one batch so the pipeline
        is primed; once the feeder owns generation it is a no-op, so the base ``step()`` calling
        this unconditionally does not double-feed. This is what lets ``trainer_base`` stay unchanged.
        """
        if self._feeder is None:
            self._feed_one_batch(self.global_steps)

    def _feed_one_batch(self, global_steps: int):
        """Pull one batch from the dataloader, tag its prompts, register them in TransferQueue, and
        dispatch generation. Tags carry both ``status`` (read by the prompt-level buffer) and ``n``
        (read by the rollout-level :class:`SessionReplayBuffer`), so either buffer works."""
        with self._dataloader_lock:
            try:
                if self.train_dataloader_it is None:
                    self.train_dataloader_it = iter(self.train_dataloader)
                batch_dict = next(self.train_dataloader_it)
            except StopIteration:
                self.train_dataloader_it = iter(self.train_dataloader)
                batch_dict = next(self.train_dataloader_it)

        batch_dict["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object)
        batch = tu.get_tensordict(batch_dict)
        tu.assign_non_tensor_data(batch, "global_steps", global_steps)

        n_sessions = int(self.config.actor_rollout_ref.rollout.n)
        tags = [{"is_prompt": True, "status": "pending", "global_steps": global_steps, "n": n_sessions}] * len(batch)
        tq.kv_batch_put(keys=list(batch["uid"]), partition_id="train", tags=tags)

        self.agent_loop_manager.generate_sequences(batch)

    def _discard_dead_prompts(self) -> int:
        """Discard prompts whose every rollout failed (no usable trajectory) so a fresh prompt can
        take their in-flight slot. No-op unless the rollout-level buffer is in use (the base buffer's
        :meth:`dead_prompt_keys` returns ``[]``). ``dead_prompt_keys`` re-syncs the buffer metadata
        under its lock, so it is safe to call concurrently with the trainer thread's ``sample``."""
        keys = self.replay_buffer.dead_prompt_keys("train")
        if not keys:
            return 0
        tq.kv_clear(partition_id="train", keys=keys)
        # one prompt key per discarded prompt (the rest are its session markers)
        return sum(1 for k in keys if "_sess" not in k)

    # ------------------------------------------------------------------ feeder thread

    def _current_param_version(self) -> int:
        with self._param_version_lock:
            return self._param_version

    def _feeder_loop(self):
        while not self._feeder_stop.is_set():
            if self._feeder_paused.is_set():
                # paused (e.g. during a weight sync): do not dispatch; generation already in flight
                # keeps running and is aborted+continued by the checkpoint engine.
                self._feeder_stop.wait(self._poll_interval)
                continue
            try:
                counts = self.replay_buffer.count_inflight("train")
                # Discard all-failed prompts (uses the state count_inflight just synced); each frees
                # its in-flight slot so the budget check below refills it with a fresh prompt.
                n = self._discard_dead_prompts()
                if n:
                    logger.info("Streaming feeder: discarded %d all-failed prompt(s)", n)
                # Bucket names are buffer-specific; the budget bounds total un-consumed prompts.
                inflight = sum(counts.values())
                if inflight < self._budget:
                    self._feed_one_batch(self._current_param_version())
                else:
                    self._feeder_stop.wait(self._poll_interval)  # interruptible sleep, no busy-wait
            except StopIteration:
                logger.info("Streaming feeder: dataset exhausted, stopping feeder")
                break
            except Exception:
                logger.exception("Streaming feeder thread crashed")
                self._feeder_error = True
                break

    def _start_feeder(self):
        self._feeder_stop.clear()
        self._feeder_paused.clear()
        self._feeder_error = False
        self._feeder = threading.Thread(target=self._feeder_loop, name="streaming-rollout-feeder", daemon=True)
        self._feeder.start()

    def _stop_feeder(self, timeout: float = 30.0):
        self._feeder_stop.set()
        if self._feeder is not None and self._feeder.is_alive():
            self._feeder.join(timeout=timeout)

    # ------------------------------------------------------------------ lifecycle

    def on_train_begin(self):
        # separate_async: prime the pipeline with num_warmup_batches (feeder is None -> warmup feeds
        # via _add_batch_to_generate). Then start the autonomous feeder.
        super().on_train_begin()
        with self._param_version_lock:
            self._param_version = self.global_steps
        # Prime the process-global TransferQueue client on THIS (main) thread before the feeder
        # thread starts. tq lazily creates the client on first use; doing that first call from the
        # background feeder thread has raced with Ray actor-context setup ("TransferQueueController
        # has not been initialized yet"). A main-thread call guarantees the client is created in a
        # valid context, and the feeder then just reuses the already-set global.
        self.replay_buffer.count_inflight("train")
        self._start_feeder()
        # print (not logger.info): verl configures no logging handler in the trainer actor, so INFO
        # is swallowed. These lifecycle markers must reach stdout to be observable/asserted.
        print(f"Streaming rollout feeder started; budget={self._budget} prompts", flush=True)

    def step(self, metrics, timing_raw):
        # fail fast instead of hanging in replay_buffer.sample if the feeder died
        if self._feeder_error:
            raise RuntimeError("Streaming feeder thread died; aborting training")
        if not self._reward_std_filter_enabled():
            return super().step(metrics, timing_raw)
        return self._step_with_reward_std_filter(metrics, timing_raw)

    def _step_with_reward_std_filter(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        # 1. add batch to generate. In fully_async this is a no-op after the feeder starts.
        self._add_batch_to_generate()

        reward_filter_attempt = 0
        while True:
            # 2. sample batch from replay buffer
            with marked_timer("gen", timing_raw, color="red"):
                self.on_sample_begin()
                batch, off_policy_metrics = self.replay_buffer.sample(
                    global_steps=self.global_steps,
                    partition_id="train",
                    batch_size=self.config.data.train_batch_size,
                )
                metrics.update(off_policy_metrics)
                batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                self.on_sample_end()

            # 3. [OPTIONAL] compute reward score with colocated reward model
            if self.reward_loop_manager.reward_loop_worker_handles is None:
                with marked_timer("reward", timing_raw, color="yellow"):
                    batch = self._compute_reward_colocate(batch, metrics=metrics)
            self._record_prompt_reward_before_expand(batch, metrics)

            filtered_batch = self._filter_reward_std_groups(batch, metrics)
            if filtered_batch is not None:
                batch = filtered_batch
                break

            reward_filter_attempt += 1
            metrics["online_filter/retry_batches"] = float(reward_filter_attempt)
            logger.warning(
                "Online reward-std filtering dropped a whole sampled batch at step %s; sampling another batch.",
                self.global_steps,
            )

        # 4. balance batch across data parallel groups
        batch = self._balance_batch(batch, metrics=metrics)

        # 5. compute old_log_prob
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            batch = self._compute_old_log_prob(batch, metrics=metrics)

        # 6. [OPTIONAL] compute ref_log_prob
        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                batch = self._compute_ref_log_prob(batch, metrics=metrics)

        # 7. [OPTIONAL] compute critic values
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                batch = self._compute_values(batch, metrics=metrics)

        # 8. compute advantage and return
        with marked_timer("adv", timing_raw, color="brown"):
            batch = self._compute_advantage(batch, metrics=metrics)

        # 9. [OPTIONAL] update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                batch = self._update_critic(batch, metrics=metrics)

        # 10. update actor
        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                batch = self._update_actor(batch, metrics=metrics)

        return batch

    def _reward_std_filter_enabled(self) -> bool:
        filter_cfg = self.config.algorithm.get("filter_groups", None)
        if not filter_cfg or not filter_cfg.get("enable", False):
            return False

        metric = filter_cfg.get("metric", "seq_reward")
        if metric not in (None, "seq_reward", "seq_final_reward", "reward", "score"):
            logger.warning(
                "algorithm.filter_groups is enabled with metric=%r; fully_async online filtering currently "
                "uses sequence rewards from rm_scores.",
                metric,
            )
        return True

    @staticmethod
    def _split_prompt_and_rollout_key(key: str, tag: dict) -> tuple[str, str]:
        parts = key.rsplit("_", 2)
        if len(parts) == 3:
            prompt_key, session_id, _ = parts
            fallback_rollout_key = f"{prompt_key}_{session_id}"
        else:
            prompt_key = key
            fallback_rollout_key = key
        return prompt_key, str(tag.get("rollout_group_id", fallback_rollout_key))

    def _filter_reward_std_groups(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta | None:
        """Drop prompt groups whose rollout rewards have zero std before trainer forward work."""
        try:
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["rm_scores"])
            rm_scores = data.to_padded_tensor()["rm_scores"]
        except Exception:
            logger.warning("algorithm.filter_groups is enabled, but rm_scores are unavailable; skipping filter.")
            return batch

        sequence_rewards = rm_scores.detach().float().sum(dim=-1).cpu()
        raw_tags = batch.tags or []
        row_tags = [
            raw_tags[i] if i < len(raw_tags) and isinstance(raw_tags[i], dict) else {} for i in range(len(batch.keys))
        ]
        prompt_rows: dict[str, list[int]] = defaultdict(list)
        prompt_rollout_rewards: dict[str, dict[str, float]] = defaultdict(dict)
        bad_prompts: set[str] = set()

        for i, key in enumerate(batch.keys):
            tag = row_tags[i]
            if tag.get("is_padding", False):
                continue

            key_str = str(key)
            prompt_key, rollout_key = self._split_prompt_and_rollout_key(key_str, tag)
            prompt_rows[prompt_key].append(i)

            if i >= sequence_rewards.numel():
                bad_prompts.add(prompt_key)
                continue
            reward = float(sequence_rewards[i].item())
            if not math.isfinite(reward):
                bad_prompts.add(prompt_key)
                continue
            # GUI multi-turn rows from one rollout share the episode reward; count that rollout once.
            prompt_rollout_rewards[prompt_key][rollout_key] = reward

        keep_prompts: set[str] = set()
        reward_stds: list[float] = []
        for prompt_key, rollout_rewards in prompt_rollout_rewards.items():
            rewards = list(rollout_rewards.values())
            if prompt_key in bad_prompts or len(rewards) < 2:
                reward_stds.append(0.0)
                continue
            reward_tensor = torch.tensor(rewards, dtype=torch.float32)
            reward_std = float(torch.std(reward_tensor, unbiased=False).item())
            reward_stds.append(reward_std)
            if reward_std > 0.0:
                keep_prompts.add(prompt_key)

        kept_indices = [i for prompt_key in keep_prompts for i in prompt_rows.get(prompt_key, [])]
        kept_indices.sort()
        kept_keys = [batch.keys[i] for i in kept_indices]
        kept_tags = [row_tags[i] for i in kept_indices]
        kept_index_set = set(kept_indices)
        dropped_keys = [key for i, key in enumerate(batch.keys) if i not in kept_index_set]

        total_prompts = len(prompt_rows)
        kept_prompts = len(keep_prompts)
        metrics["online_filter/reward_std/enabled"] = 1.0
        metrics["online_filter/reward_std/prompts_total"] = float(total_prompts)
        metrics["online_filter/reward_std/prompts_kept"] = float(kept_prompts)
        metrics["online_filter/reward_std/prompts_dropped"] = float(max(total_prompts - kept_prompts, 0))
        metrics["online_filter/reward_std/prompt_keep_frac"] = float(kept_prompts) / max(total_prompts, 1)
        metrics["online_filter/reward_std/rows_total"] = float(len(batch.keys))
        metrics["online_filter/reward_std/rows_kept"] = float(len(kept_keys))
        metrics["online_filter/reward_std/rows_dropped"] = float(len(dropped_keys))
        if reward_stds:
            metrics["online_filter/reward_std/mean"] = float(sum(reward_stds)) / len(reward_stds)
            metrics["online_filter/reward_std/max"] = float(max(reward_stds))
            metrics["online_filter/reward_std/min"] = float(min(reward_stds))

        if dropped_keys:
            try:
                from verl.utils.transferqueue_image_dedup import clear_images, fetch_image_ids

                dropped_image_ids = fetch_image_ids(dropped_keys, batch.partition_id)
                clear_images(dropped_image_ids)
                pending_image_keys = getattr(self.replay_buffer, "_pending_image_keys", None)
                if isinstance(pending_image_keys, dict) and dropped_image_ids:
                    pending = pending_image_keys.get(batch.partition_id, [])
                    dropped_image_set = set(dropped_image_ids)
                    pending_image_keys[batch.partition_id] = [key for key in pending if key not in dropped_image_set]
            except Exception:
                logger.debug("Failed to clear images for reward-std filtered rows", exc_info=True)
            tq.kv_clear(partition_id=batch.partition_id, keys=dropped_keys)

        if not kept_keys:
            return None

        filtered = KVBatchMeta(partition_id=batch.partition_id, keys=kept_keys, tags=kept_tags)
        filtered.extra_info.update(batch.extra_info)
        return filtered

    def on_step_end(self):
        # Pause the feeder around the periodic standalone weight sync so it does not dispatch prompts
        # into a server that is mid-sync. Generation already in flight is aborted+continued by the
        # checkpoint engine / FullyAsyncLLMServerClient (partial rollout), independent of this pause.
        is_sync_step = self._feeder is not None and self.global_steps % self.parameter_sync_step == 0
        if is_sync_step:
            print(f"Pausing streaming feeder for weight sync at step {self.global_steps}", flush=True)
            self._feeder_paused.set()
        try:
            super().on_step_end()  # separate_async: standalone update_weights on sync steps
        finally:
            if is_sync_step:
                self._feeder_paused.clear()
                print(f"Resumed streaming feeder after weight sync at step {self.global_steps}", flush=True)
        with self._param_version_lock:
            self._param_version = self.global_steps
        if _STEP_PROFILE:
            # logged here (not in step()) so update_weights/save_checkpoint, which run after step()
            # returns, are included in self.timing_raw.
            self._log_step_profile()

    def _log_step_profile(self):
        """Print a readable per-step timing breakdown (which phase dominates).

        For streaming, ``gen`` is the time ``replay_buffer.sample`` spent — mostly *waiting for
        rollout data*. So ``gen`` large => trainer is data-starved (rollout-bound); ``update_actor``
        large => training-compute-bound; ``update_weights`` large => the weight sync dominates. The
        replay buffer's own ``[SAMPLE_PROFILE]`` line (same step) splits that wait from collection.
        """
        t = self.timing_raw
        items = [(p, t[p]) for p in _STEP_PHASES if p in t]
        total = sum(v for _, v in items) or 1e-9
        parts = " ".join(f"{p}={v:.2f}s({v / total * 100:.0f}%)" for p, v in sorted(items, key=lambda kv: -kv[1]))
        print(f"[STEP_PROFILE] step={self.global_steps} total={total:.2f}s | {parts}", flush=True)

    def _save_checkpoint(self):
        # The feeder thread may be iterating the (non-thread-safe) dataloader; serialize against it
        # so the base's StatefulDataLoader.state_dict() is never read mid-iteration.
        with self._dataloader_lock:
            super()._save_checkpoint()

    def on_train_end(self):
        self._stop_feeder()
        super().on_train_end()
