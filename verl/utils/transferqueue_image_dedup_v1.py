# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""Config-selected subclasses that add per-image dedup to the V1 TransferQueue path.

This keeps everything under ``verl/trainer/ppo/v1/`` PRISTINE: the dedup behavior
is opted into purely via config, by pointing at the classes here.

* **Producer** — :class:`ImageDedupManagerTQ` (select with
  ``actor_rollout_ref.rollout.agent.agent_loop_manager_class``). Its worker stores
  each unique processed image once in the ``rollout_images`` partition (keyed by
  SHA1) and writes only an ``image_ids`` list per row, incrementing an
  :class:`~verl.utils.transferqueue_image_dedup.ImageRefCounter` per row reference.
* **Lifecycle** — :class:`ImageDedupReplayBuffer` (select with
  ``trainer.v1.sampler.custom_sampler.{path,name}``). It releases a deduped image
  once no live row references it: dropped rows are released in-place; consumed
  rows are released one ``sample()`` later (after the fit loop has cleared them),
  using ids captured while the rows were still alive.
* **Consume** — resolution of ``image_ids`` back into ``multi_modal_inputs`` lives
  in :func:`verl.utils.transferqueue_utils._async_meta_to_realdata` (the single
  worker-side materialization point; not under ``ppo/v1/``), gated on the presence
  of the ``image_ids`` column, so it is a no-op unless these classes are selected.

Because the worker/manager/replay-buffer methods have no finer extension seam, the
overrides here copy the upstream method bodies and add the dedup steps; if upstream
changes those methods, re-sync the copies.
"""

import logging
import os

import numpy as np
import ray
import torch
import transfer_queue as tq
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQ
from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from verl.utils.transferqueue_image_dedup import (
    IMAGE_IDS_KEY,
    PARTITION_IMAGES,
    decref_images,
    fetch_image_ids,
    get_image_refcounter,
    split_multimodal_per_image,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


# ---------------------------------------------------------------------------
# Producer: store deduped image references instead of inline pixel tensors
# ---------------------------------------------------------------------------
# ``AgentLoopWorkerTQ`` is a ``@ray.remote`` ActorClass; recover its underlying
# plain class so we can subclass it and re-apply ``ray.remote`` (the base manager
# itself stores ``ray.remote(AgentLoopWorker)``, i.e. plain-class-then-decorate).
_PlainAgentLoopWorkerTQ = AgentLoopWorkerTQ.__ray_metadata__.modified_class


class _ImageDedupWorkerTQ(_PlainAgentLoopWorkerTQ):
    """AgentLoopWorkerTQ that dedups per-image processor tensors on write (train only)."""

    async def _agent_loop_postprocess(self, output, validate, **kwargs) -> None:
        """Put agent loop outputs into TransferQueue (deduping images on the train path)."""
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if not outputs:
            logger.warning(f"Empty output for prompt {uid}_{session_id}")
            return

        await self._compute_score(outputs, kwargs=kwargs)

        final_output = outputs[-1]
        await self._compute_teacher_logprobs(
            final_output,
            prompt_ids=final_output.prompt_ids,
            response_ids=final_output.response_ids,
            validate=validate,
            sample_kwargs=kwargs,
        )

        if final_output.reward_score is not None:
            for output in outputs[:-1]:
                output.reward_score = final_output.reward_score
                output.extra_fields["reward_extra_info"] = final_output.extra_fields["reward_extra_info"]

        # key format: {uid}_{session_id}_{index}
        image_payloads: dict[str, dict] = {}
        keys, fields, tags = [], [], []
        for i, output in enumerate(outputs):
            prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(output.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)

            keys.append(f"{uid}_{session_id}_{i}")
            field = output.as_dict()
            field.update(kwargs)
            field.pop("multi_modal_data", None)  # do not store raw image/video
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            if not validate:
                # Always emit image_ids (empty for text-only rows) so every row in
                # the batch shares the same keys for list_of_dict_to_tensordict; the
                # consume hook restores multi_modal_inputs (empty dict when empty).
                image_ids, payloads = split_multimodal_per_image(multi_modal_inputs)
                image_payloads.update(payloads)
                field[IMAGE_IDS_KEY] = image_ids
            else:
                field["multi_modal_inputs"] = multi_modal_inputs
            fields.append(field)
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            tags.append(
                {
                    "status": "success",
                    "prompt_len": prompt_len,
                    "response_len": response_len,
                    "seq_len": prompt_len + response_len,
                    "global_steps": kwargs["global_steps"],
                    "min_global_steps": field["extra_fields"].get("min_global_steps"),
                    "max_global_steps": field["extra_fields"].get("max_global_steps"),
                }
            )

        if not validate and image_payloads:
            # Store each unique image once before the rows that reference it become
            # consumable, and refcount one inc per row reference.
            for sha1, payload in image_payloads.items():
                await tq.async_kv_put(key=sha1, partition_id=PARTITION_IMAGES, fields=payload)
            all_refs = [k for field in fields for k in field.get(IMAGE_IDS_KEY, [])]
            get_image_refcounter().incref.remote(all_refs)

        await tq.async_kv_batch_put(
            keys=keys,
            fields=list_of_dict_to_tensordict(fields),
            tags=tags,
            partition_id="train" if not validate else "val",
        )


# Re-decorate the plain subclass as a Ray actor (mirrors ray.remote(AgentLoopWorker)).
ImageDedupWorkerTQ = ray.remote(_ImageDedupWorkerTQ)


class ImageDedupManagerTQ(AgentLoopManagerTQ):
    """AgentLoopManagerTQ that spawns the dedup-aware worker.

    Select with ``actor_rollout_ref.rollout.agent.agent_loop_manager_class``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override the worker class; workers are created later in
        # ``_init_agent_loop_workers`` (called by ``create``), so this is in time.
        self.agent_loop_workers_class = ImageDedupWorkerTQ


# ---------------------------------------------------------------------------
# Lifecycle: release deduped images as rows leave the replay buffer
# ---------------------------------------------------------------------------
class ImageDedupReplayBuffer(ReplayBuffer):
    """ReplayBuffer that refcount-GCs deduped images.

    Select with ``trainer.v1.sampler.custom_sampler.{path,name}``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # image keys of the last-sampled (now consumed) batch, per partition
        self._pending_image_keys: dict[str, list[str]] = {}

    def sample(self, global_steps: int, partition_id: str, batch_size: int):
        batch, metrics = super().sample(global_steps, partition_id, batch_size)
        # The previously-sampled batch for this partition has been consumed and
        # cleared by the fit loop by now -> release its images.
        decref_images(self._pending_image_keys.pop(partition_id, []))
        # Capture this batch's image refs while its rows are still alive.
        self._pending_image_keys[partition_id] = fetch_image_ids(batch.keys, batch.partition_id)
        return batch, metrics

    def _drop_max_off_policy_samples(
        self, global_steps: int, partition_id: str, batch: KVBatchMeta
    ) -> tuple[KVBatchMeta, dict]:
        if not self.max_off_policy_strategy == "drop":
            return batch, {}

        kept_keys, kept_tags = [], []
        dropped_keys, dropped_tags = [], []
        for key, tag in zip(batch.keys, batch.tags, strict=False):
            prompt_global_steps = tag["global_steps"]
            if (global_steps - prompt_global_steps + 1) / self.parameter_sync_step > self.max_off_policy_threshold:
                dropped_keys.append(key)
                dropped_tags.append(tag)
            else:
                kept_keys.append(key)
                kept_tags.append(tag)

        metrics = {}
        if len(dropped_keys) > 0:
            # Release dropped rows' images BEFORE clearing the rows.
            decref_images(fetch_image_ids(dropped_keys, batch.partition_id))
            tq.kv_clear(partition_id=batch.partition_id, keys=dropped_keys)
            logger.warning(f"Dropped {len(dropped_keys)} max off policy samples from partition {batch.partition_id}")
            dropped_global_steps = np.array([tag["global_steps"] for tag in dropped_tags])
            trajectory_staleness = (global_steps - dropped_global_steps + 1) / self.parameter_sync_step
            prefix = "training" if partition_id == "train" else "validation"
            metrics[f"{prefix}/off_policy/dropped_samples"] = len(dropped_keys)
            metrics[f"{prefix}/off_policy/dropped_samples_staleness/mean"] = trajectory_staleness.mean()
            metrics[f"{prefix}/off_policy/dropped_samples_staleness/max"] = trajectory_staleness.max()
            metrics[f"{prefix}/off_policy/dropped_samples_staleness/min"] = trajectory_staleness.min()

        return KVBatchMeta(partition_id=batch.partition_id, keys=kept_keys, tags=kept_tags), metrics
