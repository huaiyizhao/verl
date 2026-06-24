# Copyright 2026 Tencent Ltd. and/or its affiliates
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

"""TransferQueue transport helpers for fully async policy training.

The transport keeps large rollout/update payloads in TransferQueue and moves only
small descriptors through Ray actor calls.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from typing import Any

import numpy as np
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack

from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import RolloutSample

try:
    import transfer_queue as tq
    from transfer_queue import KVBatchMeta
except ImportError:
    from verl.utils.transferqueue_utils import KVBatchMeta, tq

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

META_METRICS_KEY = "__fully_async_meta_metrics__"
META_REWARD_EXTRA_KEYS_KEY = "__fully_async_reward_extra_keys__"


def transfer_queue_enabled(config: Any) -> bool:
    async_training = getattr(config, "async_training", None)
    if async_training is None:
        return False
    transport = async_training.get("data_transport", "message_queue")
    return str(transport).lower() in {"transfer_queue", "tq"}


def ensure_transfer_queue_initialized(config: Any | None = None) -> None:
    """Connect this process to TransferQueue, creating it from config if needed."""
    tq_conf = None
    if config is not None and getattr(config, "transfer_queue", None) is not None:
        tq_conf = config.transfer_queue
    tq.init(tq_conf)


def _object_array(values: list[Any]) -> np.ndarray:
    arr = np.empty(len(values), dtype=object)
    for idx, value in enumerate(values):
        arr[idx] = value
    return arr


def dataproto_to_tensordict(data: DataProto, *, include_meta_columns: bool = True) -> TensorDict:
    """Convert DataProto to a TQ-storable TensorDict.

    DataProto.meta_info is batch-level and cannot be represented directly as TQ
    fields. Only row-aligned metadata needed to rebuild rollout samples is stored
    as explicit columns.
    """
    if data.batch is None:
        td = TensorDict({}, batch_size=[len(data)])
    else:
        td = TensorDict(dict(data.batch.items()), batch_size=[len(data)])

    for key, value in (data.non_tensor_batch or {}).items():
        td[key] = NonTensorStack.from_list([NonTensorData(item) for item in value])

    if include_meta_columns:
        metrics = (data.meta_info or {}).get("metrics")
        if metrics is not None:
            if not isinstance(metrics, list):
                metrics = [metrics] * len(data)
            if len(metrics) != len(data):
                raise ValueError(f"metrics length {len(metrics)} does not match DataProto length {len(data)}")
            td[META_METRICS_KEY] = NonTensorStack.from_list([NonTensorData(item) for item in metrics])

        reward_extra_keys = (data.meta_info or {}).get("reward_extra_keys")
        if reward_extra_keys is not None:
            td[META_REWARD_EXTRA_KEYS_KEY] = NonTensorStack.from_list(
                [NonTensorData(list(reward_extra_keys)) for _ in range(len(data))]
            )

    return td


def dataproto_from_tensordict(td: TensorDict, *, meta_info: dict[str, Any] | None = None) -> DataProto:
    data = DataProto.from_tensordict(td, meta_info=dict(meta_info or {}))
    metrics_col = data.non_tensor_batch.pop(META_METRICS_KEY, None)
    if metrics_col is not None:
        data.meta_info["metrics"] = [item for item in metrics_col]
    reward_extra_keys_col = data.non_tensor_batch.pop(META_REWARD_EXTRA_KEYS_KEY, None)
    if reward_extra_keys_col is not None and len(reward_extra_keys_col) > 0:
        data.meta_info["reward_extra_keys"] = list(reward_extra_keys_col[0] or [])
    return data


async def async_put_dataproto(
    *,
    data: DataProto,
    keys: list[str],
    partition_id: str,
    tags: list[dict[str, Any]] | None = None,
    config: Any | None = None,
    include_meta_columns: bool = True,
) -> KVBatchMeta:
    ensure_transfer_queue_initialized(config)
    fields = dataproto_to_tensordict(data, include_meta_columns=include_meta_columns)
    meta = await tq.async_kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=tags)
    meta.extra_info = dict(data.meta_info or {})
    return meta


async def async_get_dataproto(
    *,
    keys: list[str],
    partition_id: str,
    select_fields: list[str] | str | None = None,
    meta_info: dict[str, Any] | None = None,
    config: Any | None = None,
) -> DataProto:
    ensure_transfer_queue_initialized(config)
    td = await tq.async_kv_batch_get(keys=keys, partition_id=partition_id, select_fields=select_fields)
    return dataproto_from_tensordict(td, meta_info=meta_info)


async def async_clear_keys(*, keys: list[str], partition_id: str, config: Any | None = None) -> None:
    if not keys:
        return
    ensure_transfer_queue_initialized(config)
    await tq.async_kv_clear(keys=keys, partition_id=partition_id)


def _transport_conf(config: Any) -> Any:
    async_training = getattr(config, "async_training", None)
    return async_training.get("transfer_queue", {}) if async_training is not None else {}


def get_partition(config: Any, name: str) -> str:
    conf = _transport_conf(config)
    prefix = str(conf.get("partition_prefix", "fully_async"))
    return f"{prefix}_{name}"


def get_poll_interval(config: Any) -> float:
    conf = _transport_conf(config)
    return float(conf.get("poll_interval_s", 0.2))


class TransferQueueSampleQueueClient:
    """MessageQueue-compatible client backed by TransferQueue KV rows."""

    def __init__(self, config: Any):
        self.config = config
        self.sample_partition = get_partition(config, "samples")
        self.control_partition = get_partition(config, "control")
        self.poll_interval_s = get_poll_interval(config)

    async def put_sample(self, sample: RolloutSample | None) -> bool:
        ensure_transfer_queue_initialized(self.config)
        if sample is None:
            await tq.async_kv_put(
                key="termination",
                partition_id=self.control_partition,
                tag={"status": "terminated", "time": time.time()},
            )
            return True

        if sample.full_batch is None or len(sample.full_batch) == 0:
            return False

        keys = [f"{sample.sample_id}:row:{idx}" for idx in range(len(sample.full_batch))]
        tags = [
            {
                "status": "ready",
                "sample_id": sample.sample_id,
                "epoch": int(sample.epoch),
                "row_count": len(sample.full_batch),
                "row_index": idx,
                "rollout_status": dict(sample.rollout_status or {}),
                "image_bank_stats": dict(sample.image_bank_stats or {}),
            }
            for idx in range(len(sample.full_batch))
        ]
        await async_put_dataproto(
            data=sample.full_batch,
            keys=keys,
            partition_id=self.sample_partition,
            tags=tags,
            config=self.config,
        )
        return True

    async def get_sample(self) -> tuple[RolloutSample, int] | None:
        ensure_transfer_queue_initialized(self.config)
        while True:
            control = await tq.async_kv_list(self.control_partition)
            if control.get(self.control_partition, {}).get("termination", {}).get("status") == "terminated":
                samples = await self._ready_samples()
                if not samples:
                    return None

            samples = await self._ready_samples()
            if samples:
                sample_id = sorted(samples, key=lambda sid: min(item[1].get("row_index", 0) for item in samples[sid]))[
                    0
                ]
                rows = sorted(samples[sample_id], key=lambda item: item[1].get("row_index", 0))
                keys = [key for key, _ in rows]
                first_tag = rows[0][1]
                data = await async_get_dataproto(
                    keys=keys,
                    partition_id=self.sample_partition,
                    config=self.config,
                )
                await async_clear_keys(keys=keys, partition_id=self.sample_partition, config=self.config)
                sample = RolloutSample(
                    full_batch=data,
                    sample_id=sample_id,
                    epoch=int(first_tag.get("epoch", 0)),
                    rollout_status=dict(first_tag.get("rollout_status", {}) or {}),
                    image_bank_stats=dict(first_tag.get("image_bank_stats", {}) or {}),
                )
                queue_len = max(0, len(samples) - 1)
                return sample, queue_len

            await asyncio.sleep(self.poll_interval_s)

    async def _ready_samples(self) -> dict[str, list[tuple[str, dict[str, Any]]]]:
        listing = await tq.async_kv_list(self.sample_partition)
        rows = listing.get(self.sample_partition, {})
        grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        expected: dict[str, int] = {}
        for key, tag in rows.items():
            if tag.get("status") != "ready":
                continue
            sample_id = tag.get("sample_id")
            if not sample_id:
                continue
            grouped[sample_id].append((key, tag))
            expected[sample_id] = int(tag.get("row_count", 0) or 0)

        return {
            sample_id: items
            for sample_id, items in grouped.items()
            if expected.get(sample_id, 0) > 0 and len(items) >= expected[sample_id]
        }

    async def get_queue_size(self) -> int:
        ensure_transfer_queue_initialized(self.config)
        return len(await self._ready_samples())

    async def get_statistics(self) -> dict[str, Any]:
        queue_size = await self.get_queue_size()
        return {
            "queue_size": queue_size,
            "total_produced": 0,
            "total_consumed": 0,
            "dropped_samples": 0,
            "max_queue_size": None,
        }

    async def clear_queue(self) -> None:
        ensure_transfer_queue_initialized(self.config)
        for partition_id in (self.sample_partition, self.control_partition):
            listing = await tq.async_kv_list(partition_id)
            keys = list(listing.get(partition_id, {}).keys())
            if keys:
                await tq.async_kv_clear(keys=keys, partition_id=partition_id)

    async def shutdown(self) -> None:
        await self.put_sample(None)
