# Copyright 2025 Meituan Ltd. and/or its affiliates
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

"""TransferQueue-backed data link between the Rollouter and the Trainer.

The data plane lives entirely in TransferQueue (TQ): each rollout is expanded
into per-turn **trajectory rows** stored as tensor columns in the
``rollout_stream`` partition, and every **unique** processed image is stored
once (deduped by SHA1 key) in the ``rollout_images`` partition. The Ray object
store is no longer used to transport samples.

This module keeps a slim **control-plane** Ray actor (still named
``MessageQueue`` for import compatibility) that owns only what TQ does not model
as a single-consumer blocking FIFO:

* ordered queue of *rollout-group* handles (the TQ row-keys of one episode plus
  its referenced image-keys and ``meta_info``) — never the payloads themselves;
* ``total_produced`` / ``total_consumed`` / ``dropped_samples`` counters used by
  the Rollouter staleness/pause logic (accounting stays in **episode units**);
* an ``asyncio.Condition`` for true blocking ``get`` and a ``None`` termination
  sentinel;
* a low-volume validation queue;
* a **refcount per image key** so a screenshot shared across turns/episodes is
  cleared from ``rollout_images`` only once the last referencing group is
  consumed (the explicit analog of Ray's object-store refcounting).

``MessageQueueClient`` preserves the historical facade
(``put_sample`` / ``get_sample`` / ``put_validate`` / ``get_validate_sync`` /
``get_queue_size`` / ``get_statistics`` / ``clear_queue`` / ``shutdown``) so the
Trainer and Rollouter call sites are unchanged, and performs the TQ reads/writes
plus producer-side image dedup. ``get_sample`` returns ``(DataProto, queue_len)``
exactly as before — now the DataProto is the reconstructed, already-expanded
episode (its rows distinguished by the ``is_final`` column).
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import ray
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# TransferQueue logical partitions for the data link.
PARTITION_STREAM = "rollout_stream"
PARTITION_VAL = "rollout_val"
PARTITION_IMAGES = "rollout_images"

# meta_info carrier key for the per-rollout processed-image payloads that the
# producer wants stored (deduped) into ``rollout_images``. Mapping
# ``{sha1_key: {field_name: tensor}}``. Stripped before transport.
TQ_IMAGE_PAYLOADS_KEY = "__tq_image_payloads__"
# non_tensor_batch column holding, per row, the list of SHA1 image keys the row
# references in ``rollout_images``.
IMAGE_IDS_KEY = "image_ids"
# non_tensor_batch column with the unique per-row key used as the TQ row key.
ROW_ID_KEY = "row_id"


# ---------------------------------------------------------------------------
# TransferQueue bring-up (driver side)
# ---------------------------------------------------------------------------
def init_transfer_queue(config: DictConfig) -> Any:
    """Initialize the TransferQueue controller + storage backend on the driver.

    The storage backend is a pure config passthrough: everything under
    ``config.async_training.transfer_queue`` is forwarded verbatim as TQ's
    ``backend`` section (so ``storage_backend`` and any backend-specific subkeys
    such as ``SimpleStorage.num_data_storage_units`` or ``MooncakeStore.*`` work
    without any code change). A FIFO ``SequentialSampler`` is used; ordering and
    blocking are handled by the coordinator, not by TQ sampling.

    Safe to call once on the driver; ``MessageQueueClient`` lazily attaches in
    each consumer/producer process via ``tq.init()`` with no config.
    """
    import transfer_queue as tq

    user_backend = OmegaConf.select(config, "async_training.transfer_queue")
    overrides: dict[str, Any] = {
        "controller": {"sampler": "SequentialSampler", "polling_mode": False},
    }
    if user_backend is not None:
        backend_dict = OmegaConf.to_container(user_backend, resolve=True)
        if backend_dict:
            overrides["backend"] = backend_dict

    conf = OmegaConf.create(overrides, flags={"allow_objects": True})
    logger.info("[init_transfer_queue] initializing TransferQueue with overrides: %s", overrides)
    return tq.init(conf)


# ---------------------------------------------------------------------------
# Group handle (control-plane record; no payloads)
# ---------------------------------------------------------------------------
@dataclass
class _GroupHandle:
    """One rollout episode's location in TransferQueue (no tensors)."""

    row_keys: list[str]
    image_keys: list[str]
    meta_info: dict[str, Any] = field(default_factory=dict)


# Sentinel object enqueued by ``put_sample(None)`` to mark end-of-stream while
# preserving FIFO order against in-flight real groups.
_SENTINEL = object()


@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:
    """Control-plane coordinator for the TransferQueue data link.

    Holds ordered *group handles* (TQ key-sets), counters, a blocking condition,
    the termination sentinel, the validation queue, and per-image refcounts. It
    never stores trajectory or image tensors — those live in TransferQueue.
    """

    def __init__(self, config: DictConfig, max_queue_size: int = 1000):
        self.config = config
        if max_queue_size is None:
            raise ValueError(f"max_queue_size cannot be None, got: {max_queue_size}")
        self.max_queue_size = int(max_queue_size)
        self.queue: deque = deque()
        self.val_queue: deque = deque()

        self.running = True
        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)

        self.total_produced = 0
        self.total_consumed = 0
        self.dropped_samples = 0

        # sha1 image key -> number of live (enqueued, not-yet-released) groups
        # that reference it.
        self.image_refcounts: dict[str, int] = {}

        print(f"[MessageQueue] TransferQueue coordinator initialized with max_queue_size={self.max_queue_size}")

    # -- producer ----------------------------------------------------------
    async def enqueue(self, handle: _GroupHandle) -> tuple[bool, list[str]]:
        """Append one rollout group; increment image refcounts.

        Returns ``(accepted, evicted_image_keys)``. ``evicted_image_keys`` are
        image keys whose refcount hit 0 because the oldest group was dropped
        (queue full); the caller should ``kv_clear`` them.
        """
        async with self._lock:
            is_drop = False
            evicted: list[str] = []
            if len(self.queue) >= self.max_queue_size:
                old = self.queue.popleft()
                self.dropped_samples += 1
                is_drop = True
                if isinstance(old, _GroupHandle):
                    evicted = self._decref_images_locked(old.image_keys)
                logger.warning("Queue full, dropped oldest rollout group")

            self.queue.append(handle)
            for k in handle.image_keys:
                self.image_refcounts[k] = self.image_refcounts.get(k, 0) + 1
            self.total_produced += 1
            self._consumer_condition.notify_all()

            if self.total_produced % 100 == 0:
                print(f"MessageQueue stats: produced={self.total_produced}, queue_size={len(self.queue)}")
            return (not is_drop), evicted

    async def put_sentinel(self) -> None:
        """Mark end-of-stream (FIFO-ordered after produced groups)."""
        async with self._lock:
            self.queue.append(_SENTINEL)
            self._consumer_condition.notify_all()

    # -- consumer ----------------------------------------------------------
    async def dequeue(self) -> tuple[Any, int]:
        """Block until a group is available; return ``(handle_or_None, qlen)``.

        Returns ``(None, 0)`` on the termination sentinel or shutdown.
        """
        async with self._lock:
            while len(self.queue) == 0 and self.running:
                await self._consumer_condition.wait()

            if len(self.queue) == 0:  # shutdown with empty queue
                return None, 0

            item = self.queue.popleft()
            if item is _SENTINEL:
                return None, len(self.queue)

            self.total_consumed += 1
            return item, len(self.queue)

    async def release_images(self, image_keys: list[str]) -> list[str]:
        """Decrement refcounts for a consumed group; return keys now at 0."""
        async with self._lock:
            return self._decref_images_locked(image_keys)

    def _decref_images_locked(self, image_keys: list[str]) -> list[str]:
        to_clear: list[str] = []
        for k in image_keys:
            c = self.image_refcounts.get(k, 0) - 1
            if c <= 0:
                self.image_refcounts.pop(k, None)
                to_clear.append(k)
            else:
                self.image_refcounts[k] = c
        return to_clear

    # -- stats / lifecycle -------------------------------------------------
    async def get_queue_size(self) -> int:
        async with self._lock:
            return len(self.queue)

    async def get_statistics(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "queue_size": len(self.queue),
                "total_produced": self.total_produced,
                "total_consumed": self.total_consumed,
                "dropped_samples": self.dropped_samples,
                "max_queue_size": self.max_queue_size,
                "unique_images": len(self.image_refcounts),
            }

    async def clear_queue(self) -> list[str]:
        """Clear pending groups; return all referenced image keys for cleanup."""
        async with self._lock:
            image_keys = list(self.image_refcounts.keys())
            cleared = len(self.queue)
            self.queue.clear()
            self.image_refcounts.clear()
            logger.info("Cleared %d pending rollout groups from coordinator", cleared)
            return image_keys

    async def shutdown(self) -> None:
        async with self._lock:
            self.running = False
            self._consumer_condition.notify_all()
        logger.info("MessageQueue coordinator shutdown")

    # -- validation stream -------------------------------------------------
    async def put_validate(self, handle: _GroupHandle) -> None:
        async with self._lock:
            self.val_queue.append(handle)

    async def get_validate(self) -> Any | None:
        async with self._lock:
            return self.val_queue.popleft() if self.val_queue else None


# ---------------------------------------------------------------------------
# DataProto <-> TransferQueue conversion helpers
# ---------------------------------------------------------------------------
def _to_put_tensordict(data_proto: Any):
    """Build a batched ``TensorDict`` carrying a rollout group's rows.

    Tensor columns (``data_proto.batch``) ride as zero-copy tensor fields;
    ``non_tensor_batch`` columns ride as ``NonTensorStack`` (small: ids, masks
    metadata, ``image_ids``). The variable-size image tensors are NOT included
    here — they are deduped into ``rollout_images`` separately.
    """
    from tensordict import NonTensorStack, TensorDict

    n = len(data_proto)
    out: dict[str, Any] = {}
    if data_proto.batch is not None:
        for k, v in data_proto.batch.items():
            out[k] = v
    for k, v in (data_proto.non_tensor_batch or {}).items():
        out[k] = NonTensorStack(*list(v))
    return TensorDict(out, batch_size=[n])


def _unwrap_non_tensor(value: Any) -> Any:
    """Strip a ``NonTensorData`` wrapper to its underlying python value."""
    from tensordict.tensorclass import NonTensorData

    return value.data if isinstance(value, NonTensorData) else value


def _from_get_tensordict(td: Any, meta_info: dict[str, Any]) -> Any:
    """Inverse of :func:`_to_put_tensordict`: rebuild a DataProto."""
    import torch
    from tensordict import TensorDict

    from verl import DataProto

    n = int(td.batch_size[0])
    tensor_cols: dict[str, Any] = {}
    non_tensor_cols: dict[str, np.ndarray] = {}
    for k in td.keys():
        v = td.get(k)
        if isinstance(v, torch.Tensor):
            tensor_cols[k] = v
            continue
        # NonTensorStack / object column -> unwrap to raw python values.
        raw = [_unwrap_non_tensor(v[i]) for i in range(n)]
        arr = np.empty(n, dtype=object)
        arr[:] = raw
        non_tensor_cols[k] = arr

    batch_td = TensorDict(tensor_cols, batch_size=[n]) if tensor_cols else None
    return DataProto(batch=batch_td, non_tensor_batch=non_tensor_cols, meta_info=dict(meta_info or {}))


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------
class MessageQueueClient:
    """Asyncio-compatible facade over the TransferQueue data link.

    Preserves the historical ``MessageQueue`` client API while routing data
    through TransferQueue. The TQ client is initialized lazily per process (never
    on the driver) so the same facade instance can be serialized to both the
    Rollouter and Trainer actors.
    """

    def __init__(self, queue_actor: Any):
        self.queue_actor = queue_actor
        # per-process state (recreated after deserialization in each actor)
        self._tq_ready = False
        self._seen_images: set[str] = set()

    # -- TQ attach ---------------------------------------------------------
    def _ensure_tq(self):
        import transfer_queue as tq

        if not self._tq_ready:
            tq.init()  # attach to the existing controller (cluster-wide)
            self._tq_ready = True
        return tq

    @staticmethod
    def _group_from_dataproto(data_proto: Any) -> tuple[_GroupHandle, Any, dict[str, Any]]:
        """Split an expanded episode DataProto into (handle, rows_td, image_payloads)."""
        meta_info = dict(data_proto.meta_info or {})
        image_payloads: dict[str, Any] = meta_info.pop(TQ_IMAGE_PAYLOADS_KEY, {}) or {}

        nt = data_proto.non_tensor_batch or {}
        if ROW_ID_KEY not in nt:
            raise KeyError(f"expanded DataProto missing '{ROW_ID_KEY}' column required as TQ row key")
        row_keys = [str(x) for x in nt[ROW_ID_KEY]]

        image_keys: set[str] = set()
        if IMAGE_IDS_KEY in nt:
            for ids in nt[IMAGE_IDS_KEY]:
                for k in ids or []:
                    image_keys.add(str(k))

        rows_td = _to_put_tensordict(data_proto)
        handle = _GroupHandle(row_keys=row_keys, image_keys=sorted(image_keys), meta_info=meta_info)
        return handle, rows_td, image_payloads

    async def _write_group(
        self, tq, handle: _GroupHandle, rows_td: Any, image_payloads: dict[str, Any], partition: str
    ):
        # 1. dedup-put any unseen images into rollout_images (one key per SHA1).
        for sha1, payload in image_payloads.items():
            if sha1 in self._seen_images:
                continue
            await tq.async_kv_put(key=str(sha1), partition_id=PARTITION_IMAGES, fields=payload)
            self._seen_images.add(str(sha1))
        # 2. put the uniform trajectory rows into the stream partition.
        await tq.async_kv_batch_put(keys=handle.row_keys, partition_id=partition, fields=rows_td)

    # -- producer ----------------------------------------------------------
    async def put_sample(self, sample: Any) -> bool:
        """Put one expanded rollout episode (DataProto) or ``None`` sentinel."""
        if sample is None:
            await asyncio.wrap_future(self.queue_actor.put_sentinel.remote().future())
            return True

        tq = self._ensure_tq()
        handle, rows_td, image_payloads = self._group_from_dataproto(sample)
        await self._write_group(tq, handle, rows_td, image_payloads, PARTITION_STREAM)
        accepted, evicted = await asyncio.wrap_future(self.queue_actor.enqueue.remote(handle).future())
        if evicted:
            await tq.async_kv_clear(keys=evicted, partition_id=PARTITION_IMAGES)
        return bool(accepted)

    async def put_validate(self, data: Any) -> bool:
        tq = self._ensure_tq()
        handle, rows_td, image_payloads = self._group_from_dataproto(data)
        await self._write_group(tq, handle, rows_td, image_payloads, PARTITION_VAL)
        await asyncio.wrap_future(self.queue_actor.put_validate.remote(handle).future())
        return True

    # -- consumer ----------------------------------------------------------
    async def get_sample(self) -> Any | None:
        """Get one expanded episode as ``(DataProto, queue_len)`` or ``None``."""
        item, queue_len = await asyncio.wrap_future(self.queue_actor.dequeue.remote().future())
        if item is None:
            return None
        tq = self._ensure_tq()
        dp = await self._read_group(tq, item, PARTITION_STREAM, clear=True)
        return dp, queue_len

    def get_validate_sync(self) -> Any | None:
        handle = ray.get(self.queue_actor.get_validate.remote())
        if handle is None:
            return None
        tq = self._ensure_tq()
        return asyncio.run(self._read_group(tq, handle, PARTITION_VAL, clear=True))

    async def _read_group(self, tq, handle: _GroupHandle, partition: str, clear: bool) -> Any:
        td = await tq.async_kv_batch_get(keys=handle.row_keys, partition_id=partition)
        dp = _from_get_tensordict(td, handle.meta_info)
        # carry the referenced image keys so the trainer can release them post-step
        dp.meta_info["__tq_image_keys__"] = list(handle.image_keys)
        if clear:
            await tq.async_kv_clear(keys=handle.row_keys, partition_id=partition)
        return dp

    async def release_images(self, image_keys: list[str]) -> None:
        """Release a consumed group's image refs; clear keys that hit 0."""
        if not image_keys:
            return
        to_clear = await asyncio.wrap_future(self.queue_actor.release_images.remote(image_keys).future())
        if to_clear:
            tq = self._ensure_tq()
            await tq.async_kv_clear(keys=to_clear, partition_id=PARTITION_IMAGES)

    # -- stats / lifecycle -------------------------------------------------
    async def get_queue_size(self) -> int:
        return await asyncio.wrap_future(self.queue_actor.get_queue_size.remote().future())

    async def get_statistics(self) -> dict[str, Any]:
        return await asyncio.wrap_future(self.queue_actor.get_statistics.remote().future())

    async def clear_queue(self):
        image_keys = await asyncio.wrap_future(self.queue_actor.clear_queue.remote().future())
        if image_keys:
            tq = self._ensure_tq()
            try:
                await tq.async_kv_clear(keys=image_keys, partition_id=PARTITION_IMAGES)
            except Exception:
                logger.warning("clear_queue: failed to clear %d image keys", len(image_keys), exc_info=True)

    async def shutdown(self):
        await asyncio.wrap_future(self.queue_actor.shutdown.remote().future())
