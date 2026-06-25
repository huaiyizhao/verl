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
"""Content-hash deduplication of processed multimodal image tensors for the V1
TransferQueue (TQ) data plane.

The V1 agent-loop worker stores each trajectory row's ``multi_modal_inputs``
(processed ``pixel_values`` / ``image_grid_thw`` …) inline. For an image-heavy
multi-turn agent the same screenshot recurs across the sliding-window turns and
across the ``rollout.n`` GRPO samples of a prompt, so inline storage duplicates
large pixel tensors many times.

This module stores each *unique* processed image once in a separate TQ
partition (:data:`PARTITION_IMAGES`) keyed by the SHA1 of its content, and lets
a trajectory row carry only a small :data:`IMAGE_IDS_KEY` list (SHA1 strings) in
place of the pixel tensors:

* **Producer** (agent-loop worker): :func:`split_multimodal_per_image` slices a
  turn's ``multi_modal_inputs`` into per-image payloads keyed by content hash;
  the worker ``kv_put``\\s unique payloads into ``rollout_images`` and stores the
  row's ``image_ids`` instead of ``multi_modal_inputs``.
* **Consumer** (inside the worker that materializes a TQ batch):
  :func:`resolve_image_ids` fetches the referenced images back (deduped across
  the batch, served from a per-worker :class:`ImageLRU`) and reassembles
  ``multi_modal_inputs`` per row.
* **Lifecycle**: :class:`ImageRefCounter` (a Ray actor) refcounts image keys so
  an image is cleared only once no live row references it.

Dedup is opt-in and leaves ``verl/trainer/ppo/v1/`` untouched: it is enabled by
selecting the subclasses in :mod:`verl.utils.transferqueue_image_dedup_v1`
(``actor_rollout_ref.rollout.agent.agent_loop_manager_class`` for the producer and
``trainer.v1.sampler.custom_sampler`` for the lifecycle). When not selected, no row
carries ``image_ids`` and the consume hook is a no-op, so behavior is exactly
upstream. This module holds the shared helpers used by those subclasses.

The pure tensor helpers (:func:`content_key`, :func:`split_multimodal_per_image`,
:func:`reconstruct_row_multimodal`) depend only on ``torch`` and are unit-tested
without a running TransferQueue.
"""

import hashlib
from collections import OrderedDict
from typing import Any

import torch

# Separate TQ partitions for deduped image payloads (train / validation kept
# apart so val GC never deletes a train image and vice-versa).
PARTITION_IMAGES = "rollout_images"
PARTITION_IMAGES_VAL = "rollout_images_val"

# Per-row field that replaces inline ``multi_modal_inputs`` with a list of SHA1
# keys into the image partition.
IMAGE_IDS_KEY = "image_ids"
MULTI_MODAL_INPUTS_KEY = "multi_modal_inputs"


# ---------------------------------------------------------------------------
# Pure tensor helpers (no TransferQueue dependency)
# ---------------------------------------------------------------------------
def content_key(*tensors: torch.Tensor) -> str:
    """Deterministic content hash of one or more tensors (dtype/shape aware).

    ``bfloat16`` has no numpy dtype, so it is upcast to ``float32`` purely for
    byte extraction (the original dtype is still folded into the digest, so two
    images that differ only in dtype do not collide).
    """
    digest = hashlib.sha1()
    for t in tensors:
        a = t.detach().cpu().contiguous()
        if a.dtype == torch.bfloat16:
            a = a.to(torch.float32)
        digest.update(str((tuple(t.shape), str(t.dtype))).encode("utf-8"))
        digest.update(a.numpy().tobytes())
    return f"sha1:{digest.hexdigest()}"


def split_multimodal_per_image(
    multi_modal_inputs: dict[str, Any] | None,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Split a turn's processor output into per-image payloads keyed by content hash.

    Slices ``pixel_values`` ``[sum_patches, D]`` by the per-image patch counts in
    ``image_grid_thw`` ``[num_images, 3]`` (t, h, w). Each unique image maps to a
    single content-hash key (SHA1 of its processed ``pixel_values`` + grid), so
    the same screenshot recurring across turns/rollouts dedups to one stored
    record. PIL-free: keys come from the deterministic processed tensors.

    Returns:
        ``(image_ids, payloads)`` where ``image_ids`` is the per-image key list
        (order preserved, one per image in the row) and ``payloads`` maps each
        unique key to ``{"pixel_values", "image_grid_thw", "images_seqlens"}``.
    """
    if not multi_modal_inputs:
        return [], {}
    grid = multi_modal_inputs.get("image_grid_thw")
    pixel_values = multi_modal_inputs.get("pixel_values")
    if grid is None or pixel_values is None:
        return [], {}

    patches_per_image = (grid[:, 0] * grid[:, 1] * grid[:, 2]).tolist()
    if sum(patches_per_image) != pixel_values.shape[0]:
        raise ValueError(
            f"sum of per-image patches ({sum(patches_per_image)}) != pixel_values rows "
            f"({pixel_values.shape[0]}); image_grid_thw / pixel_values misaligned"
        )

    image_ids: list[str] = []
    payloads: dict[str, dict[str, Any]] = {}
    offset = 0
    for i in range(grid.shape[0]):
        n_patches = int(patches_per_image[i])
        pv_i = pixel_values[offset : offset + n_patches].contiguous()
        grid_i = grid[i : i + 1].contiguous()
        image_id = content_key(pv_i, grid_i)
        image_ids.append(image_id)
        if image_id not in payloads:
            payloads[image_id] = {
                "pixel_values": pv_i,
                "image_grid_thw": grid_i,
                "images_seqlens": torch.repeat_interleave(grid_i[:, 1] * grid_i[:, 2], grid_i[:, 0]),
            }
        offset += n_patches
    return image_ids, payloads


def _densify(t: Any) -> Any:
    """Convert a TransferQueue nested/jagged tensor back to a dense tensor.

    SimpleStorage reconstructs multi-key ``kv_batch_get`` results as nested
    tensors; a single image's payload is uniform, so ``stack(unbind())`` yields
    the exact original dense tensor.
    """
    if isinstance(t, torch.Tensor) and t.is_nested:
        return torch.stack(list(t.unbind()))
    return t


def reconstruct_row_multimodal(
    image_ids_per_row: list[list[str]],
    fetched: dict[str, dict[str, Any]],
) -> list[dict[str, Any] | None]:
    """Reassemble per-row ``multi_modal_inputs`` from deduped image payloads.

    Inverse of :func:`split_multimodal_per_image`: a row's images are
    concatenated back in reference order (``pixel_values`` along patches,
    ``image_grid_thw`` / ``images_seqlens`` along the image axis). Rows with no
    images yield ``None``.

    Args:
        image_ids_per_row: per-row list of SHA1 image keys.
        fetched: mapping ``{image_key: {"pixel_values", "image_grid_thw", ...}}``.
    """
    rows: list[dict[str, Any] | None] = []
    for ids in image_ids_per_row:
        if not ids:
            rows.append(None)
            continue
        parts: dict[str, list[Any]] = {}
        for key in ids:
            payload = fetched[key]
            for fname, value in payload.items():
                parts.setdefault(fname, []).append(value)
        merged: dict[str, Any] = {}
        for fname, values in parts.items():
            merged[fname] = torch.cat([_densify(v) for v in values], dim=0)
        rows.append(merged)
    return rows


# ---------------------------------------------------------------------------
# Per-worker LRU cache for resolved image payloads
# ---------------------------------------------------------------------------
class ImageLRU:
    """Count-bounded per-worker cache of resolved image payloads (keyed by SHA1).

    A hot screenshot shared across the ``rollout.n`` samples of a prompt and
    across steps is fetched from TQ once per worker, not once per row.
    """

    def __init__(self, maxsize: int = 2048):
        self.maxsize = maxsize
        self._d: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def get(self, key: str) -> dict[str, Any] | None:
        payload = self._d.get(key)
        if payload is not None:
            self._d.move_to_end(key)
        return payload

    def put(self, key: str, payload: dict[str, Any]) -> None:
        self._d[key] = payload
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)


# ---------------------------------------------------------------------------
# Consume-side resolution (worker materialization hook helper)
# ---------------------------------------------------------------------------
async def resolve_image_ids(
    tensordict: Any,
    *,
    partition: str = PARTITION_IMAGES,
    cache: ImageLRU | None = None,
) -> Any:
    """Resolve an ``image_ids`` column in a materialized TQ batch to ``multi_modal_inputs``.

    No-op if the batch carries no :data:`IMAGE_IDS_KEY` (text-only or dedup off).
    Otherwise fetches the unique referenced images from ``partition`` (deduped
    across rows, served from ``cache``), reassembles per-row
    ``multi_modal_inputs``, writes it back as a ``NonTensorStack`` column, and
    drops ``image_ids``. Callers (GPU workers) must have attached via
    ``tq.init()`` beforehand.
    """
    if IMAGE_IDS_KEY not in tensordict.keys():
        return tensordict

    import transfer_queue as tq
    from tensordict.tensorclass import NonTensorData

    from verl.utils.tensordict_utils import assign_non_tensor_stack

    n = int(tensordict.batch_size[0])
    ids_col = tensordict.get(IMAGE_IDS_KEY)
    image_ids_per_row: list[list[str]] = []
    for i in range(n):
        value = ids_col[i]
        value = value.data if isinstance(value, NonTensorData) else value
        image_ids_per_row.append(list(value) if value else [])

    cache = cache if cache is not None else ImageLRU()
    needed = sorted({k for ids in image_ids_per_row for k in ids if k not in cache})
    if needed:
        # Each key is a single image; a multi-key get is jagged (heterogeneous
        # shapes), so ``unbind()[i]`` yields each image at its native shape.
        td = await tq.async_kv_batch_get(keys=needed, partition_id=partition)
        new_payloads: dict[str, dict[str, Any]] = {k: {} for k in needed}
        for fname in td.keys():
            col = td.get(fname)
            cols = col.unbind() if getattr(col, "is_nested", False) else [col[i] for i in range(len(needed))]
            for i, key in enumerate(needed):
                new_payloads[key][fname] = cols[i]
        for key, payload in new_payloads.items():
            cache.put(key, payload)

    # Snapshot the payloads this batch needs so concurrent LRU eviction cannot
    # disturb the in-flight reconstruct.
    fetched: dict[str, dict[str, Any]] = {}
    for ids in image_ids_per_row:
        for key in ids:
            if key not in fetched:
                fetched[key] = cache.get(key)

    mm_per_row = reconstruct_row_multimodal(image_ids_per_row, fetched)
    assign_non_tensor_stack(tensordict, MULTI_MODAL_INPUTS_KEY, [m if m else {} for m in mm_per_row])
    del tensordict[IMAGE_IDS_KEY]
    return tensordict


# Per-worker LRU of resolved image payloads, shared across the
# old_log_prob / ref / update_actor passes within a step. Lazily created so the
# default (dedup-off) path never allocates it.
_WORKER_IMAGE_LRU: ImageLRU | None = None


async def maybe_resolve_image_ids(tensordict: Any) -> Any:
    """Consume-side hook: resolve a deduped ``image_ids`` column into ``multi_modal_inputs``.

    Called once per worker materialization from
    :func:`verl.utils.transferqueue_utils._async_meta_to_realdata`. A no-op unless
    the batch carries an ``image_ids`` column — only train rows written by the
    dedup producer do — so the default (dedup-off) path and validation are
    untouched. Manages a process-local :class:`ImageLRU` so a hot screenshot is
    fetched once per worker.
    """
    if IMAGE_IDS_KEY not in tensordict.keys():
        return tensordict
    global _WORKER_IMAGE_LRU
    if _WORKER_IMAGE_LRU is None:
        _WORKER_IMAGE_LRU = ImageLRU()
    return await resolve_image_ids(tensordict, cache=_WORKER_IMAGE_LRU)


# ---------------------------------------------------------------------------
# Image lifecycle: refcount shared image keys, clear at zero
# ---------------------------------------------------------------------------
IMAGE_REFCOUNTER_ACTOR = "verl_image_refcounter"


def get_image_refcounter():
    """Return the shared named ``ImageRefCounter`` Ray actor, creating it once.

    Both the producer (agent-loop worker, on ``incref``) and the trainer driver
    (on ``decref``/GC) call this to reach the same actor without threading a
    handle through constructors. ``get_if_exists`` makes concurrent creation
    race-free. Defined as a factory (not a module-level ``@ray.remote`` class)
    so importing this module stays ``ray``-free for the pure tensor helpers and
    their unit tests.
    """
    import ray

    @ray.remote
    class ImageRefCounter:
        """Refcounts deduped image keys across live trajectory rows.

        ``incref`` once per row reference (a screenshot shared by 8 GRPO rows
        gets refcount 8); ``decref`` when a row is consumed/cleared. Keys whose
        count reaches 0 are returned so the caller can ``kv_clear`` them from the
        image partition.
        """

        def __init__(self):
            self._counts: dict[str, int] = {}

        def incref(self, image_keys: list[str]) -> None:
            for k in image_keys:
                self._counts[k] = self._counts.get(k, 0) + 1

        def decref(self, image_keys: list[str]) -> list[str]:
            to_clear: list[str] = []
            for k in image_keys:
                c = self._counts.get(k, 0) - 1
                if c <= 0:
                    self._counts.pop(k, None)
                    to_clear.append(k)
                else:
                    self._counts[k] = c
            return to_clear

    return ImageRefCounter.options(
        name=IMAGE_REFCOUNTER_ACTOR,
        lifetime="detached",
        get_if_exists=True,
    ).remote()


def fetch_image_ids(row_keys, row_partition: str) -> list[str]:
    """Return the flat list of image keys referenced by ``row_keys`` (rows must be alive).

    Self-gating: returns ``[]`` unless image dedup is active (the refcounter actor
    exists) and the rows carry an ``image_ids`` column. Call this while the rows
    still exist in TransferQueue (before they are cleared) — e.g. capture a
    sampled batch's images before it is consumed, or a dropped batch's images
    before it is cleared.
    """
    import ray
    import transfer_queue as tq
    from tensordict.tensorclass import NonTensorData

    if not row_keys:
        return []
    try:
        ray.get_actor(IMAGE_REFCOUNTER_ACTOR)
    except ValueError:
        return []  # dedup not active
    try:
        td = tq.kv_batch_get(keys=list(row_keys), partition_id=row_partition, select_fields=[IMAGE_IDS_KEY])
    except Exception:  # noqa: BLE001 - rows without an image_ids column (validation / dedup off)
        return []

    col = td.get(IMAGE_IDS_KEY)
    flat: list[str] = []
    for i in range(len(row_keys)):
        value = col[i]
        value = value.data if isinstance(value, NonTensorData) else value
        if value:
            flat.extend(value)
    return flat


def decref_images(image_keys, *, image_partition: str = PARTITION_IMAGES) -> None:
    """Decrement refcounts for ``image_keys``; clear from TQ any that reach zero.

    No-op if dedup is inactive (no refcounter actor) or ``image_keys`` is empty.
    Use this with image keys captured earlier (the referencing rows may already be
    gone, so it does not re-read them).
    """
    if not image_keys:
        return
    import ray
    import transfer_queue as tq

    try:
        refcounter = ray.get_actor(IMAGE_REFCOUNTER_ACTOR)
    except ValueError:
        return
    to_clear = ray.get(refcounter.decref.remote(list(image_keys)))
    if to_clear:
        tq.kv_clear(keys=to_clear, partition_id=image_partition)


def release_row_images(row_keys, row_partition: str) -> None:
    """Decref + clear the images of still-alive rows (fetch then decref in one step)."""
    decref_images(fetch_image_ids(row_keys, row_partition))
