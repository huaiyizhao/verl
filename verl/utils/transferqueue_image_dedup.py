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
"""Content-hash deduplication of processed multimodal image tensors for the V1
TransferQueue (TQ) data plane.

The V1 agent-loop worker stores each trajectory row's ``multi_modal_inputs``
(processed ``pixel_values`` / ``image_grid_thw`` …) inline. For an image-heavy
multi-turn agent the same screenshot recurs across the sliding-window turns and
across the ``rollout.n`` GRPO samples of a prompt, so inline storage duplicates
large pixel tensors many times.

This module stores each *unique* processed image once in a separate TQ
partition (:data:`PARTITION_IMAGES`) keyed by ``{session}_{SHA1}`` (the SHA1 of
its content, namespaced by the owning rollout session), and lets a trajectory row
carry only a small :data:`IMAGE_IDS_KEY` list of those keys in place of the pixel
tensors:

* **Producer** (agent-loop worker): :func:`split_multimodal_per_image` slices a
  turn's ``multi_modal_inputs`` into per-image payloads keyed by content hash,
  namespaced by the ``{uid}_{session_id}`` of the rollout that produced them; the
  worker ``kv_put``\\s unique payloads into ``rollout_images`` and stores the
  row's ``image_ids`` instead of ``multi_modal_inputs``. Dedup is **within a
  rollout** (the dominant redundancy — a screenshot recurs across the sliding-
  window turns of one session); sibling GRPO sessions of the same prompt no longer
  share storage, which removes all cross-rollout coordination.
* **Consumer** (inside the worker that materializes a TQ batch):
  :func:`resolve_image_ids` fetches the referenced images back (deduped across
  the batch, served from a per-worker :class:`ImageLRU`) and reassembles
  ``multi_modal_inputs`` per row.
* **Lifecycle**: because each image key is owned by exactly one session, an image
  is cleared (:func:`clear_images`) unconditionally once that session's rows leave
  the replay buffer — no shared refcount actor is needed.

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
import logging
import os
import time
from collections import OrderedDict
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Master profiling switch: VERL_PROFILE=1 logs the dedup-resolve fetch/reconstruct breakdown
# (plus the heavy per-row [RESOLVE_DBG] dumps). The lighter VERL_STEP_PROFILE turns on ONLY the
# [RESOLVE_PROFILE] fetch-vs-reconstruct timing (which splits the [MATERIALIZE_PROFILE] mm_resolve),
# without the heavy dumps.
_PROFILE = os.getenv("VERL_PROFILE", "0") not in ("0", "false", "False", "")
_STEP_PROFILE = _PROFILE or os.getenv("VERL_STEP_PROFILE", "0") not in ("0", "false", "False", "")
_RESOLVE_DBG = [0]  # cap the per-row image_ids dump so it never floods the log

# Separate TQ partitions for deduped image payloads (train / validation kept
# apart so val GC never deletes a train image and vice-versa).
PARTITION_IMAGES = "rollout_images"
PARTITION_IMAGES_VAL = "rollout_images_val"

# Per-row field that replaces inline ``multi_modal_inputs`` with the image keys of
# the row, stored as a single delimiter-joined STRING (``""`` for a text-only row).
#
# Why a string and not a ``list[str]``: a per-row ``list`` column becomes a TensorDict
# "LinkedList" that treats the inner lists as nested batch structure. Variable-length
# lists — and especially the empty ``[]`` of a text row — do not survive the TQ
# serialization round-trip: the empties get dropped, the column compacts to fewer rows
# than the batch, and every row past the first text row is silently shifted (trailing
# rows read out of bounds and pick up stale images). A flat scalar-string column is the
# same shape as ``uid`` / ``data_source`` and round-trips robustly. ``\x1f`` (unit
# separator) cannot occur in a content key (``{uid}_{sid}_sha1:<hex>``).
IMAGE_IDS_KEY = "image_ids"
IMAGE_IDS_DELIM = "\x1f"
MULTI_MODAL_INPUTS_KEY = "multi_modal_inputs"


def encode_image_ids(image_ids: list[str]) -> str:
    """Join a row's image keys into the scalar string stored in the ``image_ids`` column."""
    return IMAGE_IDS_DELIM.join(image_ids)


def decode_image_ids(value: Any) -> list[str]:
    """Inverse of :func:`encode_image_ids`; tolerant of the legacy ``list`` form and ``None``."""
    if value is None:
        return []
    if isinstance(value, str):
        return value.split(IMAGE_IDS_DELIM) if value else []
    return list(value) if value else []  # back-compat: legacy list-valued image_ids


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
    namespace: str = "",
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Split a turn's processor output into per-image payloads keyed by content hash.

    Slices ``pixel_values`` ``[sum_patches, D]`` by the per-image patch counts in
    ``image_grid_thw`` ``[num_images, 3]`` (t, h, w). Each unique image maps to a
    single content-hash key (SHA1 of its processed ``pixel_values`` + grid), so
    the same screenshot recurring across the turns of one rollout dedups to one
    stored record. PIL-free: keys come from the deterministic processed tensors.

    ``namespace`` (the producing rollout's ``{uid}_{session_id}``) is prefixed onto
    every key so images are scoped to a single session: identical screenshots in
    different sessions get distinct keys and never alias, which makes lifecycle GC
    a plain per-session clear (no cross-rollout refcount). Within one ``namespace``,
    a repeated screenshot still dedups to one record.

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

    prefix = f"{namespace}_" if namespace else ""
    image_ids: list[str] = []
    payloads: dict[str, dict[str, Any]] = {}
    offset = 0
    for i in range(grid.shape[0]):
        n_patches = int(patches_per_image[i])
        pv_i = pixel_values[offset : offset + n_patches].contiguous()
        grid_i = grid[i : i + 1].contiguous()
        image_id = prefix + content_key(pv_i, grid_i)
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
        image_ids_per_row.append(decode_image_ids(value))

    # Defensive: a row may reference an image already gone from the store (a dedup-clear lifecycle
    # race can remove a still-referenced image -> the storage-unit "key ... not found" that otherwise
    # kills the whole run). Detect missing images up front and DROP the offending rows, so training
    # continues on the rest instead of crashing. Logged so the frequency is visible.
    existing = set(((tq.kv_list() or {}).get(partition) or {}).keys())
    if existing:
        missing = {k for ids in image_ids_per_row for k in ids if k not in existing}
        if missing:
            keep = [i for i, ids in enumerate(image_ids_per_row) if not any(k in missing for k in ids)]
            logger.warning(
                "[RESOLVE] dedup-clear race: %d/%d unique images missing from '%s'; dropping %d/%d rows "
                "referencing them and continuing (e.g. %s)",
                len(missing),
                len({k for ids in image_ids_per_row for k in ids}),
                partition,
                n - len(keep),
                n,
                next(iter(missing)),
            )
            if len(keep) < n:
                if not keep:
                    # whole shard unusable — extremely unlikely; surface a clear error rather than
                    # silently returning an empty batch (which would trip _balance_batch 0-items).
                    raise RuntimeError(
                        f"resolve_image_ids: ALL {n} rows reference missing images in '{partition}' "
                        f"(dedup-clear race); cannot recover this batch."
                    )
                tensordict = tensordict[keep]
                image_ids_per_row = [image_ids_per_row[i] for i in keep]
                ids_col = tensordict.get(IMAGE_IDS_KEY)
                n = len(keep)

    if _PROFILE and _RESOLVE_DBG[0] < 12:
        _RESOLVE_DBG[0] += 1
        import hashlib

        def _sig(ids: list[str]) -> str:
            # per-row signature: "<nkeys>:<short hash of the key tuple>" (text-row "-" should
            # NOT appear — every GUI-agent turn carries >=1 image).
            return f"{len(ids)}:{hashlib.sha1(chr(31).join(ids).encode()).hexdigest()[:6]}" if ids else "-"

        # Smoking-gun check: does the image_ids column actually hold n rows, or is it shorter than
        # batch_size (=> the trailing `for i in range(n)` indices read OOB and wrap/broadcast)?
        try:
            col_len = len(ids_col)
        except Exception:  # noqa: BLE001
            col_len = -1
        try:
            iid_len = len(tensordict.get("input_ids"))
        except Exception:  # noqa: BLE001
            iid_len = -1
        raw_sample = []
        for i in (0, n - 1):
            rv = ids_col[i]
            rv = rv.data if isinstance(rv, NonTensorData) else rv
            raw_sample.append(f"row{i}={type(rv).__name__}:{repr(rv)[:40]}")
        logger.warning(
            "[RESOLVE_DBG] batch_size=%d image_ids_col_len=%d input_ids_col_len=%d col_type=%s per_row=%s raw=%s",
            n,
            col_len,
            iid_len,
            type(ids_col).__name__,
            " ".join(f"{i}={_sig(image_ids_per_row[i])}" for i in range(n)),
            " ".join(raw_sample),
        )

    cache = cache if cache is not None else ImageLRU()
    needed = sorted({k for ids in image_ids_per_row for k in ids if k not in cache})
    _t = time.perf_counter()
    if needed:
        # Each key is a single image; a multi-key get is jagged (heterogeneous
        # shapes), so ``unbind()[i]`` yields each image at its native shape.
        td = await tq.async_kv_batch_get(keys=needed, partition_id=partition)
        new_payloads: dict[str, dict[str, Any]] = {k: {} for k in needed}
        _dbg_collen = {}
        for fname in td.keys():
            col = td.get(fname)
            is_nested = getattr(col, "is_nested", False)
            cols = col.unbind() if is_nested else [col[i] for i in range(len(needed))]
            _dbg_collen[fname] = (is_nested, len(cols), tuple(getattr(col, "shape", ())))
            for i, key in enumerate(needed):
                new_payloads[key][fname] = cols[i]
        for key, payload in new_payloads.items():
            cache.put(key, payload)

        if _PROFILE and _RESOLVE_DBG[0] <= 12:
            # Verify the user's hypothesis: do DISTINCT image keys come back as the SAME image
            # (a kv_batch_get / unbind misalignment for same-shape images), i.e. a fetch-side
            # broadcast — vs every key getting its own distinct image.
            def _imgfp(p: dict) -> str:
                pv = p.get("pixel_values")
                if pv is None:
                    return "?"
                f = pv.reshape(-1).to(torch.float32)
                s = f[:: max(1, f.numel() // 8)][:8]
                return f"{tuple(pv.shape)}:{float(s.sum()):.4e}"

            fps = [(k[-10:], _imgfp(new_payloads[k])) for k in needed]
            n_distinct_fp = len({fp for _, fp in fps})
            logger.warning(
                "[RESOLVE_DBG_IMG] n_needed=%d n_distinct_image_fp=%d collen=%s key->fp: %s",
                len(needed),
                n_distinct_fp,
                _dbg_collen,
                " ".join(f"{k}={fp}" for k, fp in fps[:16]),
            )
    t_fetch = time.perf_counter() - _t

    # Snapshot the payloads this batch needs so concurrent LRU eviction cannot
    # disturb the in-flight reconstruct.
    fetched: dict[str, dict[str, Any]] = {}
    for ids in image_ids_per_row:
        for key in ids:
            if key not in fetched:
                fetched[key] = cache.get(key)

    _t = time.perf_counter()
    mm_per_row = reconstruct_row_multimodal(image_ids_per_row, fetched)
    assign_non_tensor_stack(tensordict, MULTI_MODAL_INPUTS_KEY, [m if m else {} for m in mm_per_row])
    del tensordict[IMAGE_IDS_KEY]
    t_reconstruct = time.perf_counter() - _t

    if _STEP_PROFILE:
        total_imgs = sum(len(ids) for ids in image_ids_per_row)
        # print (not logger.warning): the worker logging handler swallows warnings. This splits the
        # [MATERIALIZE_PROFILE] mm_resolve into image FETCH (ZMQ get of the unique images) vs
        # RECONSTRUCT (per-row torch.cat), so we know which half of the ~77s to attack.
        print(
            f"[RESOLVE_PROFILE] n_rows={n} total_image_refs={total_imgs} unique_fetched={len(needed)} "
            f"img_fetch={t_fetch:.3f}s reconstruct={t_reconstruct:.3f}s total={t_fetch + t_reconstruct:.3f}s",
            flush=True,
        )
    return tensordict


# Per-worker LRU of resolved image payloads, shared across the
# old_log_prob / ref / update_actor passes within a step. Lazily created so the
# default (dedup-off) path never allocates it.
#
# Size (in cached images) of the persistent cross-step LRU. Each cached payload is a full
# ``pixel_values`` tensor (~10-20 MB), so at the old default of 2048 the cache alone pinned
# ~25 GB of host RAM per worker. Default 0 = DISABLED: use a per-call cache instead (still dedups
# within a batch, but drops afterward) so hot screenshots never accumulate across steps. Set
# VERL_IMAGE_LRU_MAXSIZE>0 to re-enable the cross-step cache (trades host RAM for fewer TQ fetches).
_IMAGE_LRU_MAXSIZE = int(os.getenv("VERL_IMAGE_LRU_MAXSIZE", "0"))
_WORKER_IMAGE_LRU: ImageLRU | None = None


async def maybe_resolve_image_ids(tensordict: Any) -> Any:
    """Consume-side hook: resolve a deduped ``image_ids`` column into ``multi_modal_inputs``.

    Called once per worker materialization from
    :func:`verl.utils.transferqueue_utils._async_meta_to_realdata`. A no-op unless
    the batch carries an ``image_ids`` column — only train rows written by the
    dedup producer do — so the default (dedup-off) path and validation are
    untouched. With ``VERL_IMAGE_LRU_MAXSIZE>0`` a process-local :class:`ImageLRU`
    keeps hot screenshots across steps; by default (0) a per-call cache is used so
    nothing is pinned across steps.
    """
    if IMAGE_IDS_KEY not in tensordict.keys():
        return tensordict
    if _IMAGE_LRU_MAXSIZE <= 0:
        # Persistent cache disabled: cache=None -> resolve_image_ids builds a fresh per-call cache
        # (dedups within this batch, freed on return). No cross-step accumulation.
        return await resolve_image_ids(tensordict, cache=None)
    global _WORKER_IMAGE_LRU
    if _WORKER_IMAGE_LRU is None:
        _WORKER_IMAGE_LRU = ImageLRU(maxsize=_IMAGE_LRU_MAXSIZE)
    return await resolve_image_ids(tensordict, cache=_WORKER_IMAGE_LRU)


# ---------------------------------------------------------------------------
# Image lifecycle: session-scoped, clear unconditionally
# ---------------------------------------------------------------------------
# Image keys are namespaced by ``{uid}_{session_id}`` (see split_multimodal_per_image),
# so every image is referenced only by the rows of its one owning session. The whole
# session's rows enter and leave the replay buffer together (sampled as a complete
# prompt, or dropped together by their shared staleness), so an image can be cleared
# as soon as its session's rows are gone — no cross-session refcount is required.


def fetch_image_ids(row_keys, row_partition: str) -> list[str]:
    """Return the flat list of image keys referenced by ``row_keys`` (rows must be alive).

    Returns ``[]`` unless the rows carry an ``image_ids`` column (validation rows /
    dedup off don't). Call this while the rows still exist in TransferQueue (before
    they are cleared) — e.g. capture a sampled batch's images before it is consumed,
    or a dropped batch's images before it is cleared.
    """
    import transfer_queue as tq
    from tensordict.tensorclass import NonTensorData

    if not row_keys:
        return []
    try:
        td = tq.kv_batch_get(keys=list(row_keys), partition_id=row_partition, select_fields=[IMAGE_IDS_KEY])
    except Exception:  # noqa: BLE001 - rows without an image_ids column (validation / dedup off)
        return []

    col = td.get(IMAGE_IDS_KEY)
    flat: list[str] = []
    for i in range(len(row_keys)):
        value = col[i]
        value = value.data if isinstance(value, NonTensorData) else value
        flat.extend(decode_image_ids(value))
    return flat


def clear_images(image_keys, *, image_partition: str = PARTITION_IMAGES) -> None:
    """Clear deduped image payloads from TQ.

    Use with the image keys of rows that have just been consumed or dropped. Because
    keys are session-namespaced (each image owned by a single session), clearing is
    unconditional — no refcount. Duplicate keys (one image referenced by several turns
    of its session) are de-duplicated before the clear; a no-op if empty.
    """
    if not image_keys:
        return
    import transfer_queue as tq

    tq.kv_clear(keys=list(dict.fromkeys(image_keys)), partition_id=image_partition)
