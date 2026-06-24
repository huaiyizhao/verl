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

"""Foundation smoke test for the TransferQueue-backed data link.

Exercises ONLY the ``message_queue.py`` substrate (coordinator + client + TQ
round-trip + image dedup + refcount GC) using synthetic, already-expanded
episode DataProtos — it does not require the rollouter, agent loop, or trainer.

Requires ``transfer_queue`` (pyzmq + msgspec) and ``ray``; skipped otherwise.

Run:
    python -m pytest verl/experimental/fully_async_policy/unittest/test_transfer_queue_link.py -x -s
"""

import asyncio

import numpy as np
import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("transfer_queue")
pytest.importorskip("ray")

import ray  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from verl import DataProto  # noqa: E402
from verl.experimental.fully_async_policy.message_queue import (  # noqa: E402
    IMAGE_IDS_KEY,
    PARTITION_IMAGES,
    ROW_ID_KEY,
    TQ_IMAGE_PAYLOADS_KEY,
    MessageQueue,
    MessageQueueClient,
    init_transfer_queue,
)

PROMPT_LEN, RESP_LEN = 4, 6


def _img_payload(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "pixel_values": torch.randn(8, 16, generator=g),
        "image_grid_thw": torch.tensor([[1, 2, 4]], dtype=torch.long),
    }


def _make_episode(sample_id: str, rollout_id: int, n_rows: int, image_ids_per_row, payloads):
    """Build a synthetic expanded episode: uniform-shape token rows + image refs."""
    total = PROMPT_LEN + RESP_LEN
    base = (rollout_id + 1) * 1000
    batch = TensorDict(
        {
            "input_ids": torch.arange(base, base + n_rows * total).reshape(n_rows, total),
            "attention_mask": torch.ones(n_rows, total, dtype=torch.long),
            "responses": torch.arange(base, base + n_rows * RESP_LEN).reshape(n_rows, RESP_LEN),
            "response_mask": torch.ones(n_rows, RESP_LEN, dtype=torch.long),
        },
        batch_size=[n_rows],
    )
    row_ids = np.array([f"{sample_id}:{rollout_id}:{i}" for i in range(n_rows)], dtype=object)
    is_final = np.array([i == n_rows - 1 for i in range(n_rows)], dtype=object)
    image_ids = np.empty(n_rows, dtype=object)
    image_ids[:] = image_ids_per_row
    nt = {
        ROW_ID_KEY: row_ids,
        "sample_id": np.array([sample_id] * n_rows, dtype=object),
        "rollout_id": np.array([rollout_id] * n_rows, dtype=object),
        "is_final": is_final,
        IMAGE_IDS_KEY: image_ids,
    }
    meta = {"epoch": 0, TQ_IMAGE_PAYLOADS_KEY: payloads}
    return DataProto(batch=batch, non_tensor_batch=nt, meta_info=meta)


def _img_keys_in(part: str) -> set[str]:
    import transfer_queue as tq

    listing = tq.kv_list(partition_id=part)
    return set(listing.get(part, {}).keys())


@pytest.fixture(scope="module")
def ray_tq():
    if not ray.is_initialized():
        ray.init(num_cpus=4, ignore_reinit_error=True)
    cfg = OmegaConf.create(
        {
            "async_training": {
                "transfer_queue": {
                    "storage_backend": "SimpleStorage",
                    "SimpleStorage": {"num_data_storage_units": 2},
                }
            }
        }
    )
    init_transfer_queue(cfg)
    yield cfg
    try:
        import transfer_queue as tq

        tq.close()
    except Exception:
        pass
    ray.shutdown()


def test_data_link_foundation(ray_tq):
    cfg = ray_tq
    coordinator = MessageQueue.remote(cfg, max_queue_size=1000)
    client = MessageQueueClient(coordinator)

    # Two episodes; image A (sha "A") shared across both rollouts (e.g. the
    # initial screenshot), so dedup must store it once.
    pa, pb, pc = _img_payload(1), _img_payload(2), _img_payload(3)
    ep1 = _make_episode("promptX", 0, 3, [["A"], ["A"], ["B"]], {"A": pa, "B": pb})
    ep2 = _make_episode("promptX", 1, 2, [["A"], ["C"]], {"A": pa, "C": pc})

    async def scenario():
        # ---- produce -------------------------------------------------
        assert await client.put_sample(ep1) is True
        assert await client.put_sample(ep2) is True

        # dedup: A stored once, plus B and C -> 3 unique image records.
        assert _img_keys_in(PARTITION_IMAGES) == {"A", "B", "C"}

        stats = await client.get_statistics()
        assert stats["total_produced"] == 2
        assert stats["queue_size"] == 2

        # ---- consume episode 1 (FIFO) --------------------------------
        out = await client.get_sample()
        assert out is not None
        dp1, qlen = out
        assert qlen == 1
        assert len(dp1) == 3
        # token tensors byte-equal round-trip
        torch.testing.assert_close(dp1.batch["input_ids"], ep1.batch["input_ids"])
        torch.testing.assert_close(dp1.batch["response_mask"], ep1.batch["response_mask"])
        # identity columns preserved; exactly one final
        assert list(dp1.non_tensor_batch[ROW_ID_KEY]) == list(ep1.non_tensor_batch[ROW_ID_KEY])
        assert sum(bool(x) for x in dp1.non_tensor_batch["is_final"]) == 1
        assert sorted(set(dp1.meta_info["__tq_image_keys__"])) == ["A", "B"]

        # image tensors are real tensors in TQ (zero-copy path), byte-equal
        import transfer_queue as tq

        got_a = await tq.async_kv_batch_get(keys=["A"], partition_id=PARTITION_IMAGES)
        torch.testing.assert_close(got_a["pixel_values"].squeeze(0), pa["pixel_values"])

        # consumed rows are cleared from the stream
        assert _img_keys_in("rollout_stream") == set()

        # ---- refcount GC --------------------------------------------
        # release ep1's images: B -> 0 (cleared), A still referenced by ep2.
        await client.release_images(dp1.meta_info["__tq_image_keys__"])
        assert _img_keys_in(PARTITION_IMAGES) == {"A", "C"}

        # ---- consume episode 2 --------------------------------------
        dp2, qlen2 = await client.get_sample()
        assert qlen2 == 0
        assert len(dp2) == 2
        await client.release_images(dp2.meta_info["__tq_image_keys__"])
        assert _img_keys_in(PARTITION_IMAGES) == set()  # A and C now at 0

        # ---- termination sentinel -----------------------------------
        await client.put_sample(None)
        assert await client.get_sample() is None

        # ---- statistics ---------------------------------------------
        stats = await client.get_statistics()
        assert stats["total_produced"] == 2
        assert stats["total_consumed"] == 2
        assert stats["queue_size"] == 0

    asyncio.run(scenario())


def test_validation_roundtrip(ray_tq):
    cfg = ray_tq
    coordinator = MessageQueue.remote(cfg, max_queue_size=1000)
    client = MessageQueueClient(coordinator)
    ep = _make_episode("promptV", 0, 2, [["V"], ["V"]], {"V": _img_payload(7)})

    async def produce():
        assert await client.put_validate(ep) is True

    asyncio.run(produce())
    dp = client.get_validate_sync()
    assert dp is not None
    assert len(dp) == 2
    torch.testing.assert_close(dp.batch["input_ids"], ep.batch["input_ids"])
    assert client.get_validate_sync() is None
