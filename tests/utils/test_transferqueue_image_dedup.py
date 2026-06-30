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
"""Unit tests for the pure tensor helpers of TransferQueue image dedup.

These exercise content hashing, per-image splitting and per-row reconstruction
without a running TransferQueue (no ``transfer_queue`` / GPU required).
"""

import torch

from verl.utils.transferqueue_image_dedup import (
    content_key,
    decode_image_ids,
    encode_image_ids,
    reconstruct_row_multimodal,
    split_multimodal_per_image,
)

D = 8  # small hidden dim for processed pixel patches


def _image(grid_thw: tuple[int, int, int], fill: float, dtype=torch.float32) -> dict:
    """One-image ``multi_modal_inputs`` with deterministic pixel content."""
    t, h, w = grid_thw
    n_patches = t * h * w
    pixel_values = (torch.arange(n_patches * D, dtype=torch.float32).reshape(n_patches, D) + fill).to(dtype)
    image_grid_thw = torch.tensor([[t, h, w]], dtype=torch.int64)
    return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}


def _concat(*images: dict) -> dict:
    """Merge per-image dicts into a multi-image row's ``multi_modal_inputs``."""
    return {
        "pixel_values": torch.cat([im["pixel_values"] for im in images], dim=0),
        "image_grid_thw": torch.cat([im["image_grid_thw"] for im in images], dim=0),
    }


def test_content_key_deterministic_and_distinct():
    a = _image((1, 2, 2), fill=0.0)
    a2 = _image((1, 2, 2), fill=0.0)
    b = _image((1, 2, 2), fill=100.0)
    ka = content_key(a["pixel_values"], a["image_grid_thw"])
    ka2 = content_key(a2["pixel_values"], a2["image_grid_thw"])
    kb = content_key(b["pixel_values"], b["image_grid_thw"])
    assert ka == ka2, "identical content must hash equal"
    assert ka != kb, "different content must hash differently"
    assert ka.startswith("sha1:")


def test_content_key_bfloat16_does_not_crash():
    a = _image((1, 2, 2), fill=1.0, dtype=torch.bfloat16)
    key = content_key(a["pixel_values"], a["image_grid_thw"])
    assert key.startswith("sha1:")
    # dtype is folded into the digest: same bytes, different dtype -> different key
    a_fp32 = _image((1, 2, 2), fill=1.0, dtype=torch.float32)
    assert key != content_key(a_fp32["pixel_values"], a_fp32["image_grid_thw"])


def test_split_slices_by_patch_counts():
    a = _image((1, 2, 2), fill=0.0)  # 4 patches
    b = _image((1, 2, 4), fill=50.0)  # 8 patches
    row = _concat(a, b)
    image_ids, payloads = split_multimodal_per_image(row)

    assert len(image_ids) == 2
    assert len(payloads) == 2
    # payload pixel slices match the original per-image tensors exactly
    assert torch.equal(payloads[image_ids[0]]["pixel_values"], a["pixel_values"])
    assert torch.equal(payloads[image_ids[1]]["pixel_values"], b["pixel_values"])
    # images_seqlens = t-repeated (h*w) per image
    assert torch.equal(payloads[image_ids[0]]["images_seqlens"], torch.tensor([4]))
    assert torch.equal(payloads[image_ids[1]]["images_seqlens"], torch.tensor([8]))


def test_split_dedups_repeated_image_within_row():
    a = _image((1, 2, 2), fill=7.0)
    row = _concat(a, a)  # same screenshot twice
    image_ids, payloads = split_multimodal_per_image(row)
    assert image_ids[0] == image_ids[1], "identical images share one key"
    assert len(payloads) == 1, "stored once despite two references"


def test_split_misaligned_raises():
    row = {
        "pixel_values": torch.zeros(5, D),  # 5 != 4 patches implied by grid
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.int64),
    }
    try:
        split_multimodal_per_image(row)
    except ValueError as e:
        assert "misaligned" in str(e)
    else:
        raise AssertionError("expected ValueError on misaligned pixel_values/grid")


def test_split_text_only_returns_empty():
    assert split_multimodal_per_image(None) == ([], {})
    assert split_multimodal_per_image({}) == ([], {})
    assert split_multimodal_per_image({"foo": torch.zeros(1)}) == ([], {})


def test_roundtrip_split_then_reconstruct():
    a = _image((1, 2, 2), fill=0.0)
    b = _image((1, 2, 4), fill=50.0)
    row = _concat(a, b)
    image_ids, payloads = split_multimodal_per_image(row)

    [rebuilt] = reconstruct_row_multimodal([image_ids], payloads)
    assert torch.equal(rebuilt["pixel_values"], row["pixel_values"])
    assert torch.equal(rebuilt["image_grid_thw"], row["image_grid_thw"])


def test_reconstruct_text_only_row_is_none():
    assert reconstruct_row_multimodal([[]], {}) == [None]


def test_cross_row_dedup_unique_storage():
    """A screenshot shared across two rows (same session) is stored once; both rows rebuild it."""
    a = _image((1, 2, 2), fill=0.0)
    b = _image((1, 2, 4), fill=50.0)
    row1 = _concat(a, b)
    row2 = _image((1, 2, 2), fill=0.0)  # same as `a`

    ids1, pay1 = split_multimodal_per_image(row1)
    ids2, pay2 = split_multimodal_per_image(row2)
    bank = {**pay1, **pay2}  # merge as the producer would across rows

    assert ids2[0] == ids1[0], "shared image -> shared key"
    assert len(bank) == 2, "A and B stored once each despite A used twice"

    rebuilt = reconstruct_row_multimodal([ids1, ids2], bank)
    assert torch.equal(rebuilt[0]["pixel_values"], row1["pixel_values"])
    assert torch.equal(rebuilt[1]["pixel_values"], row2["pixel_values"])


def test_image_ids_string_encoding_roundtrip():
    """image_ids round-trips through the scalar-string encoding, incl. the empty (text) row.

    The column is stored as a delimiter-joined string (not a list) so empty/variable-length
    rows survive the TransferQueue serialization round-trip without dropping/shifting rows.
    """
    for ids in ([], ["k0"], ["k0", "k1"], ["uidX_0_sha1:deadbeef", "uidX_0_sha1:cafe"]):
        assert decode_image_ids(encode_image_ids(ids)) == ids
    # a text row encodes to the empty string (a scalar, not an empty list)
    assert encode_image_ids([]) == ""
    # decode is tolerant of the legacy list form and of None
    assert decode_image_ids(["k0", "k1"]) == ["k0", "k1"]
    assert decode_image_ids([]) == []
    assert decode_image_ids(None) == []


def test_namespace_dedups_within_but_isolates_across_sessions():
    """Within one namespace identical images share a key; across namespaces they don't.

    Models the within-rollout dedup design: a screenshot recurring across the turns of one
    session (same ``{uid}_{session_id}``) stores once, while sibling GRPO sessions get
    distinct keys so lifecycle GC can clear a session's images without touching another's.
    """
    a = _image((1, 2, 2), fill=0.0)
    turn0 = _image((1, 2, 2), fill=0.0)  # same screenshot, later turn of session A
    turn1 = _image((1, 2, 2), fill=0.0)  # same screenshot again, session B

    ids_a0, _ = split_multimodal_per_image(a, namespace="uidX_0")
    ids_a1, _ = split_multimodal_per_image(turn0, namespace="uidX_0")
    ids_b, _ = split_multimodal_per_image(turn1, namespace="uidX_1")

    assert ids_a0[0] == ids_a1[0], "same content within a session -> one shared key"
    assert ids_a0[0] != ids_b[0], "same content in a different session -> distinct key"
    assert ids_a0[0].startswith("uidX_0_sha1:")
    assert ids_b[0].startswith("uidX_1_sha1:")
