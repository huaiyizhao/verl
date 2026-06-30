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
"""CPU unit tests for the synthetic-padding template construction.

Focus: a padding row is a text-only 2-token sequence with NO image placeholder, so it must
reference NO images — for both the inline ``multi_modal_inputs`` path and the dedup ``image_ids``
path. A regression here attaches a real image to a 0-image-token row, which crashes training with
"Image features and image tokens do not match".
"""

import torch

from verl.trainer.ppo.padding_utils import construct_minimal_padding_template

EOS = 151643


def _source_sample(**overrides):
    src = {
        "input_ids": torch.arange(100),
        "prompts": torch.arange(60),
        "responses": torch.arange(40),
        "position_ids": torch.zeros(3, 100, dtype=torch.long),
        "extra_fields": {"min_global_steps": 1},
    }
    src.update(overrides)
    return src


def test_padding_template_is_text_only_minimal():
    tmpl, tag = construct_minimal_padding_template(_source_sample(), {"global_steps": 5}, eos_token_id=EOS)
    assert tmpl["input_ids"].numel() == 2  # one prompt token + one response token
    assert torch.equal(tmpl["input_ids"], torch.full((2,), EOS, dtype=torch.int64))
    assert tag["is_padding"] is True
    assert tag["prompt_len"] == 1 and tag["response_len"] == 1


def test_padding_template_clears_dedup_image_ids():
    """Dedup path: image_ids must be reset to "" (no images), not inherited from the template."""
    src = _source_sample(image_ids="uidA_0_sha1:deadbeef\x1fuidA_0_sha1:cafe")
    tmpl, _ = construct_minimal_padding_template(src, {"global_steps": 5}, eos_token_id=EOS)
    # "" decodes to an empty image list -> a text-only padding row with no image placeholder.
    assert tmpl["image_ids"] == ""


def test_padding_template_clears_inline_multi_modal_inputs():
    """Non-dedup path: inline multi_modal_inputs must be emptied."""
    src = _source_sample(multi_modal_inputs={"pixel_values": torch.zeros(8160, 1536)})
    tmpl, _ = construct_minimal_padding_template(src, {"global_steps": 5}, eos_token_id=EOS)
    assert tmpl["multi_modal_inputs"] == {}
