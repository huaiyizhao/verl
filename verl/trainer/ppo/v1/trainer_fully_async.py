# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os
import threading

from omegaconf import DictConfig

from verl.trainer.ppo.v1.streaming_feeder import StreamingFeeder, compute_max_inflight_prompts
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@register_trainer("fully_async")
class PPOTrainerFullyAsync(PPOTrainerSeparateAsync):
    """Streaming asynchronous PPO trainer.

    Builds on ``separate_async`` (standalone rollout + checkpoint-engine weight sync). The
    only behavioral difference is *who drives prompt feeding*: instead of ``step()`` feeding
    exactly one batch per training step, an autonomous background feeder thread continuously
    streams prompts into TransferQueue (bounded by a staleness / in-flight budget), while
    ``step()`` only samples + trains. This decouples production rate from consumption rate so
    that rollout overlaps training.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)  # inherits separate_async asserts + bypass_mode
        # base step() skips _add_batch_to_generate; the feeder owns generation
        self._feeder_owns_generation = True
        # parameter version used to tag fed prompts; tracks self.global_steps (see on_step_end)
        self._param_version = 0
        self._param_version_lock = threading.Lock()
        self._feeder: StreamingFeeder | None = None

    def _max_inflight_prompts(self) -> int:
        cfg = self.config.trainer.v1.fully_async
        return compute_max_inflight_prompts(
            cfg.staleness_threshold, cfg.parameter_sync_step, self.config.data.train_batch_size
        )

    def _current_param_version(self) -> int:
        with self._param_version_lock:
            return self._param_version

    def on_train_begin(self):
        # separate_async: prime the pipeline with num_warmup_batches (runs before the thread)
        super().on_train_begin()
        with self._param_version_lock:
            self._param_version = self.global_steps
        budget = self._max_inflight_prompts()
        self._feeder = StreamingFeeder(
            count_inflight=lambda: self.replay_buffer.count_inflight("train"),
            feed_one_batch=self._feed_one_batch,
            param_version=self._current_param_version,
            budget=budget,
            poll_interval=self.config.trainer.v1.fully_async.feeder_poll_interval,
        )
        self._feeder.start()
        logger.info("Streaming rollout feeder started; budget=%d prompts", budget)

    def step(self, metrics, timing_raw):
        # fail fast instead of hanging in replay_buffer.sample if the feeder died
        if self._feeder is not None and self._feeder.error:
            raise RuntimeError("Streaming feeder thread died; aborting training")
        return super().step(metrics, timing_raw)

    def on_step_end(self):
        # Pause the feeder around the periodic standalone weight sync so it does not dispatch
        # prompts into a server that is mid-sync. Generation already in flight is aborted and
        # continued by the checkpoint engine / FullyAsyncLLMServerClient (partial rollout),
        # independent of this pause.
        is_sync_step = self._feeder is not None and self.global_steps % self.parameter_sync_step == 0
        if is_sync_step:
            logger.info("Pausing streaming feeder for weight sync at step %d", self.global_steps)
            self._feeder.pause()
        try:
            super().on_step_end()  # separate_async: standalone update_weights on sync steps
        finally:
            if is_sync_step:
                self._feeder.resume()
                logger.info("Resumed streaming feeder after weight sync at step %d", self.global_steps)
        with self._param_version_lock:
            self._param_version = self.global_steps

    def on_train_end(self):
        if self._feeder is not None:
            self._feeder.stop()
        super().on_train_end()
