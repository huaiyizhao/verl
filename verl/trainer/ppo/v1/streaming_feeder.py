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
"""Autonomous streaming feeder for the V1 PPO trainer's ``fully_async`` mode.

This module is intentionally dependency-light (standard library only) so that the
concurrency / throttling logic can be unit tested on a CPU-only machine without
importing the heavy trainer stack (ray / transfer_queue / vllm). All interaction
with the trainer happens through injected callables.
"""

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


def compute_max_inflight_prompts(staleness_threshold: float, parameter_sync_step: int, train_batch_size: int) -> int:
    """Compute the in-flight prompt budget for the streaming feeder.

    The budget caps how many un-consumed prompts (pending + running + finished + failure)
    may exist in TransferQueue at once, bounding how far the rollouter runs ahead of
    training. Mirrors the ``fully_async_policy`` reference formula with ``require_batches=1``.

    Args:
        staleness_threshold: Allowed off-policy staleness (in parameter-sync units).
        parameter_sync_step: Number of training steps between weight syncs.
        train_batch_size: Prompts consumed per training step.

    Returns:
        The maximum number of un-consumed prompts allowed in flight.
    """
    budget = int((1 + staleness_threshold) * parameter_sync_step * train_batch_size)
    assert budget >= train_batch_size, (
        f"in-flight budget ({budget}) must be >= train_batch_size ({train_batch_size}); "
        f"check staleness_threshold/parameter_sync_step"
    )
    return budget


class StreamingFeeder:
    """Background producer that continuously feeds prompts under an in-flight budget.

    A daemon thread runs :meth:`_loop`, which repeatedly checks the current in-flight
    prompt counts and, while below ``budget``, calls ``feed_one_batch`` to stream another
    batch of prompts into generation. When the budget is full it sleeps on an interruptible
    event (``poll_interval``) instead of busy-waiting. The loop terminates on ``stop()``,
    on dataset exhaustion (``StopIteration`` from ``feed_one_batch``), or on any other
    exception (which sets :attr:`error` so the trainer can fail fast instead of hanging).

    Args:
        count_inflight: Returns a dict with keys ``pending``/``running``/``finished``/
            ``failure`` giving the current TransferQueue prompt counts.
        feed_one_batch: Called as ``feed_one_batch(global_steps)`` to stream one batch of
            prompts, tagged with the given parameter version.
        param_version: Returns the current parameter version used to tag fed prompts.
        budget: Maximum number of un-consumed prompts allowed in flight.
        poll_interval: Seconds to sleep when the budget is full (avoids busy-waiting).
    """

    def __init__(
        self,
        count_inflight: Callable[[], dict[str, int]],
        feed_one_batch: Callable[[int], None],
        param_version: Callable[[], int],
        budget: int,
        poll_interval: float,
    ):
        self._count_inflight = count_inflight
        self._feed_one_batch = feed_one_batch
        self._param_version = param_version
        self._budget = budget
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._paused = threading.Event()  # when set, the loop stops dispatching new prompts
        self.error = False  # set True if the feeder thread dies unexpectedly
        self._thread: threading.Thread | None = None

    def _loop(self):
        while not self._stop.is_set():
            if self._paused.is_set():
                # paused (e.g. during a weight sync): do not dispatch; generation already in
                # flight keeps running and is aborted+continued by the checkpoint engine.
                self._stop.wait(self._poll_interval)
                continue
            try:
                counts = self._count_inflight()
                inflight = counts["pending"] + counts["running"] + counts["finished"] + counts["failure"]
                if inflight < self._budget:
                    self._feed_one_batch(self._param_version())
                else:
                    # interruptible sleep: avoids a tight busy loop while the budget is full
                    self._stop.wait(self._poll_interval)
            except StopIteration:
                logger.info("Streaming feeder: dataset exhausted, stopping feeder")
                break
            except Exception:
                logger.exception("Streaming feeder thread crashed")
                self.error = True
                break

    def start(self):
        """Start the background feeder thread."""
        self._stop.clear()
        self.error = False
        self._thread = threading.Thread(target=self._loop, name="streaming-rollout-feeder", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0):
        """Signal the feeder to stop and join the thread."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def pause(self):
        """Pause dispatching new prompts (e.g. during a weight sync).

        Generation already in flight is unaffected (it is aborted+continued by the checkpoint
        engine / FullyAsyncLLMServerClient). Idempotent.
        """
        self._paused.set()

    def resume(self):
        """Resume dispatching after :meth:`pause`. Idempotent."""
        self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()
