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
"""CPU-only unit tests for the streaming rollout feeder used by the V1 ``fully_async`` trainer.

These exercise the highest-risk part of the feature (deadlock / busy-wait / silent error /
thread lifecycle) with fakes, so no GPU / ray / transfer_queue is required.

``verl/trainer/ppo/v1/streaming_feeder.py`` is loaded directly by file path so that importing
it does NOT trigger the ``verl.trainer.ppo.v1`` package ``__init__`` (which pulls ray / vllm).
The file is dependency-light (stdlib only), so this test also runs without pytest:

    /cbs/cua/.venv/bin/python tests/trainer/ppo/v1/test_streaming_feeder_on_cpu.py
"""

import importlib.util
import threading
import time
from pathlib import Path


def _load_streaming_feeder():
    """Load streaming_feeder.py by path, bypassing the heavy package __init__."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "verl" / "trainer" / "ppo" / "v1" / "streaming_feeder.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("streaming_feeder_under_test", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("could not locate verl/trainer/ppo/v1/streaming_feeder.py")


_sf = _load_streaming_feeder()
StreamingFeeder = _sf.StreamingFeeder
compute_max_inflight_prompts = _sf.compute_max_inflight_prompts


class _FakeState:
    """Thread-safe fake backing both count_inflight and feed_one_batch."""

    def __init__(self, budget, version=0, feed_sleep=0.0):
        self.budget = budget
        self.version = version
        self.feed_sleep = feed_sleep
        self.fed = 0
        self.fed_versions = []
        self.count_calls = 0
        self._lock = threading.Lock()

    def count_inflight(self):
        with self._lock:
            self.count_calls += 1
            inflight = self.fed
        # inflight reported entirely as "pending"; the others stay at 0
        return {"pending": inflight, "running": 0, "finished": 0, "failure": 0}

    def feed(self, global_steps):
        if self.feed_sleep:
            time.sleep(self.feed_sleep)
        with self._lock:
            self.fed += 1
            self.fed_versions.append(global_steps)

    def param_version(self):
        with self._lock:
            return self.version


def _make_feeder(state, budget=None, poll_interval=0.01):
    return StreamingFeeder(
        count_inflight=state.count_inflight,
        feed_one_batch=state.feed,
        param_version=state.param_version,
        budget=state.budget if budget is None else budget,
        poll_interval=poll_interval,
    )


# --------------------------------------------------------------------------------------
# 8. pure budget function
# --------------------------------------------------------------------------------------
def test_compute_max_inflight_prompts_values():
    # (1 + staleness) * parameter_sync_step * train_batch_size
    assert compute_max_inflight_prompts(0, 1, 32) == 32
    assert compute_max_inflight_prompts(1, 4, 8) == 64
    assert compute_max_inflight_prompts(0.5, 2, 16) == 48


def test_compute_max_inflight_prompts_floor_and_assert():
    # 0 staleness, sync step 1 → exactly train_batch_size (boundary, must not assert)
    assert compute_max_inflight_prompts(0, 1, 10) == 10
    # impossible budget < train_batch_size should assert; only reachable with bad sync step
    raised = False
    try:
        compute_max_inflight_prompts(0, 0, 10)  # 0 * 10 = 0 < 10
    except AssertionError:
        raised = True
    assert raised, "expected AssertionError when budget < train_batch_size"


# --------------------------------------------------------------------------------------
# 1. budget gating
# --------------------------------------------------------------------------------------
def test_feeds_until_budget_then_stops():
    state = _FakeState(budget=5)
    feeder = _make_feeder(state)
    feeder.start()
    time.sleep(0.2)  # plenty of poll cycles
    feeder.stop()
    # inflight grows one per feed; loop stops feeding once fed == budget (5 < 5 is False)
    assert state.fed == 5, f"expected exactly budget feeds, got {state.fed}"
    assert not feeder.error


# --------------------------------------------------------------------------------------
# 2. param version passthrough
# --------------------------------------------------------------------------------------
def test_param_version_passed_to_feed():
    state = _FakeState(budget=4, version=7)
    feeder = _make_feeder(state)
    feeder.start()
    time.sleep(0.2)
    feeder.stop()
    assert state.fed == 4
    assert state.fed_versions == [7, 7, 7, 7], state.fed_versions


def test_param_version_read_live():
    # budget large enough to keep feeding; flip the version mid-run and confirm it's observed
    state = _FakeState(budget=10_000, version=1, feed_sleep=0.005)
    feeder = _make_feeder(state, poll_interval=0.01)
    feeder.start()
    time.sleep(0.05)
    state.version = 99
    time.sleep(0.05)
    feeder.stop()
    assert 1 in state.fed_versions and 99 in state.fed_versions, state.fed_versions[:5]


# --------------------------------------------------------------------------------------
# 3. clean stop
# --------------------------------------------------------------------------------------
def test_stop_joins_quickly_when_budget_full():
    state = _FakeState(budget=0)  # always "full" → feeder only ever sleeps
    feeder = _make_feeder(state, poll_interval=0.05)
    feeder.start()
    time.sleep(0.02)
    t0 = time.time()
    feeder.stop(timeout=5)
    elapsed = time.time() - t0
    assert elapsed < 1.0, f"stop took too long: {elapsed:.3f}s"
    assert state.fed == 0


# --------------------------------------------------------------------------------------
# 4. exception sets error flag and stops (no hang)
# --------------------------------------------------------------------------------------
def test_exception_sets_error():
    def boom(_global_steps):
        raise ValueError("kaboom")

    feeder = StreamingFeeder(
        count_inflight=lambda: {"pending": 0, "running": 0, "finished": 0, "failure": 0},
        feed_one_batch=boom,
        param_version=lambda: 0,
        budget=10,
        poll_interval=0.01,
    )
    feeder.start()
    time.sleep(0.1)
    assert feeder.error is True
    feeder.stop(timeout=5)
    assert feeder._thread is None or not feeder._thread.is_alive()


# --------------------------------------------------------------------------------------
# 5. StopIteration (dataset exhausted) → clean exit, no error
# --------------------------------------------------------------------------------------
def test_stopiteration_clean_exit():
    def exhausted(_global_steps):
        raise StopIteration

    feeder = StreamingFeeder(
        count_inflight=lambda: {"pending": 0, "running": 0, "finished": 0, "failure": 0},
        feed_one_batch=exhausted,
        param_version=lambda: 0,
        budget=10,
        poll_interval=0.01,
    )
    feeder.start()
    time.sleep(0.1)
    assert feeder.error is False
    feeder.stop(timeout=5)
    assert feeder._thread is None or not feeder._thread.is_alive()


# --------------------------------------------------------------------------------------
# 6. avoids busy-wait when budget is full
# --------------------------------------------------------------------------------------
def test_no_busy_wait_when_full():
    state = _FakeState(budget=0)  # inflight (0) is never < budget (0) → always sleeps
    poll = 0.05
    feeder = _make_feeder(state, poll_interval=poll)
    feeder.start()
    window = 0.3
    time.sleep(window)
    feeder.stop()
    assert state.fed == 0
    # if it were busy-spinning, count_calls would be huge; with sleeping it is ~window/poll
    assert state.count_calls <= (window / poll) + 5, f"too many polls ({state.count_calls}); busy-waiting?"


# --------------------------------------------------------------------------------------
# 7. concurrent smoke: external thread mutates counts while feeder runs
# --------------------------------------------------------------------------------------
def test_concurrent_mutation_smoke():
    state = _FakeState(budget=50, feed_sleep=0.001)
    feeder = _make_feeder(state, poll_interval=0.01)

    stop_mutator = threading.Event()

    def mutator():
        # periodically "consume" prompts so the feeder keeps producing
        while not stop_mutator.is_set():
            with state._lock:
                if state.fed > 10:
                    state.fed -= 5
            time.sleep(0.002)

    m = threading.Thread(target=mutator, daemon=True)
    feeder.start()
    m.start()
    time.sleep(0.3)
    stop_mutator.set()
    m.join(timeout=2)
    feeder.stop(timeout=5)

    assert not feeder.error
    # bounded: never wildly exceeds budget despite concurrent mutation
    assert state.fed <= state.budget + 5, f"inflight overshoot: {state.fed}"


# --------------------------------------------------------------------------------------
# 9. pause / resume (feeder pause during weight sync)
# --------------------------------------------------------------------------------------
def test_pause_stops_feeding_then_resume():
    # budget huge so the feeder always *wants* to feed; only pausing should stop it
    state = _FakeState(budget=10_000, feed_sleep=0.002)
    feeder = _make_feeder(state, poll_interval=0.01)
    feeder.start()
    time.sleep(0.05)
    feeder.pause()
    assert feeder.paused
    time.sleep(0.03)  # let any in-flight feed finish and the loop observe the pause
    with state._lock:
        fed_at_pause = state.fed
    time.sleep(0.12)  # while paused, the count must not grow
    with state._lock:
        fed_during_pause = state.fed
    assert fed_during_pause == fed_at_pause, f"fed while paused: {fed_at_pause} -> {fed_during_pause}"

    feeder.resume()
    assert not feeder.paused
    time.sleep(0.12)
    with state._lock:
        fed_after_resume = state.fed
    feeder.stop()
    assert fed_after_resume > fed_during_pause, "feeder did not resume feeding after resume()"


def test_pause_then_stop_is_quick():
    state = _FakeState(budget=10_000, feed_sleep=0.001)
    feeder = _make_feeder(state, poll_interval=0.05)
    feeder.start()
    feeder.pause()
    time.sleep(0.02)
    t0 = time.time()
    feeder.stop(timeout=5)
    assert time.time() - t0 < 1.0, "stop() hung while paused"


def test_pause_resume_idempotent():
    state = _FakeState(budget=10_000, feed_sleep=0.001)
    feeder = _make_feeder(state, poll_interval=0.01)
    feeder.resume()  # before start, no-op
    feeder.start()
    feeder.pause()
    feeder.pause()  # idempotent
    assert feeder.paused
    feeder.resume()
    feeder.resume()  # idempotent
    assert not feeder.paused
    feeder.stop()


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    return failures


if __name__ == "__main__":
    import sys

    sys.exit(1 if _run_all() else 0)
