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
"""CPU integration tests for the streaming rollout pipeline (rollout-level dispatch).

These wire the *real* ``StreamingFeeder`` and the *real* ``ReplayBuffer`` against a *real*
TransferQueue (SimpleStorage / ZMQ in-memory, no GPU) backed by a local Ray cluster. The only
simulated pieces are the parts that inherently need a GPU rollout server:

- the trainer-side ``_feed_one_batch`` glue (here: register a prompt metadata key in TQ,
  mirroring trainer_base._feed_one_batch: ``{is_prompt, global_steps, n}``), and
- the agent-loop worker (here: a thread that writes per-session completion markers +
  trajectory data keys, mirroring AgentLoopWorkerTQ._execute_rollout).

Readiness is **session-counting**: a prompt becomes sampleable once all ``n`` of its sessions
have written a completion marker (``success`` or ``failure``). This validates the streaming
throttle end to end against the real TQ counts read by ReplayBuffer.count_inflight, and that the
full produce -> complete -> sample -> clear cycle reaches a bounded steady state without deadlock.

Run:  /cbs/cua/.venv/bin/python -m pytest \
          tests/trainer/ppo/v1/test_streaming_feeder_tq_integration_on_cpu.py -v
"""

import importlib.util
import threading
import time
import uuid
from pathlib import Path

import pytest
import ray
import transfer_queue as tq
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve()
while not (_ROOT / "verl" / "trainer" / "ppo" / "v1").exists():
    _ROOT = _ROOT.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_sf = _load("streaming_feeder_itest", "verl/trainer/ppo/v1/streaming_feeder.py")
_rb = _load("replay_buffer_itest", "verl/trainer/ppo/v1/replay_buffer.py")
StreamingFeeder = _sf.StreamingFeeder
ReplayBuffer = _rb.ReplayBuffer


def setup_module(module):
    ray.init(num_cpus=6, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    conf = OmegaConf.create(
        {
            "enable": True,
            "metrics": {"enabled": False, "port": 0},
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {"total_storage_size": 10000, "num_data_storage_units": 2},
            },
        }
    )
    tq.init(conf)


def teardown_module(module):
    try:
        tq.close()
    finally:
        ray.shutdown()


def _clear_all():
    data = tq.kv_list() or {}
    for partition_id, items in data.items():
        if items:
            tq.kv_clear(partition_id=partition_id, keys=list(items.keys()))


@pytest.fixture(autouse=True)
def _clean_tq():
    _clear_all()
    yield
    _clear_all()


def _make_replay_buffer(strategy="drop", threshold=8, parameter_sync_step=1, session_counting=True):
    # session_counting=True selects the rollout-level (session-counting) readiness; False selects
    # the legacy prompt-status readiness. The flag is read from trainer_config.rollout_level_dispatch.
    return ReplayBuffer(
        trainer_mode="fully_async",
        trainer_config=OmegaConf.create(
            {"parameter_sync_step": parameter_sync_step, "rollout_level_dispatch": session_counting}
        ),
        max_off_policy_threshold=threshold,
        max_off_policy_strategy=strategy,
        sampler_kwargs=OmegaConf.create({}),
        poll_interval=0.05,
    )


def _register_prompt(uid, global_steps, n=1, partition_id="train"):
    """Mirror trainer_base._feed_one_batch's prompt metadata write (no status; carries n)."""
    tq.kv_batch_put(
        keys=[uid],
        partition_id=partition_id,
        tags=[{"is_prompt": True, "global_steps": global_steps, "n": n}],
    )


def _register_session(uid, session_id=0, status="success", global_steps=1, partition_id="train"):
    """Mirror AgentLoopWorkerTQ completing one rollout: write the trajectory data key (success
    only) then the per-session completion marker."""
    if status == "success":
        tq.kv_batch_put(
            keys=[f"{uid}_{session_id}_0"],
            partition_id=partition_id,
            tags=[{"global_steps": global_steps, "seq_len": 8, "status": "success"}],
        )
    tq.kv_batch_put(
        keys=[f"{uid}_sess{session_id}"],
        partition_id=partition_id,
        tags=[{"is_session": True, "session_id": session_id, "status": status}],
    )


# ======================================================================================
# I1: ReplayBuffer.count_inflight reflects real TransferQueue state (incomplete/complete)
# ======================================================================================
def test_count_inflight_reflects_real_tq():
    rb = _make_replay_buffer()
    for uid in ["p0", "p1", "p2", "p3", "p4"]:
        _register_prompt(uid, 1, n=1)
    # complete two of them by writing their single session marker
    _register_session("p2", 0)
    _register_session("p3", 0)

    counts = rb.count_inflight("train")
    assert counts == {"incomplete": 3, "complete": 2}, counts


# ======================================================================================
# I1b: a prompt is not ready until ALL n sessions complete (no head-of-line on partial groups)
# ======================================================================================
def test_prompt_not_ready_until_all_sessions_complete():
    rb = _make_replay_buffer(strategy="none")
    _register_prompt("g", 1, n=3)
    _register_session("g", 0)
    _register_session("g", 1)
    assert rb.count_inflight("train") == {"incomplete": 1, "complete": 0}
    _register_session("g", 2)
    assert rb.count_inflight("train") == {"incomplete": 0, "complete": 1}


# ======================================================================================
# I1c: a failed session still completes the group (no permanent stall), data is partial
# ======================================================================================
def test_failed_session_counts_toward_completion():
    rb = _make_replay_buffer(strategy="none")
    _register_prompt("g", 1, n=2)
    _register_session("g", 0, status="success")
    _register_session("g", 1, status="failure")  # no data key, but marker completes the group
    assert rb.count_inflight("train")["complete"] == 1

    batch, _ = rb.sample(global_steps=1, partition_id="train", batch_size=1)
    # only the successful session contributes a trajectory data key
    assert {k.split("_")[0] for k in batch.keys} == {"g"}, batch.keys
    assert len(batch.keys) == 1, batch.keys


# ======================================================================================
# I1d: sample() clears the prompt metadata key + its session markers (trainer clears data keys)
# ======================================================================================
def test_sample_clears_prompt_and_session_markers():
    rb = _make_replay_buffer(strategy="none")
    _register_prompt("g", 1, n=1)
    _register_session("g", 0)

    batch, _ = rb.sample(global_steps=1, partition_id="train", batch_size=1)
    tq.kv_clear(keys=batch.keys, partition_id="train")  # trainer clears the data keys post-consume

    leftover = list(((tq.kv_list() or {}).get("train") or {}).keys())
    assert leftover == [], leftover  # prompt key + marker cleared by sample, data by trainer


# ======================================================================================
# I1e: pure session-counting helper (no TQ involved)
# ======================================================================================
def test_compute_complete_uids_pure():
    f = _rb.compute_complete_uids
    assert f({"a": 2, "b": 1}, {"a": {0, 1}, "b": set()}) == {"a"}
    assert f({"a": 3}, {"a": {0, 1}}) == set()
    assert f({"a": 1, "b": 1}, {"a": {0}, "b": {0}}) == {"a", "b"}
    assert f({}, {}) == set()


# ======================================================================================
# I2: the real feeder throttles to the budget against the real TQ
# ======================================================================================
def test_feeder_throttles_to_budget_against_real_tq():
    rb = _make_replay_buffer()
    budget = _sf.compute_max_inflight_prompts(staleness_threshold=0, parameter_sync_step=1, train_batch_size=5)
    assert budget == 5

    fed = []

    def feed_one_batch(global_steps):
        uid = f"uid_{len(fed)}"
        _register_prompt(uid, global_steps, n=1)  # never completed => never consumed
        fed.append(uid)

    feeder = StreamingFeeder(
        count_inflight=lambda: rb.count_inflight("train"),
        feed_one_batch=feed_one_batch,
        param_version=lambda: 1,
        budget=budget,
        poll_interval=0.02,
    )
    feeder.start()
    time.sleep(0.6)
    feeder.stop()

    # nothing ever completes/consumed, so the feeder must stop exactly at the budget
    assert len(fed) == budget, f"expected {budget} fed, got {len(fed)}"
    counts = rb.count_inflight("train")
    assert counts["incomplete"] == budget, counts


# ======================================================================================
# I3: full streaming steady state — feeder + simulated worker + consumer, bounded, no deadlock
# ======================================================================================
def test_streaming_steady_state_bounded_and_progresses():
    rb = _make_replay_buffer(strategy="drop", threshold=8, parameter_sync_step=1)
    budget = 6
    batch_size = 2

    state_lock = threading.Lock()
    pending_uids = []  # uids waiting for the "worker" to complete them
    fed_count = 0
    max_inflight_seen = 0
    param_version = [1]

    def feed_one_batch(global_steps):
        nonlocal fed_count
        uid = uuid.uuid4().hex[:8]
        _register_prompt(uid, global_steps, n=1)
        with state_lock:
            pending_uids.append((uid, global_steps))
            fed_count += 1

    feeder = StreamingFeeder(
        count_inflight=lambda: rb.count_inflight("train"),
        feed_one_batch=feed_one_batch,
        param_version=lambda: param_version[0],
        budget=budget,
        poll_interval=0.02,
    )

    stop_worker = threading.Event()

    def worker():
        # simulate the agent-loop worker: complete prompts by writing their session marker + data
        while not stop_worker.is_set():
            item = None
            with state_lock:
                if pending_uids:
                    item = pending_uids.pop(0)
            if item is None:
                time.sleep(0.01)
                continue
            uid, gs = item
            time.sleep(0.01)  # simulate generation latency
            _register_session(uid, 0, "success", gs)  # completes the (n=1) group -> sampleable

    consumed = [0]
    stop_consumer = threading.Event()

    def consumer():
        # simulate the trainer step: sample complete prompts (which kv_clears them)
        while not stop_consumer.is_set():
            try:
                # ReplayBuffer.sample returns (KVBatchMeta, off_policy_metrics), matching the
                # trainer's `batch, off_policy_metrics = self.replay_buffer.sample(...)`.
                batch, _metrics = rb.sample(global_steps=param_version[0], partition_id="train", batch_size=batch_size)
            except Exception:
                time.sleep(0.02)
                continue
            if not batch.keys:
                continue
            consumed[0] += len(batch.keys)
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)

    # monitor: record peak in-flight (sum of all count_inflight buckets) while everything runs
    stop_mon = threading.Event()

    def monitor():
        nonlocal max_inflight_seen
        while not stop_mon.is_set():
            total = sum(rb.count_inflight("train").values())
            with state_lock:
                max_inflight_seen = max(max_inflight_seen, total)
            time.sleep(0.01)

    threads = [
        threading.Thread(target=worker, daemon=True),
        threading.Thread(target=consumer, daemon=True),
        threading.Thread(target=monitor, daemon=True),
    ]
    feeder.start()
    for t in threads:
        t.start()

    # let it run; bump the param version a couple of times like periodic weight syncs
    time.sleep(0.4)
    param_version[0] = 2
    time.sleep(0.4)
    param_version[0] = 3
    time.sleep(0.4)

    stop_worker.set()
    stop_consumer.set()
    stop_mon.set()
    feeder.stop()
    for t in threads:
        t.join(timeout=2)

    assert not feeder.error
    # forward progress: prompts were produced and consumed through the full pipeline
    assert fed_count > budget, f"feeder should have produced many batches, got {fed_count}"
    assert consumed[0] > 0, "consumer should have sampled complete prompts"
    # bounded: the feeder never let in-flight blow far past the budget
    # (small slack for the batch in flight between count and feed)
    assert max_inflight_seen <= budget + batch_size + 2, f"in-flight overshoot: {max_inflight_seen}"


# ======================================================================================
# I4: pausing the feeder (weight-sync window) blocks dispatch against the real TQ
# ======================================================================================
def test_feeder_pause_blocks_dispatch_against_real_tq():
    rb = _make_replay_buffer()
    budget = 100000  # large so the feeder always wants to feed; only pause should stop it
    fed = []

    def feed_one_batch(global_steps):
        uid = f"u_{len(fed)}"
        _register_prompt(uid, global_steps, n=1)
        fed.append(uid)

    feeder = StreamingFeeder(
        count_inflight=lambda: rb.count_inflight("train"),
        feed_one_batch=feed_one_batch,
        param_version=lambda: 1,
        budget=budget,
        poll_interval=0.01,
    )
    feeder.start()
    time.sleep(0.15)
    feeder.pause()
    time.sleep(0.05)
    n_at_pause = len(fed)
    time.sleep(0.25)  # simulate the weight-sync window
    n_during_pause = len(fed)
    feeder.resume()
    time.sleep(0.15)
    feeder.stop()  # join first so len(fed) and the TQ count are read from a quiesced state
    n_after_resume = len(fed)

    assert n_during_pause == n_at_pause, f"dispatched during pause: {n_at_pause} -> {n_during_pause}"
    assert n_after_resume > n_during_pause, "feeder did not resume dispatching"
    # TQ reflects exactly what was fed (all left incomplete, nothing consumed)
    counts = rb.count_inflight("train")
    assert counts["incomplete"] == n_after_resume, counts


# ======================================================================================
# I5: "none" strategy applies no staleness gate (streaming default; TIS corrects off-policy)
# ======================================================================================
def test_strategy_none_no_staleness_gate():
    """`none`: sample the oldest batch_size ready prompts regardless of staleness — nothing is
    dropped. Streaming bounds staleness via the feeder budget; TIS corrects the off-policyness.

    Note: uids must not contain '_' because sample() matches trajectories via key.split('_')[0].
    """
    rb = _make_replay_buffer(strategy="none", threshold=2, parameter_sync_step=1)
    global_steps = 10

    # Very stale (gs=1 -> staleness 10, far over any threshold) + fresh (gs=10) prompts.
    for i in range(3):
        uid = f"old{i}"
        _register_prompt(uid, 1, n=1)
        _register_session(uid, 0, "success", 1)
    for i in range(3):
        uid = f"new{i}"
        _register_prompt(uid, 10, n=1)
        _register_session(uid, 0, "success", 10)

    batch, metrics = rb.sample(global_steps=global_steps, partition_id="train", batch_size=4)

    selected = {k.split("_")[0] for k in batch.keys}
    # Full non-empty batch, oldest-first (the stale gs=1 prompts are kept, not dropped), and the
    # buffer reports no drops.
    assert len(selected) == 4, selected
    assert {"old0", "old1", "old2"}.issubset(selected), selected
    assert metrics == {}, metrics


# ======================================================================================
# L1/L2: legacy prompt-status readiness (session_counting=False) — backward compatibility
# ======================================================================================
def _register_prompt_legacy(uid, status, global_steps, partition_id="train"):
    """Mirror the legacy trainer_base._feed_one_batch / _run_prompt prompt-status tag."""
    tq.kv_batch_put(
        keys=[uid],
        partition_id=partition_id,
        tags=[{"is_prompt": True, "status": status, "global_steps": global_steps}],
    )


def _register_trajectory_legacy(uid, global_steps, partition_id="train"):
    """Legacy data key (no session marker), as the legacy worker postprocess would write."""
    tq.kv_batch_put(
        keys=[f"{uid}_0_0"],
        partition_id=partition_id,
        tags=[{"global_steps": global_steps, "seq_len": 8, "status": "success"}],
    )


def test_legacy_count_inflight_status_buckets():
    rb = _make_replay_buffer(session_counting=False)
    _register_prompt_legacy("p_pending", "pending", 1)
    _register_prompt_legacy("p_running", "running", 1)
    _register_prompt_legacy("p_fin1", "finished", 1)
    _register_prompt_legacy("p_fin2", "finished", 1)
    _register_prompt_legacy("p_fail", "failure", 1)
    counts = rb.count_inflight("train")
    assert counts == {"pending": 1, "running": 1, "finished": 2, "failure": 1}, counts


def test_legacy_sample_selects_finished_oldest():
    rb = _make_replay_buffer(strategy="none", session_counting=False)
    for i in range(3):
        uid = f"old{i}"
        _register_trajectory_legacy(uid, 1)
        _register_prompt_legacy(uid, "finished", 1)
    for i in range(2):
        uid = f"new{i}"
        _register_trajectory_legacy(uid, 10)
        _register_prompt_legacy(uid, "finished", 10)
    batch, metrics = rb.sample(global_steps=10, partition_id="train", batch_size=3)
    selected = {k.split("_")[0] for k in batch.keys}
    assert selected == {"old0", "old1", "old2"}, selected  # oldest-first, finished prompts
    assert metrics == {}, metrics


def _run_all():
    setup_module(None)
    try:
        tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
        failures = 0
        for fn in tests:
            _clear_all()
            try:
                fn()
                print(f"PASS {fn.__name__}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                import traceback

                traceback.print_exc()
                print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            finally:
                _clear_all()
        print(f"\n{len(tests) - failures}/{len(tests)} passed")
        return failures
    finally:
        teardown_module(None)


if __name__ == "__main__":
    import sys

    sys.exit(1 if _run_all() else 0)
