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
"""CPU integration tests for the streaming rollout pipeline.

These wire the *real* ``StreamingFeeder`` and the *real* ``ReplayBuffer`` against a *real*
TransferQueue (SimpleStorage / ZMQ in-memory, no GPU) backed by a local Ray cluster. The only
simulated pieces are the parts that inherently need a GPU rollout server:

- the trainer-side ``_feed_one_batch`` glue (here: register a pending prompt + a running
  trajectory key in TQ, exactly mirroring trainer_base._feed_one_batch's TQ writes), and
- the agent-loop worker (here: a thread that flips prompts pending->finished after a delay).

This validates the streaming throttle end to end: that the feeder bounds in-flight prompts to
the budget against the real TQ counts read by ReplayBuffer.count_inflight, and that the full
produce -> finish -> sample -> clear cycle reaches a bounded steady state without deadlock.

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


def _make_replay_buffer(strategy="drop", threshold=8, parameter_sync_step=1):
    return ReplayBuffer(
        trainer_mode="fully_async",
        trainer_config=OmegaConf.create({"parameter_sync_step": parameter_sync_step}),
        max_off_policy_threshold=threshold,
        max_off_policy_strategy=strategy,
        sampler_kwargs=OmegaConf.create({}),
        poll_interval=0.05,
    )


def _register_prompt(uid, status, global_steps, partition_id="train"):
    """Mirror what trainer_base._feed_one_batch writes for the prompt status marker."""
    tq.kv_batch_put(
        keys=[uid],
        partition_id=partition_id,
        tags=[{"is_prompt": True, "status": status, "global_steps": global_steps}],
    )


def _register_trajectory(uid, global_steps, partition_id="train"):
    """Register one trajectory value key for a prompt (non-prompt tag), as a worker would."""
    traj_key = f"{uid}_0_0"
    tq.kv_batch_put(
        keys=[traj_key],
        partition_id=partition_id,
        tags=[{"global_steps": global_steps, "seq_len": 8, "status": "finished"}],
    )
    return traj_key


# ======================================================================================
# I1: ReplayBuffer.count_inflight reflects real TransferQueue state
# ======================================================================================
def test_count_inflight_reflects_real_tq():
    rb = _make_replay_buffer()
    _register_prompt("p_pending", "pending", 1)
    _register_prompt("p_running", "running", 1)
    _register_prompt("p_fin1", "finished", 1)
    _register_prompt("p_fin2", "finished", 1)
    _register_prompt("p_fail", "failure", 1)

    counts = rb.count_inflight("train")
    assert counts == {"pending": 1, "running": 1, "finished": 2, "failure": 1}, counts


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
        _register_prompt(uid, "pending", global_steps)  # stays pending => never consumed
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

    # nothing ever finishes/consumed, so the feeder must stop exactly at the budget
    assert len(fed) == budget, f"expected {budget} fed, got {len(fed)}"
    counts = rb.count_inflight("train")
    assert counts["pending"] == budget, counts


# ======================================================================================
# I3: full streaming steady state — feeder + simulated worker + consumer, bounded, no deadlock
# ======================================================================================
def test_streaming_steady_state_bounded_and_progresses():
    rb = _make_replay_buffer(strategy="drop", threshold=8, parameter_sync_step=1)
    budget = 6
    batch_size = 2

    state_lock = threading.Lock()
    pending_uids = []  # uids waiting for the "worker" to finish them
    fed_count = 0
    max_inflight_seen = 0
    param_version = [1]

    def feed_one_batch(global_steps):
        nonlocal fed_count
        uid = uuid.uuid4().hex[:8]
        _register_prompt(uid, "pending", global_steps)
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
        # simulate the agent-loop worker: turn pending prompts into finished trajectories
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
            _register_trajectory(uid, gs)
            _register_prompt(uid, "finished", gs)  # flip prompt status -> sampleable

    consumed = [0]
    stop_consumer = threading.Event()

    def consumer():
        # simulate the trainer step: sample finished prompts (which kv_clears them)
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

    # sampler watchdog: record peak in-flight while everything runs
    stop_mon = threading.Event()

    def monitor():
        nonlocal max_inflight_seen
        while not stop_mon.is_set():
            c = rb.count_inflight("train")
            total = c["pending"] + c["running"] + c["finished"] + c["failure"]
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
    assert consumed[0] > 0, "consumer should have sampled finished prompts"
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
        _register_prompt(uid, "pending", global_steps)
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
    # TQ reflects exactly what was fed (all left pending, nothing consumed)
    counts = rb.count_inflight("train")
    assert counts["pending"] == n_after_resume, counts


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
