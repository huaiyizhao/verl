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

import os
from math import ceil

_PATCHED = False


def _is_disabled(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().lower() in {"", "0", "false", "none", "null", "off"}


def _get_resource_amount() -> float:
    raw_amount = os.getenv("VERL_TQ_STORAGE_RESOURCE_AMOUNT", "0.001")
    try:
        amount = float(raw_amount)
    except ValueError as exc:
        raise ValueError(f"VERL_TQ_STORAGE_RESOURCE_AMOUNT must be a float, got {raw_amount!r}") from exc
    if amount <= 0:
        raise ValueError(f"VERL_TQ_STORAGE_RESOURCE_AMOUNT must be > 0, got {amount}")
    return amount


def _get_alive_storage_nodes() -> list[tuple[str, str]]:
    import ray

    balance_resource = os.getenv("VERL_TQ_STORAGE_BALANCE_RESOURCE")
    if _is_disabled(balance_resource):
        balance_resource = None
    else:
        assert balance_resource is not None
        balance_resource = balance_resource.strip()

    nodes = []
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        resources = node.get("Resources", {})
        if resources.get("CPU", 0) <= 0:
            continue
        if balance_resource and resources.get(balance_resource, 0) <= 0:
            continue
        nodes.append((node["NodeID"], node.get("NodeManagerAddress", "<unknown>")))
    return sorted(nodes, key=lambda item: item[1])


def _assert_resource_exists(resource_name: str) -> None:
    import ray

    matched_nodes = []
    for node in ray.nodes():
        resources = node.get("Resources", {})
        amount = resources.get(resource_name, 0)
        if node.get("Alive") and amount > 0:
            matched_nodes.append((node.get("NodeManagerAddress", "<unknown>"), amount))

    if matched_nodes:
        print(f"[TQ_PLACEMENT] storage resource {resource_name!r} found on nodes: {matched_nodes}", flush=True)
        return

    alive_resources = {
        node.get("NodeManagerAddress", "<unknown>"): sorted(node.get("Resources", {}).keys())
        for node in ray.nodes()
        if node.get("Alive")
    }
    raise RuntimeError(
        f"VERL_TQ_STORAGE_NODE_RESOURCE={resource_name!r}, but no alive Ray node advertises that resource. "
        "Start the rollout node with e.g. `ray start ... --resources='{\"rollout_node\": 1}'`, "
        "or disable this pin with `tq_storage_node_resource=none`. "
        f"Alive node resources: {alive_resources}"
    )


def _patch_balanced_simple_storage() -> None:
    import ray
    from transfer_queue.storage.bootstrap.provider import StorageBootstrapProvider
    from transfer_queue.storage.simple_storage import SimpleStorageUnit
    from transfer_queue.utils.zmq_utils import process_zmq_server_info

    def initialize_balanced_simple_storage(conf):
        simple_storage_handles = {}
        num_data_storage_units = conf.backend.SimpleStorage.num_data_storage_units
        total_storage_size = conf.backend.SimpleStorage.total_storage_size
        storage_nodes = _get_alive_storage_nodes()
        if not storage_nodes:
            balance_resource = os.getenv("VERL_TQ_STORAGE_BALANCE_RESOURCE")
            raise RuntimeError(
                "No alive Ray node is available for TransferQueue SimpleStorage "
                f"(VERL_TQ_STORAGE_BALANCE_RESOURCE={balance_resource!r})."
            )

        plan = []
        for storage_unit_rank in range(num_data_storage_units):
            node_id, node_ip = storage_nodes[storage_unit_rank % len(storage_nodes)]
            plan.append(node_ip)
            storage_node = SimpleStorageUnit.options(  # type: ignore[attr-defined]
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                ),
                name=f"TransferQueueStorageUnit#{storage_unit_rank}",
            ).remote(
                storage_unit_size=ceil(total_storage_size / num_data_storage_units),
            )
            simple_storage_handles[f"TransferQueueStorageUnit#{storage_unit_rank}"] = storage_node

        print(
            "[TQ_PLACEMENT] SimpleStorage balanced placement: "
            f"actors={num_data_storage_units} nodes={[ip for _, ip in storage_nodes]} plan={plan}",
            flush=True,
        )

        storage_zmq_info = process_zmq_server_info(simple_storage_handles)
        backend_name = conf.backend.storage_backend
        conf.backend[backend_name].zmq_info = storage_zmq_info
        return simple_storage_handles

    StorageBootstrapProvider.register_provider("SimpleStorage")(initialize_balanced_simple_storage)


def _patch_node_resource_simple_storage() -> None:
    resource_name = os.getenv("VERL_TQ_STORAGE_NODE_RESOURCE")
    if _is_disabled(resource_name):
        print("[TQ_PLACEMENT] SimpleStorage node-resource pin disabled", flush=True)
        return

    assert resource_name is not None
    resource_name = resource_name.strip()
    resource_amount = _get_resource_amount()
    _assert_resource_exists(resource_name)

    from transfer_queue.storage.bootstrap import simple_storage_bootstrap

    def get_placement_group(num_ray_actors: int, num_cpus_per_actor: int = 1):
        import ray

        bundle = {"CPU": num_cpus_per_actor, resource_name: resource_amount}
        bundles = [bundle.copy() for _ in range(num_ray_actors)]
        placement_group = ray.util.placement_group(bundles, strategy="SPREAD")
        print(
            "[TQ_PLACEMENT] SimpleStorage node-resource placement group: "
            f"actors={num_ray_actors} bundle={bundle} strategy=SPREAD",
            flush=True,
        )
        ray.get(placement_group.ready())
        return placement_group

    simple_storage_bootstrap.get_placement_group = get_placement_group


def patch_transfer_queue_simple_storage_placement() -> None:
    """Configure TransferQueue SimpleStorage placement.

    TransferQueue 0.1.x creates SimpleStorage with a CPU-only SPREAD placement
    group. That is role-agnostic and can place most in-memory storage on the
    wrong nodes. VERL_TQ_STORAGE_PLACEMENT=balanced replaces the SimpleStorage
    bootstrap with deterministic round-robin placement across alive Ray nodes.
    VERL_TQ_STORAGE_PLACEMENT=node_resource keeps the older custom-resource pin.
    """

    global _PATCHED
    if _PATCHED:
        return

    placement = os.getenv("VERL_TQ_STORAGE_PLACEMENT", "none").strip().lower()
    if placement in {"", "0", "false", "none", "null", "off"}:
        print("[TQ_PLACEMENT] SimpleStorage placement patch disabled", flush=True)
        return

    if placement == "balanced":
        _patch_balanced_simple_storage()
    elif placement == "node_resource":
        _patch_node_resource_simple_storage()
    else:
        raise ValueError(f"VERL_TQ_STORAGE_PLACEMENT must be one of balanced, node_resource, none; got {placement!r}")

    print(f"[TQ_PLACEMENT] SimpleStorage placement patch enabled: mode={placement}", flush=True)
    _PATCHED = True
