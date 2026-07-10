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

from collections.abc import Callable
from functools import wraps
from typing import TypeVar

T = TypeVar("T")


def guard_stop_iteration(fn: Callable[[], T]) -> Callable[[], T]:
    """Make executor callables safe for asyncio Futures.

    A callable passed to ``run_in_executor`` must not raise ``StopIteration``
    directly. Python cannot store that exception on an asyncio Future, which can
    leave the awaiting coroutine pending forever. Converting it lets normal
    exception handling run and, in rollout workers, write failure markers.
    """

    @wraps(fn)
    def wrapper() -> T:
        try:
            return fn()
        except StopIteration as exc:
            raise RuntimeError("executor callable raised StopIteration") from exc

    return wrapper
