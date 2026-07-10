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

import base64
import contextlib
import functools
import inspect
import io
import json
import os
from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel

from verl.utils.executor_guard import guard_stop_iteration
from verl.utils.ray_utils import get_event_loop

_trace_enabled: ContextVar[bool] = ContextVar("_trace_enabled", default=True)
_trace_attributes: ContextVar[dict | None] = ContextVar("_trace_attributes", default=None)


class RolloutTraceConfig:
    """Configuration for rollout tracing with various backends.

    Singleton configuration class for managing rollout trace settings across different
            tracing backends like Weave, MLflow, and Trackio.

    Args:
        backend (Optional[str]): Tracing backend to use ('weave', 'mlflow', or None).
        client (Optional[object]): Client instance for the selected backend.
        token2text (bool): Whether to convert tokens to text in traces. Defaults to False.
        project_name (str): Name of the project for tracing.
        experiment_name (str): Name of the experiment for tracing.
        max_samples_per_step_per_worker (Optional[int]): Maximum number of unique samples to trace
            per worker per step. If None, all samples are traced. If set, each worker will randomly
            select up to this many unique samples to trace (including all their rollouts for GRPO).
            Total traces = max_samples_per_step_per_worker * num_workers * n_rollouts_per_sample.
    """

    _instance: Optional["RolloutTraceConfig"] = None
    backend: str | None = None
    client: object | None = None
    token2text: bool = False
    _initialized: bool = False
    project_name: str = None
    experiment_name: str = None
    max_samples_per_step_per_worker: int | None = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> "RolloutTraceConfig":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def init(
        cls,
        project_name: str,
        experiment_name: str,
        backend: str,
        token2text: bool = False,
        max_samples_per_step_per_worker: int | None = None,
    ):
        config = cls.get_instance()
        if config._initialized:
            return

        config.backend = backend
        config.token2text = token2text
        config.project_name = project_name
        config.experiment_name = experiment_name
        config.max_samples_per_step_per_worker = max_samples_per_step_per_worker

        if backend == "weave":
            import weave

            config.client = weave.init(project_name)
        elif backend == "mlflow":
            import mlflow

            mlflow.config.enable_async_logging()
            config.client = mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            mlflow.set_experiment(project_name)
        elif backend == "trackio":
            import trackio
            from trackio import context_vars

            if context_vars.current_run.get() is None:
                trackio.init(project=project_name, name=experiment_name, config={"framework": "verl"})
            config.client = trackio
        else:
            config.client = None

        config._initialized = True

    @classmethod
    def get_backend(cls) -> str | None:
        return cls.get_instance().backend

    @classmethod
    def get_client(cls) -> object | None:
        return cls.get_instance().client

    @classmethod
    def enable_token2text(cls) -> bool | None:
        return cls.get_instance().token2text

    @classmethod
    def reset(cls):
        cls._instance = None


@contextlib.contextmanager
def rollout_trace_attr(
    sample_index=None, step=None, rollout_n=None, name="rollout_trace", validate=False, trace: bool = True
):
    """A context manager to add attributes to a trace for the configured backend.

    Args:
        sample_index: Sample index for the trace.
        step: Training step number.
        rollout_n: Rollout number (for GRPO with multiple rollouts per sample).
        name: Name for the trace span (used by mlflow backend).
        validate: Whether this is a validation run.
        trace: If False, disables tracing for the duration of the context.
    """
    backend = RolloutTraceConfig.get_backend()

    should_skip = backend is not None and not trace

    if should_skip:
        token = _trace_enabled.set(False)
        try:
            yield
        finally:
            _trace_enabled.reset(token)
        return

    # Build attributes for the trace
    attributes = {}
    if backend:
        if sample_index is not None:
            attributes["sample_index"] = sample_index
        if step is not None:
            attributes["step"] = step
        if rollout_n is not None:
            attributes["rollout_n"] = rollout_n
        attributes["validate"] = validate
        attributes["experiment_name"] = RolloutTraceConfig.get_instance().experiment_name

    if not attributes or backend is None:
        yield
        return

    token = _trace_attributes.set(attributes)
    if backend == "weave":
        import weave

        try:
            with weave.attributes(attributes):
                yield
        finally:
            _trace_attributes.reset(token)
    elif backend == "mlflow":
        import mlflow

        try:
            with mlflow.start_span(name=name) as span:
                trace_id = span.trace_id
                for key, value in attributes.items():
                    mlflow.set_trace_tag(trace_id, str(key), str(value))
                yield
        finally:
            _trace_attributes.reset(token)
    else:
        try:
            yield
        finally:
            _trace_attributes.reset(token)


def _json_trace_content(value):
    if isinstance(value, BaseModel):
        value = value.model_dump()
    return json.dumps(value, default=str, ensure_ascii=False)


def _json_trace_metadata(value):
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): _json_trace_metadata(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_trace_metadata(v) for v in value]
    return str(value)


def _trackio_message_dict(message):
    if not isinstance(message, dict):
        return None
    role = message.get("role")
    if not isinstance(role, str):
        return None
    return dict(message)


def _trackio_output_dict(output):
    if isinstance(output, BaseModel):
        return output.model_dump()
    if isinstance(output, dict):
        return output
    if hasattr(output, "__dict__"):
        return dict(vars(output))
    return None


def _trackio_trace_key(op_name):
    return "rollout_trace/" + "".join(char if char.isalnum() or char in "._-" else "_" for char in op_name)


def _trackio_trace_step(attributes):
    step = attributes.get("step")
    if step is None:
        return None
    try:
        return int(step)
    except (TypeError, ValueError):
        return None


def _log_trackio_trace(op_name, inputs, output=None, exception=None):
    trackio = RolloutTraceConfig.get_client()
    attributes = _current_trace_attributes()
    metadata_inputs = {key: value for key, value in inputs.items() if key != "messages"}
    output_dict = _trackio_output_dict(output)
    metadata = {
        "op": op_name,
        "backend": "trackio",
        "experiment_name": RolloutTraceConfig.get_instance().experiment_name,
        "inputs": _json_trace_metadata(metadata_inputs),
        **{key: _json_trace_metadata(value) for key, value in attributes.items()},
    }
    if exception is not None:
        metadata["status"] = "error"
        metadata["exception_type"] = type(exception).__name__
    else:
        metadata["status"] = "success"
        metadata["output"] = _json_trace_metadata(output_dict if output_dict is not None else output)

    messages = []
    input_messages = inputs.get("messages") if isinstance(inputs, dict) else None
    if isinstance(input_messages, list):
        messages = [
            message for message in (_trackio_message_dict(message) for message in input_messages) if message is not None
        ]

    if not messages:
        messages = [
            {"role": "system", "content": f"verl rollout trace operation: {op_name}"},
            {"role": "user", "content": _json_trace_content({"inputs": inputs})},
        ]

    if exception is not None:
        messages.append(
            {
                "role": "assistant",
                "content": _json_trace_content(
                    {
                        "exception_type": type(exception).__name__,
                        "exception": str(exception),
                    }
                ),
            }
        )
    elif output_dict is not None and output_dict.get("response_text"):
        messages.append({"role": "assistant", "content": str(output_dict["response_text"])})
    elif output_dict is not None and output_dict.get("answer"):
        messages.append({"role": "assistant", "content": str(output_dict["answer"])})
    else:
        messages.append({"role": "assistant", "content": _json_trace_content({"output": output})})

    trackio.log(
        {_trackio_trace_key(op_name): trackio.Trace(messages=messages, metadata=metadata)},
        step=_trackio_trace_step(attributes),
    )


def _current_trace_attributes():
    backend = RolloutTraceConfig.get_backend()
    if backend == "weave":
        from weave.trace.context import call_context

        return {**call_context.call_attributes.get()}
    return {**(_trace_attributes.get() or {})}


# --- MLflow multimodal trace rendering (ported from verl-async fully_async_policy) -------------
# MLflow's trace UI renders images inline ONLY when inputs/outputs follow the OpenAI chat-messages
# schema with {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}} content parts
# (https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/multimodal/). Raw PIL objects
# get stored as repr strings. We walk the payload, emit one image_url per PIL image, and dump the
# rest as a JSON text part. Pydantic models / __dict__ objects are expanded so images nested inside
# them (e.g. AgentLoopOutput.multi_modal_data, incl. a *list* of outputs from the GUI loop) are found.


def _is_pil_image(obj) -> bool:
    try:
        from PIL import Image as _PILImage

        return isinstance(obj, _PILImage.Image)
    except ImportError:
        return False


def _pil_to_data_uri(img) -> str:
    # Downscale + JPEG-compress so a full-trace run (every screenshot base64'd) doesn't bloat the
    # MLflow trace store. 1280 long-side JPEG q60 is ~100-150KB vs a 1080p PNG's several MB. Tunable.
    max_side = int(os.getenv("VERL_TRACE_IMG_MAX_SIDE", "1280"))
    quality = int(os.getenv("VERL_TRACE_IMG_QUALITY", "60"))
    buf = io.BytesIO()
    try:
        im = img if img.mode in ("RGB", "L") else img.convert("RGB")
        longest = max(im.size)
        if longest > max_side:
            scale = max_side / longest
            im = im.resize((max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale))))
        im.save(buf, format="JPEG", quality=quality)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _as_walkable(obj):
    """Expand pydantic models / plain objects to dicts so nested images are reachable."""
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return dict(vars(obj))
    return obj


def _collect_images_as_content_parts(obj, _depth=0, _parts=None):
    """Collect every PIL image in ``obj`` (any depth) as an OpenAI ``image_url`` content part."""
    if _parts is None:
        _parts = []
    if _depth > 12:
        return _parts
    if _is_pil_image(obj):
        _parts.append({"type": "image_url", "image_url": {"url": _pil_to_data_uri(obj)}})
        return _parts
    obj = _as_walkable(obj)
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_images_as_content_parts(v, _depth + 1, _parts)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _collect_images_as_content_parts(v, _depth + 1, _parts)
    return _parts


_TOKEN_ID_FIELDS = {
    "input_ids",
    "output_ids",
    "prompt_ids",
    "response_ids",
    "token_ids",
}


def _token_ids_to_list(token_ids):
    if token_ids is None or isinstance(token_ids, str | bytes):
        return None
    if hasattr(token_ids, "detach"):
        token_ids = token_ids.detach().cpu()
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)
    if not isinstance(token_ids, list):
        return None
    if token_ids and isinstance(token_ids[0], list | tuple):
        return None
    try:
        return [int(x) for x in token_ids]
    except (TypeError, ValueError):
        return None


def _summarize_token_ids(value):
    ids = _token_ids_to_list(value)
    if ids is None:
        return value
    return {"num_token_ids": len(ids), "head": ids[:8], "tail": ids[-8:] if len(ids) > 8 else []}


def _extract_reward(payload):
    walk = _as_walkable(payload)
    if not isinstance(walk, dict):
        return None
    if walk.get("reward") is not None:
        return walk["reward"]
    if walk.get("reward_score") is not None:
        return walk["reward_score"]
    extra_fields = walk.get("extra_fields")
    if isinstance(extra_fields, dict):
        if extra_fields.get("reward") is not None:
            return extra_fields["reward"]
        reward_extra = extra_fields.get("reward_extra_info")
        if isinstance(reward_extra, dict) and reward_extra.get("reward") is not None:
            return reward_extra["reward"]
    return None


def _strip_images_and_summarize_tokens(obj, _depth=0):
    """Return ``obj`` with PIL images removed and large token-id arrays summarized."""
    if _depth > 12:
        return repr(obj)
    if _is_pil_image(obj):
        return None
    walk = _as_walkable(obj)
    if isinstance(walk, dict):
        out = {}
        for k, v in walk.items():
            if k in _TOKEN_ID_FIELDS:
                out[k] = _summarize_token_ids(v)
            else:
                out[k] = _strip_images_and_summarize_tokens(v, _depth + 1)
        return out
    if isinstance(walk, list | tuple):
        return [_strip_images_and_summarize_tokens(v, _depth + 1) for v in walk]
    return walk


def _append_text_fields(payload, lines: list[str], _depth=0):
    if _depth > 8:
        return
    walk = _as_walkable(payload)
    if isinstance(walk, dict):
        reward = _extract_reward(walk)
        if reward is not None:
            lines.append(f"reward: {reward}")
        for key in ("prompt_text", "response_text", "answer", "text"):
            value = walk.get(key)
            if isinstance(value, str) and value:
                lines.append(f"{key}:\n{value}")
        extra_fields = walk.get("extra_fields")
        if isinstance(extra_fields, dict):
            reward_extra = extra_fields.get("reward_extra_info")
            if isinstance(reward_extra, dict) and reward_extra:
                try:
                    lines.append("reward_extra_info:\n" + json.dumps(reward_extra, ensure_ascii=False, indent=2))
                except Exception:
                    lines.append(f"reward_extra_info:\n{reward_extra!r}")
        for key, value in walk.items():
            if key in _TOKEN_ID_FIELDS:
                continue
            if isinstance(_as_walkable(value), dict | list | tuple):
                _append_text_fields(value, lines, _depth + 1)
    elif isinstance(walk, list | tuple):
        for idx, item in enumerate(walk):
            before = len(lines)
            _append_text_fields(item, lines, _depth + 1)
            if len(lines) > before:
                lines.insert(before, f"[item {idx}]")


def _to_mlflow_chat_messages(payload, *, role: str) -> dict:
    """Wrap a payload into MLflow's OpenAI chat schema.

    Put decoded text/reward first so the MLflow Chat tab is readable. The full
    structured payload is still present below it, but bulky token-id arrays are
    summarized instead of overwhelming the trace with raw ids.
    """
    image_parts = _collect_images_as_content_parts(payload)
    display_lines: list[str] = []
    _append_text_fields(payload, display_lines)
    try:
        metadata = json.dumps(_strip_images_and_summarize_tokens(payload), default=repr, ensure_ascii=False, indent=2)
    except Exception:
        metadata = repr(payload)
    if display_lines:
        text = "\n\n".join(display_lines) + "\n\nmetadata:\n" + metadata
    else:
        text = metadata
    content_parts: list = [{"type": "text", "text": text}]
    content_parts.extend(image_parts)
    return {"messages": [{"role": role, "content": content_parts}]}


def rollout_trace_op(func):
    @functools.wraps(func)
    async def async_wrapper(self, *args, **kwargs):
        if not _trace_enabled.get():
            return await func(self, *args, **kwargs)

        backend = RolloutTraceConfig.get_backend()
        enable_token2text = RolloutTraceConfig.enable_token2text()
        if backend is None:
            return await func(self, *args, **kwargs)

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        async def add_token2text(self, result):
            """Decode prompt_ids/response_ids/token_ids to text. Handles a single AgentLoopOutput /
            TokenOutput, and a *list* of them (the GUI multi-trajectory loop returns a list)."""
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None or not hasattr(tokenizer, "decode"):
                return result
            if isinstance(result, list | tuple):
                decoded = [await add_token2text(self, r) for r in result]
                return list(decoded)
            if isinstance(result, BaseModel):
                _result = result.model_dump()
            elif hasattr(result, "__dict__"):
                _result = dict(vars(result))
            else:
                return result
            loop = get_event_loop()
            if hasattr(result, "prompt_ids"):
                prompt_ids = _token_ids_to_list(result.prompt_ids)
                if prompt_ids is None:
                    prompt_ids = result.prompt_ids
                _result["prompt_text"] = await loop.run_in_executor(
                    None, guard_stop_iteration(lambda: tokenizer.decode(prompt_ids))
                )
            if hasattr(result, "response_ids"):
                response_ids = _token_ids_to_list(result.response_ids)
                if response_ids is None:
                    response_ids = result.response_ids
                _result["response_text"] = await loop.run_in_executor(
                    None, guard_stop_iteration(lambda: tokenizer.decode(response_ids))
                )
            # TokenOutput (generate): token_ids holds the response tokens.
            if hasattr(result, "token_ids") and not hasattr(result, "prompt_ids"):
                token_ids = _token_ids_to_list(result.token_ids)
                if token_ids is None:
                    token_ids = result.token_ids
                _result["response_text"] = await loop.run_in_executor(
                    None, guard_stop_iteration(lambda: tokenizer.decode(token_ids))
                )
            reward = _extract_reward(_result)
            if reward is not None and "reward" not in _result:
                _result["reward"] = reward
            return _result

        async def add_token2text_to_inputs(self, inputs_dict):
            """Decode a prompt_ids field in the inputs dict so the trace shows readable text."""
            tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None or not hasattr(tokenizer, "decode"):
                return inputs_dict
            prompt_ids = inputs_dict.get("prompt_ids")
            prompt_ids = _token_ids_to_list(prompt_ids)
            if prompt_ids is not None and len(prompt_ids) > 0:
                loop = get_event_loop()
                prompt_text = await loop.run_in_executor(
                    None, guard_stop_iteration(lambda: tokenizer.decode(prompt_ids))
                )
                inputs_dict = {**inputs_dict, "prompt_text": prompt_text}
            return inputs_dict

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()

            cur_attributes = _current_trace_attributes()
            call = tracer.create_call(op=func.__qualname__, inputs=inputs, attributes=cur_attributes)
            try:
                result = await func(self, *args, **kwargs)

                if enable_token2text:
                    _result = await add_token2text(self, result)
                    tracer.finish_call(call, output=_result)
                else:
                    tracer.finish_call(call, output=result)

                return result

            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            with mlflow.start_span(name=func.__qualname__) as span:
                # Wrap into the OpenAI chat schema so MLflow renders screenshots inline (image_url
                # parts) and the decoded prompt/response as the text part.
                if enable_token2text:
                    inputs = await add_token2text_to_inputs(self, inputs)
                span.set_inputs(_to_mlflow_chat_messages(inputs, role="user"))
                result = await func(self, *args, **kwargs)
                _out = await add_token2text(self, result) if enable_token2text else result
                span.set_outputs(_to_mlflow_chat_messages(_out, role="assistant"))

            return result
        elif backend == "trackio":
            try:
                result = await func(self, *args, **kwargs)
                if enable_token2text:
                    _result = await add_token2text(self, result)
                    _log_trackio_trace(func.__qualname__, inputs, output=_result)
                else:
                    _log_trackio_trace(func.__qualname__, inputs, output=result)
                return result
            except Exception as e:
                _log_trackio_trace(func.__qualname__, inputs, exception=e)
                raise e

        else:
            return await func(self, *args, **kwargs)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not _trace_enabled.get():
            return func(self, *args, **kwargs)

        backend = RolloutTraceConfig.get_backend()
        if backend is None:
            return func(self, *args, **kwargs)

        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()
        inputs = dict(bound_args.arguments)
        del inputs["self"]

        if backend == "weave":
            tracer = RolloutTraceConfig.get_client()

            cur_attributes = _current_trace_attributes()
            call = tracer.create_call(op=func.__qualname__, inputs=inputs, attributes=cur_attributes)
            try:
                result = func(self, *args, **kwargs)
                tracer.finish_call(call, output=result)
                return result
            except Exception as e:
                tracer.finish_call(call, exception=e)
                raise e
        elif backend == "mlflow":
            import mlflow

            return mlflow.trace(func)(self, *args, **kwargs)
        elif backend == "trackio":
            try:
                result = func(self, *args, **kwargs)
                _log_trackio_trace(func.__qualname__, inputs, output=result)
                return result
            except Exception as e:
                _log_trackio_trace(func.__qualname__, inputs, exception=e)
                raise e
        else:
            return func(self, *args, **kwargs)

    return async_wrapper if inspect.iscoroutinefunction(func) else wrapper
