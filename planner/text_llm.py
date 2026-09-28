"""
The single text LLM of the intelligent stack (Session 29).

One local text model owns every symbolic NLP job on the online path:

  * ``:goal`` authoring (``goal_generator.backends.local_llm``)
  * domain template selection + ``complete`` | ``incomplete`` (``domain_llm``)
  * closed-catalog domain enrichment (``domain_llm``)
  * R1 affordance enrichment (``planner.r1``, opt-in)

They differ only by prompt. Design §3.3.1 forbids a second text model (e.g. a
small goal-only network *plus* a larger enricher): the budget that matters is
joint VRAM with the perceptual VLM, so the same weights are shared through
:func:`get_shared_text_client`. When ``VLMRP_TEXT_LLM_BASE_URL`` is set,
:class:`OpenAICompatibleClient` is used instead of local transformers (L-1).

Model id resolution order: explicit argument → ``VLMRP_TEXT_LLM_MODEL`` →
``VLMRP_GOAL_LLM_MODEL`` (legacy, goal-only sessions) → ``DEFAULT_TEXT_LLM_MODEL_ID``.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Mapping, Protocol

from planner.call_timings import llm_span

ENV_TEXT_LLM_MODEL = "VLMRP_TEXT_LLM_MODEL"
# Session 9b–9c name; still honoured so existing goal recipes keep working.
ENV_GOAL_LLM_MODEL = "VLMRP_GOAL_LLM_MODEL"
ENV_TEXT_LLM_BASE_URL = "VLMRP_TEXT_LLM_BASE_URL"
ENV_TEXT_LLM_API_KEY = "VLMRP_TEXT_LLM_API_KEY"
ENV_TEXT_LLM_TIMEOUT_S = "VLMRP_TEXT_LLM_TIMEOUT_S"
ENV_TEXT_LLM_TEMPERATURE = "VLMRP_TEXT_LLM_TEMPERATURE"
# 0/false → send enable_thinking=false (Tesla llama.cpp / R1). Unset → omit field.
ENV_TEXT_LLM_ENABLE_THINKING = "VLMRP_TEXT_LLM_ENABLE_THINKING"

# Default stays the Session 9c recommendation so no run silently changes model.
# Labs with VRAM headroom raise it once (one env var) for goal + select + enrich.
DEFAULT_TEXT_LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PERCEPTUAL_VLM_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
# llama.cpp 32B/72B at 150–200 ms/token can exceed 120 s on llm_pddl (1536 tok).
DEFAULT_API_TIMEOUT_S = 600.0
DEFAULT_TEXT_LLM_TEMPERATURE = 0.0

GenerateFn = Callable[[str, str], str]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

__all__ = [
    "DEFAULT_API_TIMEOUT_S",
    "DEFAULT_PERCEPTUAL_VLM_MODEL_ID",
    "DEFAULT_TEXT_LLM_MODEL_ID",
    "DEFAULT_TEXT_LLM_TEMPERATURE",
    "ENV_GOAL_LLM_MODEL",
    "ENV_TEXT_LLM_API_KEY",
    "ENV_TEXT_LLM_BASE_URL",
    "ENV_TEXT_LLM_ENABLE_THINKING",
    "ENV_TEXT_LLM_MODEL",
    "ENV_TEXT_LLM_TEMPERATURE",
    "ENV_TEXT_LLM_TIMEOUT_S",
    "GenerateFn",
    "LocalTextClient",
    "OpenAICompatibleClient",
    "TransformersLocalClient",
    "apply_thinking_to_payload",
    "get_shared_text_client",
    "reset_shared_text_client",
    "resolve_api_timeout_s",
    "resolve_enable_thinking",
    "resolve_temperature",
    "resolve_text_llm_model_id",
    "strip_think_tags",
]


class LocalTextClient(Protocol):
    """Minimal local text-only completion interface (no images)."""

    def complete(self, system: str, user: str) -> str:
        ...


def resolve_text_llm_model_id(
    model_id: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the one text-LLM model id for goal / select / enrich."""
    if model_id and str(model_id).strip():
        return str(model_id).strip()
    source = env if env is not None else os.environ
    for key in (ENV_TEXT_LLM_MODEL, ENV_GOAL_LLM_MODEL):
        raw = str(source.get(key, "") or "").strip()
        if raw:
            return raw
    return DEFAULT_TEXT_LLM_MODEL_ID


def resolve_api_timeout_s(
    timeout_s: float | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> float:
    """HTTP timeout for ``OpenAICompatibleClient`` (Tesla 32B+ needs minutes)."""
    if timeout_s is not None:
        return float(timeout_s)
    source = env if env is not None else os.environ
    raw = str(source.get(ENV_TEXT_LLM_TIMEOUT_S, "") or "").strip()
    if raw:
        return float(raw)
    return DEFAULT_API_TIMEOUT_S


def resolve_temperature(
    temperature: float | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> float:
    """Sampling temperature. Campaign / eval default is 0.0 (greedy)."""
    if temperature is not None:
        return float(temperature)
    source = env if env is not None else os.environ
    raw = str(source.get(ENV_TEXT_LLM_TEMPERATURE, "") or "").strip()
    if raw:
        return float(raw)
    return DEFAULT_TEXT_LLM_TEMPERATURE


def resolve_enable_thinking(
    *,
    env: Mapping[str, str] | None = None,
) -> bool | None:
    """
    Explicit thinking switch. ``None`` = omit extra JSON fields (OpenAI
    cloud rejects unknown keys). ``0``/``1`` is sent both as llama.cpp
    ``enable_thinking`` and vLLM ``chat_template_kwargs.enable_thinking``
    (Qwen3 defaults thinking ON and ignores the top-level field).
    """
    source = env if env is not None else os.environ
    raw = str(source.get(ENV_TEXT_LLM_ENABLE_THINKING, "") or "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return None


def apply_thinking_to_payload(
    payload: dict[str, Any], enable_thinking: bool | None
) -> None:
    """Mutate a chat-completions body with llama.cpp + vLLM thinking flags."""
    if enable_thinking is None:
        return
    flag = bool(enable_thinking)
    payload["enable_thinking"] = flag
    payload["chat_template_kwargs"] = {"enable_thinking": flag}


def strip_think_tags(text: str) -> str:
    """Drop ``<think>…</think>`` blocks some distilled R1 models emit."""
    return _THINK_RE.sub("", text).strip()


def _message_text(body: Mapping[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"text LLM API returned no choices: {body!r}"[:400])
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(part))
        content = "".join(parts)
    text = str(content or "").strip()
    if not text:
        fallback = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or ""
        )
        text = str(fallback).strip()
    return strip_think_tags(text)


class TransformersLocalClient:
    """
    HuggingFace transformers chat client (lazy load).

    Requires ``transformers`` + torch and downloaded weights. Used for live
    goal generation / domain selection / enrichment and optional
    ``@pytest.mark.llm`` / ``@pytest.mark.gpu`` smoke — CI mocks ``generate_fn``
    instead.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_TEXT_LLM_MODEL_ID,
        *,
        max_new_tokens: int = 256,
        device_map: str | None = "auto",
        dtype: str | None = "auto",
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.dtype = dtype
        self.device = device  # "auto" | "cuda" | "cpu"
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[text-llm] Loading {self.model_id} …", flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        want_cpu = self.device == "cpu" or (
            self.device == "auto" and not torch.cuda.is_available()
        )
        use_cuda = (not want_cpu) and torch.cuda.is_available()

        load_kwargs: dict = {}
        if self.dtype == "auto":
            load_kwargs["torch_dtype"] = torch.float16 if use_cuda else torch.float32
        elif self.dtype in {"float16", "fp16"}:
            load_kwargs["torch_dtype"] = torch.float16
        elif self.dtype in {"bfloat16", "bf16"}:
            load_kwargs["torch_dtype"] = torch.bfloat16
        elif self.dtype in {"float32", "fp32"}:
            load_kwargs["torch_dtype"] = torch.float32

        if use_cuda and self.device_map is not None:
            load_kwargs["device_map"] = self.device_map
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, **load_kwargs
            )
            self._device = None  # device_map manages placement
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, **load_kwargs
            )
            self._device = torch.device("cuda" if use_cuda else "cpu")
            self._model.to(self._device)

        self._model.eval()
        where = "cuda" if use_cuda else "cpu"
        print(f"[text-llm] Ready on {where}.", flush=True)

    def close(self) -> None:
        """Drop weights and free CUDA cache (battery arm boundaries)."""
        model = self._model
        self._model = None
        self._tokenizer = None
        self._device = None
        if model is None:
            return
        import gc

        del model
        gc.collect()
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        print("[text-llm] Released weights.", flush=True)

    def complete(
        self, system: str, user: str, *, max_new_tokens: int | None = None
    ) -> str:
        with llm_span():
            return self._complete_impl(system, user, max_new_tokens=max_new_tokens)

    def _complete_impl(
        self, system: str, user: str, *, max_new_tokens: int | None = None
    ) -> str:
        import torch

        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None
        budget = int(
            max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"System: {system}\n\nUser: {user}\n\nAssistant:"

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self._device is not None:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        else:
            # device_map="auto" — move inputs to first parameter device
            try:
                first = next(self._model.parameters()).device
                inputs = {k: v.to(first) for k, v in inputs.items()}
            except StopIteration:
                pass

        pad_id = self._tokenizer.eos_token_id
        # Greedy decode (temperature 0). do_sample=False ignores temperature.
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=budget,
                do_sample=False,
                pad_token_id=pad_id,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


class OpenAICompatibleClient:
    """
    Chat Completions client (OpenAI, vLLM, llama.cpp, Azure-compatible).

    Activated when ``VLMRP_TEXT_LLM_BASE_URL`` is set. Same ``complete(system,
    user)`` surface as ``TransformersLocalClient`` so select / enrich / R1 /
    goal share one adapter. Stdlib only (no extra pip dependency).
    """

    def __init__(
        self,
        model_id: str,
        *,
        base_url: str,
        api_key: str | None = None,
        max_new_tokens: int = 256,
        timeout_s: float = DEFAULT_API_TIMEOUT_S,
        enable_thinking: bool | None = None,
        temperature: float = DEFAULT_TEXT_LLM_TEMPERATURE,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.max_new_tokens = max_new_tokens
        self.timeout_s = float(timeout_s)
        self.enable_thinking = enable_thinking
        self.temperature = float(temperature)

    def close(self) -> None:
        """API client holds no local weights."""

    def complete(
        self, system: str, user: str, *, max_new_tokens: int | None = None
    ) -> str:
        with llm_span():
            return self._complete_impl(system, user, max_new_tokens=max_new_tokens)

    def _complete_impl(
        self, system: str, user: str, *, max_new_tokens: int | None = None
    ) -> str:
        import json
        import urllib.error
        import urllib.request

        url = self.base_url
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        budget = int(
            max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        )
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": budget,
            "temperature": float(self.temperature),
        }
        if self.enable_thinking is not None:
            apply_thinking_to_payload(payload, self.enable_thinking)
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"text LLM API HTTP {exc.code} at {url}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"text LLM API connect/timeout at {url} "
                f"after {self.timeout_s:.0f}s: {exc}"
            ) from exc
        return _message_text(body)


# Keyed by (kind, model_id, base_url). Decode budget is per complete() call
# so select (256) and R1 enrich (768) share one weight load.
_SHARED_CLIENTS: dict[tuple[str, str, str], LocalTextClient] = {}


def get_shared_text_client(
    model_id: str | None = None,
    *,
    max_new_tokens: int = 256,
    env: Mapping[str, str] | None = None,
) -> LocalTextClient:
    """
    Return the process-wide client for the single text LLM (lazy weights).

    Sharing is by model id (and API base URL), not by ``max_new_tokens``.
    Select / enrich / R1 / goal used to pass 128–768 and each budget loaded
    a second 7B (CPU offload, minutes per call). Pass the decode cap to
    ``complete(..., max_new_tokens=)`` instead.

    If ``VLMRP_TEXT_LLM_BASE_URL`` is set, an OpenAI-compatible HTTP client is
    used instead of local transformers (L-1).
    """
    source = env if env is not None else os.environ
    resolved = resolve_text_llm_model_id(model_id, env=source)
    base_url = str(source.get(ENV_TEXT_LLM_BASE_URL, "") or "").strip()
    key = ("api" if base_url else "local", resolved, base_url)
    client = _SHARED_CLIENTS.get(key)
    if client is None:
        if base_url:
            client = OpenAICompatibleClient(
                resolved,
                base_url=base_url,
                api_key=str(source.get(ENV_TEXT_LLM_API_KEY, "") or "").strip()
                or None,
                max_new_tokens=int(max_new_tokens),
                timeout_s=resolve_api_timeout_s(env=source),
                enable_thinking=resolve_enable_thinking(env=source),
                temperature=resolve_temperature(env=source),
            )
        else:
            client = TransformersLocalClient(
                resolved, max_new_tokens=int(max_new_tokens)
            )
        _SHARED_CLIENTS[key] = client
    return client


def reset_shared_text_client() -> None:
    """Close cached clients and drop the process-wide cache (tests / arm swap)."""
    for client in list(_SHARED_CLIENTS.values()):
        close = getattr(client, "close", None)
        if callable(close):
            close()
    _SHARED_CLIENTS.clear()
