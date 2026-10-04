from __future__ import annotations

from .base import InferenceBackend, BackendError  # noqa: F401
from .vllm_backend import VLLMBackend, build_vllm_backend  # noqa: F401
from .transformers_backend import TransformersBackend, build_transformers_backend  # noqa: F401
from .sglang_backend import SGLangBackend, build_sglang_backend  # noqa: F401


def build_backend(name: str, model_path: str, backend_cfg: dict | None = None, logger=None) -> InferenceBackend:
    backend_cfg = backend_cfg or {}
    if name == "vllm":
        return build_vllm_backend(model_path, backend_cfg, logger)
    if name == "transformers":
        return build_transformers_backend(model_path, backend_cfg, logger)
    if name == "sglang":
        return build_sglang_backend(model_path, backend_cfg, logger)
    raise BackendError(f"unknown backend '{name}' (expected: vllm | transformers | sglang)")
