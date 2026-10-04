from __future__ import annotations

from .base import InferenceBackend, BackendError


def build_sglang_backend(model_path: str, backend_cfg: dict, logger=None) -> "SGLangBackend":
    return SGLangBackend(model_path=model_path, cfg=backend_cfg or {}, logger=logger)


class SGLangBackend(InferenceBackend):
    """Status: blocked. sglang is not installed in the current environment.

    The interface matches DESIGN.md §6.1. Raising BackendError (not silently
    falling back to another engine) is intentional per the design contract.
    """

    name = "sglang"

    def __init__(self, model_path: str, cfg: dict, logger=None):
        try:
            import sglang  # noqa: F401

            self._sglang_available = True
        except Exception:
            self._sglang_available = False
        self.model_path = model_path
        self.cfg = cfg

    def _blocked(self):
        if not self._sglang_available:
            raise BackendError(
                "SGLang backend is blocked: sglang is not installed in this environment. "
                "Install sglang or use backend='vllm'. No silent fallback is performed."
            )

    def capabilities(self) -> dict:
        return {
            "backend": "sglang",
            "generation": False,
            "token_logprobs": False,
            "token_last_hidden": False,
            "status": "blocked",
            "note": "sglang not installed; interface reserved per DESIGN.md §6.1",
        }

    def sample(self, *a, **k):
        self._blocked()

    def score(self, *a, **k):
        self._blocked()

    def replay_last_hidden(self, *a, **k):
        self._blocked()
