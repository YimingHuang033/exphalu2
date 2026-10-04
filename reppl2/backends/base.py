from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BackendError(RuntimeError):
    pass


class InferenceBackend(ABC):
    name: str = "base"

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def sample(self, prompt_token_ids: list[int], sampling_config: dict, n: int, seed: int | None = None) -> dict:
        """Returns {'token_ids': [[int]], 'texts': [str], 'logprobs': [[float]], 'finish_reasons': [str]}"""

    @abstractmethod
    def score(self, context_token_ids: list[int], output_token_ids: list[int]) -> list[float]:
        """Per-token conditional logprobs of output tokens given the context."""

    @abstractmethod
    def replay_last_hidden(self, context_token_ids: list[int], output_token_ids: list[int]) -> dict:
        """Returns {'hidden': (L_out, H) float32 last-hidden states for output token positions, 'dtype': str}"""

    def close(self):
        pass

    def release_for_replay(self):
        """Hook: free generation-phase resources before state-replay phase (no-op by default)."""
        pass

    @staticmethod
    def validate_ids(context_token_ids, output_token_ids):
        if not context_token_ids:
            raise BackendError("context_token_ids is empty")
        if not output_token_ids:
            raise BackendError("output_token_ids is empty (K<2/empty-output guard)")
        for t in output_token_ids:
            if not isinstance(t, int) or t < 0:
                raise BackendError(f"invalid output token id: {t}")
