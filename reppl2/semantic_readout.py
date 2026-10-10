"""Fixed suffix observer using native last-token replay, not a judgment head."""
from __future__ import annotations

import time
import numpy as np

from .methods.reppl_c import unit_vector


class SemanticReadout:
    def __init__(self, backend, tokenizer, config):
        if backend.name not in ("vllm", "sglang"):
            raise ValueError("C/D require native vLLM or SGLang; no Transformers fallback")
        self.backend, self.tokenizer, self.config = backend, tokenizer, config
        self.cache = {}
        self.calls = self.tokens = self.cache_hits = 0
        self.seconds = 0.0

    def _read(self, text):
        ids = list(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(ids) < 2 or len(ids) > self.config["max_readout_tokens"]:
            raise ValueError("readout length invalid; no silent truncation")
        key = tuple(ids)
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        # Tokenize the COMPLETE string once; the observer is its last token.
        # Splitting token IDs rather than tokenizing prefix/suffix avoids BPE drift.
        t0 = time.monotonic()
        result = self.backend.replay_last_hidden(ids[:-1], ids[-1:])
        h = np.asarray(result["hidden"], dtype=np.float64)
        if h.ndim != 2 or h.shape[0] != 1:
            raise ValueError("native replay did not return exactly the observer token")
        value = unit_vector(h[0], self.config["vector_floor"])
        self.calls += 1
        self.tokens += len(ids)
        self.seconds += time.monotonic() - t0
        self.cache[key] = value
        return value

    def observe(self, question, candidate):
        return self._read(self.config["readout_template"].format(question=question, candidate=candidate))

    def concept(self, candidate):
        return self._read(self.config["concept_template"].format(candidate=candidate))

    def cost(self):
        return {"replay_calls": self.calls, "replay_tokens": self.tokens,
                "replay_seconds": self.seconds, "cache_hits": self.cache_hits}
