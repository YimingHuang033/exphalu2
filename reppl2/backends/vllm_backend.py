from __future__ import annotations

from .base import InferenceBackend, BackendError


def build_vllm_backend(model_path: str, backend_cfg: dict, logger=None) -> "VLLMBackend":
    return VLLMBackend(model_path=model_path, cfg=backend_cfg or {}, logger=logger)


class VLLMBackend(InferenceBackend):
    name = "vllm"

    def __init__(self, model_path: str, cfg: dict, logger=None):
        try:
            import vllm
        except Exception as e:
            raise BackendError(f"vllm is not importable: {e}") from e
        self._vllm = vllm
        self.model_path = model_path
        self.cfg = cfg
        self.logger = logger
        self._gen_llm = None
        self._pool_llm = None
        self._gen_meta = {}

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(f"[vllm] {msg}")

    def _gen_engine(self):
        if self._gen_llm is None:
            from vllm import LLM

            kwargs = dict(
                model=self.model_path,
                runner="generate",
                gpu_memory_utilization=float(self.cfg.get("gpu_memory_utilization", 0.85)),
                max_model_len=int(self.cfg.get("max_model_len", 4096)),
                enforce_eager=bool(self.cfg.get("enforce_eager", True)),
                dtype=str(self.cfg.get("dtype", "bfloat16")),
                seed=int(self.cfg.get("seed", 42)),
            )
            self._log(f"loading generate runner: {kwargs}")
            self._gen_llm = LLM(**kwargs)
        return self._gen_llm

    def _pool_engine(self):
        if self._pool_llm is None:
            from vllm import LLM

            kwargs = dict(
                model=self.model_path,
                runner="pooling",
                convert="embed",
                gpu_memory_utilization=float(self.cfg.get("gpu_memory_utilization", 0.85)),
                max_model_len=int(self.cfg.get("max_model_len", 4096)),
                enforce_eager=bool(self.cfg.get("enforce_eager", True)),
                dtype=str(self.cfg.get("dtype", "bfloat16")),
                seed=int(self.cfg.get("seed", 42)),
            )
            self._log(f"loading pooling runner: {kwargs}")
            self._pool_llm = LLM(**kwargs)
        return self._pool_llm

    def release_generate_engine(self):
        if self._gen_llm is not None:
            self._log("releasing generate runner (engine core is a separate process; waiting for VRAM)")
            del self._gen_llm
            self._gen_llm = None
            self._wait_gpu_free()

    def release_pool_engine(self):
        if self._pool_llm is not None:
            del self._pool_llm
            self._pool_llm = None
            self._wait_gpu_free()

    def _wait_gpu_free(self, timeout_s: float = 180.0):
        import gc
        import time

        import torch

        gc.collect()
        torch.cuda.empty_cache()
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            free_b, total_b = torch.cuda.mem_get_info()
            free_gib = free_b / (1024 ** 3)
            if free_gib >= total_b / (1024 ** 3) * 0.5:
                return
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2.0)
        free_b, total_b = torch.cuda.mem_get_info()
        raise BackendError(
            f"GPU memory did not free after engine release "
            f"({free_b/(1024**3):.1f}/{total_b/(1024**3):.1f} GiB free after {timeout_s}s); "
            "an engine core process is still resident"
        )

    def release_for_replay(self):
        self.release_generate_engine()

    def close(self):
        self.release_generate_engine()
        self.release_pool_engine()

    def capabilities(self) -> dict:
        return {
            "backend": "vllm",
            "vllm_version": getattr(self._vllm, "__version__", "unknown"),
            "generation": True,
            "token_logprobs": True,
            "token_last_hidden": True,
            "note": "generate runner and pooling runner are loaded in separate phases; not simultaneously resident",
        }

    def sample(self, prompt_token_ids, sampling_config, n, seed=None):
        import time

        from vllm import SamplingParams

        llm = self._gen_engine()

        def _one_call(nn, sd):
            sp = SamplingParams(
                temperature=float(sampling_config.get("temperature", 1.0)),
                top_p=float(sampling_config.get("top_p", 1.0)),
                top_k=int(sampling_config.get("top_k", -1)),
                max_tokens=int(sampling_config.get("max_new_tokens", 64)),
                logprobs=1,
                n=nn,
                seed=sd,
                stop_token_ids=sampling_config.get("stop_token_ids") or None,
            )
            t0 = time.time()
            outs = llm.generate(
                prompts=[{"prompt_token_ids": list(prompt_token_ids)}],
                sampling_params=sp,
                use_tqdm=False,
            )
            dt = time.time() - t0
            if len(outs) != 1:
                raise BackendError(f"vllm returned {len(outs)} outputs for 1 prompt")
            return outs[0].outputs, dt

        if n > 1 and seed is not None:
            o = []
            dt = 0.0
            for i in range(n):
                cand, dt_i = _one_call(1, int(seed) + i)
                o.append(cand[0])
                dt += dt_i
        else:
            o, dt = _one_call(n, seed)
        token_ids, texts, lps, finish = [], [], [], []
        for cand in o:
            ids = list(cand.token_ids)
            cand_lp = []
            if cand.logprobs:
                for pos, step in enumerate(cand.logprobs):
                    if step is None:
                        cand_lp.append(float("nan"))
                        continue
                    tok = ids[pos] if pos < len(ids) else 0
                    entry = step.get(tok)
                    if entry is None and len(step) > 0:
                        entry = next(iter(step.values()))
                    cand_lp.append(float(entry.logprob) if entry is not None else float("nan"))
            while len(cand_lp) < len(ids):
                cand_lp.append(float("nan"))
            token_ids.append(ids)
            texts.append(cand.text)
            lps.append(cand_lp)
            finish.append(cand.finish_reason or "")
        self._gen_meta["last_sample_time_s"] = dt
        return {
            "token_ids": token_ids,
            "texts": texts,
            "logprobs": lps,
            "finish_reasons": finish,
            "time_s": dt,
        }

    def generate_greedy(self, prompt_token_ids, max_new_tokens):
        return self.sample(
            prompt_token_ids,
            {"temperature": 0.0, "max_new_tokens": max_new_tokens},
            n=1,
        )

    def score(self, context_token_ids, output_token_ids):
        import time

        from vllm import SamplingParams
        from vllm.pooling_params import PoolingParams  # noqa: F401

        llm = self._gen_engine()
        full_ids = list(context_token_ids) + list(output_token_ids)
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        t0 = time.time()
        outs = llm.generate(
            prompts=[{"prompt_token_ids": full_ids}],
            sampling_params=sp,
            use_tqdm=False,
        )
        dt = time.time() - t0
        o = outs[0]
        plp = o.prompt_logprobs
        if plp is None:
            raise BackendError("vllm did not return prompt_logprobs")
        if len(plp) != len(full_ids):
            raise BackendError(
                f"prompt_logprobs length {len(plp)} != sequence length {len(full_ids)}"
            )
        ctx_len = len(context_token_ids)
        out_lps = []
        for pos in range(ctx_len, len(full_ids)):
            step = plp[pos]
            if step is None:
                raise BackendError(f"prompt_logprobs missing at position {pos}")
            tok = full_ids[pos]
            entry = step.get(tok)
            if entry is None:
                raise BackendError(f"prompt_logprobs lacks token {tok} at position {pos}")
            out_lps.append(float(entry.logprob))
        self._gen_meta["last_score_time_s"] = dt
        return out_lps

    def replay_last_hidden(self, context_token_ids, output_token_ids):
        import time

        import numpy as np
        from vllm import PoolingParams

        llm = self._pool_engine()
        full_ids = list(context_token_ids) + list(output_token_ids)
        pp = PoolingParams(task="token_embed", use_activation=False)
        t0 = time.time()
        res = llm.encode(
            prompts=[{"prompt_token_ids": full_ids}],
            pooling_params=pp,
            pooling_task="token_embed",
            use_tqdm=False,
        )
        dt = time.time() - t0
        data = res[0].outputs.data
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 2:
            raise BackendError(f"token_embed returned ndim={arr.ndim}, expected 2")
        if arr.shape[0] != len(full_ids):
            raise BackendError(
                f"token_embed rows {arr.shape[0]} != sequence length {len(full_ids)}"
            )
        ctx_len = len(context_token_ids)
        out_states = arr[ctx_len:]
        if not np.isfinite(out_states).all():
            raise BackendError("non-finite values in last hidden states")
        if float(np.abs(out_states).sum()) == 0.0:
            raise BackendError("all-zero pooled states guard triggered")
        return {
            "hidden": out_states,
            "dtype": str(arr.dtype),
            "time_s": dt,
        }
