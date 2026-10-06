from __future__ import annotations

import time

import numpy as np
import torch

from .base import InferenceBackend, BackendError


def build_transformers_backend(model_path: str, backend_cfg: dict, logger=None) -> "TransformersBackend":
    return TransformersBackend(model_path=model_path, cfg=backend_cfg or {}, logger=logger)


class TransformersBackend(InferenceBackend):
    """Legacy port kept from old exphalu: HF Transformers forward passes.

    This is the compatibility backend for architectures the inference engine does
    not natively support (e.g. Qwen3.5 on vLLM 0.15.1). It is NOT the primary
    path of RePPL 2.0; results from this backend are tagged backend='transformers'
    and must be reported separately from vLLM native results.
    """

    name = "transformers"

    def __init__(self, model_path: str, cfg: dict, logger=None):
        self.model_path = model_path
        self.cfg = cfg
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.device = cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")

    def _log(self, msg):
        if self.logger is not None:
            self.logger.info(f"[transformers] {msg}")

    def load(self):
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
            str(self.cfg.get("dtype", "bfloat16"))
        ]
        self._log(f"loading {self.model_path} on {self.device} dtype={dtype}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            attn_implementation=str(self.cfg.get("attn_implementation", "eager")),
        ).to(self.device)
        self.model.eval()

    def close(self):
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()

    def capabilities(self) -> dict:
        return {
            "backend": "transformers",
            "generation": True,
            "token_logprobs": True,
            "token_last_hidden": True,
            "note": "legacy port; HF forward; kept for architecture compatibility",
        }

    def _chat_template_hash(self):
        tok = self.tokenizer
        tmpl = getattr(tok, "chat_template", None)
        import hashlib

        return hashlib.sha256((tmpl or "").encode()).hexdigest()[:12]

    def sample(self, prompt_token_ids, sampling_config, n, seed=None):
        self.load()
        input_ids = torch.tensor([prompt_token_ids], device=self.device)
        attn = torch.ones_like(input_ids)
        temp = float(sampling_config.get("temperature", 1.0))
        top_k = int(sampling_config.get("top_k", -1))
        gen_cfg = dict(
            max_new_tokens=int(sampling_config.get("max_new_tokens", 64)),
            do_sample=temp > 0,
            temperature=temp if temp > 0 else None,
            top_p=float(sampling_config.get("top_p", 1.0)) or None,
            top_k=top_k if top_k > 0 else None,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            output_logits=True,
            return_dict_in_generate=True,
        )
        for key in ("temperature", "top_p", "top_k"):
            if gen_cfg[key] is None:
                gen_cfg.pop(key)
        if seed is not None:
            torch.manual_seed(int(seed))
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(input_ids, attention_mask=attn, **gen_cfg)
        dt = time.time() - t0
        in_len = input_ids.shape[1]
        eos_id = self.tokenizer.eos_token_id
        token_ids, texts, lps, finish = [], [], [], []
        n_steps = len(out.logits)
        for i in range(n):
            full = out.sequences[i, in_len:].tolist()
            seq = full[:n_steps] if len(full) > n_steps else full
            if eos_id is not None and eos_id in seq:
                seq = seq[: seq.index(eos_id) + 1]
                finish.append("stop")
            else:
                finish.append("length")
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            step_lps = []
            for t, tok in enumerate(seq):
                lg = out.logits[t][i].float()
                lp = torch.log_softmax(lg, dim=-1)[tok].item()
                step_lps.append(float(lp))
            while len(step_lps) < len(seq):
                step_lps.append(float("nan"))
            token_ids.append(seq)
            texts.append(text)
            lps.append(step_lps)
        return {"token_ids": token_ids, "texts": texts, "logprobs": lps,
                "finish_reasons": finish, "time_s": dt}

    def score(self, context_token_ids, output_token_ids):
        self.load()
        full = list(context_token_ids) + list(output_token_ids)
        input_ids = torch.tensor([full], device=self.device)
        t0 = time.time()
        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits[0].float()
        dt = time.time() - t0
        ctx_len = len(context_token_ids)
        log_probs = torch.log_softmax(logits, dim=-1)
        out_lps = []
        for pos in range(ctx_len, len(full)):
            tok = full[pos]
            out_lps.append(float(log_probs[pos - 1, tok].item()))
        return out_lps

    def replay_last_hidden(self, context_token_ids, output_token_ids):
        self.load()
        full = list(context_token_ids) + list(output_token_ids)
        input_ids = torch.tensor([full], device=self.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model(input_ids=input_ids, output_hidden_states=True)
        dt = time.time() - t0
        last = out.hidden_states[-1][0].float()
        ctx_len = len(context_token_ids)
        ctx_states = last[:ctx_len].cpu().numpy().astype(np.float32)
        states = last[ctx_len:].cpu().numpy().astype(np.float32)
        if not np.isfinite(states).all() or not np.isfinite(ctx_states).all():
            raise BackendError("non-finite values in last hidden states")
        if float(np.abs(states).sum()) == 0.0:
            raise BackendError("all-zero pooled states guard triggered")
        return {"hidden": states, "hidden_context": ctx_states,
                "dtype": "float32", "time_s": dt}

    def replay_attention(self, context_token_ids, output_token_ids):
        """Reference-environment signal for attention-based baselines (RAUQ).

        One forward pass over [context, output] with output_attentions; returns
        (n_layers, n_heads, T_total, T_total) float32 attention weights. Requires
        attn_implementation="eager" (set in backend_cfg). The native vLLM path
        does NOT provide this - capability errors there are expected and honest.
        """
        self.load()
        full = list(context_token_ids) + list(output_token_ids)
        input_ids = torch.tensor([full], device=self.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model(input_ids=input_ids, output_attentions=True)
        dt = time.time() - t0
        att = torch.stack([a[0] for a in out.attentions]).float().cpu().numpy()
        if not np.isfinite(att).all():
            raise BackendError("non-finite values in attention weights")
        if att.shape[-1] != len(full):
            raise BackendError(
                f"attention length {att.shape[-1]} != sequence length {len(full)}")
        return {"attentions": att, "n_layers": int(att.shape[0]),
                "n_heads": int(att.shape[1]), "dtype": "float32", "time_s": dt}
