"""Kernel choice for the Qwen3.5 linear-attention layers on Windows.

Measured on the RTX 2070 Super (bench.speedtest_kernels, 400 LegalBench
prompts): plain PyTorch 55 ms per prompt, flash-linear-attention's
chunk_gated_delta_rule 74 ms, plus its Triton causal conv1d 70 ms (max |dP|
0.003 against plain). fla's Triton kernels are tuned for newer GPUs and
recompile for new prompt lengths, so on this Turing card the plain path is
faster and stays the default even though fla is installed.

    BENCH_FLA=1        use fla's delta rule (transformers picks it on import)
    BENCH_FAST_CONV=1  with BENCH_FLA=1, also route the causal conv to fla's Triton kernel

configure() must run before transformers' Qwen3.5 model code is imported;
bench.judges calls it at import time.
"""

from __future__ import annotations

import importlib.abc
import os
import sys

_state = None


class _HideFla(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "fla" or name.startswith("fla."):
            raise ImportError("flash-linear-attention disabled (BENCH_FLA=1 enables it)")
        return None


def configure() -> str:
    global _state
    if _state is not None:
        return _state
    if os.environ.get("BENCH_FLA", "0") != "1":
        sys.meta_path.insert(0, _HideFla())
        _state = "plain PyTorch (fla hidden; faster on this GPU)"
        return _state
    _state = "fla delta rule"
    if os.environ.get("BENCH_FAST_CONV", "0") == "1":
        from fla.modules.convolution import causal_conv1d
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

        def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
            y, _ = causal_conv1d(hidden_states.transpose(1, 2).contiguous(), weight=weight, bias=bias,
                                 activation=activation)
            return y.transpose(1, 2).to(hidden_states.dtype)

        modeling.causal_conv1d_fn = causal_conv1d_fn
        _state += " + fla Triton causal conv1d"
    return _state


def install() -> str:  # kept for callers of the first version
    return configure()
