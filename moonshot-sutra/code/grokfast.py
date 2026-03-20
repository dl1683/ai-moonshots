"""Grokfast: EMA gradient filter for accelerated generalization.

Amplifies slow (generalization-inducing) gradient modes via exponential moving average.
Chrome-validated: +11.0% BPT improvement at dim=128 with alpha=0.95, lambda=2.0.

Paper: arXiv 2405.20233 (Seoul National University, May 2024)
Reports 50x grokking acceleration on standard benchmarks.

Usage in training loop:
    from grokfast import GrokfastFilter
    gf = GrokfastFilter(model, alpha=0.95, lam=2.0)

    # In training loop, after loss.backward() and before opt.step():
    gf.apply()

Integration with sutra_v05_train.py:
    Insert gf.apply() at line 212 (after NaN check, before clip_grad_norm_).
    Or at line 198 (after loss.backward(), before NaN check — gradients will be
    amplified before NaN detection, which is fine since the amplification is bounded).

Chrome results:
    alpha=0.95, lam=2.0:  +11.0% BPT (6.468 vs 7.265)  PASS
    alpha=0.98, lam=5.0:  NaN (too aggressive)            KILL
    alpha=0.99, lam=10.0: +0.0% (too conservative)        NEUTRAL
"""

import torch


class GrokfastFilter:
    """EMA gradient filter that amplifies slow generalization modes.

    The filter maintains an exponential moving average of gradients per parameter.
    After each backward pass, it adds the amplified EMA back to the gradients.
    This amplifies slow-varying components (which correlate with generalization)
    relative to fast-varying components (which correlate with memorization).

    Math:
        grads_ema = alpha * grads_ema + (1 - alpha) * grad
        grad += lam * grads_ema

    Args:
        model: The model whose gradients to filter
        alpha: EMA decay rate (0.95 recommended for Sutra)
        lam: Amplification factor (2.0 recommended for Sutra)
    """

    def __init__(self, model, alpha=0.95, lam=2.0):
        self.model = model
        self.alpha = alpha
        self.lam = lam
        self.grads_ema = {}

    def apply(self):
        """Apply Grokfast filter to current gradients. Call after loss.backward()."""
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            if name not in self.grads_ema:
                self.grads_ema[name] = torch.zeros_like(p.grad)
            self.grads_ema[name].mul_(self.alpha).add_(p.grad, alpha=1 - self.alpha)
            p.grad.add_(self.grads_ema[name], alpha=self.lam)

    def state_dict(self):
        """Save EMA state for checkpoint resume."""
        return {name: ema.clone() for name, ema in self.grads_ema.items()}

    def load_state_dict(self, state):
        """Load EMA state from checkpoint."""
        for name, ema in state.items():
            self.grads_ema[name] = ema
