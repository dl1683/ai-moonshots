From [SCRATCHPAD.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/SCRATCHPAD.md):

- Code-like structured reps: `N3/F2/H3`. Interesting, but mostly a neuro-symbolic repackaging unless you find a very soft differentiable form. Could help routing and uncertainty, but discrete types/bindings are optimization poison at `475M`. Keep as a later auxiliary/interface idea, not the core thesis.

- Recursive compression: `N4/F4/H5`. Highest core-idea potential. It gives one unifying principle for hierarchy, routing, abstraction, and verification instead of another component. Also plausible at `475M`: multiscale processing is implementable without requiring brittle symbolic machinery.

- Differentiable database memory: `N2/F3/H3`. Useful subsystem, weak central idea. It solves storage/update/retrieval better than reasoning itself. Keep only if factual persistence or inference-time updating is a known bottleneck; otherwise it risks becoming expensive architecture without changing the model’s actual intelligence.

- Position-free architecture: `N3/F1/H1`. Kill immediately in the strong form. Language is too order-sensitive for “position doesn’t matter” to carry the system. The only salvageable piece is a narrow ablation: content-only retrieval across patches while keeping local sequential structure.

Core bet: `#2 Recursive compression`.

Kill now: `#4` as stated. Also don’t let `#3` become the thesis.

Missing fifth angle: `iterative constraint satisfaction / belief propagation`. Treat language understanding as finding a globally consistent state under many local constraints; that could unify routing, uncertainty, and verification more directly than database memory or position removal.