**Definitive 7-Stage Pipeline Reference**

**Stage 1. Segmentation / Compression**  
Definition: Convert the raw stream into units that preserve predictive structure while minimizing unnecessary sequence length.  
Current best-in-class: `BLT` dynamic byte patching.  
Sutra v0.4 approach: raw bytes with fixed 4-byte patches.  
Gap severity: Medium.  
v0.5 fix: None; defer until routing bottlenecks are fixed.  
Theoretical optimal: learned MDL-style adaptive segmentation that changes with local entropy and task demands.  
Biological parallel: retina-style sensory compression before higher cognition.

**Stage 2. State Init / Addressing**  
Definition: Initialize latent state and positional coordinates so the model knows what each unit is and where it sits relative to others.  
Current best-in-class: `RoFormer` / `RoPE`.  
Sutra v0.4 approach: byte embeddings plus in-patch GRU order signal, but weak inter-patch relative addressing.  
Gap severity: High.  
v0.5 fix: add `RoPE` to patch-summary interactions and retrieval `Q/K`.  
Theoretical optimal: hierarchical relative addressing across bytes, patches, scratchpad slots, and memory states.  
Biological parallel: hippocampal place/grid coding plus cortical topographic maps.

**Stage 3. Local State Construction**  
Definition: Build strong local features before long-range communication begins.  
Current best-in-class: `Mamba-2`-style selective local sequence modeling.  
Sutra v0.4 approach: 1-layer GRU inside each 4-byte patch.  
Gap severity: Low.  
v0.5 fix: None; later add a small depthwise causal conv before the GRU.  
Theoretical optimal: a content-adaptive local block combining short convolution with selective recurrence/SSM.  
Biological parallel: V1/cortical microcircuits extracting local edges, phonemes, and motifs.

**Stage 4. Communication / Routing**  
Definition: Move information across positions so distant evidence can enter working state with low distortion.  
Current best-in-class: dense causal self-attention from `Attention Is All You Need` as the quality ceiling.  
Sutra v0.4 approach: windowed local message passing plus sparse retrieval.  
Gap severity: High.  
v0.5 fix: add one causal multi-head attention refresh over patch summaries every 2 local rounds.  
Theoretical optimal: low-diameter hybrid routing with mostly local diffusion, sparse targeted jumps, and periodic dense global refresh.  
Biological parallel: long-range cortical tracts and mycorrhizal hub-and-spoke transport.

**Stage 5. State Update / Memory Write**  
Definition: Decide what to retain, erase, and overwrite after new evidence arrives.  
Current best-in-class: `Gated DeltaNet`.  
Sutra v0.4 approach: additive retrieval merge plus ungated MLP/residual update.  
Gap severity: High.  
v0.5 fix: replace additive merge with a gated write, e.g. `GRUCell(msgs + retrieved, h)` or DeltaNet-style gating.  
Theoretical optimal: explicit retain/erase/write dynamics with minimal interference and durable useful state.  
Biological parallel: synaptic plasticity, immune memory formation, and fungal tube thickening/atrophy.

**Stage 6. Compute Control / Deliberation**  
Definition: Allocate extra passes only when the sequence still needs more computation.  
Current best-in-class: `DeepSeek-R1`-style extra reasoning compute.  
Sutra v0.4 approach: `PonderNet` adaptive halting over 1-8 rounds with a `min_rounds` safeguard.  
Gap severity: Medium.  
v0.5 fix: None; later replace per-step halting with a sequence-level pass controller choosing `{2,4,8}`.  
Theoretical optimal: an uncertainty-calibrated controller that buys full extra passes only when expected value exceeds cost.  
Biological parallel: basal ganglia/thalamic action gating and immune escalation proportional to threat.

**Stage 7. Readout / Decode / Verify**  
Definition: Convert final state into outputs, then optionally decode, verify, or rerank them.  
Current best-in-class: verifier-guided decoding from `Let's Verify Step by Step`.  
Sutra v0.4 approach: linear LM head with plain autoregressive decoding.  
Gap severity: Medium.  
v0.5 fix: None; later add a verifier head and rerank `N=4` samples.  
Theoretical optimal: calibrated readout plus an external verifier/search/tool loop separate from the core forward pass.  
Biological parallel: motor readout with feedback correction, or immune effector selection after recognition.

**Definitive Improvement Order**  
1. Stage 2: add `RoPE` / relative addressing.  
2. Stage 4: add periodic global causal attention refresh.  
3. Stage 5: replace additive merge with gated write/update.  
4. Stage 7: add verifier head plus reranking.  
5. Stage 1: move from fixed 4-byte patches to adaptive segmentation.  
6. Stage 6: replace per-step halting with sequence-level pass control.  
7. Stage 3: add a small depthwise causal conv before the GRU.

**Definitive v0.5 Design**  
Keep byte input, fixed 4-byte patches, the GRU patch processor, current halting, and current readout. Make exactly three routing-centric changes: add `RoPE` to patch-summary and retrieval addressing, insert a causal multi-head attention refresh over patch summaries every 2 local rounds, and replace the current additive summary merge with a gated write/update. This is the cleanest `2 + 4 + 5` upgrade.

**Definitive v1.0 Dream Design**  
Optimal Sutra is a multiscale hybrid, not a pure message-passing model: adaptive byte segmentation; hierarchical relative addressing; a local block built from depthwise conv plus selective GRU/SSM; a repeated inner loop of local message passing, sparse retrieval, small global scratchpad tokens, and occasional dense causal attention refresh; gated write/update at every inner-loop step; a sequence-level compute controller choosing budgets like `{2,4,8,16}`; and LM readout followed by verifier-guided reranking and optional tool-assisted checking outside the core forward pass.

**Framework Summary**  
The definitive Sutra view is: front-end compression and addressing set up the state, then a repeated inner loop does local construction, global routing, memory writing, and compute allocation, and only then does the system read out and optionally verify. The core bottleneck is not local modeling or depth by itself; it is global information flow under bandwidth constraints, especially the `Stage 2 -> Stage 4 -> Stage 5` interface. That is why `v0.5` should be a routing-focused release with `RoPE`, periodic global attention refresh, and gated writes, while `v1.0` should become a fully multiscale hybrid architecture built around principled communication and memory, not pure local diffusion.

I could not write this into the repo because the workspace is currently read-only.