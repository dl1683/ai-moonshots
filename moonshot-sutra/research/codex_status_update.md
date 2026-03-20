Revised architecture: a sequential-first hybrid, not an attention replacement.

Use:
- `byte/BPE embeddings -> MSBP local mixer`
- `GRU + message passing` as the main state-transition core
- `recursive compression` for multiscale memory
- `grown sparsity` only as a selective retrieval/read mechanism
- `bf16 + token-level training` as the practical training stack

Remove entirely:
- Kalman uncertainty
- OT routing
- phase-sync / linear-attention variants
- position-free ideas

Math-wise, the surviving pieces now have distinct roles instead of competing:
- MSBP = parameter-efficient local structure
- GRU+MP = ordered reasoning/state update
- compression = long-context storage
- sparsity = cheap lookup

So yes: ship Combo 5 now. It is the correct pragmatist move.

If there is still novelty worth pursuing, it is not a new routing theory. It is a clean hierarchical recurrent architecture where compression and sparse retrieval are integrated around the GRU+MP core. That is the only novel lane still justified.