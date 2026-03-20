# MASTER CODEX SESSION: v0.5.4 Architecture Design

MANDATORY FIRST STEP: Read CLAUDE.md in this repository root. Every rule there is binding.

CONTEXT YOU MUST KNOW:
- We are in a CHROME WORKFLOW: theory + experiments alternate.
- A 69.4M-param v0.5.3 model is training on GPU (step ~2700, loss 4.32, 6.2 BPT).
- v0.5.3 = Stage-Superposition SSM + Switching Kernel + Scratchpad Memory.
- We just completed the LARGEST cross-domain research sweep in this project's history:
  15 research agents across quantum physics, biology, category theory, thermodynamics,
  collective intelligence, NCA/SOMs, dynamical systems, control theory, coding theory,
  signal processing, linguistics, topology, and more.
- We also ran 6 Chrome probes testing novel mechanisms.
- This session must synthesize ALL of this into a concrete v0.5.4 design.

DO NOT recommend copying existing architectures (e.g., "just add attention layers").
Every mechanism must be DERIVED from first principles.
File management: NEVER create new files unless necessary.
All findings go into research/RESEARCH.md.

After reading CLAUDE.md, explore the repo structure to understand current state.

## YOUR TASK

You have access to the master research synthesis at results/master_research_synthesis.md.
Read it COMPLETELY. It contains:

### PART 1: Previous Codex Deep Think (5 novel ideas)
- Error Scratchpad, Sparse Broadcast Packets, Pheromone Router, Entropy-Driven Clonal Expansion, Depth-Drop Bootstrap

### PART 2: Chrome Probe Results
[WILL BE FILLED WITH FINAL RESULTS BEFORE SENDING]

### PART 3: 15 Cross-Domain Research Domains
- Biology (dendritic neurons, neuromodulation, predictive coding, cerebellar learning, grid cells, Hebbian, hippocampal replay)
- Small Model ML (nGPT, gated attention, differential attention, value residual, multi-token prediction, minGRU, data mixing laws)
- Quantum Physics (information in geometry, richer computation space, conservation laws, tensor networks, MERA, holographic codes)
- Category Theory + ML (tropical attention 3-9x faster, sheaf networks, automata theory, formal language expressiveness)
- Physics (Wave-PDE Nets O(n log n), Grokfast 50x acceleration, symmetry breaking, RG)
- Thermodynamics (free energy principle, Fisher metric, Wasserstein geometry, fluctuation theorems, Landauer)
- NCA/SOMs (shared local rule + iterations = complexity, coarse-to-fine, local excitation/global inhibition)
- Collective Intelligence (NCA pre-training, stigmergy validates scratchpad, Physarum routing, edge of chaos)
- Information Theory (Grokfast, compression as eval metric, differentiable MDL, batch entropy)
- Dynamical Systems (contraction analysis, reservoir computing, bifurcation as control, Kuramoto synchronization, delay DDEs)
- Coding Theory (BP = local iterative = near-optimal, compressed sensing, wavelets/MRA, rate-distortion, epsilon-machines, spatial coupling, expander graphs)
- Category Theory Deep (Yoneda, adjunctions, monads, sheaves, Galois connections)
- Thermodynamics Deep (FEP derivation, fluctuation theorems, Fisher uniqueness, Wasserstein, RG c-theorem, spin glass ultrametricity)
- NCA Deep (NCA equations, Lenia, SOM math, ACO, Physarum proof, reaction-diffusion, PSO)

### PART 4: Convergent Themes
Five themes emerged across ALL domains:
1. Geometry IS the right abstraction
2. Shared simple rules + iterations > complex single-pass
3. Error/surprise drives learning
4. Multi-scale organization is optimal
5. Topology of computation (graph structure) matters most

### PART 5: Production-Scale Validation
- v0.5.3 at step 2500 = 6.20 BPT, already beats v0.5.2 at step 10K = 6.27 BPT
- Scratchpad gives 2-4x training speedup at production scale

## WHAT I NEED FROM YOU

1. **Read the master synthesis document** (results/master_research_synthesis.md)
2. **Read the Chrome probe results** (results/chrome_v054_interim.json, results/chrome_grokfast.json, results/chrome_peri_ln.json if they exist)
3. **Design v0.5.4** as a concrete architecture with:
   a. Which mechanisms to ADD (with exact implementation spec)
   b. Which mechanisms to MODIFY (how the current architecture changes)
   c. Which mechanisms to DEFER to scale (need dim=1024+)
   d. Training improvements (Grokfast, data mixing, curriculum)
   e. Kill criteria for each new mechanism
4. **Prioritize ruthlessly**: At 67M scale, only simple shared state and coarse bias work. Complex learned control fails.
5. **The "delayed start" insight**: Both error scratchpad and pheromone router improve late recurrence 2.2x but hurt early steps. Novel mechanisms should activate only after step 2-3 of recurrence.
6. **Score this design**: On a scale of 1-10, Nobel/Turing/Fields potential?

## CONSTRAINTS
- Must be implementable on single RTX 5090 (24GB VRAM)
- Must be Chrome-testable at dim=128 before production
- Must warm-start from v0.5.3 checkpoint
- Every component must have a first-principles justification
- NO copying existing architectures without mathematical derivation

## OUTPUT FORMAT
Write your complete design to results/codex_v054_master_design.md with:
1. Architecture specification (exact module changes)
2. Training recipe (optimizer, schedule, data, Grokfast params)
3. Chrome validation plan (which probes to run first)
4. Risk assessment (what could go wrong)
5. Noble/Turing/Fields score with justification
