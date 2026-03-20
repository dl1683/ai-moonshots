**Verdict**

As a primary training strategy, heavy transformer-to-Sutra distillation is more likely a trap than a clean win right now. As a bounded auxiliary, it can help. The safest version is: keep Sutra’s architecture-native objectives primary, use the teacher late and lightly, and prefer teacher-assisted data curation over hidden-state imitation.

A key practical blocker first: Sutra currently trains on GPT-2 vocab `50257` and loads the GPT-2 tokenizer in [sutra_v05_train.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_train.py#L43) and [sutra_v05_train.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_train.py#L128). Naive full-vocab logit KD from Pythia or Qwen is not clean, because Pythia uses GPT-NeoX tokenization and Qwen uses its own tokenizer family.

1. **Risks**
- Yes, the student can learn transformer-like behavior in function space even without copying weights. KD does not force transformer internals, but it strongly rewards “match the teacher’s one-shot next-token policy,” which can make `pi`, `lambda`, rerouting, and adaptive compute become decorative rather than causal.
- Soft labels can suppress stage dynamics if they are applied everywhere and early. Sutra’s thesis depends on stage occupancy, content-dependent transitions, and verify→reroute being real state variables, per [CLAUDE.md](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/CLAUDE.md#L131), [research/VISION.md](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/VISION.md#L9), and [research/codex_v05_design.md](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/codex_v05_design.md#L5). A teacher only supervises final distributions, so it gives no native incentive to preserve those dynamics.
- Feature-level KD is the highest-risk form. There is no principled layer↔stage correspondence. Matching transformer hidden states or attention patterns would import sequential-layer geometry into a state-graph model.
- The current implementation makes this worse: [sutra_v05_ssm.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L271) explicitly says “No verify/reroute yet” and uses fixed recurrent steps, with zero real compute cost in [sutra_v05_ssm.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L318). KD now would mostly optimize the shortcut version of Sutra, not the full thesis.

2. **Best mechanism for Sutra**
- Best default: **late, terminal, hard-position KD**. Distill only the final accepted readout, and only on positions where Sutra is unresolved: low `lambda`, high entropy, or failed verify. That uses the teacher as extra evidence, not as a controller.
- Best schedule: **CE-first, KD-second, CE-last**. Let stage differentiation emerge first, add KD mid-training, then anneal it down so the model finishes on data rather than teacher imitation.
- Best Sutra-specific variant:
  `L = CE + α·1[low_lambda or verify_fail]·KL(p_teacher^T || p_student^T) + β·L_interstep + γ·L_stage_native`
  where `L_stage_native` preserves stage entropy/sparsity, lambda monotonicity, reroute usage, and content-dependent transition diversity.
- I would avoid hidden-state KD. If you want something richer than logits, use very weak relational signals only at the output side, not layerwise feature matching.

3. **What to steal vs not steal**
- Steal: lexical knowledge, next-token uncertainty structure, semantic alternatives, topic priors, broad world knowledge, code idioms, and difficulty signals.
- Do not steal: attention maps, layer chronology, positional encoding geometry, hidden-state manifolds, chain-of-thought style, or chat/instruction alignment behavior.
- The separation rule is simple: distill only what can be expressed at the readout or dataset level; keep internal routing, memory update, compute control, and verification architecture-native. Otherwise you are compressing a transformer policy into a new shell.

4. **Practical choice**
- Between your two candidates, **Pythia-160M is the safer KD teacher**. It is a plain research LM, smaller, less post-trained, and less likely to inject chat/reasoning artifacts. Use [EleutherAI/pythia-160m](https://huggingface.co/EleutherAI/pythia-160m).
- **Qwen3-0.6B is the stronger knowledge source, but the riskier direct teacher**. The main `Qwen3-0.6B` card is explicitly pretraining + post-training with thinking/non-thinking behavior. If you use Qwen, prefer the base model for offline data work, not the aligned chat model: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base).
- Start with `α_KD = 0.1–0.2`, `α_CE = 1.0`, temperature `T = 2`. Do not start above `0.3` KD weight. Anneal KD down in the last 20-30% of training.
- Because of tokenizer mismatch, I would not start with full-vocab KL at all. Start with candidate-set KD or data-level distillation.

5. **Does this violate the thesis?**
- It does not violate the letter of “no pre-trained weights” in [CLAUDE.md](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/CLAUDE.md#L18). It does weaken the story if it becomes necessary for viability.
- The strong skeptic argument is: “If Sutra needs a transformer teacher, then scale discovered the knowledge and Sutra is just a compression vessel.” That is a real risk.
- The defensible position is narrower: geometry may still be the better **compute structure**, while the teacher supplies reusable **corpus knowledge**. But to defend that, you need ablations showing CE-only Sutra preserves the stage behaviors and teacher help is additive, not foundational.

6. **Alternative: teacher for data, not logits**
- This is probably the cleaner path.
- Use the teacher to score/filter corpus quality, mine hard spans, relabel ambiguous or corrupted examples, and generate synthetic tasks targeted at Sutra’s weak points: long-range routing, evidence accumulation, verify-and-revise, code variable binding.
- This avoids tokenizer mismatch, preserves internal freedom, and aligns better with the “novel architecture stays novel” thesis.
- The danger is style laundering: synthetic data can quietly turn Sutra into “Qwen-sounding text with a different compute graph.” Keep synthetic ratios bounded and mix with raw corpus.

**Bottom line**

Direct hidden-state or heavy logit KD into Stage-Superposition is a trap. Light, late, hard-position output KD is defensible. Teacher-driven data curation and synthetic stage-targeted tasks are the better first experiment.

If you do try KD, success is not “lower BPT” alone. It must also preserve or improve stage metrics already central to the thesis: content-dependent transition differences, lambda as a halting signal, stage diversity, and real verify/reroute behavior in [research/RESEARCH.md](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1466), [research/RESEARCH.md](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1488), [research/RESEARCH.md](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1572), and [research/RESEARCH.md](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1629). If KD improves loss while those collapse, the skeptic wins.