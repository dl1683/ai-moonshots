# Sutra Research Log

## Chrome Cycle 1: Eval Set Design (2026-03-19)

### Theory: What Makes a Discriminating Eval?

From Item Response Theory (IRT), each question has discrimination `a_i` and difficulty `b_i`. The information function `I_i(theta) = a_i^2 * P_i(theta) * Q_i(theta)` peaks when question difficulty matches model ability. A good eval is a spectrum analyzer across the ability range.

### Key Design Decision: Zero Fact Lookup

Current benchmarks (MMLU, ARC, HellaSwag) primarily test knowledge retrieval. This favors models trained on more data, not models with better architecture. For Sutra, we need to test the REASONING ENGINE, not the knowledge database.

**Principle**: If a model knows WHAT to look up and HOW to combine information, fact lookup becomes a trivial tool call. The hard part is the thinking process.

### Taxonomy (500 questions, 7 categories)

| Category | Count | Tests |
|----------|-------|-------|
| Strategic Reasoning | 100 | Planning, trade-offs, game theory, resource allocation |
| Synthesis & Combination | 100 | Cross-domain connection, creative problem solving |
| Critical Analysis | 80 | Flaw detection, assumption identification, evidence evaluation |
| Instruction Following | 80 | Precision under complex interacting constraints |
| Drafting Under Constraints | 60 | Generation quality with multiple simultaneous requirements |
| Code & Algorithmic Thinking | 50 | Algorithm design, debugging reasoning, optimization |
| Meta-Cognition | 30 | Self-awareness, knowledge gap identification, calibrated uncertainty |

Difficulty distribution within each: 20% easy, 30% medium, 30% hard, 20% extreme.

### Scoring Framework

Three modes:
1. **exact_match** — one correct answer (math, logic, some code)
2. **constraint_check** — binary per-constraint, score = fraction met (instruction following)
3. **rubric** — multi-dimensional 0-3 scoring (drafting, synthesis, analysis)

Automated scoring: exact_match (~40%) + constraint_check (~30%) + LLM-as-judge with explicit rubrics (~30%).

### Dead Ends

*(None yet)*
