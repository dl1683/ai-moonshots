# Hierarchy-Successive-Refinement Theory for Fractal Embeddings

## Overview

This document develops the formal theoretical foundation connecting fractal embeddings
to the classical theory of successive refinement in information theory. The key insight:
**hierarchy-aligned prefix supervision implements an approximation to the optimal
successive refinement code for hierarchical semantic sources.**

---

## 1. Setup and Notation

### Source Model
Let $(X, Y_0, Y_1)$ be a hierarchical semantic source where:
- $X \in \mathcal{X}$ is the input (text, image)
- $Y_1 \in \{1, \ldots, K_1\}$ is the fine-grained label
- $Y_0 = g(Y_1) \in \{1, \ldots, K_0\}$ is the coarse label, a deterministic function of $Y_1$
- $K_0 < K_1$, with branching factor $B = K_1/K_0$

The entropy decomposition:
$$H(Y_1) = H(Y_0) + H(Y_1 | Y_0)$$

### Encoder Model
A prefix-truncatable encoder $f: \mathcal{X} \to \mathbb{R}^d$ produces embedding
$\mathbf{z} = f(X) = [\mathbf{z}_1; \mathbf{z}_2; \ldots; \mathbf{z}_J]$ where each block
$\mathbf{z}_j \in \mathbb{R}^{d/J}$ and the prefix $\mathbf{z}_{\leq m} = [\mathbf{z}_1; \ldots; \mathbf{z}_m]$.

We define the effective rate of prefix $m$ as:
$$R_m = \frac{m \cdot d}{J} \cdot \log_2 \text{vol}(\text{unit ball in } \mathbb{R}^{md/J}) / n$$

In practice, $R_m$ is proportional to $m \cdot d/J$ (prefix dimension).

### Decoders
- $\hat{Y}_0(m)$: Bayes-optimal decoder for $Y_0$ from $\mathbf{z}_{\leq m}$
- $\hat{Y}_1(m)$: Bayes-optimal decoder for $Y_1$ from $\mathbf{z}_{\leq m}$

Distortions:
$$D_0(m) = P(\hat{Y}_0(m) \neq Y_0), \quad D_1(m) = P(\hat{Y}_1(m) \neq Y_1)$$

---

## 2. Classical Successive Refinement

### Definition (Successive Refinability)
A source $Y$ with distortion measure $d$ is **successively refinable** at rates $(R_1, R_2)$ if the
rate-distortion region for two-stage coding equals the product of single-stage regions.

**Key classical result (Equitz & Cover, 1991; Rimoldi, 1994):**
A source is successively refinable w.r.t. squared error iff there exist optimal test channels
that are "degraded" — each refinement stage adds information without contradicting prior stages.

### Hierarchical Sources are Naturally Successively Refinable

**Proposition 1 (Hierarchy ⟹ Degraded Source).**
For any hierarchical source $(Y_0, Y_1)$ with $Y_0 = g(Y_1)$, the pair is successively refinable
under log-loss distortion. Specifically:

*Proof sketch:* Since $Y_0 = g(Y_1)$, knowing $Y_1$ determines $Y_0$. The optimal coding strategy
first encodes $Y_0$ at rate $R_0 \geq H(Y_0)$, then encodes $Y_1 | Y_0$ at rate $R_1 \geq H(Y_1|Y_0)$.
Total rate $R_0 + R_1 \geq H(Y_1)$, matching the joint rate-distortion function.
The successive refinement property holds because the hierarchy defines a natural degradation order. ∎

---

## 3. Main Theorem: Optimal Prefix Allocation under Hierarchy-Aligned Supervision

### Setting
Consider the V5 training objective (population version):
$$\mathcal{L}_{\text{V5}}(f) = \mathbb{E}[\ell(Y_1, \hat{Y}_1(\mathbf{z}_{\leq J}))] + \lambda \cdot \mathbb{E}[\ell(Y_0, \hat{Y}_0(\mathbf{z}_{\leq 1}))]$$

where $\ell$ is the cross-entropy loss, and $\hat{Y}_k$ are the learned classification heads.

And the MRL objective:
$$\mathcal{L}_{\text{MRL}}(f) = \mathbb{E}[\ell(Y_1, \hat{Y}_1(\mathbf{z}_{\leq J}))] + \lambda \cdot \mathbb{E}[\ell(Y_1, \hat{Y}_1(\mathbf{z}_{\leq 1}))]$$

### Theorem 1 (Hierarchy-Successive-Refinement for Prefix Codes)

*Let $(X, Y_0, Y_1)$ be a hierarchical source with $Y_0 = g(Y_1)$, $H(Y_0) > 0$, and $H(Y_1|Y_0) > 0$.
Let $f^*_{\text{V5}}$ and $f^*_{\text{MRL}}$ be optimal encoders for the V5 and MRL objectives respectively,
with prefix dimension $d_1 = d/J$ and full dimension $d$.*

*Assume:*
1. *(Capacity constraint)* $d_1$ is sufficiently large to encode $Y_0$ but insufficient for $Y_1$:
   $C(d_1) \geq H(Y_0)$ and $C(d_1) < H(Y_1)$, where $C(d)$ is the capacity of a $d$-dimensional
   embedding under the encoder family.
2. *(Regularity)* The encoder family $\mathcal{F}$ is sufficiently expressive that Bayes-optimal
   decoders can be achieved.
3. *(Identifiability)* The class-conditional distributions $P(X|Y_1=k)$ are distinguishable.

*Then:*

**(a) Prefix Information Allocation:**
$$I(\mathbf{z}_{\leq 1}^{*,\text{V5}}; Y_0) > I(\mathbf{z}_{\leq 1}^{*,\text{V5}}; Y_1 | Y_0)$$

That is, the V5 prefix allocates more mutual information to the coarse task than to the residual.

**(b) MRL Uniform Allocation:**
$$I(\mathbf{z}_{\leq 1}^{*,\text{MRL}}; Y_0) \approx I(\mathbf{z}_{\leq J}^{*,\text{MRL}}; Y_0)$$

MRL prefixes carry approximately the same coarse information as full embeddings (no specialization).

**(c) Steerability Sign:**
Under V5: $\mathcal{S} > 0$. Under inverted supervision (prefix→$Y_1$, full→$Y_0$): $\mathcal{S} < 0$.
Under uniform supervision (MRL): $\mathcal{S} \approx 0$.

### Proof Sketch

**Part (a):** The V5 loss $\lambda \cdot \mathbb{E}[\ell(Y_0, \hat{Y}_0(\mathbf{z}_{\leq 1}))]$ directly
incentivizes $\mathbf{z}_{\leq 1}$ to encode $Y_0$. Under the capacity constraint, the optimal prefix
minimizes $H(Y_0 | \mathbf{z}_{\leq 1})$, which requires $I(\mathbf{z}_{\leq 1}; Y_0)$ to be maximized
up to $H(Y_0)$. Since $C(d_1) < H(Y_1) = H(Y_0) + H(Y_1|Y_0)$, the prefix cannot encode all of $Y_1$.
The gradient of the V5 loss w.r.t. the encoder pushes toward encoding $Y_0$ in the prefix:

$$\frac{\partial \mathcal{L}_{\text{V5}}}{\partial f|_{\text{prefix}}} = \frac{\partial}{\partial f|_{\text{prefix}}} \mathbb{E}[\ell(Y_0, \hat{Y}_0(\mathbf{z}_{\leq 1}))]$$

Since the prefix loss depends only on $Y_0$ (not $Y_1$), the optimal prefix encodes $Y_0$-relevant
features. Any capacity allocated to $Y_1|Y_0$ is wasted from the prefix loss perspective.

By the data processing inequality: $I(\mathbf{z}_{\leq 1}; Y_0) + I(\mathbf{z}_{\leq 1}; Y_1|Y_0) \leq C(d_1)$.
Maximizing $I(\mathbf{z}_{\leq 1}; Y_0)$ subject to this constraint yields $I(\mathbf{z}_{\leq 1}; Y_0) > I(\mathbf{z}_{\leq 1}; Y_1|Y_0)$
when $H(Y_0) < C(d_1) < H(Y_1)$.

**Part (b):** The MRL prefix loss $\lambda \cdot \mathbb{E}[\ell(Y_1, \hat{Y}_1(\mathbf{z}_{\leq 1}))]$
incentivizes the prefix to encode $Y_1$, which includes both $Y_0$ and $Y_1|Y_0$ components. Under
the capacity constraint $C(d_1) < H(Y_1)$, the prefix allocates capacity proportionally to
reduce $H(Y_1 | \mathbf{z}_{\leq 1})$. Since $Y_0$ is a coarser version of $Y_1$, the prefix
naturally recovers $Y_0$ as a byproduct, but without preferential allocation. Hence
$I(\mathbf{z}_{\leq 1}; Y_0) \approx I(\mathbf{z}_{\leq J}; Y_0)$: both prefix and full encode $Y_0$
roughly equally well. No specialization emerges.

**Part (c):** Steerability $\mathcal{S}$ measures the asymmetry between prefix and full embedding.
From parts (a) and (b):
- V5: prefix specializes for $Y_0$, full for $Y_1$ → $\mathcal{S} > 0$
- Inverted: prefix specializes for $Y_1$, full for $Y_0$ → $\mathcal{S} < 0$
- MRL: no specialization → $\mathcal{S} \approx 0$ ∎

---

## 4. The Goldilocks Theorem (Capacity-Demand Matching)

### Theorem 2 (Steerability Peaks at Capacity-Demand Match)

*Under the setup of Theorem 1, with varying coarse partition $K_0$ (hence varying $H(Y_0) = \log_2 K_0$)
while holding $H(Y_1)$ and the text distribution $P(X)$ fixed:*

*The steerability $\mathcal{S}(K_0)$ is a non-monotonic function of $H(Y_0)$ with:*
1. *(Rising phase)* When $H(Y_0) \ll C(d_1)$: $\partial \mathcal{S}/\partial H(Y_0) > 0$.
   More coarse classes → richer prefix codebook → better coarse separation → higher $\mathcal{S}$.
2. *(Peak)* $\mathcal{S}$ peaks at $H^*(Y_0) \approx C(d_1)$, the effective prefix capacity.
3. *(Falling phase)* When $H(Y_0) \gg C(d_1)$: $\partial \mathcal{S}/\partial H(Y_0) < 0$.
   Too many coarse classes → prefix errors → degraded steerability.

*Corollary:* The optimal prefix dimension for a hierarchy with coarse entropy $H(Y_0)$
satisfies $d_{\text{opt}} \propto 2^{H(Y_0)}$.

### Proof Sketch

**Rising phase:** When $H(Y_0)$ is small (few coarse classes), the prefix can perfectly encode $Y_0$
with spare capacity. This spare capacity leaks $Y_1|Y_0$ information into the prefix, reducing
the specialization gap. As $H(Y_0)$ increases, the prefix dedicates more capacity to the coarse
task, reducing leakage and increasing $\mathcal{S}$.

Formally: $\mathcal{S} \propto \text{SpecGap} = I(\mathbf{z}_{\leq 1}; Y_0) - I(\mathbf{z}_{\leq 1}; Y_1|Y_0)$.
When $H(Y_0)$ is small, the prefix has capacity $C(d_1) - H(Y_0)$ available for $Y_1|Y_0$,
so $I(\mathbf{z}_{\leq 1}; Y_1|Y_0)$ is large and SpecGap is small.

**Falling phase:** When $H(Y_0) > C(d_1)$, the prefix cannot perfectly encode $Y_0$.
By Fano's inequality:
$$D_0(1) = P(\hat{Y}_0(\mathbf{z}_{\leq 1}) \neq Y_0) \geq \frac{H(Y_0) - C(d_1) - 1}{\log_2 K_0}$$

Coarse classification errors in the prefix reduce $I(\mathbf{z}_{\leq 1}; Y_0)$ and hence $\mathcal{S}$.

**Peak:** At $H(Y_0) \approx C(d_1)$, all prefix capacity is devoted to $Y_0$ (no leakage)
and $Y_0$ is still recoverable (no errors). This is the capacity-demand match point.

The quadratic approximation follows from Taylor expansion around $H^*$:
$$\mathcal{S}(H) \approx \mathcal{S}^* - \alpha(H - H^*)^2$$

which matches our empirical fit ($R^2 = 0.964$ on synthetic data). ∎

---

## 5. Rate-Distortion Connection

### Multi-Resolution Rate-Distortion

The fractal embedding problem maps naturally to the **multi-resolution source coding** framework
(Rimoldi, 1994; Effros, 1999):

- **Rate 1** (prefix): $R_1 = C(d_1)$ bits, distortion $D_0$ on coarse task
- **Rate 2** (full): $R_2 = C(d)$ bits, distortion $D_1$ on fine task

The rate-distortion region for successively refinable sources satisfies:
$$\mathcal{R} = \{(R_1, R_2, D_0, D_1): R_1 \geq R(D_0), R_1 + R_2 \geq R(D_0) + R_{\text{res}}(D_1|D_0)\}$$

where $R(D_0)$ is the rate-distortion function for $Y_0$ and $R_{\text{res}}$ is the residual rate.

### Fractal Embeddings as Learned Successive Refinement

**Claim:** V5 training approximates the successive refinement code for the hierarchical source:
- Block 1 encodes $Y_0$ at rate $R_1 \approx C(d_1)$
- Blocks 2-J encode the residual $Y_1|Y_0$ at rate $R_2 - R_1 \approx C(d) - C(d_1)$

MRL training approximates single-resolution coding at both rates:
- Block 1 encodes $Y_1$ at rate $R_1$ (lossy)
- Full encodes $Y_1$ at rate $R_2$ (less lossy)

The successive refinement code achieves:
$$D_0^{\text{V5}}(R_1) \leq D_0^{\text{MRL}}(R_1) \text{ and } D_1^{\text{V5}}(R_2) \approx D_1^{\text{MRL}}(R_2)$$

That is, V5 achieves better coarse distortion at the prefix rate while maintaining the same
fine distortion at the full rate — matching our empirical finding of "accuracy parity + steerability gains."

### Prediction: When Does V5 Fail?

The theory predicts V5 ≈ MRL when:
1. $H(Y_0) \approx 0$: trivial coarse hierarchy (e.g., binary L0)
2. $H(Y_1|Y_0) \approx 0$: hierarchy is degenerate (L0 ≈ L1)
3. $C(d_1) \gg H(Y_1)$: prefix can encode everything (no bottleneck)
4. $C(d_1) < H(Y_0)$ and $H(Y_0) \gg H(Y_1|Y_0)$: prefix overwhelmed by coarse task

Our empirical boundary conditions match: DBPedia (ceiling), Yahoo (shallow hierarchy).

---

## 6. Empirical Predictions from Theory

The theory makes testable predictions beyond our current data:

1. **Prefix dimension sensitivity:** Doubling prefix dimension should shift the Goldilocks peak
   rightward (to higher $K_0$). Testable via capacity sweep ablation.

2. **Deeper hierarchies (3+ levels):** With L0, L1, L2 and three prefix groups,
   steerability should decompose: S01 from blocks 1-2, S12 from blocks 3-4.

3. **Continuous hierarchies:** For taxonomic trees with varying depth, the optimal prefix
   allocation should follow the entropy decomposition at each level.

4. **Cross-modal transfer:** The theory is modality-agnostic — same hierarchy structure
   in vision (CIFAR-100 superclasses) or audio should produce the same steerability pattern.

5. **Fine-grained Goldilocks:** With more data points on the synthetic curve, the peak
   should track $C(d_1)$, and doubling $d_1$ should shift the peak.

---

## 7. Connection to Information Bottleneck

The V5 objective can be rewritten as a **hierarchical information bottleneck**:

$$\min_{f} \left[ H(Y_1 | \mathbf{z}) + \lambda H(Y_0 | \mathbf{z}_{\leq 1}) + \beta I(X; \mathbf{z}) \right]$$

The first term forces the full embedding to encode $Y_1$. The second forces the prefix to encode $Y_0$.
The third (implicit, via limited dimensionality) constrains total information.

This is a special case of the **multi-task information bottleneck** where:
- Task 1 (prefix): compress $X$ to predict $Y_0$
- Task 2 (full): compress $X$ to predict $Y_1$
- Tasks share an encoder but operate at different rates

The hierarchy constraint $Y_0 = g(Y_1)$ makes this bottleneck **successively decodable**:
the prefix information is a necessary input for the full decoding.

### Lagrangian Formulation

$$\mathcal{L} = I(\mathbf{z}; Y_1) + \lambda I(\mathbf{z}_{\leq 1}; Y_0) - \beta I(X; \mathbf{z}) - \beta_0 I(X; \mathbf{z}_{\leq 1})$$

At the optimum (saddle point of the Lagrangian):
$$\frac{\partial I(\mathbf{z}_{\leq 1}; Y_0)}{\partial R_1} = \lambda^{-1} \frac{\partial I(\mathbf{z}; Y_1)}{\partial R_1}$$

This is the **water-filling** condition: marginal information about $Y_0$ from the prefix
equals $\lambda^{-1}$ times the marginal information about $Y_1$ from the prefix.
Since $\lambda > 0$ (V5 has nonzero prefix supervision), the prefix prioritizes $Y_0$.

---

## 8. Open Problems (Toward Fields-Medal-Level Math)

1. **Distribution-free characterization:** Under what conditions on $P(X, Y_0, Y_1)$ is
   the hierarchical source successively refinable under nonlinear deep encoders?
   (Current results assume Bayes-optimal decoders.)

2. **Finite-sample rates:** What is the sample complexity of learning the successive
   refinement code? How does it depend on $H(Y_0)$, $H(Y_1|Y_0)$, and $d$?

3. **Phase transition:** Is there a sharp phase transition in steerability at $H(Y_0) = C(d_1)$?
   Can we prove a "threshold phenomenon" analogous to those in random coding theory?

4. **Optimal prefix allocation for trees:** Given a tree-structured hierarchy with $L$ levels,
   what is the optimal allocation of dimensions to each level?
   Conjecture: $d_\ell \propto H(Y_\ell | Y_{\ell-1})$ (entropy-proportional allocation).

5. **Universal successive refinement:** Does there exist a single encoder that is optimal
   for ALL hierarchies over a given text distribution? (Universal coding analog.)

---

## References

- Equitz, W.H.R. & Cover, T.M. (1991). Successive Refinement of Information. IEEE Trans. IT.
- Rimoldi, B. (1994). Successive Refinement of Information: Characterization of Achievable Rates. IEEE Trans. IT.
- Effros, M. (1999). Distortion-Rate Bounds for Fixed- and Variable-Rate Multiresolution Source Codes. IEEE Trans. IT.
- Tishby, N., Pereira, F., & Bialek, W. (2000). The Information Bottleneck Method.
- Kusupati, A. et al. (2022). Matryoshka Representation Learning. NeurIPS.
