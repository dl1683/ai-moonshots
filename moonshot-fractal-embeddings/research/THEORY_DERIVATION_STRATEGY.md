# Theoretical Derivation Strategy: Spectral Phase Transition Law

**Designed by Codex (GPT-5.3-codex, xhigh reasoning), Feb 20 2026**
**Confidence: 6/10 mathematical program, 9/10 Nobel-potential if completed**

## Common Asymptotic Setup

All theorems use one model:
- X | Y=k ~ N(mu_k, Sigma_W), Pr(Y=k) = 1/K, n train points/class
- S_B = sum_k (mu_k - mu_bar)(mu_k - mu_bar)^T
- S_W = sum_k Sigma_W
- kappa = tr(S_B) / tr(S_W)
- eta = tr(S_W)^2 / (d * tr(S_W^2))
- Regime: d, n, K -> infinity, log(n) = o(d), balanced classes

---

## Theorem 1: Sigmoid Law for Isotropic Gaussian Clusters (Confidence: 7/10)

**Statement**: If Sigma_W = sigma^2 * I_d, class means are asymptotically isotropic, then:
```
q = (Acc_1NN - 1/K) / (1 - 1/K) = sigmoid(a * kappa/sqrt(K) + c) + epsilon
```
where epsilon -> 0.

**Proof Strategy**:
1. Write classwise nearest distances M_k = min_i ||X - X_{k,i}||^2
2. For true class: central chi^2 tail; wrong class: noncentral chi^2 tail
3. Poissonization of lower-tail exceedances gives Gumbel limits for M_y and M_j
4. Joint margin process M_j - M_y has dominant common mode + idiosyncratic mode (equicorrelated)
5. Correct classification {M_y < min_{j != y} M_j} = difference of two extreme-value variables
6. **KEY INSIGHT**: Difference of two Gumbels -> LOGISTIC distribution -> SIGMOID CDF
7. Identify location term with kappa, scale term with sqrt(K)

**Key Lemmas**:
- L1: Uniform saddlepoint approximation for central/noncentral chi^2 left tails
- L2: Uniform Gumbel convergence of classwise minima in k
- L3: Dependence-decoupling lemma for shared-query randomness
- L4: Trace-to-margin lemma: mean margin depends on S_B, S_W only through kappa
- L5: Extreme-difference lemma: difference of aligned Gumbels gives logistic CDF

---

## Theorem 2: kappa is Sufficient Statistic (Confidence: 8/10)

**Statement**: For any parameterizations theta, theta' with same kappa:
```
|Acc_1NN(theta) - Acc_1NN(theta')| <= r_{d,n,K} -> 0
```

**Proof Strategy**:
1. Orthogonal invariance reduces dependence to spectral invariants of S_B
2. Higher spectral-shape terms enter margins only as o(1) under concentration
3. Prove contiguity (Le Cam style) of experiments with same kappa
4. Transfer contiguity to kNN risk functional

**Key Lemmas**:
- L1: Orthogonal invariance of distance process
- L2: Hanson-Wright bounds for quadratic-form fluctuations
- L3: Spectral-shape remainder bound
- L4: kNN risk is Lipschitz in pairwise-distance empirical process

---

## Theorem 3: Why sqrt(K) (Confidence: 6/10)

**Statement**: Transition width in kappa-space grows like sqrt(K):
```
logit(q) = a * kappa/sqrt(K) + c + o(1)
```

**Proof Strategy**:
1. Write margin vector Z in R^{K-1}, Z_j = M_j - M_y
2. Decompose covariance: Cov(Z) = lambda_1 * P_1 + lambda_2 * (I - P_1)
   where lambda_1 ~ K, lambda_2 ~ 1
3. Classification event min_j Z_j > 0 controlled by projection on all-ones mode
4. Effective noise scale ~ sqrt(K), signal ~ kappa, yielding kappa/sqrt(K)

**Key Insight**: The equicorrelated structure of multiclass margins means the
common mode (all-ones direction) dominates near the decision boundary.
Its eigenvalue scales as K, so the noise standard deviation scales as sqrt(K).

---

## Theorem 4: Anisotropy Correction via eta (Confidence: 7/10 asymptotic, 5/10 finite)

**Statement**: For general Sigma_W with bounded condition number:
```
logit(q) = a * kappa * eta^{1/2} / sqrt(K) + c + delta
|delta| <= C * ((1-eta)^2 + d^{-1/2})
```
Empirically: q ~ sigmoid(a * kappa * eta^b + c), b ~ 0.3 at finite (d,n,K).

**Proof Strategy**:
1. E||u-v||^2 = 2*tr(Sigma_W), Var(||u-v||^2) = 8*tr(Sigma_W^2)
2. Replace d by d_eff = d*eta in concentration/EV scales
3. Repeat Theorem 1 with d_eff: slope scales as sqrt(eta)
4. Finite-sample correction from second-order EV terms gives b ~ 0.3

---

## Theorem 5: Universality (Confidence: 4/10)

**Statement**: For wide networks under standard conditions, penultimate features satisfy:
```
W_2(L(Z|Y=k), N(mu_k, Sigma_k)) <= epsilon -> 0
```
Therefore Theorems 1-4 apply to trained network representations.

**Proof Strategy**:
1. Feature CLT at width m -> infinity
2. SGD diffusion gives covariance homogenization
3. Cross-entropy drives class means toward simplex/ETF (neural collapse attractor)
4. Wasserstein stability of kNN risk transfers GMM predictions to network features

---

## Hardest Steps
1. Joint extreme-value limit for dependent noncentral distance minima (Thm 1/3 core)
2. Rigorous proof that only kappa survives (not higher spectral descriptors) (Thm 2)
3. Full universality bridge from SGD-trained features to Gaussian mixture (Thm 5)

## Implementation Priority
1. **Theorem 1 + numerical verification** (START HERE - most impactful, most provable)
2. Theorem 3 (the sqrt(K) explanation is the most novel contribution)
3. Theorem 2 (sufficiency of kappa - straightforward with Hanson-Wright)
4. Theorem 4 (eta correction - good but less novel)
5. Theorem 5 (universality - defer, cite existing CLT results)
