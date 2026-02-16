# Proof: kNN Classification Error Upper Bound via Fisher Q

## Theorem (Upper Bound)

Let Z_1, ..., Z_n be i.i.d. random variables in R^d with labels Y_1, ..., Y_n in {1,...,C}.
Assume:
(A1) Class-conditional distributions: Z|Y=c ~ sub-Gaussian(mu_c, sigma^2 I_d)
(A2) Balanced classes: P(Y=c) = 1/C
(A3) Normalized representations: ||Z|| = 1 a.s. (or approximately, after projection)

Define the multivariate Fisher ratio:
  Q = tr(Sigma_B) / tr(Sigma_W)

where:
  Sigma_B = (1/C) sum_c (mu_c - mu)(mu_c - mu)^T  [between-class scatter]
  Sigma_W = (1/C) sum_c sigma_c^2 I_d = sigma^2 I_d  [within-class scatter, homoscedastic]
  mu = (1/C) sum_c mu_c  [grand mean]

Then Q = (1/(C*sigma^2)) sum_c ||mu_c - mu||^2.

## Step 1: Bayes-Optimal Error

For the nearest-centroid classifier (asymptotic kNN limit):
  P(error | Z=z, Y=c) = P(exists c' != c : ||z - mu_{c'}|| < ||z - mu_c||)

By union bound:
  P(error | Y=c) <= sum_{c'!=c} P(||Z - mu_{c'}|| < ||Z - mu_c|| | Y=c)

## Step 2: Pairwise Error Probability

For classes c and c', the event ||Z - mu_{c'}|| < ||Z - mu_c|| is equivalent to:
  <Z, mu_c - mu_{c'}> < (||mu_c||^2 - ||mu_{c'}||^2) / 2

Define Delta_{cc'} = mu_c - mu_{c'} and t_{cc'} = (||mu_c||^2 - ||mu_{c'}||^2) / 2.

Then: P(error between c and c') = P(<Z, Delta_{cc'}> < t_{cc'} | Y=c)

Since Z|Y=c ~ sub-Gaussian(mu_c, sigma^2 I):
  <Z, Delta_{cc'}> | Y=c ~ sub-Gaussian(<mu_c, Delta_{cc'}>, sigma^2 ||Delta_{cc'}||^2)

The mean of <Z, Delta_{cc'}> given Y=c is:
  <mu_c, mu_c - mu_{c'}> = ||mu_c||^2 - <mu_c, mu_{c'}>

And t_{cc'} = (||mu_c||^2 - ||mu_{c'}||^2) / 2

So the "margin" (distance from mean to threshold) is:
  m_{cc'} = <mu_c, Delta_{cc'}> - t_{cc'}
           = ||mu_c||^2 - <mu_c, mu_{c'}> - ||mu_c||^2/2 + ||mu_{c'}||^2/2
           = ||mu_c||^2/2 - <mu_c, mu_{c'}> + ||mu_{c'}||^2/2
           = ||mu_c - mu_{c'}||^2 / 2
           = ||Delta_{cc'}||^2 / 2

And the variance is sigma^2 ||Delta_{cc'}||^2.

## Step 3: Sub-Gaussian Tail Bound

P(error between c and c') <= exp(-m_{cc'}^2 / (2 * sigma^2 * ||Delta_{cc'}||^2))
                            = exp(-||Delta_{cc'}||^4 / (8 * sigma^2 * ||Delta_{cc'}||^2))
                            = exp(-||Delta_{cc'}||^2 / (8 * sigma^2))

## Step 4: Union Bound

P(error | Y=c) <= sum_{c'!=c} exp(-||mu_c - mu_{c'}||^2 / (8*sigma^2))

Taking expectation over Y:
P(error) = (1/C) sum_c sum_{c'!=c} exp(-||mu_c - mu_{c'}||^2 / (8*sigma^2))

## Step 5: Connect to Fisher Q

By Cauchy-Schwarz / AM-GM:
||mu_c - mu_{c'}||^2 >= (2/C) * sum_c ||mu_c - mu||^2 [for well-separated centroids]

More precisely, for balanced classes:
(1/C(C-1)) sum_{c!=c'} ||mu_c - mu_{c'}||^2 = (2/C) * tr(Sigma_B)

So the MINIMUM pairwise distance satisfies:
min_{c!=c'} ||mu_c - mu_{c'}||^2 >= ... [depends on centroid geometry]

For the WORST CASE (two closest centroids):
Delta_min^2 = min_{c!=c'} ||mu_c - mu_{c'}||^2

P(error) <= (C-1) * exp(-Delta_min^2 / (8*sigma^2))

And since tr(Sigma_B) = sigma^2 * Q * C (from definition of Q):
Delta_min^2 >= (2/C) * tr(Sigma_B) - max deviation
            >= 2*sigma^2*Q - correction

For a simplex arrangement (Neural Collapse endpoint):
Delta_min^2 = (2C/(C-1)) * ||mu_c - mu||^2 = (2/(C-1)) * C * ||mu_c - mu||^2
            = (2/(C-1)) * tr(Sigma_B) / (1/C)  [if all centroids equidistant]
            = 2C/(C-1) * sigma^2 * Q

So: P(error) <= (C-1) * exp(-C*Q / (4*(C-1)))

## Final Result

**Theorem**: Under (A1)-(A3), for the nearest-centroid classifier:

P(error) <= (C-1) * exp(-Q / (4*(1-1/C)))

where Q = tr(Sigma_B) / tr(Sigma_W) is the multivariate Fisher ratio.

For large C: P(error) <= (C-1) * exp(-Q/4)

**Corollary** (kNN with n samples):
For k-NN with n training samples, by Cover-Hart (1967) and finite-sample corrections:

P_kNN(error) <= P_Bayes(error) + O(sqrt(k/n)) + O(n^{-2/(d+2)})
             <= (C-1)*exp(-Q/4) + O(sqrt(k/n)) + O(n^{-2/(d+2)})

## Discussion

1. This bound is EXPONENTIAL in Q. Doubling Q roughly squares the error reduction.
2. The bound assumes homoscedastic, sub-Gaussian class-conditional distributions.
3. For real representations, this may be approximate but the MONOTONIC relationship
   between Q and error should hold robustly.
4. The simplex arrangement (Neural Collapse) maximizes Q for fixed total variance,
   confirming that NC is optimal in our framework.

## Novelty Assessment

What's known:
- Fisher discriminant and its connection to classification (Fisher 1936)
- Sub-Gaussian tail bounds (Hoeffding, etc.)
- Cover-Hart kNN asymptotics (1967)
- Neural Collapse (Papyan 2020)

What's new here:
- Explicit error bound in terms of MULTIVARIATE Fisher Q (not just 2-class LDA)
- Connection to Neural Collapse (NC = Q -> optimal)
- Integration with kNN finite-sample rates
- Framework for CONTROLLABILITY (next proof)

Rigor: 5/10 (still has handwavy steps, esp. the Delta_min -> Q connection)
The simplex assumption is strong; real centroids may be arbitrarily arranged.
Need: proper handling of arbitrary centroid configurations.
