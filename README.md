# SU(2) Cat Probes in Mach–Zehnder Interferometry

## Matricial Bayes risk vs matricial information bounds with $\hat J_z$ readout

This repository contains code to numerically study the problem of **point
estimation** in a two-parameter Mach–Zehnder interferometer model implemented on
a fixed spin - $j$ representation of $SU(2)$. The main focus is the comparison
between:

- the **Bayes risk matrix** of the posterior-mean estimator,
- a **coarse-grained Cramér–Rao benchmark** (CG-CRB) averaged over a local
  neighbourhood prior (Jeffreys/Fisher–Rao weighting), and
- a **van Trees** (Bayesian CRB) lower bound.

The analysis is tailored to regimes where the Fisher information becomes
**ill-conditioned or singular**, especially near symmetry-induced degeneracies
such as $\phi=\pi$.

## Mathematical context

### Physical model

We work in a fixed spin-$j$ irreducible representation $\mathcal{H}_j$ with
angular momentum operators $\hat J_x$, $\hat J_y$, and $\hat J_z$.

The interferometer is modeled in axis–angle form as
$$R(\beta,\phi)=\exp\left(-i\phi\bigl(\cos\beta\,\hat J_z+\sin\beta\,\hat J_y\bigr)\right),$$
where $\theta=(\beta,\phi)$ is the unknown parameter and $\beta\in[0,\pi)$,
$\phi\in[0,2\pi)$.

### Probe: spin-cat state

The input is a “cat” state in the $\hat J_z$ basis:
$$\ket{\psi_{\mathrm{in}}}=\frac{1}{\sqrt2}\Bigl(\ket{j,m_c}+e^{i\chi}\ket{j,-m_c}\Bigr),$$
with fixed design parameters $j$, $m_c$, and $\chi$.

### Measurement: $\hat J_z$ projective readout

We measure in the $\hat J_z$ eigenbasis $\set{\lvert j,m\rangle}_{m=-j}^{j}.$ 

The induced pmf is $$p(m\mid \beta,\phi) = \left|\langle j,m \mid R(\beta,\phi)\mid \psi_{\mathrm{in}}\rangle\right|^2.$$

### Fisher information

For $\theta=(\beta,\phi)$, the Fisher information matrix is

$$I(\theta)_{ab} = \sum_m \frac{\partial_a p(m\mid\theta)\,\partial_b p(m\mid\theta)}{p(m\mid\theta)}$$,

for $a,b \in \set{\beta,\phi}$.

## What the code computes

For each sample size $n$, the script computes:

### 1) Bayes estimator and Bayes risk matrix

Using a neighbourhood prior $\pi(\theta)$, the Bayes estimator under squared
loss is the posterior mean $$\hat\theta_B(X^n)=\mathbb E[\theta\mid X^n],$$
computed numerically on the same grid used to define $\pi$. The Bayes risk
matrix is estimated by Monte Carlo:

$$R_n = \mathbb E_{\theta\sim\pi},\mathbb E_{X^n\sim P_\theta^{\otimes n}}
\bigl[(\hat\theta_B-\theta)(\hat\theta_B-\theta)^{\mathsf T}\bigr]$$.


### 2) Coarse-grained CRB benchmark (CG-CRB)

This is the prior average of the local inverse Fisher information:
$$B_{CG}(n)=\frac{1}{n},\mathbb E_{\theta\sim\pi}[I(\theta)^{-1}],$$ where the
inverse is taken pointwise on a “regularized” set (grid points must satisfy
support/conditioning filters).

+ **Interpretation:** $$B_{CG}$$ is a *geometric benchmark* summarizing typical
  local conditioning of $$I(\theta)^{-1}$$ over the region where inference is
  intended to operate.

### 3) van Trees (Bayesian CRB)

A matrix lower bound on Bayes risk: $$B_{VT}(n) = \bigl(n,\mathbb
E_{\pi}[I(\theta)] + I_\pi\bigr)^{-1},$$ where $$I_\pi = \mathbb
E_\pi\bigl[(\nabla\log\pi)(\nabla\log\pi)^{\mathsf T}\bigr]$$ is the prior
Fisher information. In the code, $I_\pi$ is estimated via finite differences on
the discrete grid representation of $\pi$.

## Neighbourhood priors, unwrapped phase, and tapering

All computations are carried out on the same local chart
$\theta=(\beta,\phi_u)$, where $\phi_u$ is an **unwrapped** phase coordinate
centered at $\phi_0$: $$\phi_u = \phi_0 +
\mathrm{wrap}(\phi-\phi_0)\in(\phi_0-\pi,\phi_0+\pi]$$.

We define neighbourhoods around $\theta_0=(\beta_0,\phi_0)$ in the
$(\beta,\phi_u)$ chart:

+ **ball:** $r < \epsilon_2$
+ **ring:** $\epsilon_1 < r < \epsilon_2$
+ **half-ball / half-ring:** additionally restrict to $\phi_u\ge \phi_0$ or
  $\phi_u\le \phi_0$ to break symmetry near $\phi=\pi$.

The Fisher–Rao metric (Jeffreys) weight is $\sqrt{\det I(\theta)}$ on the
regular set, and we optionally apply a smooth taper function $w_R$ to avoid hard
truncation effects, so that,
$$\pi(\theta)\ \propto\ \sqrt{\det I(\theta)},
w_R(\theta).$$

Supported taper options:

+ `none` (hard truncation),
+ `bump` ((C^\infty) bump),
+ `poly` (polynomial taper (\propto (1-\rho^2)^k)).

---

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── scripts/
│   └── matricial_risk_vs_bounds.py
```
---

## Installation

### Python environment

Tested with Python 3.10+ (anything compatible with QuTiP should work).

### Install dependencies:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt`:

```txt
numpy
pandas
matplotlib
qutip
```

---

## Quick start

The main entry point is the function:

* `study_matricial_risk_vs_bounds(...)`

and there are example runs at the bottom of the script under `if __name__ ==
"__main__":`.

### Example A: regular point (ball)

Set `theta0=(π/3, π/4)` and use a ball neighbourhood (or `neighborhood_mode="auto"`), then run:

```bash
python scripts/matricial_risk_vs_bounds.py
```

### Example B: singular point near (\phi=\pi) (half-ring)

The included example uses:

* `theta0=(π/3, π)`,
* `neighborhood_mode="half_ring"`,
* `half_plane="phi_ge"`,
* two inner radii `eps1_list=[0.01, 0.05]` (fixed `eps2=0.10`),

to study how excising the immediate neighbourhood of the singular locus affects convergence.

---

## Outputs

Each run writes an output directory (timestamped unless you set `output_root`) containing:

* `config.txt` (all parameters)
* per-(n) folders `n_XXXX/` with:

  * `bound_cg_matrix.txt`, `bound_vt_matrix.txt`
  * `risk_matrix_mean.txt`, `risk_matrix_se.txt`
  * Monte Carlo replicate folders `mc_YYY/` with samples, errors, optional posterior plots
* summary CSV:

  * `summary_matricial.csv`
* plots:

  * trace comparisons (linear/log-y)
  * eigenvalue comparisons (min/max, linear/log-y)
  * PSD diagnostics: mean $\lambda_{\min}(R_n - B)$

---

## Reproducibility and numerical details

* **Grid-based computation:** the prior and the posterior are computed on the *same* $(\beta,\phi_u)$ grid.
* **Regularity filters:** grid points are discarded if:

  * Fisher information is non-finite,
  * $\det I(\theta)\le \mathrm{det_tol}$,
  * or the pmf has near-zero support mass (controlled by `eps_support`).
* **Monte Carlo design:** outer replicates (`m_batch`) × inner experiments (`r_batch`) estimate $R_n$ with uncertainty.
* **Phase treatment:** phase is handled on the unwrapped chart for inference; phase errors can be wrapped for reporting.

---




