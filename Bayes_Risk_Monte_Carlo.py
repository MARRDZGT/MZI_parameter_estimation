"""
Matricial Bayes risk vs matricial bounds for SU(2) cat probes with Jz readout.

This script computes, for each sample size n:
  - Bayes risk matrix R_n = E[(theta_hat - theta)(theta_hat - theta)^T]
  - CG-CRB matrix B_CG(n) = (1/n) E_pi[ I(theta)^(-1) ]
  - Van Trees (simple) matrix B_VT(n) = ( n E_pi[I(theta)] + I_pi )^(-1)

All computations are carried out on the SAME local chart:
  theta = (beta, phi_u), where phi_u is unwrapped around phi0.

Outputs
-------
For each n:
  - Monte Carlo estimates of R_n with uncertainty (outer replicates)
  - Deterministic bound matrices B_CG(n), B_VT(n)
  - Comparisons via trace and eigenvalues
  - Diagnostics for PSD ordering (min eigenvalue of R_n - bound)

Plots (saved):
  - Trace comparison: linear-y and log-y
  - Eigenvalue comparisons (lambda_min and lambda_max): linear-y and log-y
  - PSD diagnostic plots (optional; enabled by default)

Dependencies: numpy, pandas, matplotlib, qutip
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---- Robust backend selection (prevents Qt errors on headless machines)
import matplotlib

if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qutip as qt


TWO_PI = 2.0 * math.pi
PI = math.pi

NeighborhoodType = Literal["ball", "ring", "half_ball", "half_ring"]
NeighborhoodMode = Literal["auto", "ball", "ring", "half_ball", "half_ring"]
HalfPlane = Literal["phi_ge", "phi_le"]
TaperType = Literal["none", "bump", "poly"]

# =============================================================================
# Utilities
# =============================================================================
def wrap_phi(phi: float) -> float:
    """Map angle to [0, 2π)."""
    return float(phi % TWO_PI)


def wrap_angle(angle: float) -> float:
    """Map angle to (-π, π]."""
    return float((angle + PI) % TWO_PI - PI)


def unwrap_relative(phi: float, phi_ref: float) -> float:
    """
    Unwrap phi to the local chart centered at phi_ref:
      phi_u = phi_ref + wrap_angle(phi - phi_ref).
    """
    return float(phi_ref + wrap_angle(phi - phi_ref))


def det_2x2(a: np.ndarray) -> np.ndarray:
    """Determinant of (..., 2, 2) array."""
    return a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]


def inv_2x2(a: np.ndarray, det: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Inverse of (..., 2, 2) array for invertible matrices.
    Uses the closed-form 2x2 inverse (fast and vectorizable).
    """
    if det is None:
        det = det_2x2(a)
    inv = np.empty_like(a, dtype=float)
    inv[..., 0, 0] = a[..., 1, 1] / det
    inv[..., 1, 1] = a[..., 0, 0] / det
    inv[..., 0, 1] = -a[..., 0, 1] / det
    inv[..., 1, 0] = -a[..., 1, 0] / det
    return inv


def eigvals_psd_2x2(a: np.ndarray) -> np.ndarray:
    """Sorted eigenvalues (ascending) of a symmetric 2x2 matrix."""
    a = 0.5 * (a + a.T)
    return np.linalg.eigvalsh(a)


def safe_logsumexp(x: np.ndarray) -> float:
    """Stable log-sum-exp for 1D array."""
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def theta_distance(beta: np.ndarray, phi_u: np.ndarray, beta0: float, phi0: float) -> np.ndarray:
    """Euclidean distance in (beta, phi_u) coordinates on the local chart."""
    return np.sqrt((beta - beta0) ** 2 + (phi_u - phi0) ** 2)


def _taper_outer_bump(r: np.ndarray, eps2: float, alpha: float) -> np.ndarray:
    """C-infinity bump supported on r < eps2."""
    x = r / eps2
    out = np.zeros_like(r, dtype=float)
    inside = x < 1.0
    z = 1.0 - x[inside] ** 2
    out[inside] = np.exp(-alpha / z)
    return out


def _taper_inner_bump(r: np.ndarray, eps1: float, alpha: float) -> np.ndarray:
    """C-infinity 'bump' that is 0 for r <= eps1 and >0 for r > eps1."""
    out = np.zeros_like(r, dtype=float)
    outside = r > eps1
    x = eps1 / r[outside]
    z = 1.0 - x ** 2
    out[outside] = np.exp(-alpha / z)
    return out


def _taper_outer_poly(r: np.ndarray, eps2: float, k: int) -> np.ndarray:
    """Polynomial taper supported on r < eps2."""
    x2 = (r / eps2) ** 2
    out = np.maximum(0.0, 1.0 - x2) ** k
    return out


def _taper_inner_poly(r: np.ndarray, eps1: float, k: int) -> np.ndarray:
    """Polynomial taper that is 0 for r <= eps1 and >0 for r > eps1."""
    out = np.zeros_like(r, dtype=float)
    outside = r > eps1
    x2 = (eps1 / r[outside]) ** 2
    out[outside] = np.maximum(0.0, 1.0 - x2) ** k
    return out


def taper_weight(
    r: np.ndarray,
    *,
    eps1: float,
    eps2: float,
    neighborhood: str,
    taper_type: TaperType,
    use_taper: bool,
    bump_alpha: float = 1.0,
    poly_k: int = 4,
) -> np.ndarray:
    """
    Return w(r) used in pi(theta) ∝ sqrt(det FI(theta)) * w(r).

    Note: for half_ball / half_ring, this function returns ONLY the radial part.
    The half-plane truncation is applied in build_cg_cache.
    """
    r = np.asarray(r, dtype=float)

    if neighborhood not in ("ball", "ring", "half_ball", "half_ring"):
        raise ValueError("neighborhood must be one of 'ball','ring','half_ball','half_ring'")

    # Map half_* to its radial base shape
    base = neighborhood
    if neighborhood.startswith("half_"):
        base = neighborhood.split("_", 1)[1]  # "ball" or "ring"

    if not use_taper or taper_type == "none":
        if base == "ball":
            return (r < eps2).astype(float)
        return ((r > eps1) & (r < eps2)).astype(float)

    if taper_type == "bump":
        w_out = _taper_outer_bump(r, eps2, bump_alpha)
        if base == "ball":
            return w_out
        w_in = _taper_inner_bump(r, eps1, bump_alpha)
        return w_in * w_out

    if taper_type == "poly":
        if poly_k < 1:
            raise ValueError("poly_k must be >= 1")
        w_out = _taper_outer_poly(r, eps2, poly_k)
        if base == "ball":
            return w_out
        w_in = _taper_inner_poly(r, eps1, poly_k)
        return w_in * w_out

    raise ValueError(f"Unknown taper_type={taper_type!r}")

# =============================================================================
# Your physical model: SU(2) cat + Jz readout
# =============================================================================
def spin_m_values(j: float) -> np.ndarray:
    """Return m values in the Jz basis: [-j, -j+1, ..., j]."""
    two_j = int(round(2.0 * j))
    if abs(two_j - 2.0 * j) > 1e-12:
        raise ValueError(f"j must be integer or half-integer, got j={j}")
    return np.arange(-two_j, two_j + 1, 2, dtype=int) / 2.0


@dataclass(frozen=True)
class SU2CatJzModel:
    """
    MZI axis-angle interferometer with fixed cat probe and Jz readout.

    Probe: |psi> = (|m_c> + e^{i chi} |-m_c>)/sqrt(2)
    Readout: projective in Jz eigenbasis {|j,m>}.

    Parameters
      j   spin (integer or half-integer)
      m_c cat component index (admissible m)
      chi relative phase
    """

    j: float
    m_c: float
    chi: float = 0.0

    def __post_init__(self) -> None:
        m_vals = spin_m_values(self.j)
        if float(self.m_c) not in set(map(float, m_vals)):
            raise ValueError(f"m_c={self.m_c} not admissible for j={self.j}")

        object.__setattr__(self, "_m_vals", m_vals)
        object.__setattr__(self, "_dim", len(m_vals))
        object.__setattr__(self, "_m_to_idx", {float(m): i for i, m in enumerate(m_vals)})

        jx = qt.jmat(self.j, "x")
        jy = qt.jmat(self.j, "y")
        jz = qt.jmat(self.j, "z")
        object.__setattr__(self, "_Jx", jx)
        object.__setattr__(self, "_Jy", jy)
        object.__setattr__(self, "_Jz", jz)

        ket_mc = qt.basis(self._dim, self._m_to_idx[float(self.m_c)])
        ket_mcm = qt.basis(self._dim, self._m_to_idx[float(-self.m_c)])
        psi_in = (ket_mc + np.exp(1j * self.chi) * ket_mcm).unit()
        object.__setattr__(self, "_psi_in", psi_in)

    @property
    def dim(self) -> int:
        return self._dim

    def R_axis_angle(self, beta: float, phi: float) -> qt.Qobj:
        """R(β,φ) = exp(-i φ (cosβ Jz + sinβ Jy))."""
        a = math.cos(beta) * self._Jz + math.sin(beta) * self._Jy
        return (-1j * float(phi) * a).expm()

    def state_out(self, beta: float, phi: float) -> qt.Qobj:
        """|psi_{β,φ}> = R(β,φ)|psi_in>."""
        return self.R_axis_angle(float(beta), float(wrap_phi(phi))) * self._psi_in

    def pmf(self, beta: float, phi: float) -> np.ndarray:
        """p_m(β,φ) = |<m|psi_{β,φ}>|^2 in Jz basis ordering."""
        psi = self.state_out(beta, phi)
        amps = psi.full().reshape(-1)
        p = np.abs(amps) ** 2
        s = float(p.sum())
        if s <= 0.0:
            raise RuntimeError("PMF sum non-positive (numerical failure).")
        return p / s

    def fisher_information(
        self,
        beta: float,
        phi: float,
        *,
        eps_support: float = 1e-15,
        return_extras: bool = False,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Classical Fisher information I(θ) for θ=(β,φ) using a compact factorization.

        I(θ) = 4 N(θ)^T C(θ) N(θ),
        where u_m = Im( <m|J|ψθ> / <m|ψθ> ), C = Σ p_m u_m u_m^T.
        """
        beta = float(beta)
        phi = float(wrap_phi(phi))

        psi = self.state_out(beta, phi)
        amps = psi.full().reshape(-1)
        p = np.abs(amps) ** 2
        p_sum = float(p.sum())
        if p_sum <= 0.0:
            raise RuntimeError("PMF sum non-positive (numerical failure).")
        p = p / p_sum

        if np.any(p <= eps_support):
            i_mat = np.full((2, 2), np.nan, dtype=float)
            extras = {"p": p, "amps": amps} if return_extras else None
            return i_mat, extras

        jx_psi = (self._Jx * psi).full().reshape(-1)
        jy_psi = (self._Jy * psi).full().reshape(-1)
        jz_psi = (self._Jz * psi).full().reshape(-1)

        w_x = jx_psi / amps
        w_y = jy_psi / amps
        w_z = jz_psi / amps
        u = np.stack([np.imag(w_x), np.imag(w_y), np.imag(w_z)], axis=1)  # (dim,3)

        c_mat = (u.T * p) @ u  # 3x3

        n_phi = np.array([0.0, math.sin(beta), math.cos(beta)], dtype=float)
        n_beta = np.array(
            [
                -(1.0 - math.cos(phi)),
                math.sin(phi) * math.cos(beta),
                -math.sin(phi) * math.sin(beta),
            ],
            dtype=float,
        )
        n_mat = np.column_stack([n_beta, n_phi])  # (3,2)

        i_mat = 4.0 * (n_mat.T @ c_mat @ n_mat)

        extras = None
        if return_extras:
            extras = {"p": p, "amps": amps}
        return i_mat, extras


# =============================================================================
# Coarse-grained grid cache (Jeffreys prior + likelihood cache)
# =============================================================================
@dataclass(frozen=True)
class CGCache:
    # neighborhood meta
    beta0: float
    phi0: float
    eps1: float
    eps2: float
    neighborhood: NeighborhoodType
    half_plane: Optional[str]

    # grid axes and spacings
    beta_vals: np.ndarray              # (Nb,)
    phi_u_vals: np.ndarray             # (Np,)
    phi_w_vals: np.ndarray             # (Np,)
    d_beta: float
    d_phi: float
    cell_area: float

    # full grids
    beta_mesh: np.ndarray              # (Nb,Np)
    phi_u_mesh: np.ndarray             # (Nb,Np)
    phi_w_mesh: np.ndarray             # (Nb,Np)

    # validity mask and Jeffreys weights (full grid)
    valid_mask: np.ndarray             # (Nb,Np) bool
    det_fi_grid: np.ndarray            # (Nb,Np) float
    mass_grid: np.ndarray              # (Nb,Np) probability mass, sums to 1 on valid

    # vectorized active points
    valid_flat_idx: np.ndarray         # (M,) indices into ravel grid
    beta_vec: np.ndarray               # (M,)
    phi_u_vec: np.ndarray              # (M,)
    phi_w_vec: np.ndarray              # (M,)
    mass_vec: np.ndarray               # (M,) sums to 1

    fi_vec: np.ndarray                 # (M,2,2)
    inv_fi_vec: np.ndarray             # (M,2,2)
    pmf_vec: np.ndarray                # (M,dim)
    log_pmf_vec: np.ndarray            # (M,dim)
    log_mass_vec: np.ndarray           # (M,)


def build_cg_cache(
    model: SU2CatJzModel,
    theta0: Tuple[float, float],
    eps1: float,
    eps2: float,
    *,
    num_beta: int = 161,
    num_phi: int = 241,
    det_tol: float = 1e-12,
    eps_support: float = 1e-15,
    like_floor: float = 1e-300,
    # ---- NEW: taper controls
    use_taper: bool = False,
    taper_type: TaperType = "none",
    bump_alpha: float = 1.0,
    poly_k: int = 4,
    neighborhood_mode: NeighborhoodMode = "auto",
    half_plane: HalfPlane = "phi_ge",
) -> CGCache:
    """
    Build a coarse-grained neighborhood and the CG-Jeffreys prior on a local chart.

    Neighborhood selection rule:
      - if det(FI(theta0)) > det_tol -> ball of radius eps2
      - else -> ring eps1 < r < eps2
    plus excision of any additional points with det(FI) <= det_tol or invalid FI.
    """
    if num_beta < 3 or num_phi < 3:
        raise ValueError("num_beta and num_phi must be >= 3.")
    if not (eps2 > eps1 >= 0.0):
        raise ValueError("Require eps2 > eps1 >= 0.")

    beta0 = float(theta0[0])
    phi0 = float(wrap_phi(theta0[1]))
    eps1 = float(eps1)
    eps2 = float(eps2)

    #fi0, _ = model.fisher_information(beta0, phi0, eps_support=eps_support)
    #det0 = float(det_2x2(fi0)) if np.all(np.isfinite(fi0)) else 0.0
    #neighborhood: NeighborhoodType = "ball" if det0 > det_tol else "ring"

    fi0, _ = model.fisher_information(beta0, phi0, eps_support=eps_support)
    det0 = float(det_2x2(fi0)) if np.all(np.isfinite(fi0)) else 0.0

    if neighborhood_mode == "auto":
        neighborhood: NeighborhoodType = "ball" if det0 > det_tol else "ring"
    else:
        neighborhood = neighborhood_mode  # type: ignore[assignment]

    beta_min = max(0.0, beta0 - eps2)
    beta_max = min(PI, beta0 + eps2)
    beta_vals = np.linspace(beta_min, beta_max, num_beta, dtype=float)

    # Local unwrapped chart around phi0
    phi_u_min = phi0 - eps2
    phi_u_max = phi0 + eps2
    phi_u_vals = np.linspace(phi_u_min, phi_u_max, num_phi, dtype=float)
    phi_w_vals = np.array([wrap_phi(x) for x in phi_u_vals], dtype=float)

    beta_mesh, phi_u_mesh = np.meshgrid(beta_vals, phi_u_vals, indexing="ij")
    _, phi_w_mesh = np.meshgrid(beta_vals, phi_w_vals, indexing="ij")

    # r = theta_distance(beta_mesh, phi_u_mesh, beta0=beta0, phi0=phi0)
    # if neighborhood == "ball":
    #     coarse_mask = r < eps2
    # else:
    #     coarse_mask = (r > eps1) & (r < eps2)

    r = theta_distance(beta_mesh, phi_u_mesh, beta0=beta0, phi0=phi0)

    # Radial taper / radial support
    w_r = taper_weight(
        r,
        eps1=eps1,
        eps2=eps2,
        neighborhood=neighborhood,
        taper_type=taper_type,
        use_taper=use_taper,
        bump_alpha=bump_alpha,
        poly_k=poly_k,
    )

    # ---- NEW: half-plane truncation in phi_u around phi0
    hp: Optional[str] = None
    if neighborhood.startswith("half_"):
        hp = half_plane
        if half_plane == "phi_ge":
            half_mask = phi_u_mesh >= phi0
        elif half_plane == "phi_le":
            half_mask = phi_u_mesh <= phi0
        else:
            raise ValueError("half_plane must be 'phi_ge' or 'phi_le'")
        w_r = w_r * half_mask.astype(float)

    # support of the prior (where we bother computing FI/PMF)
    coarse_mask = w_r > 0.0


    d_beta = float(beta_vals[1] - beta_vals[0])
    d_phi = float(phi_u_vals[1] - phi_u_vals[0])
    cell_area = d_beta * d_phi

    nb, np_ = beta_mesh.shape
    dim = model.dim

    fi_grid = np.full((nb, np_, 2, 2), np.nan, dtype=float)
    det_fi_grid = np.full((nb, np_), np.nan, dtype=float)
    pmf_grid = np.full((nb, np_, dim), np.nan, dtype=float)

    flat_beta = beta_mesh.ravel()
    flat_phi_w = phi_w_mesh.ravel()
    flat_mask = coarse_mask.ravel()
    active_flat = np.flatnonzero(flat_mask)

    # Compute FI + PMF only on coarse_mask points (still needs a loop: qutip)
    for k in active_flat:
        b = float(flat_beta[k])
        phiw = float(flat_phi_w[k])
        i_mat, extras = model.fisher_information(
            b, phiw, eps_support=eps_support, return_extras=True
        )
        ii, jj = np.unravel_index(k, (nb, np_))
        fi_grid[ii, jj] = i_mat
        if extras is not None:
            pmf_grid[ii, jj] = extras["p"]

    det_fi_grid = det_2x2(fi_grid)
    valid_mask = coarse_mask & np.isfinite(det_fi_grid) & (det_fi_grid > det_tol)

    # Jeffreys weight proportional to sqrt(det FI)
    sqrt_det = np.zeros_like(det_fi_grid, dtype=float)
    sqrt_det[valid_mask] = np.sqrt(det_fi_grid[valid_mask])

    unnorm_mass_grid = np.zeros_like(det_fi_grid, dtype=float)
    # unnorm_mass_grid[valid_mask] = sqrt_det[valid_mask] * cell_area
    unnorm_mass_grid[valid_mask] = (sqrt_det[valid_mask] * w_r[valid_mask]) * cell_area

    z = float(np.sum(unnorm_mass_grid))
    if not (math.isfinite(z) and z > 0.0):
        raise RuntimeError("CG-Jeffreys normalization failed: zero mass region.")
    mass_grid = unnorm_mass_grid / z  # probability mass per cell, sums to 1

    valid_flat_idx = np.flatnonzero(valid_mask.ravel())
    m = valid_flat_idx.size
    if m == 0:
        raise RuntimeError("No valid grid points. Relax det_tol or enlarge eps2.")

    # Vectorized views of active points
    beta_vec = beta_mesh.ravel()[valid_flat_idx]
    phi_u_vec = phi_u_mesh.ravel()[valid_flat_idx]
    phi_w_vec = phi_w_mesh.ravel()[valid_flat_idx]
    mass_vec = mass_grid.ravel()[valid_flat_idx]

    fi_vec = fi_grid.reshape(-1, 2, 2)[valid_flat_idx]
    det_vec = det_fi_grid.ravel()[valid_flat_idx]
    inv_fi_vec = inv_2x2(fi_vec, det=det_vec)

    pmf_vec = pmf_grid.reshape(-1, dim)[valid_flat_idx]
    log_pmf_vec = np.log(np.clip(pmf_vec, like_floor, 1.0))
    log_mass_vec = np.log(np.clip(mass_vec, like_floor, 1.0))

    return CGCache(
        beta0=beta0,
        phi0=phi0,
        eps1=eps1,
        eps2=eps2,
        neighborhood=neighborhood,
        half_plane=hp,
        beta_vals=beta_vals,
        phi_u_vals=phi_u_vals,
        phi_w_vals=phi_w_vals,
        d_beta=d_beta,
        d_phi=d_phi,
        cell_area=cell_area,
        beta_mesh=beta_mesh,
        phi_u_mesh=phi_u_mesh,
        phi_w_mesh=phi_w_mesh,
        valid_mask=valid_mask,
        det_fi_grid=det_fi_grid,
        mass_grid=mass_grid,
        valid_flat_idx=valid_flat_idx,
        beta_vec=beta_vec,
        phi_u_vec=phi_u_vec,
        phi_w_vec=phi_w_vec,
        mass_vec=mass_vec,
        fi_vec=fi_vec,
        inv_fi_vec=inv_fi_vec,
        pmf_vec=pmf_vec,
        log_pmf_vec=log_pmf_vec,
        log_mass_vec=log_mass_vec,
    )

def build_coarse_grained_grid(*args, **kwargs) -> CGCache:
    """
    Backward-compatible alias.

    Historically this project used build_coarse_grained_grid for the prior+grid cache.
    The current implementation uses build_cg_cache; this wrapper keeps the old name.
    """
    return build_cg_cache(*args, **kwargs)


# =============================================================================
# Prior Fisher information I_pi (finite differences on log pi)
# =============================================================================
def prior_information_i_pi(
    cg: CGCache,
    *,
    mass_floor: float = 1e-300,
) -> np.ndarray:
    """
    Approximate I_pi = E_pi[(∇ log pi)(∇ log pi)^T] on (beta, phi_u) chart.

    IMPORTANT:
    We compute log pi from the *actual discrete prior* built in cg.mass_grid.
    This automatically incorporates:
      - Jeffreys factor sqrt(det FI)
      - hard truncation (if no taper)
      - smooth taper (if enabled)
    up to an additive constant (which cancels in gradients).
    """
    mask = cg.valid_mask
    mass = cg.mass_grid  # probability mass per cell, sums to 1 over valid cells

    # log pi differs from log(mass) by an additive constant (log cell_area), irrelevant for gradients.
    log_pi = np.full_like(mass, np.nan, dtype=float)
    log_pi[mask] = np.log(np.clip(mass[mask], mass_floor, 1.0))

    gb = np.full_like(log_pi, np.nan, dtype=float)
    gp = np.full_like(log_pi, np.nan, dtype=float)

    nb, np_ = log_pi.shape
    for i in range(nb):
        for j in range(np_):
            if not mask[i, j] or not np.isfinite(log_pi[i, j]):
                continue

            # beta derivative (one-sided near boundary)
            if 0 < i < nb - 1 and mask[i - 1, j] and mask[i + 1, j]:
                gb[i, j] = (log_pi[i + 1, j] - log_pi[i - 1, j]) / (2.0 * cg.d_beta)
            elif i < nb - 1 and mask[i + 1, j]:
                gb[i, j] = (log_pi[i + 1, j] - log_pi[i, j]) / cg.d_beta
            elif i > 0 and mask[i - 1, j]:
                gb[i, j] = (log_pi[i, j] - log_pi[i - 1, j]) / cg.d_beta

            # phi_u derivative (one-sided near boundary)
            if 0 < j < np_ - 1 and mask[i, j - 1] and mask[i, j + 1]:
                gp[i, j] = (log_pi[i, j + 1] - log_pi[i, j - 1]) / (2.0 * cg.d_phi)
            elif j < np_ - 1 and mask[i, j + 1]:
                gp[i, j] = (log_pi[i, j + 1] - log_pi[i, j]) / cg.d_phi
            elif j > 0 and mask[i, j - 1]:
                gp[i, j] = (log_pi[i, j] - log_pi[i, j - 1]) / cg.d_phi

    ok = mask & np.isfinite(gb) & np.isfinite(gp)
    if not np.any(ok):
        return np.zeros((2, 2), dtype=float)

    w = mass[ok]
    gbv = gb[ok]
    gpv = gp[ok]

    i00 = float(np.sum(w * gbv * gbv))
    i11 = float(np.sum(w * gpv * gpv))
    i01 = float(np.sum(w * gbv * gpv))
    return np.array([[i00, i01], [i01, i11]], dtype=float)

# def prior_information_i_pi(
#     cg: CGCache,
#     *,
#     det_floor: float = 1e-300,
# ) -> np.ndarray:
#     """
#     Approximate I_pi = E_pi[(∇ log pi)(∇ log pi)^T] on (beta, phi_u) chart.

#     Here pi(beta, phi_u) ∝ sqrt(det FI(beta, phi_u)) inside the CG neighborhood.
#     The constant normalization drops out of ∇log pi, so we use log pi = 0.5 log det FI.
#     """
#     det_fi = cg.det_fi_grid
#     mask = cg.valid_mask
#     mass = cg.mass_grid  # probability mass per cell

#     safe_det = np.where(mask, np.maximum(det_fi, det_floor), np.nan)
#     log_pi = 0.5 * np.log(safe_det)  # +const ignored

#     gb = np.full_like(log_pi, np.nan, dtype=float)
#     gp = np.full_like(log_pi, np.nan, dtype=float)

#     nb, np_ = log_pi.shape
#     for i in range(nb):
#         for j in range(np_):
#             if not mask[i, j] or not np.isfinite(log_pi[i, j]):
#                 continue

#             # beta derivative
#             if 0 < i < nb - 1 and mask[i - 1, j] and mask[i + 1, j]:
#                 gb[i, j] = (log_pi[i + 1, j] - log_pi[i - 1, j]) / (2.0 * cg.d_beta)
#             elif i < nb - 1 and mask[i + 1, j]:
#                 gb[i, j] = (log_pi[i + 1, j] - log_pi[i, j]) / cg.d_beta
#             elif i > 0 and mask[i - 1, j]:
#                 gb[i, j] = (log_pi[i, j] - log_pi[i - 1, j]) / cg.d_beta

#             # phi_u derivative
#             if 0 < j < np_ - 1 and mask[i, j - 1] and mask[i, j + 1]:
#                 gp[i, j] = (log_pi[i, j + 1] - log_pi[i, j - 1]) / (2.0 * cg.d_phi)
#             elif j < np_ - 1 and mask[i, j + 1]:
#                 gp[i, j] = (log_pi[i, j + 1] - log_pi[i, j]) / cg.d_phi
#             elif j > 0 and mask[i, j - 1]:
#                 gp[i, j] = (log_pi[i, j] - log_pi[i, j - 1]) / cg.d_phi

#     ok = mask & np.isfinite(gb) & np.isfinite(gp)
#     if not np.any(ok):
#         return np.zeros((2, 2), dtype=float)

#     w = mass[ok]
#     gbv = gb[ok]
#     gpv = gp[ok]

#     i00 = float(np.sum(w * gbv * gbv))
#     i11 = float(np.sum(w * gpv * gpv))
#     i01 = float(np.sum(w * gbv * gpv))
#     return np.array([[i00, i01], [i01, i11]], dtype=float)


# =============================================================================
# Matricial bounds
# =============================================================================
def expected_fisher(cg: CGCache) -> np.ndarray:
    """E_pi[FI(theta)]."""
    return np.einsum("i,ijk->jk", cg.mass_vec, cg.fi_vec)


def expected_inv_fisher(cg: CGCache) -> np.ndarray:
    """E_pi[FI(theta)^{-1}] (with inversion performed pointwise)."""
    return np.einsum("i,ijk->jk", cg.mass_vec, cg.inv_fi_vec)


def cg_crb_matrix(cg: CGCache, n: int) -> np.ndarray:
    """B_CG(n) = (1/n) E_pi[FI^{-1}]."""
    if n <= 0:
        raise ValueError("n must be positive.")
    return (1.0 / float(n)) * expected_inv_fisher(cg)


def van_trees_matrix(cg: CGCache, n: int, i_pi: np.ndarray) -> np.ndarray:
    """B_VT(n) = (n E_pi[FI] + I_pi)^{-1}."""
    if n <= 0:
        raise ValueError("n must be positive.")
    info = float(n) * expected_fisher(cg) + i_pi
    det_info = float(det_2x2(info))
    if not (math.isfinite(det_info) and det_info > 0.0):
        return np.full((2, 2), np.nan, dtype=float)
    return inv_2x2(info, det=np.array(det_info))


# =============================================================================
# Posterior + Bayes estimator (quadratic loss on local chart)
# =============================================================================
def posterior_from_counts(cg: CGCache, counts: np.ndarray) -> np.ndarray:
    """
    Posterior over grid points (vector form, sums to 1):
      post(i) ∝ mass(i) * prod_m p(m|theta_i)^{counts[m]}.
    """
    if counts.ndim != 1 or counts.size != cg.log_pmf_vec.shape[1]:
        raise ValueError("counts shape mismatch.")

    # log_post = log_prior + sum_m counts[m]*log p_m(theta_i)
    log_post = cg.log_mass_vec + (cg.log_pmf_vec @ counts.astype(float))
    log_z = safe_logsumexp(log_post)
    post = np.exp(log_post - log_z)
    return post


def bayes_estimator_local_mean(cg: CGCache, post: np.ndarray) -> np.ndarray:
    """
    Bayes estimator under quadratic loss on the local chart:
      theta_hat = E[theta | X] in (beta, phi_u).

    Returns a length-2 array [beta_hat, phi_u_hat].
    """
    if post.ndim != 1 or post.size != cg.mass_vec.size:
        raise ValueError("posterior shape mismatch.")
    beta_hat = float(np.dot(post, cg.beta_vec))
    phi_u_hat = float(np.dot(post, cg.phi_u_vec))
    return np.array([beta_hat, phi_u_hat], dtype=float)


def posterior_cov_local(cg: CGCache, post: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Posterior covariance on the local chart (beta, phi_u)."""
    d0 = cg.beta_vec - float(mean[0])
    d1 = cg.phi_u_vec - float(mean[1])
    c00 = float(np.dot(post, d0 * d0))
    c11 = float(np.dot(post, d1 * d1))
    c01 = float(np.dot(post, d0 * d1))
    return np.array([[c00, c01], [c01, c11]], dtype=float)


# =============================================================================
# Monte Carlo: Bayes risk matrix R_n
# =============================================================================
@dataclass(frozen=True)
class BatchResult:
    """One Monte Carlo batch result (r_batch experiments) for a fixed n."""
    errors: np.ndarray          # (r_batch,2) errors in local chart
    risk_matrix: np.ndarray     # (2,2) = E[e e^T] (empirical over r_batch)


def run_bayes_batch(
    cg: CGCache,
    *,
    n: int,
    r_batch: int,
    rng: np.random.Generator,
    save_dir: Optional[Path] = None,
    num_posteriors_to_save: int = 0,
) -> BatchResult:
    """
    Run r_batch independent Bayesian experiments:
      theta ~ prior, X|theta ~ Multinomial(n, p(.|theta)),
      theta_hat = posterior mean on local chart.

    Returns:
      errors array and risk_matrix = (E^T E)/r_batch.
    """
    if n <= 0 or r_batch <= 0:
        raise ValueError("n and r_batch must be positive.")

    m = cg.mass_vec.size
    dim = cg.pmf_vec.shape[1]

    # sample theta indices from prior
    idx = rng.choice(m, size=r_batch, replace=True, p=cg.mass_vec)

    errors = np.zeros((r_batch, 2), dtype=float)

    # optional plotting directory
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for t in range(r_batch):
        k = int(idx[t])
        beta_true = float(cg.beta_vec[k])
        phi_u_true = float(cg.phi_u_vec[k])

        # true likelihood (theta is on-grid, so use cached pmf for speed)
        p_true = cg.pmf_vec[k, :]
        counts = rng.multinomial(int(n), p_true).astype(int)

        post = posterior_from_counts(cg, counts)
        theta_hat = bayes_estimator_local_mean(cg, post)

        errors[t, 0] = float(theta_hat[0] - beta_true)
        errors[t, 1] = float(theta_hat[1] - phi_u_true)

        if save_dir is not None and t < num_posteriors_to_save:
            _save_posterior_heatmaps(
                cg=cg,
                post=post,
                theta_true=(beta_true, phi_u_true),
                out_path_prefix=save_dir / f"posterior_{t:04d}",
            )

    risk_matrix = (errors.T @ errors) / float(r_batch)
    return BatchResult(errors=errors, risk_matrix=risk_matrix)


def _save_posterior_heatmaps(
    cg: CGCache,
    post: np.ndarray,
    theta_true: Tuple[float, float],
    out_path_prefix: Path,
) -> None:
    """Save posterior heatmaps on the (phi_u, beta) chart (linear and log)."""
    post_grid = np.zeros_like(cg.mass_grid, dtype=float)
    post_grid.ravel()[cg.valid_flat_idx] = post

    img = np.where(cg.valid_mask, post_grid, np.nan)
    img_log = np.log(np.maximum(img, 1e-300))

    beta_true, phi_u_true = float(theta_true[0]), float(theta_true[1])

    for arr, suffix, title in [
        (img, "png", "Posterior"),
        (img_log, "log.png", "Posterior (log-scale)"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        pcm = ax.pcolormesh(cg.phi_u_mesh, cg.beta_mesh, arr, shading="auto")
        fig.colorbar(pcm, ax=ax)
        ax.scatter([phi_u_true], [beta_true], marker="o")
        ax.set_xlabel("phi (unwrapped)")
        ax.set_ylabel("beta")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(str(out_path_prefix) + f"_{suffix}", dpi=200, bbox_inches="tight")
        plt.close(fig)


@dataclass(frozen=True)
class RiskMCResult:
    """Outer Monte Carlo result for a fixed n."""
    risk_mats: np.ndarray       # (m_batch,2,2)
    trace_vals: np.ndarray      # (m_batch,)
    eig_min_vals: np.ndarray    # (m_batch,)
    eig_max_vals: np.ndarray    # (m_batch,)


def monte_carlo_risk_matrix(
    cg: CGCache,
    *,
    n: int,
    r_batch: int,
    m_batch: int,
    base_seed: int,
    save_dir: Optional[Path] = None,
    num_posteriors_to_save_per_mc: int = 0,
) -> RiskMCResult:
    """
    Outer Monte Carlo: for m=1..m_batch, run a batch of size r_batch and store R_n^{(m)}.
    """
    if n <= 0 or r_batch <= 0 or m_batch <= 0:
        raise ValueError("n, r_batch, m_batch must be positive.")
    risk_mats = np.zeros((m_batch, 2, 2), dtype=float)
    trace_vals = np.zeros(m_batch, dtype=float)
    eig_min_vals = np.zeros(m_batch, dtype=float)
    eig_max_vals = np.zeros(m_batch, dtype=float)

    for m in range(m_batch):
        rng = np.random.default_rng(int(base_seed + 1_000_000 * m))

        mc_dir = None
        if save_dir is not None:
            mc_dir = save_dir / f"mc_{m:03d}"
            mc_dir.mkdir(parents=True, exist_ok=True)

        batch = run_bayes_batch(
            cg,
            n=n,
            r_batch=r_batch,
            rng=rng,
            save_dir=mc_dir,
            num_posteriors_to_save=num_posteriors_to_save_per_mc,
        )
        rmat = 0.5 * (batch.risk_matrix + batch.risk_matrix.T)
        risk_mats[m] = rmat

        ev = eigvals_psd_2x2(rmat)
        trace_vals[m] = float(np.trace(rmat))
        eig_min_vals[m] = float(ev[0])
        eig_max_vals[m] = float(ev[1])

        if mc_dir is not None:
            # Save batch errors + risk matrix
            np.savetxt(mc_dir / "risk_matrix.txt", rmat, fmt="%.18e")
            df_err = pd.DataFrame(batch.errors, columns=["err_beta", "err_phi_u"])
            df_err.to_csv(mc_dir / "errors.csv", index=False)

            # Histogram of squared error (trace loss per experiment)
            se = np.sum(batch.errors**2, axis=1)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(se, bins=30)
            ax.set_xlabel("||error||^2 (local chart)")
            ax.set_ylabel("count")
            ax.set_title(f"Squared-error histogram (n={n}, mc={m})")
            fig.tight_layout()
            fig.savefig(mc_dir / "hist_squared_error.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

    return RiskMCResult(
        risk_mats=risk_mats,
        trace_vals=trace_vals,
        eig_min_vals=eig_min_vals,
        eig_max_vals=eig_max_vals,
    )


# =============================================================================
# Study vs n: compute matrices + compare by trace and eigenvalues
# =============================================================================
@dataclass(frozen=True)
class SummaryRow:
    """Convenient container for one-n summary (stored later in a DataFrame)."""
    n: int

    # risk (MC)
    risk_trace_mean: float
    risk_trace_se: float
    risk_eigmin_mean: float
    risk_eigmin_se: float
    risk_eigmax_mean: float
    risk_eigmax_se: float

    # mean risk matrix (elementwise)
    r00: float
    r01: float
    r11: float
    r00_se: float
    r01_se: float
    r11_se: float

    # bounds (deterministic)
    cg_trace: float
    cg_eigmin: float
    cg_eigmax: float
    vt_trace: float
    vt_eigmin: float
    vt_eigmax: float

    # PSD diagnostics (risk - bound): mean of min eigenvalue over MC replicates
    min_eig_r_minus_cg_mean: float
    min_eig_r_minus_vt_mean: float


def study_matricial_risk_vs_bounds(
    *,
    j: float,
    m_c: float,
    chi: float,
    theta0: Tuple[float, float],
    eps1: float,
    eps2: float,
    n_values: Sequence[int],
    r_batch: int,
    m_batch: int,
    grid_num_beta: int = 161,
    grid_num_phi: int = 241,
    det_tol: float = 1e-12,
    eps_support: float = 1e-15,
    base_seed: int = 1234,
    num_posteriors_to_save_per_mc: int = 0,
    output_root: Optional[Path] = None,
    make_psd_diagnostic_plots: bool = True,
    # ---- NEW: taper options
    use_taper: bool = False,
    taper_type: TaperType = "none",
    bump_alpha: float = 1.0,
    poly_k: int = 4,
    neighborhood_mode: NeighborhoodMode = "auto",
    half_plane: HalfPlane = "phi_ge",
    #slope_window: int = 3,
    #slope_tail_points: int = 4,
    #asymptotic_slope_tol: float = 0.10,
    #asymptotic_ratio_tol: float = 0.20,
    #asymptotic_min_run: int = 2,
    tail_fit_points: int = 4,
    tail_plot_extension_factor: float = 10.0,
) -> Tuple[pd.DataFrame, Path]:
    """
    Main entry point: for each n in n_values, compute:
      - MC estimate of Bayes risk matrix R_n (with uncertainty)
      - matricial bounds B_CG(n), B_VT(n)
      - comparisons by trace and eigenvalues

    Saves plots and a summary CSV under output_root.
    """
    model = SU2CatJzModel(j=float(j), m_c=float(m_c), chi=float(chi))
    cg = build_cg_cache(
        model=model,
        theta0=theta0,
        eps1=eps1,
        eps2=eps2,
        num_beta=int(grid_num_beta),
        num_phi=int(grid_num_phi),
        det_tol=det_tol,
        eps_support=eps_support,
        use_taper=use_taper,
        taper_type=taper_type,
        bump_alpha=bump_alpha,
        poly_k=poly_k,
        neighborhood_mode=neighborhood_mode,
        half_plane=half_plane,
    )
    i_pi = prior_information_i_pi(cg)

    if output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = Path(f"matricial_risk_outputs_{stamp}")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Save config
    (output_root / "config.txt").write_text(
        "\n".join(
            [
                f"j={j}",
                f"m_c={m_c}",
                f"chi={chi}",
                f"theta0={theta0}",
                f"eps1={eps1}",
                f"eps2={eps2}",
                f"neighborhood={cg.neighborhood}",
                f"grid_num_beta={grid_num_beta}",
                f"grid_num_phi={grid_num_phi}",
                f"det_tol={det_tol}",
                f"eps_support={eps_support}",
                f"r_batch={r_batch}",
                f"m_batch={m_batch}",
                f"use_taper={use_taper}",
                f"taper_type={taper_type}",
                f"bump_alpha={bump_alpha}",
                f"poly_k={poly_k}",
                f"neighborhood_mode={neighborhood_mode}",
                f"half_plane={half_plane}",
            ]
        )
        + "\n"
    )

    n_arr = np.array(list(map(int, n_values)), dtype=int)
    n_arr.sort()

    # Deterministic bound arrays (for plotting)
    cg_tr = np.zeros_like(n_arr, dtype=float)
    cg_e1 = np.zeros_like(n_arr, dtype=float)
    cg_e2 = np.zeros_like(n_arr, dtype=float)

    vt_tr = np.zeros_like(n_arr, dtype=float)
    vt_e1 = np.zeros_like(n_arr, dtype=float)
    vt_e2 = np.zeros_like(n_arr, dtype=float)

    # Risk (MC) arrays
    risk_tr_mean = np.zeros_like(n_arr, dtype=float)
    risk_tr_se = np.zeros_like(n_arr, dtype=float)
    risk_e1_mean = np.zeros_like(n_arr, dtype=float)
    risk_e1_se = np.zeros_like(n_arr, dtype=float)
    risk_e2_mean = np.zeros_like(n_arr, dtype=float)
    risk_e2_se = np.zeros_like(n_arr, dtype=float)

    # PSD diagnostics
    min_eig_r_minus_cg = np.zeros_like(n_arr, dtype=float)
    min_eig_r_minus_vt = np.zeros_like(n_arr, dtype=float)

    rows: list[dict] = []

    for idx, n in enumerate(n_arr):
        n_dir = output_root / f"n_{n:04d}"
        n_dir.mkdir(parents=True, exist_ok=True)

        # Bounds
        b_cg = cg_crb_matrix(cg, int(n))
        b_vt = van_trees_matrix(cg, int(n), i_pi=i_pi)

        ev_cg = eigvals_psd_2x2(b_cg)
        ev_vt = eigvals_psd_2x2(b_vt)

        cg_tr[idx] = float(np.trace(b_cg))
        cg_e1[idx] = float(ev_cg[0])
        cg_e2[idx] = float(ev_cg[1])

        vt_tr[idx] = float(np.trace(b_vt))
        vt_e1[idx] = float(ev_vt[0])
        vt_e2[idx] = float(ev_vt[1])

        # Risk MC
        mc = monte_carlo_risk_matrix(
            cg,
            n=int(n),
            r_batch=int(r_batch),
            m_batch=int(m_batch),
            base_seed=int(base_seed + 10_000 * idx),
            save_dir=n_dir,
            num_posteriors_to_save_per_mc=int(num_posteriors_to_save_per_mc),
        )
        mats = mc.risk_mats

        # elementwise mean and SE (symmetric entries only)
        mean_mat = np.mean(mats, axis=0)
        se_mat = np.std(mats, axis=0, ddof=1) / math.sqrt(m_batch) if m_batch > 1 else np.full((2, 2), np.nan)

        r00 = float(mean_mat[0, 0])
        r01 = float(mean_mat[0, 1])
        r11 = float(mean_mat[1, 1])
        r00_se = float(se_mat[0, 0])
        r01_se = float(se_mat[0, 1])
        r11_se = float(se_mat[1, 1])

        # trace/eigen summaries from replicate-level scalars (correct uncertainty propagation)
        tr_vals = mc.trace_vals
        e1_vals = mc.eig_min_vals
        e2_vals = mc.eig_max_vals

        risk_tr_mean[idx] = float(np.mean(tr_vals))
        risk_e1_mean[idx] = float(np.mean(e1_vals))
        risk_e2_mean[idx] = float(np.mean(e2_vals))

        if m_batch > 1:
            risk_tr_se[idx] = float(np.std(tr_vals, ddof=1) / math.sqrt(m_batch))
            risk_e1_se[idx] = float(np.std(e1_vals, ddof=1) / math.sqrt(m_batch))
            risk_e2_se[idx] = float(np.std(e2_vals, ddof=1) / math.sqrt(m_batch))
        else:
            risk_tr_se[idx] = float("nan")
            risk_e1_se[idx] = float("nan")
            risk_e2_se[idx] = float("nan")

        # PSD-order diagnostics: min eig of (R - bound), averaged over MC replicates
        d_cg = np.zeros(m_batch, dtype=float)
        d_vt = np.zeros(m_batch, dtype=float)
        for m in range(m_batch):
            d_cg[m] = float(eigvals_psd_2x2(mats[m] - b_cg)[0])
            d_vt[m] = float(eigvals_psd_2x2(mats[m] - b_vt)[0])
        min_eig_r_minus_cg[idx] = float(np.mean(d_cg))
        min_eig_r_minus_vt[idx] = float(np.mean(d_vt))

        # Save matrices
        np.savetxt(n_dir / "bound_cg_matrix.txt", b_cg, fmt="%.18e")
        np.savetxt(n_dir / "bound_vt_matrix.txt", b_vt, fmt="%.18e")
        np.savetxt(n_dir / "risk_matrix_mean.txt", mean_mat, fmt="%.18e")
        np.savetxt(n_dir / "risk_matrix_se.txt", se_mat, fmt="%.18e")

        rows.append(
            SummaryRow(
                n=int(n),
                risk_trace_mean=float(risk_tr_mean[idx]),
                risk_trace_se=float(risk_tr_se[idx]),
                risk_eigmin_mean=float(risk_e1_mean[idx]),
                risk_eigmin_se=float(risk_e1_se[idx]),
                risk_eigmax_mean=float(risk_e2_mean[idx]),
                risk_eigmax_se=float(risk_e2_se[idx]),
                r00=r00,
                r01=r01,
                r11=r11,
                r00_se=r00_se,
                r01_se=r01_se,
                r11_se=r11_se,
                cg_trace=float(cg_tr[idx]),
                cg_eigmin=float(cg_e1[idx]),
                cg_eigmax=float(cg_e2[idx]),
                vt_trace=float(vt_tr[idx]),
                vt_eigmin=float(vt_e1[idx]),
                vt_eigmax=float(vt_e2[idx]),
                min_eig_r_minus_cg_mean=float(min_eig_r_minus_cg[idx]),
                min_eig_r_minus_vt_mean=float(min_eig_r_minus_vt[idx]),
            ).__dict__
        )

        summary_df = pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Tail power-law fits for trace curves
    # -------------------------------------------------------------------------
    fit_risk = _fit_power_law_tail(
        n_arr, risk_tr_mean, tail_points=int(tail_fit_points)
    )
    fit_cg = _fit_power_law_tail(
        n_arr, cg_tr, tail_points=int(tail_fit_points)
    )
    fit_vt = _fit_power_law_tail(
        n_arr, vt_tr, tail_points=int(tail_fit_points)
    )

    risk_cg_intersection = _power_law_intersection(fit_risk, fit_cg)
    risk_vt_intersection = _power_law_intersection(fit_risk, fit_vt)

    summary_df["tail_fit_points"] = int(tail_fit_points)
    summary_df["risk_trace_tail_slope"] = float(fit_risk["slope"])
    summary_df["cg_trace_tail_slope"] = float(fit_cg["slope"])
    summary_df["vt_trace_tail_slope"] = float(fit_vt["slope"])

    summary_df["risk_cg_intersection_n"] = (
        float(risk_cg_intersection["x"]) if risk_cg_intersection is not None else float("nan")
    )
    summary_df["risk_vt_intersection_n"] = (
        float(risk_vt_intersection["x"]) if risk_vt_intersection is not None else float("nan")
    )

    summary_df.to_csv(output_root / "summary_matricial.csv", index=False)

    # Compact text report
    report_lines = [
        f"tail_fit_points={tail_fit_points}",
        "",
        f"risk_tail_slope={fit_risk['slope']:.8f}",
        f"risk_tail_slope_se={fit_risk['slope_se']:.8f}",
        f"cg_tail_slope={fit_cg['slope']:.8f}",
        f"cg_tail_slope_se={fit_cg['slope_se']:.8f}",
        f"vt_tail_slope={fit_vt['slope']:.8f}",
        f"vt_tail_slope_se={fit_vt['slope_se']:.8f}",
        "",
        "Intersections from tail-line extrapolation:",
        f"risk_cg_intersection_n={None if risk_cg_intersection is None else risk_cg_intersection['x']}",
        f"risk_vt_intersection_n={None if risk_vt_intersection is None else risk_vt_intersection['x']}",
    ]
    (output_root / 'tail_extrapolation_report.txt').write_text(
        '\n'.join(report_lines) + '\n'
    )

    # summary_df = pd.DataFrame(rows)

    # slope_window_eff = max(2, min(int(slope_window), len(n_arr)))
    # slope_tail_eff = max(2, min(int(slope_tail_points), len(n_arr)))

    # risk_slope, risk_slope_se = _trailing_loglog_slopes(
    #     n_arr, risk_tr_mean, window=slope_window_eff
    # )
    # cg_slope, cg_slope_se = _trailing_loglog_slopes(
    #     n_arr, cg_tr, window=slope_window_eff
    # )
    # vt_slope, vt_slope_se = _trailing_loglog_slopes(
    #     n_arr, vt_tr, window=slope_window_eff
    # )

    # risk_to_cg_ratio = _safe_ratio(risk_tr_mean, cg_tr)
    # vt_to_cg_ratio = _safe_ratio(vt_tr, cg_tr)

    # summary_df["risk_trace_slope_trailing"] = risk_slope
    # summary_df["risk_trace_slope_trailing_se"] = risk_slope_se
    # summary_df["cg_trace_slope_trailing"] = cg_slope
    # summary_df["cg_trace_slope_trailing_se"] = cg_slope_se
    # summary_df["vt_trace_slope_trailing"] = vt_slope
    # summary_df["vt_trace_slope_trailing_se"] = vt_slope_se
    # summary_df["risk_trace_over_cg_trace"] = risk_to_cg_ratio
    # summary_df["vt_trace_over_cg_trace"] = vt_to_cg_ratio

    # # Tail fits (last slope_tail_eff points)
    # _, risk_tail_slope, risk_tail_slope_se = _loglog_ols(
    #     n_arr[-slope_tail_eff:], risk_tr_mean[-slope_tail_eff:]
    # )
    # _, cg_tail_slope, cg_tail_slope_se = _loglog_ols(
    #     n_arr[-slope_tail_eff:], cg_tr[-slope_tail_eff:]
    # )
    # _, vt_tail_slope, vt_tail_slope_se = _loglog_ols(
    #     n_arr[-slope_tail_eff:], vt_tr[-slope_tail_eff:]
    # )

    # # Heuristic "onset of asymptotic regime"
    # # For the Bayes risk we require:
    # #   slope ~ -1 and risk/CG ~ 1
    # idx_risk_onset = _first_asymptotic_index(
    #     n_arr,
    #     risk_slope,
    #     target_slope=-1.0,
    #     slope_tol=float(asymptotic_slope_tol),
    #     curve=risk_tr_mean,
    #     ref_curve=cg_tr,
    #     ratio_tol=float(asymptotic_ratio_tol),
    #     min_run=int(asymptotic_min_run),
    # )
    # n_risk_onset = None if idx_risk_onset is None else int(n_arr[idx_risk_onset])

    # # For VT we use the same criterion against CG
    # idx_vt_onset = _first_asymptotic_index(
    #     n_arr,
    #     vt_slope,
    #     target_slope=-1.0,
    #     slope_tol=float(asymptotic_slope_tol),
    #     curve=vt_tr,
    #     ref_curve=cg_tr,
    #     ratio_tol=float(asymptotic_ratio_tol),
    #     min_run=int(asymptotic_min_run),
    # )
    # n_vt_onset = None if idx_vt_onset is None else int(n_arr[idx_vt_onset])

    # summary_df.to_csv(output_root / "summary_matricial.csv", index=False)

    # # Save a compact slope report
    # slope_report_lines = [
    #     f"slope_window={slope_window_eff}",
    #     f"slope_tail_points={slope_tail_eff}",
    #     "",
    #     f"risk_tail_slope={risk_tail_slope:.8f}",
    #     f"risk_tail_slope_se={risk_tail_slope_se:.8f}",
    #     f"cg_tail_slope={cg_tail_slope:.8f}",
    #     f"cg_tail_slope_se={cg_tail_slope_se:.8f}",
    #     f"vt_tail_slope={vt_tail_slope:.8f}",
    #     f"vt_tail_slope_se={vt_tail_slope_se:.8f}",
    #     "",
    #     f"target_asymptotic_slope=-1.0",
    #     f"asymptotic_slope_tol={asymptotic_slope_tol}",
    #     f"asymptotic_ratio_tol={asymptotic_ratio_tol}",
    #     f"asymptotic_min_run={asymptotic_min_run}",
    #     f"risk_asymptotic_onset_n={n_risk_onset}",
    #     f"vt_asymptotic_onset_n={n_vt_onset}",
    # ]
    # (output_root / "slope_report.txt").write_text("\n".join(slope_report_lines) + "\n")

    title = (
        f"Matricial Bayes risk vs bounds | neighborhood={cg.neighborhood}, "
        f"j={j}, m_c={m_c}, chi={chi}"
    )

    # Trace plots
    _plot_compare_with_errorbars(
        x=n_arr,
        y_mean=risk_tr_mean,
        y_se=risk_tr_se,
        y_cg=cg_tr,
        y_vt=vt_tr,
        ylabel="trace",
        title=title + " | trace",
        out_linear=output_root / "compare_trace_linear.png",
        out_log=output_root / "compare_trace_logy.png",
    )

    # Eigenvalue plots: lambda_min and lambda_max
    _plot_compare_with_errorbars(
        x=n_arr,
        y_mean=risk_e1_mean,
        y_se=risk_e1_se,
        y_cg=cg_e1,
        y_vt=vt_e1,
        ylabel="lambda_min",
        title=title + " | eigenvalue min",
        out_linear=output_root / "compare_eigmin_linear.png",
        out_log=output_root / "compare_eigmin_logy.png",
    )
    _plot_compare_with_errorbars(
        x=n_arr,
        y_mean=risk_e2_mean,
        y_se=risk_e2_se,
        y_cg=cg_e2,
        y_vt=vt_e2,
        ylabel="lambda_max",
        title=title + " | eigenvalue max",
        out_linear=output_root / "compare_eigmax_linear.png",
        out_log=output_root / "compare_eigmax_logy.png",
    )

    # PSD diagnostics
    if make_psd_diagnostic_plots:
        _plot_psd_diagnostic(
            x=n_arr,
            y=min_eig_r_minus_cg,
            title=title + " | PSD diagnostic: min eig(R - CG)",
            ylabel="mean min eigenvalue",
            out_path=output_root / "psd_diag_r_minus_cg.png",
        )
        _plot_psd_diagnostic(
            x=n_arr,
            y=min_eig_r_minus_vt,
            title=title + " | PSD diagnostic: min eig(R - VT)",
            ylabel="mean min eigenvalue",
            out_path=output_root / "psd_diag_r_minus_vt.png",
        )

    # -------------------------------------------------------------------------
    # Slope and ratio
    # -------------------------------------------------------------------------
    # _plot_trace_slopes(
    #     x=n_arr,
    #     risk_slope=risk_slope,
    #     cg_slope=cg_slope,
    #     vt_slope=vt_slope,
    #     out_path=output_root / "trace_slopes_loglog.png",
    #     title=title + " | trailing log-log slopes of trace",
    # )

    _plot_trace_tail_extrapolation(
        x=n_arr,
        risk_trace=risk_tr_mean,
        cg_trace=cg_tr,
        vt_trace=vt_tr,
        fit_risk=fit_risk,
        fit_cg=fit_cg,
        fit_vt=fit_vt,
        risk_cg_intersection=risk_cg_intersection,
        risk_vt_intersection=risk_vt_intersection,
        out_path=output_root / "trace_tail_extrapolation.png",
        title=title + " | tail power-law extrapolation",
        x_extension_factor=float(tail_plot_extension_factor),
    )

    _plot_trace_ratios_to_cg(
        x=n_arr,
        risk_trace=risk_tr_mean,
        vt_trace=vt_tr,
        cg_trace=cg_tr,
        out_path=output_root / "trace_ratios_to_cg.png",
        title=title + " | trace ratios to CG",
    )

    return summary_df, output_root


# =============================================================================
# Plotting functions
# =============================================================================
def _plot_compare_with_errorbars(
    *,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_se: np.ndarray,
    y_cg: np.ndarray,
    y_vt: np.ndarray,
    ylabel: str,
    title: str,
    out_linear: Path,
    out_log: Path,
) -> None:
    """Two plots: linear and log-y."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        x,
        y_mean,
        yerr=1.96 * y_se,
        marker="o",
        linestyle="-",
        capsize=3,
        label="Bayes risk (mean ± 1.96 SE)",
    )
    ax.plot(x, y_cg, marker="s", label="CG-CRB")
    ax.plot(x, y_vt, marker="^", label="Van Trees (simple)")
    ax.set_xlabel("n")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_linear, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # log-y
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        x,
        _safe_positive(y_mean),
        yerr=1.96 * y_se,
        marker="o",
        linestyle="-",
        capsize=3,
        label="Bayes risk (mean ± 1.96 SE)",
    )
    ax.plot(x, _safe_positive(y_cg), marker="s", label="CG-CRB")
    ax.plot(x, _safe_positive(y_vt), marker="^", label="Van Trees (simple)")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel(ylabel)
    ax.set_title(title + " (log-y)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_log, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_psd_diagnostic(
    *,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Plot mean min eigenvalue of (R - bound). Negative implies PSD-order violation."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker="o")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("n")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _safe_positive(y: np.ndarray) -> np.ndarray:
    y2 = np.asarray(y, dtype=float).copy()
    y2[~np.isfinite(y2)] = np.nan
    y2[y2 <= 0.0] = np.nan
    return y2

def _fit_power_law_tail(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tail_points: int = 4,
) -> dict:
    """
    Fit a power law y ~ exp(a) * x^b on the last tail_points valid points.

    The fit is performed as:
        log(y) = a + b log(x)

    Returns a dict with:
        intercept, slope, intercept_se, slope_se, n_used
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    xv = x[mask]
    yv = y[mask]

    if xv.size < 2:
        return {
            "intercept": float("nan"),
            "slope": float("nan"),
            "intercept_se": float("nan"),
            "slope_se": float("nan"),
            "n_used": 0,
        }

    k = max(2, min(int(tail_points), xv.size))
    xv = xv[-k:]
    yv = yv[-k:]

    lx = np.log(xv)
    ly = np.log(yv)

    xbar = float(np.mean(lx))
    ybar = float(np.mean(ly))

    dx = lx - xbar
    dy = ly - ybar

    ssx = float(np.dot(dx, dx))
    if ssx <= 0.0:
        return {
            "intercept": float("nan"),
            "slope": float("nan"),
            "intercept_se": float("nan"),
            "slope_se": float("nan"),
            "n_used": k,
        }

    slope = float(np.dot(dx, dy) / ssx)
    intercept = float(ybar - slope * xbar)

    if k <= 2:
        slope_se = float("nan")
        intercept_se = float("nan")
    else:
        resid = ly - (intercept + slope * lx)
        sigma2 = float(np.dot(resid, resid) / (k - 2))
        slope_se = float(math.sqrt(sigma2 / ssx))
        intercept_se = float(math.sqrt(sigma2 * (1.0 / k + xbar * xbar / ssx)))

    return {
        "intercept": intercept,
        "slope": slope,
        "intercept_se": intercept_se,
        "slope_se": slope_se,
        "n_used": k,
    }


def _power_law_eval(x: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    """Evaluate y = exp(intercept) * x^slope."""
    x = np.asarray(x, dtype=float)
    return np.exp(intercept) * np.power(x, slope)


def _power_law_intersection(
    fit_a: dict,
    fit_b: dict,
) -> Optional[dict]:
    """
    Intersection of two fitted log-log lines:
        log y = a + b log x

    Returns:
        {"x": x_star, "y": y_star}
    or None if slopes are equal / intersection invalid.
    """
    a1 = float(fit_a["intercept"])
    b1 = float(fit_a["slope"])
    a2 = float(fit_b["intercept"])
    b2 = float(fit_b["slope"])

    if not (np.isfinite(a1) and np.isfinite(b1) and np.isfinite(a2) and np.isfinite(b2)):
        return None

    denom = b1 - b2
    if abs(denom) < 1e-12:
        return None

    log_x_star = (a2 - a1) / denom
    x_star = float(np.exp(log_x_star))
    if not (np.isfinite(x_star) and x_star > 0.0):
        return None

    y_star = float(np.exp(a1 + b1 * log_x_star))
    if not np.isfinite(y_star):
        return None

    return {"x": x_star, "y": y_star}

def _loglog_ols(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Ordinary least squares fit on log-log scale:
        log y = a + b log x

    Returns:
        intercept a, slope b, slope standard error
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    xv = np.log(x[mask])
    yv = np.log(y[mask])

    m = xv.size
    if m < 2:
        return float("nan"), float("nan"), float("nan")

    xbar = float(np.mean(xv))
    ybar = float(np.mean(yv))

    dx = xv - xbar
    dy = yv - ybar
    ssx = float(np.dot(dx, dx))
    if ssx <= 0.0:
        return float("nan"), float("nan"), float("nan")

    slope = float(np.dot(dx, dy) / ssx)
    intercept = float(ybar - slope * xbar)

    if m <= 2:
        slope_se = float("nan")
    else:
        resid = yv - (intercept + slope * xv)
        sigma2 = float(np.dot(resid, resid) / (m - 2))
        slope_se = float(math.sqrt(sigma2 / ssx))

    return intercept, slope, slope_se


def _trailing_loglog_slopes(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trailing-window log-log slopes.

    For each index i >= window-1, fit log y vs log x on the window
    [i-window+1, ..., i]. The returned slope is aligned with x[i].

    Returns:
        slopes, slope_ses
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if window < 2:
        raise ValueError("window must be >= 2")

    n = x.size
    slopes = np.full(n, np.nan, dtype=float)
    slope_ses = np.full(n, np.nan, dtype=float)

    for i in range(window - 1, n):
        _, slope, slope_se = _loglog_ols(
            x[i - window + 1 : i + 1],
            y[i - window + 1 : i + 1],
        )
        slopes[i] = slope
        slope_ses[i] = slope_se

    return slopes, slope_ses


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise ratio with NaN where division is unsafe."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0.0)
    out[mask] = num[mask] / den[mask]
    return out


def _first_asymptotic_index(
    x: np.ndarray,
    slope: np.ndarray,
    *,
    target_slope: float = -1.0,
    slope_tol: float = 0.10,
    curve: Optional[np.ndarray] = None,
    ref_curve: Optional[np.ndarray] = None,
    ratio_tol: float = 0.20,
    min_run: int = 2,
) -> Optional[int]:
    """
    First index i such that for min_run consecutive points:
      - slope is within slope_tol of target_slope
      - and, if curve/ref_curve are provided, curve/ref_curve is within ratio_tol of 1

    Returns:
        index i_start or None
    """
    x = np.asarray(x, dtype=float)
    slope = np.asarray(slope, dtype=float)

    good = np.isfinite(slope) & (np.abs(slope - target_slope) <= slope_tol)

    if curve is not None and ref_curve is not None:
        ratio = _safe_ratio(curve, ref_curve)
        good &= np.isfinite(ratio) & (np.abs(ratio - 1.0) <= ratio_tol)

    run = 0
    for i, ok in enumerate(good):
        run = run + 1 if ok else 0
        if run >= min_run:
            return i - min_run + 1

    return None


def _plot_trace_slopes(
    *,
    x: np.ndarray,
    risk_slope: np.ndarray,
    cg_slope: np.ndarray,
    vt_slope: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Plot trailing log-log slopes for the trace curves."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, risk_slope, marker="o", label="Risk slope")
    ax.plot(x, cg_slope, marker="s", label="CG slope")
    ax.plot(x, vt_slope, marker="^", label="VT slope")
    ax.axhline(-1.0, linestyle="--", linewidth=1.0, label="slope = -1")
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel(r"d log(trace) / d log(n)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_trace_ratios_to_cg(
    *,
    x: np.ndarray,
    risk_trace: np.ndarray,
    vt_trace: np.ndarray,
    cg_trace: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """
    Plot trace ratios relative to CG:
      risk / CG, VT / CG
    """
    risk_ratio = _safe_ratio(risk_trace, cg_trace)
    vt_ratio = _safe_ratio(vt_trace, cg_trace)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, risk_ratio, marker="o", label="Risk / CG")
    ax.plot(x, vt_ratio, marker="^", label="VT / CG")
    ax.axhline(1.0, linestyle="--", linewidth=1.0, label="ratio = 1")
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("trace ratio")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _plot_trace_two_eps2(
    *,
    x: np.ndarray,
    # eps2 = a
    y_mean_a: np.ndarray,
    y_se_a: np.ndarray,
    y_cg_a: np.ndarray,
    y_vt_a: np.ndarray,
    eps2_a: float,
    # eps2 = b
    y_mean_b: np.ndarray,
    y_se_b: np.ndarray,
    y_cg_b: np.ndarray,
    y_vt_b: np.ndarray,
    eps2_b: float,
    title: str,
    out_linear: Path,
    out_log: Path,
) -> None:
    """Overlay trace curves for two eps2 values (linear-y and log-y)."""
    # ---- linear-y
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(
        x,
        y_mean_a,
        yerr=1.96 * y_se_a,
        marker="o",
        linestyle="-",
        capsize=3,
        label=f"Risk (eps2={eps2_a:g})",
    )
    ax.plot(x, y_cg_a, marker="s", linestyle="--", label=f"CG (eps2={eps2_a:g})")
    ax.plot(x, y_vt_a, marker="^", linestyle="--", label=f"VT (eps2={eps2_a:g})")

    ax.errorbar(
        x,
        y_mean_b,
        yerr=1.96 * y_se_b,
        marker="o",
        linestyle="-",
        capsize=3,
        label=f"Risk (eps2={eps2_b:g})",
    )
    ax.plot(x, y_cg_b, marker="s", linestyle="--", label=f"CG (eps2={eps2_b:g})")
    ax.plot(x, y_vt_b, marker="^", linestyle="--", label=f"VT (eps2={eps2_b:g})")

    ax.set_xlabel("n")
    ax.set_ylabel("trace")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_linear, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- log-y
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(
        x,
        _safe_positive(y_mean_a),
        yerr=1.96 * y_se_a,
        marker="o",
        linestyle="-",
        capsize=3,
        label=f"Risk (eps2={eps2_a:g})",
    )
    ax.plot(x, _safe_positive(y_cg_a), marker="s", linestyle="--", label=f"CG (eps2={eps2_a:g})")
    ax.plot(x, _safe_positive(y_vt_a), marker="^", linestyle="--", label=f"VT (eps2={eps2_a:g})")

    ax.errorbar(
        x,
        _safe_positive(y_mean_b),
        yerr=1.96 * y_se_b,
        marker="o",
        linestyle="-",
        capsize=3,
        label=f"Risk (eps2={eps2_b:g})",
    )
    ax.plot(x, _safe_positive(y_cg_b), marker="s", linestyle="--", label=f"CG (eps2={eps2_b:g})")
    ax.plot(x, _safe_positive(y_vt_b), marker="^", linestyle="--", label=f"VT (eps2={eps2_b:g})")

    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("trace")
    ax.set_title(title + " (log-y)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_log, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _plot_trace_two_eps1(
    *,
    x: np.ndarray,
    # eps1 = a
    y_mean_a: np.ndarray,
    y_se_a: np.ndarray,
    y_cg_a: np.ndarray,
    y_vt_a: np.ndarray,
    eps1_a: float,
    # eps1 = b
    y_mean_b: np.ndarray,
    y_se_b: np.ndarray,
    y_cg_b: np.ndarray,
    y_vt_b: np.ndarray,
    eps1_b: float,
    title: str,
    out_linear: Path,
    out_log: Path,
) -> None:
    """Overlay trace curves for two eps1 values (linear-y and log-y)."""
    # ---- linear-y
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(
        x, y_mean_a, yerr=1.96 * y_se_a,
        marker="o", linestyle="-", capsize=3,
        label=f"Risk (eps1={eps1_a:g})",
    )
    ax.plot(x, y_cg_a, marker="s", linestyle="--", label=f"CG (eps1={eps1_a:g})")
    ax.plot(x, y_vt_a, marker="^", linestyle="--", label=f"VT (eps1={eps1_a:g})")

    ax.errorbar(
        x, y_mean_b, yerr=1.96 * y_se_b,
        marker="o", linestyle="-", capsize=3,
        label=f"Risk (eps1={eps1_b:g})",
    )
    ax.plot(x, y_cg_b, marker="s", linestyle="--", label=f"CG (eps1={eps1_b:g})")
    ax.plot(x, y_vt_b, marker="^", linestyle="--", label=f"VT (eps1={eps1_b:g})")

    ax.set_xlabel("n")
    ax.set_ylabel("trace")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_linear, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- log-y
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.errorbar(
        x, _safe_positive(y_mean_a), yerr=1.96 * y_se_a,
        marker="o", linestyle="-", capsize=3,
        label=f"Risk (eps1={eps1_a:g})",
    )
    ax.plot(x, _safe_positive(y_cg_a), marker="s", linestyle="--", label=f"CG (eps1={eps1_a:g})")
    ax.plot(x, _safe_positive(y_vt_a), marker="^", linestyle="--", label=f"VT (eps1={eps1_a:g})")

    ax.errorbar(
        x, _safe_positive(y_mean_b), yerr=1.96 * y_se_b,
        marker="o", linestyle="-", capsize=3,
        label=f"Risk (eps1={eps1_b:g})",
    )
    ax.plot(x, _safe_positive(y_cg_b), marker="s", linestyle="--", label=f"CG (eps1={eps1_b:g})")
    ax.plot(x, _safe_positive(y_vt_b), marker="^", linestyle="--", label=f"VT (eps1={eps1_b:g})")

    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("trace")
    ax.set_title(title + " (log-y)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_log, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _plot_trace_tail_extrapolation(
    *,
    x: np.ndarray,
    risk_trace: np.ndarray,
    cg_trace: np.ndarray,
    vt_trace: np.ndarray,
    fit_risk: dict,
    fit_cg: dict,
    fit_vt: dict,
    risk_cg_intersection: Optional[dict],
    risk_vt_intersection: Optional[dict],
    out_path: Path,
    title: str,
    x_extension_factor: float = 10.0,
) -> None:
    """
    Plot the observed trace curves together with tail power-law extrapolations.
    """
    x = np.asarray(x, dtype=float)

    xmax = float(np.nanmax(x))
    xmin = float(np.nanmin(x))
    x_fit = np.geomspace(xmin, xmax * float(x_extension_factor), 400)

    fig, ax = plt.subplots(figsize=(7, 4))

    # observed curves
    ax.plot(x, risk_trace, marker="o", linestyle="-", label="Risk (observed)")
    ax.plot(x, cg_trace, marker="s", linestyle="-", label="CG (observed)")
    ax.plot(x, vt_trace, marker="^", linestyle="-", label="VT (observed)")

    # fitted tail lines
    if np.isfinite(fit_risk["intercept"]) and np.isfinite(fit_risk["slope"]):
        ax.plot(
            x_fit,
            _power_law_eval(x_fit, fit_risk["intercept"], fit_risk["slope"]),
            linestyle="--",
            label=f"Risk tail fit (slope={fit_risk['slope']:.3f})",
        )

    if np.isfinite(fit_cg["intercept"]) and np.isfinite(fit_cg["slope"]):
        ax.plot(
            x_fit,
            _power_law_eval(x_fit, fit_cg["intercept"], fit_cg["slope"]),
            linestyle="--",
            label=f"CG tail fit (slope={fit_cg['slope']:.3f})",
        )

    if np.isfinite(fit_vt["intercept"]) and np.isfinite(fit_vt["slope"]):
        ax.plot(
            x_fit,
            _power_law_eval(x_fit, fit_vt["intercept"], fit_vt["slope"]),
            linestyle="--",
            label=f"VT tail fit (slope={fit_vt['slope']:.3f})",
        )

    # intersections
    if risk_cg_intersection is not None:
        ax.axvline(
            risk_cg_intersection["x"],
            linestyle=":",
            linewidth=1.2,
            label=f"Risk-CG intersect ~ n={risk_cg_intersection['x']:.2e}",
        )

    if risk_vt_intersection is not None:
        ax.axvline(
            risk_vt_intersection["x"],
            linestyle=":",
            linewidth=1.2,
            label=f"Risk-VT intersect ~ n={risk_vt_intersection['x']:.2e}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("trace")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# Example usage
# Regular Point
# =============================================================================

# if __name__ == "__main__":
#     j = 3
#     m_c = 1
#     chi = 0.0

#     theta0 = (PI / 3.0, PI / 4.0)
#     eps1 = 0.01
#     eps2_list = [0.10, 0.50]

#     n_values = [50, 100, 200, 400, 800, 1600]
#     r_batch = 300
#     m_batch = 20

#     # Use one parent folder, create two subfolders inside
#     stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     root = Path(f"compare_eps2_trace_{stamp}")
#     root.mkdir(parents=True, exist_ok=True)

#     # ---- run eps2=0.10
#     summary_a, out_a = study_matricial_risk_vs_bounds(
#         j=j,
#         m_c=m_c,
#         chi=chi,
#         theta0=theta0,
#         eps1=eps1,
#         eps2=eps2_list[0],
#         n_values=n_values,
#         r_batch=r_batch,
#         m_batch=m_batch,
#         grid_num_beta=161,
#         grid_num_phi=241,
#         base_seed=1234,
#         output_root=root / f"eps2_{eps2_list[0]:g}",
#         make_psd_diagnostic_plots=True,
#         # taper (optional)
#         use_taper=True,
#         taper_type="bump",
#         bump_alpha=2.0,
#         # neighborhood choice: keep whatever you want here
#         neighborhood_mode="auto",
#         half_plane="phi_ge",
#     )

#     # ---- run eps2=0.50
#     summary_b, out_b = study_matricial_risk_vs_bounds(
#         j=j,
#         m_c=m_c,
#         chi=chi,
#         theta0=theta0,
#         eps1=eps1,
#         eps2=eps2_list[1],
#         n_values=n_values,
#         r_batch=r_batch,
#         m_batch=m_batch,
#         grid_num_beta=161,
#         grid_num_phi=241,
#         base_seed=5678,  # use a different seed to avoid identical MC streams
#         output_root=root / f"eps2_{eps2_list[1]:g}",
#         make_psd_diagnostic_plots=True,
#         # taper (optional)
#         use_taper=True,
#         taper_type="bump",
#         bump_alpha=2.0,
#         neighborhood_mode="auto",
#         half_plane="phi_ge",
#     )

#     # ---- combined trace plot
#     n_arr = np.array(sorted(set(summary_a["n"].tolist())), dtype=int)

#     # Ensure alignment by n (robust if you later vary n grids)
#     a = summary_a.set_index("n").loc[n_arr]
#     b = summary_b.set_index("n").loc[n_arr]

#     title = f"Trace comparison at theta0=(pi/3, pi/4), j={j}, m_c={m_c}, chi={chi}"

#     _plot_trace_two_eps2(
#         x=n_arr,
#         y_mean_a=a["risk_trace_mean"].to_numpy(float),
#         y_se_a=a["risk_trace_se"].to_numpy(float),
#         y_cg_a=a["cg_trace"].to_numpy(float),
#         y_vt_a=a["vt_trace"].to_numpy(float),
#         eps2_a=float(eps2_list[0]),
#         y_mean_b=b["risk_trace_mean"].to_numpy(float),
#         y_se_b=b["risk_trace_se"].to_numpy(float),
#         y_cg_b=b["cg_trace"].to_numpy(float),
#         y_vt_b=b["vt_trace"].to_numpy(float),
#         eps2_b=float(eps2_list[1]),
#         title=title,
#         out_linear=root / "trace_compare_eps2_linear.png",
#         out_log=root / "trace_compare_eps2_logy.png",
#     )

#     print("Saved eps2=0.10 outputs to:", out_a)
#     print("Saved eps2=0.50 outputs to:", out_b)
#     print("Saved combined trace plots to:", root)



# =============================================================================
# Example usage
# Singular Point
# =============================================================================

if __name__ == "__main__":
    j = 3
    m_c = 1
    chi = 0.0

    theta0 = (PI / 3.0, PI)
    eps2 = 0.10
    eps1_list = [0.01, 0.05]

    n_values = [50, 100, 200, 400, 800, 1600, 3200, 6400]
    r_batch = 300
    m_batch = 20

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(f"compare_eps1_trace_{stamp}")
    root.mkdir(parents=True, exist_ok=True)

    # Fixed neighborhood choice: half-ring
    neighborhood_mode = "half_ring"
    half_plane = "phi_ge"  # or "phi_le"

    # ---- run eps1=0.01
    summary_a, out_a = study_matricial_risk_vs_bounds(
        j=j,
        m_c=m_c,
        chi=chi,
        theta0=theta0,
        eps1=eps1_list[0],
        eps2=eps2,
        n_values=n_values,
        r_batch=r_batch,
        m_batch=m_batch,
        grid_num_beta=161,
        grid_num_phi=241,
        base_seed=1234,
        output_root=root / f"eps1_{eps1_list[0]:g}",
        make_psd_diagnostic_plots=True,
        # taper (optional)
        use_taper=True,
        taper_type="bump",
        bump_alpha=2.0,
        # half-ring
        neighborhood_mode=neighborhood_mode,
        half_plane=half_plane,
    )

    # ---- run eps1=0.05
    summary_b, out_b = study_matricial_risk_vs_bounds(
        j=j,
        m_c=m_c,
        chi=chi,
        theta0=theta0,
        eps1=eps1_list[1],
        eps2=eps2,
        n_values=n_values,
        r_batch=r_batch,
        m_batch=m_batch,
        grid_num_beta=161,
        grid_num_phi=241,
        base_seed=5678,  # different stream
        output_root=root / f"eps1_{eps1_list[1]:g}",
        make_psd_diagnostic_plots=True,
        # taper (optional)
        use_taper=True,
        taper_type="bump",
        bump_alpha=2.0,
        # half-ring
        neighborhood_mode=neighborhood_mode,
        half_plane=half_plane,
    )

    # ---- combined trace plot
    n_arr = np.array(sorted(set(summary_a["n"].tolist())), dtype=int)
    a = summary_a.set_index("n").loc[n_arr]
    b = summary_b.set_index("n").loc[n_arr]

    title = (
        f"Trace at theta0=(pi/3, pi), eps2={eps2:g}, {neighborhood_mode}, "
        f"half_plane={half_plane}, j={j}, m_c={m_c}, chi={chi}"
    )

    _plot_trace_two_eps1(
        x=n_arr,
        y_mean_a=a["risk_trace_mean"].to_numpy(float),
        y_se_a=a["risk_trace_se"].to_numpy(float),
        y_cg_a=a["cg_trace"].to_numpy(float),
        y_vt_a=a["vt_trace"].to_numpy(float),
        eps1_a=float(eps1_list[0]),
        y_mean_b=b["risk_trace_mean"].to_numpy(float),
        y_se_b=b["risk_trace_se"].to_numpy(float),
        y_cg_b=b["cg_trace"].to_numpy(float),
        y_vt_b=b["vt_trace"].to_numpy(float),
        eps1_b=float(eps1_list[1]),
        title=title,
        out_linear=root / "trace_compare_eps1_linear.png",
        out_log=root / "trace_compare_eps1_logy.png",
    )

    print("Saved eps1=0.01 outputs to:", out_a)
    print("Saved eps1=0.05 outputs to:", out_b)
    print("Saved combined trace plots to:", root)
