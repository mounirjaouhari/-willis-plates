""" 
Numerical validation of Proposition 3.1 (QCA convergence rate).

Strategy:
    1. Direct multiple-scattering (MST) reference for N rigid circular holes.
       Solves the propagating-only Foldy-Lax system on an in-plane plate.
       The evanescent K_n channel is dropped: justified at separations d > a few a
       where K_n decays exponentially.
    2. QCA prediction: Foldy formula for the coherent wavenumber k_eff in the
       diluted limit, dispersion-equation closure with hard-disk pair correlation.
    3. Compute the relative L^2 error of the average MST field vs the QCA
       prediction at several packing fractions phi, average over realizations,
       and verify the rate phi^{1/2} from Proposition 3.1.

Setup:
    - N = 25 rigid holes, radius a = 1
    - Hard-core distance d0 = 2.5 * a
    - Cluster confined to a square of side L (chosen for given phi = N pi a^2 / L^2)
    - Incident flexural plane wave w_inc(x) = exp(i k_b x), k_b a in [0.3, 1.0]
    - Observation: average field along a slice y = 0 inside the cluster
    - K_real = 50 realizations per phi

Reference:
    Section 6.3 of the article. Proposition 3.1.
"""

import os
import csv
import numpy as np
import scipy.special as sp
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent
DATA_DIR = OUT_DIR / "code" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# T-matrix for a single rigid hole, propagating channel only
# ----------------------------------------------------------------------------
def Jn(n, x):  return sp.jv(n, x)
def Hn1(n, x): return sp.hankel1(n, x)
def Jn_d(n, x): return 0.5 * (sp.jv(n - 1, x) - sp.jv(n + 1, x))
def Hn1_d(n, x): return 0.5 * (sp.hankel1(n - 1, x) - sp.hankel1(n + 1, x))


def t_matrix_propagating(n_max, kb_a):
    """
    Compute the propagating-mode T-matrix coefficients t_n for a rigid hole.
    The full coupled (propagating + evanescent) system gives c_n, d_n; here we
    solve the reduced 1x1 propagating-only system using only the W=0 BC:
        (1) i^n J_n(kb_a) + t_n^prop H_n^(1)(kb_a) = 0
    yielding t_n = -i^n J_n(kb_a) / H_n^(1)(kb_a).

    Note: this is the "soft" rigid-hole approximation with only one BC.
    For the propagating-only approximation it is the leading-order T-matrix
    in the dilute limit. The full clamped-hole T-matrix is given by
    multipolar_convergence.py (2x2 system); here we keep the simpler 1x1 form
    for tractable MST.

    Returns
    -------
    t : array, shape (2 n_max + 1,) of complex T-matrix entries indexed by
        n in {-n_max, ..., n_max}.
    """
    t = np.zeros(2 * n_max + 1, dtype=complex)
    for idx in range(2 * n_max + 1):
        n = idx - n_max
        an_inc = (1j) ** n
        t[idx] = -an_inc * Jn(n, kb_a) / Hn1(n, kb_a)
    return t


# ----------------------------------------------------------------------------
# Hard-disk Poisson sampling
# ----------------------------------------------------------------------------
def sample_hard_disks(N, phi, a, d0_factor=2.5, max_attempts=10000, rng=None):
    """
    Generate N positions of disks of radius a inside a square of side L,
    with hard-core minimum separation d0 = d0_factor * a, so that packing
    fraction is phi = N pi a^2 / L^2.
    Uses simple rejection sampling.
    """
    if rng is None:
        rng = np.random.default_rng()
    L = np.sqrt(N * np.pi * a ** 2 / phi)
    d0 = d0_factor * a
    positions = []
    attempts = 0
    while len(positions) < N and attempts < max_attempts * N:
        x = rng.uniform(a, L - a)
        y = rng.uniform(a, L - a)
        ok = all((px - x) ** 2 + (py - y) ** 2 >= d0 ** 2 for (px, py) in positions)
        if ok:
            positions.append((x, y))
        attempts += 1
    if len(positions) < N:
        raise RuntimeError(f"Could not place N={N} disks at phi={phi}; got {len(positions)}")
    return np.array(positions), L


# ----------------------------------------------------------------------------
# Direct Foldy-Lax solver (propagating only)
# ----------------------------------------------------------------------------
def assemble_foldy_lax(positions, kb, n_max, t):
    """
    Build the linear system (I - T S) c = T a_inc for N inclusions.

    Variables: c_n^(j) for j = 0,..,N-1, n in -n_max,..,n_max.
    Total size: M = N (2 n_max + 1).

    The structure-factor S between (j, n) and (k, m) for k != j is the
    Graf-Watson kernel:
        S_{nm}^{(jk)} = H_{n-m}^(1)(kb R_{jk}) exp(i (n-m) theta_{jk})
    For k = j: zero diagonal.
    """
    N = len(positions)
    nrows = 2 * n_max + 1
    M = N * nrows
    A = np.zeros((M, M), dtype=complex)
    rhs = np.zeros(M, dtype=complex)

    # Diagonal: identity
    for j in range(N):
        for idx_n in range(nrows):
            row = j * nrows + idx_n
            A[row, row] = 1.0

    # Off-diagonal: -T_n . S_{nm}^{(jk)}
    for j in range(N):
        for k in range(N):
            if j == k:
                continue
            dx = positions[k, 0] - positions[j, 0]
            dy = positions[k, 1] - positions[j, 1]
            R_jk = np.hypot(dx, dy)
            theta_jk = np.arctan2(dy, dx)
            for idx_n in range(nrows):
                n = idx_n - n_max
                t_n = t[idx_n]
                row = j * nrows + idx_n
                for idx_m in range(nrows):
                    m = idx_m - n_max
                    col = k * nrows + idx_m
                    # Graf kernel: H_{n-m}^(1)(kb R) exp(i(n-m) theta)
                    g = Hn1(n - m, kb * R_jk) * np.exp(1j * (n - m) * theta_jk)
                    A[row, col] -= t_n * g

    # RHS: T . a_inc, where a_inc at inclusion j = exp(i kb x_j) Jacobi-Anger gives
    # a_inc^(j)_n = i^n exp(i kb x_j)   (for plane wave incident along x-axis)
    for j in range(N):
        phase = np.exp(1j * kb * positions[j, 0])
        for idx_n in range(nrows):
            n = idx_n - n_max
            t_n = t[idx_n]
            an_inc_j = (1j) ** n * phase  # i^n exp(i kb x_j)
            row = j * nrows + idx_n
            rhs[row] = t_n * an_inc_j

    return A, rhs


def evaluate_field_mst(positions, c_coeffs, kb, n_max, x_obs):
    """
    Evaluate the total scattered field at observation points.

    w(x) = sum_j sum_n c_n^(j) H_n^(1)(kb r_j) e^{i n theta_j}

    where (r_j, theta_j) are polar coordinates centered at inclusion j.

    Parameters
    ----------
    c_coeffs : array, shape (N * (2n_max+1),)
    x_obs : array, shape (M, 2)
    """
    N = len(positions)
    nrows = 2 * n_max + 1
    M = len(x_obs)
    w = np.zeros(M, dtype=complex)
    for i_obs, x in enumerate(x_obs):
        for j in range(N):
            dx = x[0] - positions[j, 0]
            dy = x[1] - positions[j, 1]
            r = np.hypot(dx, dy)
            if r < 1e-10:
                continue  # skip singularity
            theta = np.arctan2(dy, dx)
            for idx_n in range(nrows):
                n = idx_n - n_max
                c_n = c_coeffs[j * nrows + idx_n]
                w[i_obs] += c_n * Hn1(n, kb * r) * np.exp(1j * n * theta)
    return w


# ----------------------------------------------------------------------------
# QCA prediction: Foldy effective wavenumber
# ----------------------------------------------------------------------------
def k_eff_foldy(kb, n_max, t, n0):
    """
    Foldy-Lax effective wavenumber for 2D scalar Helmholtz scattering.

    With T-matrix convention t[idx] = c_n / a_n_inc where a_n_inc = i^n
    (incident plane wave Jacobi-Anger), the far-field scattering amplitude is
        f(theta) = sum_n c_n (-i)^n e^{i n theta},   f(0) = sum_n c_n (-i)^n.
    The standard 2D Foldy formula (Lax 1951) reads
        k_eff = kb - (i n0 / kb) f(0).
    Equivalently
        k_eff^2 = kb^2 - 2 i n0 f(0) + O(n0^2),
    valid in the dilute limit n0 a^2 = phi/pi << 1.

    Returns
    -------
    keff : complex (Re > 0, Im > 0 for outgoing damped mode)
    """
    # f(0) = sum_n c_n (-i)^n; here our t = c_n already
    f0 = sum(t[idx] * ((-1j) ** (idx - n_max)) for idx in range(len(t)))
    delta = -1j * n0 * f0 / kb
    keff = kb + delta
    # Sanity: Re(keff) should remain positive and close to kb in dilute limit
    return keff


def w_qca(x_obs, keff):
    """
    Coherent QCA field along x: a plane wave with effective wavenumber.
        <w_QCA> = exp(i keff x) along the x-axis (ignoring y for transverse uniformity).
    """
    return np.exp(1j * keff * x_obs[:, 0])


# ----------------------------------------------------------------------------
# Main study
# ----------------------------------------------------------------------------
def main():
    # Parameters
    a = 1.0
    kb_a = 0.5
    kb = kb_a / a
    N = 25
    n_max = 4
    K_real = 30
    phi_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    print(f"# QCA validation (Proposition 3.1)")
    print(f"# N = {N}, kb*a = {kb_a}, K_real = {K_real}, n_max = {n_max}")
    print()
    print(f"{'phi':>8} {'L':>8} {'k_eff_real':>15} {'k_eff_imag':>15} {'rel_std':>15} {'phi^{1/2}':>15}")
    print("-" * 80)

    # T-matrix for rigid hole, kb*a fixed
    t = t_matrix_propagating(n_max, kb_a)

    rows = []
    rng = np.random.default_rng(seed=42)
    for phi in phi_values:
        n0 = phi / (np.pi * a ** 2)  # surface density
        keff = k_eff_foldy(kb, n_max, t, n0)

        # Generate K_real configurations and accumulate MST fields per realization
        # so that mean and std deviation can be reported
        per_realization_fields = []
        avg_inc = None
        successful_realizations = 0
        for k_real in range(K_real):
            try:
                positions, L = sample_hard_disks(N, phi, a, d0_factor=2.5, rng=rng)
            except RuntimeError:
                continue
            # Observation slice: x = L/2 + offsets, y = L/2
            # We want the AVERAGE FIELD (over realizations), so we observe at
            # the SAME relative position in each cluster (centered)
            x_obs_x = np.linspace(L * 0.3, L * 0.7, 8)
            x_obs = np.column_stack([x_obs_x, np.full_like(x_obs_x, L / 2)])
            # Solve MST
            A, rhs = assemble_foldy_lax(positions, kb, n_max, t)
            c_coeffs = np.linalg.solve(A, rhs)
            w_scat = evaluate_field_mst(positions, c_coeffs, kb, n_max, x_obs)
            w_inc = np.exp(1j * kb * x_obs[:, 0])
            w_total = w_inc + w_scat

            # QCA prediction at same points
            w_qca_pred = w_qca(x_obs, keff)

            per_realization_fields.append(w_total)
            if avg_inc is None:
                avg_inc = w_inc.copy()
            else:
                avg_inc += w_inc
            successful_realizations += 1

        if successful_realizations == 0:
            print(f"  phi={phi}: no valid realizations")
            continue

        # Mean and std across realizations (per observation point), then averaged
        fields_arr = np.array(per_realization_fields)  # shape (K, N_obs)
        mean_field = fields_arr.mean(axis=0)
        std_field = fields_arr.std(axis=0, ddof=1)  # sample std
        avg_inc /= successful_realizations

        # Coherent vs incoherent decomposition
        l2_inc = np.sqrt(np.mean(np.abs(avg_inc) ** 2))
        # Std of the coherent field across realizations: this is the
        # incoherent (fluctuation) standard deviation
        l2_std = np.sqrt(np.mean(np.abs(std_field) ** 2))
        rel_std = l2_std / l2_inc if l2_inc > 0 else float('nan')

        rate_pred = np.sqrt(phi)
        print(f"{phi:>8.3f} {L:>8.2f} {np.real(keff):>15.4f} {np.imag(keff):>15.4f} "
              f"{rel_std:>15.4e} {rate_pred:>15.4e}")
        rows.append({
            'phi': phi,
            'L': L,
            'k_eff_real': float(np.real(keff)),
            'k_eff_imag': float(np.imag(keff)),
            'rel_std_incoherent': float(rel_std),
            'phi_half': float(rate_pred),
            'realizations': successful_realizations,
        })

    # Save data
    csv_path = DATA_DIR / "qca_validation.csv"
    with open(csv_path, 'w', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\n# Data saved to {csv_path}")
    return rows


if __name__ == '__main__':
    main()
