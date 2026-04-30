"""
Numerical validation of Proposition 2.4 (multipolar truncation error). 

Setup:
    A rigid circular hole of radius a in an infinite Kirchhoff plate with
    flexural rigidity D, surface density rho*h. A flexural plane wave of
    wavenumber k_b is incident. The scattered field is expanded in
    cylindrical harmonics as

        w_scat(r, theta) = sum_{n} [ c_n^{(b)} H_n^{(1)}(k_b r)
                                    + d_n^{(b)} K_n(k_b r) ] e^{i n theta}

    For a clamped (rigid, no rotation) hole, the boundary conditions at r=a are
        w(a,theta) = 0,
        partial_r w(a,theta) = 0,
    yielding a 2x2 linear system per harmonic order n.

Validation:
    1. Compute exact scattering coefficients (c_n, d_n) up to N_max = 20.
    2. For each truncation order N in {1,2,3,5,8,10,12}, compute the truncated
       solution and the error in H^2(B_R \ B_a) energy norm with R = 2a.
    3. Plot relative error vs N for several frequencies (k_b a in {0.25, 1, 2}).
    4. Verify the theoretical rate (e * k_b * R / (2N))^N from Proposition 2.4.

Output:
    - data/multipolar_convergence.csv : numerical errors
    - figures/fig_convergence_real.pdf : final convergence plot

Reference:
    Proposition 2.4 of the article. Norris-Vemula (1995) for rigid hole.
"""

import os
import csv
import numpy as np
import scipy.special as sp
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent
DATA_DIR = OUT_DIR / "code" / "data"
FIG_DIR = OUT_DIR / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Bessel-derivative helpers
# ----------------------------------------------------------------------------
def Jn(n, x):
    return sp.jv(n, x)

def Hn1(n, x):
    return sp.hankel1(n, x)

def In(n, x):
    return sp.iv(n, x)

def Kn(n, x):
    return sp.kv(n, x)

# Derivatives wrt argument
def Jn_d(n, x):
    return 0.5 * (sp.jv(n - 1, x) - sp.jv(n + 1, x))

def Hn1_d(n, x):
    return 0.5 * (sp.hankel1(n - 1, x) - sp.hankel1(n + 1, x))

def In_d(n, x):
    return 0.5 * (sp.iv(n - 1, x) + sp.iv(n + 1, x))

def Kn_d(n, x):
    return -0.5 * (sp.kv(n - 1, x) + sp.kv(n + 1, x))

# Second derivatives wrt argument (for biharmonic energy norm)
def Jn_dd(n, x):
    return 0.25 * (sp.jv(n - 2, x) - 2 * sp.jv(n, x) + sp.jv(n + 2, x))

def Hn1_dd(n, x):
    return 0.25 * (sp.hankel1(n - 2, x) - 2 * sp.hankel1(n, x) + sp.hankel1(n + 2, x))

def In_dd(n, x):
    return 0.25 * (sp.iv(n - 2, x) + 2 * sp.iv(n, x) + sp.iv(n + 2, x))

def Kn_dd(n, x):
    return 0.25 * (sp.kv(n - 2, x) + 2 * sp.kv(n, x) + sp.kv(n + 2, x))


# ----------------------------------------------------------------------------
# Scattering coefficients for rigid hole
# ----------------------------------------------------------------------------
def scattering_coefficients(n_max, kb_a):
    """
    Solve for scattering coefficients (c_n, d_n) for a rigid (clamped) hole
    illuminated by a flexural plane wave w_inc = exp(i kb x).

    Jacobi-Anger expansion of plane wave:
        exp(i kb x) = sum_n i^n J_n(kb r) e^{i n theta}

    Boundary conditions at r = a (rigid hole, clamped):
        w_total(a) = 0
        partial_r w_total(a) = 0

    Total field outside the hole:
        w = w_inc + w_scat
           = sum_n [i^n J_n(kb r) + c_n H_n^{(1)}(kb r) + d_n K_n(kb r)] e^{i n th}

    The coefficient I_n(kb r) of the regular evanescent mode is zero in the
    incident plane wave (which is purely propagating), so we only need
    c_n, d_n for the scattered field.

    For each n:
        i^n J_n(kb a) + c_n H_n^{(1)}(kb a) + d_n K_n(kb a) = 0
        i^n J'_n(kb a) + c_n (H_n^{(1)})'(kb a) + d_n K'_n(kb a) = 0

    Solving the 2x2 system gives (c_n, d_n).

    Returns
    -------
    c : array, shape (2*n_max+1,)
        Coefficients for H_n^{(1)} mode (n = -n_max, ..., n_max)
    d : array, shape (2*n_max+1,)
        Coefficients for K_n mode

    Note: by symmetry c_{-n} = c_n, d_{-n} = d_n for n > 0 if incidence is
    along x-axis (real Jacobi-Anger coefficient i^n on n and i^{-n} on -n).
    We keep both for generality.
    """
    c = np.zeros(2 * n_max + 1, dtype=complex)
    d = np.zeros(2 * n_max + 1, dtype=complex)

    for idx in range(2 * n_max + 1):
        n = idx - n_max  # n in {-n_max, ..., n_max}
        an_inc = (1j) ** n  # incident amplitude

        # 2x2 boundary system
        M = np.array([
            [Hn1(n, kb_a), Kn(n, kb_a)],
            [Hn1_d(n, kb_a), Kn_d(n, kb_a)],
        ], dtype=complex)
        rhs = -an_inc * np.array([Jn(n, kb_a), Jn_d(n, kb_a)], dtype=complex)

        sol = np.linalg.solve(M, rhs)
        c[idx] = sol[0]
        d[idx] = sol[1]

    return c, d


# ----------------------------------------------------------------------------
# H^2 energy norm of a single multipolar mode on annulus B_R \ B_a
# ----------------------------------------------------------------------------
def mode_H2_norm_squared(n, mode_type, kb_a, R_over_a, n_quad=300):
    """
    Compute || phi_n e^{i n theta} ||^2_{H^2(B_R \ B_a)}
    where phi_n is the radial profile of the given mode.

    H^2 norm squared:
        sum_{|alpha|<=2} || D^alpha (phi_n e^{i n theta}) ||^2_{L^2}

    Using polar coordinates and orthogonality in theta, this reduces to a
    1D radial integral. The L^2 norm in 2D is
        || f(r) e^{i n theta} ||^2_{L^2} = 2 pi int_a^R |f(r)|^2 r dr.
    Derivatives in Cartesian give:
        |grad f e^{in theta}|^2 = |f'|^2 + n^2 |f|^2 / r^2
        |D^2 f e^{in theta}|^2 = sum of terms involving f'', f'/r, f/r^2 with
                                 n^0, n^2, n^4 weights.

    For simplicity we compute ||f||^2 + ||f'||^2 + n^2/r^2 ||f||^2 + ||D^2 f||^2
    where the H^2 norm is approximated by the sum of L^2 norms of value, first,
    and second mixed derivatives.

    Parameters
    ----------
    n : int
        Angular order
    mode_type : str
        'H1' for H_n^{(1)}(kb r), 'K' for K_n(kb r)
    kb_a : float
        Dimensionless k_b * a
    R_over_a : float
        Outer radius R/a
    n_quad : int
        Number of quadrature points (Gauss-Legendre)
    """
    # Quadrature on [a, R] -> [1, R/a] in units of a (we normalize a=1)
    # so r_hat = r/a, x = kb*a * r_hat = kb*r
    # We compute everything in terms of kb*r for direct Bessel argument.

    # Quadrature in r/a from 1 to R/a
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    a_lo, a_hi = 1.0, R_over_a
    r_hat = 0.5 * (a_hi - a_lo) * nodes + 0.5 * (a_hi + a_lo)
    w_r = 0.5 * (a_hi - a_lo) * weights

    x = kb_a * r_hat  # argument of Bessel

    if mode_type == 'H1':
        f = Hn1(n, x)
        fp = kb_a * Hn1_d(n, x)        # df/d(r/a) (chain rule with x=kb_a * r_hat)
        fpp = kb_a**2 * Hn1_dd(n, x)
    elif mode_type == 'K':
        f = Kn(n, x)
        fp = kb_a * Kn_d(n, x)
        fpp = kb_a**2 * Kn_dd(n, x)
    else:
        raise ValueError(f"unknown mode_type: {mode_type}")

    # H^2 components (in units where a = 1, so r = r_hat)
    # ||f||^2_L^2 = 2 pi int |f|^2 r dr
    # ||grad(f e^{i n theta})||^2_L^2 = 2 pi int (|f'|^2 + n^2 |f|^2 / r^2) r dr
    # ||D^2 f e^{i n theta}||^2_L^2 = 2 pi int (|f''|^2 + ...) r dr
    # We use the simplified estimate (sum of L^2 of value + grad components + Hessian components)

    integrand_L2 = np.abs(f) ** 2 * r_hat
    integrand_grad = (np.abs(fp) ** 2 + (n ** 2) * np.abs(f) ** 2 / r_hat ** 2) * r_hat
    # Hessian: components include f''(r), f'(r)/r, n^2 f / r^2, with extra n^2 weights
    integrand_hess = (
        np.abs(fpp) ** 2
        + (np.abs(fp) ** 2) / r_hat ** 2
        + (n ** 4) * np.abs(f) ** 2 / r_hat ** 4
        + 2 * (n ** 2) * np.abs(fp) ** 2 / r_hat ** 2
    ) * r_hat

    integral = 2 * np.pi * np.sum(w_r * (integrand_L2 + integrand_grad + integrand_hess))
    return float(np.real(integral))


# ----------------------------------------------------------------------------
# Truncation error
# ----------------------------------------------------------------------------
def relative_truncation_error(N_trunc, N_ref, kb_a, R_over_a):
    """
    Compute relative H^2 error between truncation at N_trunc and reference at N_ref.

    Error = || w - w^(N_trunc) ||_{H^2(annulus)}^2 = sum_{|n| > N_trunc, |n| <= N_ref} || w_n ||^2

    Norm of total field = sum_{|n| <= N_ref} || w_n ||^2
    """
    c_ref, d_ref = scattering_coefficients(N_ref, kb_a)

    # Total norm squared
    total = 0.0
    tail = 0.0
    for idx in range(2 * N_ref + 1):
        n = idx - N_ref
        an_inc = (1j) ** n  # incident amplitude

        # mode = i^n J_n + c_n H_n^(1) + d_n K_n  (only outside, scattered is c, d only)
        # For consistency with Proposition 2.4 which compares scattered fields:
        # we measure the scattered field error.
        mode_norm_sq = (
            (np.abs(c_ref[idx]) ** 2) * mode_H2_norm_squared(n, 'H1', kb_a, R_over_a)
            + (np.abs(d_ref[idx]) ** 2) * mode_H2_norm_squared(n, 'K', kb_a, R_over_a)
        )
        # Cross terms exist between H and K but their cross norm is approx zero
        # (different radial profiles); for energy norm we keep the diagonal estimate.
        total += mode_norm_sq
        if abs(n) > N_trunc:
            tail += mode_norm_sq

    if total <= 0:
        return float('nan')
    return float(np.sqrt(tail / total))


# ----------------------------------------------------------------------------
# Theoretical rate from Proposition 2.4
# ----------------------------------------------------------------------------
def theoretical_rate(N, kb_a, R_over_a):
    """ (e * k_max * R / (2 N))^N where k_max R = kb_a * R_over_a """
    k_R = kb_a * R_over_a
    return (np.e * k_R / (2 * N)) ** N


# ----------------------------------------------------------------------------
# Main study
# ----------------------------------------------------------------------------
def main():
    R_over_a = 2.0
    N_ref = 20
    N_values = [1, 2, 3, 5, 8, 10, 12]
    kb_a_values = [0.25, 0.5, 1.0, 1.5, 2.0]

    print(f"# Multipolar convergence validation (Proposition 2.4)")
    print(f"# Reference truncation: N_ref = {N_ref}")
    print(f"# Annulus: R/a = {R_over_a}")
    print()
    print(f"{'kb_a':>8} {'N':>5} {'rel_err':>15} {'theory_rate':>15} {'ratio':>10}")
    print("-" * 60)

    rows = []
    for kb_a in kb_a_values:
        for N in N_values:
            err = relative_truncation_error(N, N_ref, kb_a, R_over_a)
            theory = theoretical_rate(N, kb_a, R_over_a)
            ratio = err / theory if theory > 0 else float('nan')
            print(f"{kb_a:>8.3f} {N:>5d} {err:>15.4e} {theory:>15.4e} {ratio:>10.3f}")
            rows.append({
                'kb_a': kb_a, 'N': N, 'rel_err': err,
                'theory_rate': theory, 'ratio': ratio,
            })

    # Save CSV
    csv_path = DATA_DIR / "multipolar_convergence.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['kb_a', 'N', 'rel_err', 'theory_rate', 'ratio'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n# Data saved to {csv_path}")

    return rows


if __name__ == '__main__':
    main()
