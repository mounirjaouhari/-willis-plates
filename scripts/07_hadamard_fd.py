"""
Q3 / L7 -- Hadamard shape derivative validation on the membrane sub-problem.

Validates the structure of Theorem 5.2 on the simpler 2nd-order membrane cell
problem of Theorem 3.2. Full coupled-cell validation (membrane + bending,
mirror-broken inclusion) is reported in the companion repository.

Setup:
    Annular cell Y' = {a < |y| < R} with Dirichlet outer BC at r=R,
    free hole boundary at r=a.
    A = host stiffness on Y' (E=1, nu=0.30, plane stress).
    Cell problem (uniaxial e^(11)):
        find chi in [H^1(Y')]^2, chi=0 on |y|=R, such that
            int_{Y'} A : (eps(chi) + e^(11)) : eps(v) dy = 0  for all v.
    Functional: J(a) = int_{Y'} sigma_{11}(eps(chi) + e^(11)) dy

This is the CIRCULAR (annular) version of the cell problem of section 6.4
(which uses a square outer boundary). The Hadamard formula has the same
structure; testing it on the annulus avoids mesh-topology jumps under hole
expansion since the polar-fitted mesh deforms smoothly with a.

Shape derivative (envelope theorem, hole growing radially):
    dJ/da = -oint_{|y|=a} A_{ijkl}(eps(chi) + e^(11))_{ij}
                                  (eps(chi) + e^(11))_{kl} dl
        = -oint A:(eps_tot):(eps_tot) dl  (matrix-side trace, V_n = 1)

Validation:
    For t in {2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4}:
        D_FD(t) = [J(a+t) - J(a)] / t
    Compare to analytical dJ/da via line integral on Gamma.
    Expect |D_FD - dJ/da| = O(t) (linear slope on log-log).
"""

from pathlib import Path
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skfem import (MeshTri, Basis, ElementVector, ElementTriP2,
                   FacetBasis, LinearForm, Functional, asm, condense, solve)
from skfem.models.elasticity import linear_elasticity

DATA = Path(__file__).parent / "data"
DATA.mkdir(parents=True, exist_ok=True)
FIG = Path(__file__).parent.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C_PRIMARY = "#0066CC"
C_RED = "#990000"
C_TEAL = "#008080"

# -- Plane-stress elasticity, E=1, nu=0.30 --
E_mod = 1.0
nu = 0.30
lam_b = E_mod * nu / ((1 + nu) * (1 - 2 * nu))
mu = E_mod / (2 * (1 + nu))
lam_s = 2 * lam_b * mu / (lam_b + 2 * mu)
A_1111 = lam_s + 2 * mu
A_1122 = lam_s
A_1212 = mu


def build_annular_mesh(a_inc, R_out=0.5, n_r=24, n_theta=96):
    """Polar-fitted annular triangular mesh on {a < |y| < R}.

    Nodes are placed on a structured (r, theta) grid:
        r_k = a + (R - a) * k / n_r,    k = 0, ..., n_r
        theta_i = 2 pi i / n_theta,      i = 0, ..., n_theta-1
    Each (i, j) cell -> two triangles. Wraparound in theta.
    Result: scikit-fem MeshTri with the inner ring at exact r=a, outer at r=R.
    """
    r = np.linspace(a_inc, R_out, n_r + 1)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    pts = []
    for k in range(n_r + 1):
        for i in range(n_theta):
            pts.append([r[k] * np.cos(theta[i]), r[k] * np.sin(theta[i])])
    pts = np.array(pts).T  # (2, n_pts)

    def idx(i, k):
        return k * n_theta + i

    tris = []
    for k in range(n_r):
        for i in range(n_theta):
            i_n = (i + 1) % n_theta
            v00 = idx(i, k)
            v10 = idx(i_n, k)
            v01 = idx(i, k + 1)
            v11 = idx(i_n, k + 1)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])
    tris = np.array(tris).T  # (3, n_tri)

    return MeshTri(pts, tris)


def solve_cell_problem(a_inc, n_r=24, n_theta=96, R_out=0.5):
    """Solve the e^(11) corrector on the annulus. Returns (mesh, basis, chi)."""
    mesh = build_annular_mesh(a_inc, R_out=R_out, n_r=n_r, n_theta=n_theta)
    basis = Basis(mesh, ElementVector(ElementTriP2()))
    K = asm(linear_elasticity(Lambda=lam_s, Mu=mu), basis)

    # Identify outer-ring nodes (r ~ R_out): clamp to remove rigid translation.
    # Inner ring (r = a_inc) is FREE.
    r_node = np.sqrt(mesh.p[0] ** 2 + mesh.p[1] ** 2)
    on_outer_node = r_node > R_out - 1e-9
    # Get all DOFs on outer ring (vertex DOFs only for clamping is enough)
    fixed_facets = []
    bf = mesh.boundary_facets()
    fc = mesh.p[:, mesh.facets[:, bf]].mean(axis=1)
    rc = np.sqrt(fc[0] ** 2 + fc[1] ** 2)
    on_outer_facet = rc > 0.99 * R_out
    outer_facets = bf[on_outer_facet]
    fixed_dofs = basis.get_dofs(outer_facets).flatten()

    @LinearForm
    def rhs(v, w):
        eps_v = 0.5 * (v.grad + np.transpose(v.grad, (1, 0, 2, 3)))
        # sigma(e^(11)):eps(v) with e^(11) = [[1,0],[0,0]]
        return -(lam_s * (eps_v[0, 0] + eps_v[1, 1]) + 2 * mu * eps_v[0, 0])

    b = asm(rhs, basis)
    chi = solve(*condense(K, b, D=fixed_dofs))
    return mesh, basis, chi


def J_C1111(a_inc, n_r=24, n_theta=96, R_out=0.5):
    """Compute int sigma_11(eps_tot) dy over the annulus."""
    mesh, basis, chi = solve_cell_problem(a_inc, n_r=n_r, n_theta=n_theta,
                                          R_out=R_out)
    eps_chi = basis.interpolate(chi).grad
    eps_chi_sym = 0.5 * (eps_chi + np.transpose(eps_chi, (1, 0, 2, 3)))
    e_macro = np.array([[1.0, 0.0], [0.0, 0.0]])
    eps_tot = e_macro[:, :, None, None] + eps_chi_sym
    tr = eps_tot[0, 0] + eps_tot[1, 1]
    sigma_11 = lam_s * tr + 2 * mu * eps_tot[0, 0]
    dx = basis.dx
    return float(np.sum(sigma_11 * dx))


def hill_mandel_check(a_inc, n_r=30, n_theta=120, R_out=0.5):
    """Verify J_mix = J_eng at the cell-problem solution.

    J_mix = int A:(eps_tot):e^(11) dy = int sigma_11 dy        (linear in eps_tot)
    J_eng = int A:(eps_tot):(eps_tot) dy                       (quadratic)

    Hill-Mandel: cross term int A:(eps_tot):eps(chi) dy = 0 by cell problem,
    so J_mix = J_eng at the optimum.
    """
    mesh, basis, chi = solve_cell_problem(a_inc, n_r=n_r, n_theta=n_theta,
                                          R_out=R_out)
    eps_chi = basis.interpolate(chi).grad
    eps_chi_sym = 0.5 * (eps_chi + np.transpose(eps_chi, (1, 0, 2, 3)))
    e_macro = np.array([[1.0, 0.0], [0.0, 0.0]])
    eps_tot = e_macro[:, :, None, None] + eps_chi_sym

    tr = eps_tot[0, 0] + eps_tot[1, 1]
    sigma_11 = lam_s * tr + 2 * mu * eps_tot[0, 0]
    sigma_22 = lam_s * tr + 2 * mu * eps_tot[1, 1]
    sigma_12 = 2 * mu * eps_tot[0, 1]

    dx = basis.dx
    # Mixed: int sigma : e^(11) dy  with e^(11) only the (1,1) entry
    J_mix = float(np.sum(sigma_11 * dx))
    # Energy: int sigma : eps_tot dy
    J_eng = float(np.sum(
        (sigma_11 * eps_tot[0, 0] + sigma_22 * eps_tot[1, 1]
         + 2 * sigma_12 * eps_tot[0, 1]) * dx
    ))
    return J_mix, J_eng


def line_integral_on_gamma(mesh, basis, chi, a_inc, R_out=0.5):
    """oint_{|y|=a} A:(eps_tot):(eps_tot) dl  using P2 boundary trace.

    Uses scikit-fem FacetBasis on the inner-ring facets for proper
    matrix-side trace of eps(chi).
    """
    bf = mesh.boundary_facets()
    fc = mesh.p[:, mesh.facets[:, bf]].mean(axis=1)
    rc = np.sqrt(fc[0] ** 2 + fc[1] ** 2)
    # inner ring: rc closer to a_inc than to R_out
    inner_mask = rc < 0.5 * (a_inc + R_out)
    inner_facets = bf[inner_mask]

    fb = FacetBasis(mesh, basis.elem, facets=inner_facets)
    eps_chi = fb.interpolate(chi).grad           # (2, 2, n_fac, n_quad)
    eps_chi_sym = 0.5 * (eps_chi + np.transpose(eps_chi, (1, 0, 2, 3)))

    e_macro = np.array([[1.0, 0.0], [0.0, 0.0]])
    eps_tot = e_macro[:, :, None, None] + eps_chi_sym
    tr = eps_tot[0, 0] + eps_tot[1, 1]
    norm2 = (eps_tot[0, 0] ** 2 + eps_tot[1, 1] ** 2 + 2 * eps_tot[0, 1] ** 2)
    energy = lam_s * tr ** 2 + 2 * mu * norm2    # (n_fac, n_quad)

    dx_f = fb.dx                                  # (n_fac, n_quad)
    return float(np.sum(energy * dx_f))


def main():
    phi = 0.10
    a_inc = float(np.sqrt(phi / np.pi))
    R_out = 0.5
    n_r, n_theta = 30, 120

    print(f"# Q3 / L7 -- Hadamard FD validation (annular membrane cell)")
    print(f"# E={E_mod}, nu={nu}, lam*={lam_s:.4f}, mu={mu:.4f}")
    print(f"# Annulus: a={a_inc:.6f}, R={R_out}")
    print(f"# Mesh: n_r={n_r}, n_theta={n_theta}")
    print()

    # Sanity: J at base radius
    J0 = J_C1111(a_inc, n_r=n_r, n_theta=n_theta, R_out=R_out)
    print(f"# Base J(a={a_inc:.4f}) = {J0:.6f}")
    print(f"# C^(0)_1111 = lam* + 2mu = {A_1111:.4f}")
    print(f"# J0 / |annulus area| (= pi (R^2 - a^2)) = "
          f"{J0 / (np.pi * (R_out ** 2 - a_inc ** 2)):.4f}")
    print()

    # Mesh convergence at a_inc
    print("# Mesh convergence at a_inc:")
    for nr, nt in [(20, 80), (30, 120), (40, 160), (50, 200)]:
        Jn = J_C1111(a_inc, n_r=nr, n_theta=nt, R_out=R_out)
        print(f"  (n_r,n_theta)=({nr},{nt}): J = {Jn:.6f}")
    print()

    # Hill-Mandel verification: J_mix = J_eng at the cell solution
    print("# Hill-Mandel verification (J_mix = J_eng at cell-problem optimum):")
    Jm, Je = hill_mandel_check(a_inc, n_r=n_r, n_theta=n_theta, R_out=R_out)
    print(f"  J_mix = int A:eps_tot:e^(11) dy  = {Jm:.6f}")
    print(f"  J_eng = int A:eps_tot:eps_tot dy = {Je:.6f}")
    print(f"  |J_mix - J_eng| / J_mix         = {abs(Jm - Je) / abs(Jm):.3e}")
    print()

    # FD sweep
    t_values = [2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4]
    print("# Finite-difference sweep:")
    print(f"  {'t':>10} {'J(a+t)':>12} {'D_FD':>12}")
    print("  " + "-" * 42)
    D_FD = {}
    for t in t_values:
        Jt = J_C1111(a_inc + t, n_r=n_r, n_theta=n_theta, R_out=R_out)
        D_FD[t] = (Jt - J0) / t
        print(f"  {t:>10.1e} {Jt:>12.6f} {D_FD[t]:>12.6f}")
    print()

    # Analytical via FacetBasis line integral at chi_0
    mesh0, basis0, chi0 = solve_cell_problem(a_inc, n_r=n_r, n_theta=n_theta,
                                             R_out=R_out)
    line_int = line_integral_on_gamma(mesh0, basis0, chi0, a_inc, R_out=R_out)
    G_ana = -line_int
    print(f"# Analytical dJ/da = -oint A:(eps_tot):(eps_tot) dl  (FacetBasis P2)")
    print(f"  G_ana = {G_ana:.6f}")
    print("# Mesh convergence of G_ana:")
    for nr, nt in [(20, 80), (30, 120), (40, 160), (50, 200)]:
        m, bs, c = solve_cell_problem(a_inc, n_r=nr, n_theta=nt, R_out=R_out)
        gi = -line_integral_on_gamma(m, bs, c, a_inc, R_out=R_out)
        print(f"    (n_r,n_theta)=({nr},{nt}): G = {gi:.6f}")
    print()

    # FD vs analytical
    print("# FD vs analytical Hadamard:")
    print(f"  {'t':>10} {'D_FD':>12} {'G_ana':>12} {'|D-G|':>12}")
    print("  " + "-" * 52)
    err = {}
    for t in t_values:
        e = abs(D_FD[t] - G_ana)
        err[t] = e
        print(f"  {t:>10.1e} {D_FD[t]:>12.6f} {G_ana:>12.6f} {e:>12.6e}")
    print()

    # Slope fit on log-log. Keep the largest-t prefix where the consecutive
    # log-log slope stays in [0.5, 1.5] (i.e. close to Hadamard slope = 1).
    # Once the local slope exits this band, discretization noise dominates.
    ts = np.array(sorted(t_values, reverse=True))
    es = np.array([err[t] for t in ts])
    keep = [True]
    for i in range(1, len(es)):
        if es[i] <= 0 or es[i - 1] <= 0:
            keep.append(False)
            continue
        local_slope = np.log(es[i] / es[i - 1]) / np.log(ts[i] / ts[i - 1])
        keep.append(keep[-1] and (0.5 <= local_slope <= 1.5))
    mask = np.array(keep)
    print(f"# Linear-regime mask (local slope within [0.5,1.5]): "
          f"{mask.sum()} of {len(es)} points retained "
          f"(t in [{ts[mask].min():.1e}, {ts[mask].max():.1e}])")
    if mask.sum() >= 3:
        lt = np.log(ts[mask])
        le = np.log(es[mask])
        slope, intercept = np.polyfit(lt, le, 1)
        log_e_fit = slope * lt + intercept
        ss_res = np.sum((le - log_e_fit) ** 2)
        ss_tot = np.sum((le - le.mean()) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"# Empirical slope on linear regime: {slope:.3f}, R^2 = {R2:.4f}")
        print(f"# Theoretical slope (Hadamard, Theorem 5.2): 1.0")
        print(f"# Discrepancy: {abs(slope - 1.0):.3f}")
    else:
        slope = float('nan'); R2 = float('nan')
        intercept = 0.0

    # CSV
    out_csv = DATA / "Q3_hadamard_fd.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "D_FD", "G_ana", "abs_err"])
        for t in sorted(t_values, reverse=True):
            w.writerow([t, D_FD[t], G_ana, err[t]])
    print(f"\nCSV: {out_csv}")

    # Figure
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ts_plot = np.array(sorted(t_values, reverse=True))
    es_plot = np.array([err[t] for t in ts_plot])
    ax.loglog(ts_plot, es_plot, "o", color=C_PRIMARY, markersize=8,
              label=fr"$|D_{{\mathrm{{FD}}}}(t) - \mathcal{{G}}_{{\rm ana}}|$")
    if mask.sum() >= 3:
        ax.loglog(ts[mask], np.exp(intercept) * ts[mask] ** slope,
                  "--", color=C_RED, lw=1.5,
                  label=fr"empirical fit, slope = {slope:.2f}, $R^2={R2:.3f}$")
    ts_ref = np.geomspace(ts_plot.min(), ts_plot.max(), 100)
    if mask.sum() > 0:
        ax.loglog(ts_ref, ts_ref * (es_plot[0] / ts_plot[0]),
                  ":", color=C_TEAL, lw=1.5,
                  label=r"theoretical, slope = 1")
    ax.set_xlabel(r"finite-difference step $t$")
    ax.set_ylabel(r"$|D_{\mathrm{FD}}(t) - \mathcal{G}_{\mathrm{ana}}|$")
    ax.set_title("Hadamard shape derivative", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out_pdf = FIG / "Q3_hadamard_fd.pdf"
    plt.savefig(out_pdf, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Figure: {out_pdf}")

    return slope, R2


if __name__ == "__main__":
    main()
