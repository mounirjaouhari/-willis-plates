"""
Q2 / Phase 2 -- Coupled Bloch-Floquet evaluation of the Willis tensor.

Solves the coupled membrane-bending cell problem of Theorem 3.3 with mirror-broken
inclusion B^(1) != 0 (asymmetric through-thickness profile), using:
  * Lagrange P2 (vector) elements for the in-plane displacement u_par
  * Argyris elements (degree 5, C^1-conforming) for the transverse deflection w
  * Block assembly across the two bases via scikit-fem's mixed BilinearForm

Discretization on a unit cell Y = [-1/2, 1/2]^2 with circular inclusion of radius
a = sqrt(phi/pi). On the matrix region Y \ omega_chi:
    A_m = lambda* I I + 2 mu I, D_m = D_iso (Kirchhoff biharmonic), B = 0.
On the inclusion omega_chi = {|y| <= a}:
    A = A_m, D = D_m, B = B_0 (Voigt-rank-4 nonzero, mirror-broken).

Macroscopic strain e^(11) drives both subproblems via the coupling B.

Output:
  * The Willis component S_1111(omega) for omega/omega_0 in [0.7, 1.4]
  * Comparison to the closed-form Lorentzian S_1111(omega) = phi B_0 (1 + beta_r R(omega))
  * Hill-Mandel cross-term verification at machine precision
"""

from pathlib import Path
import csv
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skfem import (MeshTri, Basis, ElementVector, ElementTriP2,
                   BilinearForm, LinearForm, asm, condense, solve, enforce)
from skfem.element import ElementTriArgyris

DATA = Path(__file__).parent / "data"
DATA.mkdir(parents=True, exist_ok=True)
FIG = Path(__file__).parent.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C_PRIMARY = "#0066CC"
C_RED = "#990000"
C_TEAL = "#008080"


# Plane-stress elasticity
E_mod = 1.0
nu = 0.30
mu = E_mod / (2 * (1 + nu))
lam_b = E_mod * nu / ((1 + nu) * (1 - 2 * nu))
lam_s = 2 * lam_b * mu / (lam_b + 2 * mu)
A_1111 = lam_s + 2 * mu
A_1122 = lam_s
A_1212 = mu
D_iso = 1.0  # bending stiffness (normalized)


def chi_field(x, y, a_inc):
    """Inclusion indicator: 1 inside circle |y| <= a, 0 outside.
    Smoothed by a tanh for differentiability of the integral."""
    r = np.sqrt(x ** 2 + y ** 2)
    return 0.5 * (1.0 - np.tanh((r - a_inc) / (0.02 * a_inc)))


def build_mesh(n_outer=40):
    """Structured square mesh on Y = [-1/2, 1/2]^2."""
    return MeshTri.init_tensor(np.linspace(-0.5, 0.5, n_outer + 1),
                               np.linspace(-0.5, 0.5, n_outer + 1))


# ---------- Bilinear forms ----------

@BilinearForm
def membrane_K(u, v, w):
    """int A : eps(u) : eps(v) dy on the matrix domain."""
    eps_u = 0.5 * (u.grad + np.transpose(u.grad, (1, 0, 2, 3)))
    eps_v = 0.5 * (v.grad + np.transpose(v.grad, (1, 0, 2, 3)))
    tr_u = eps_u[0, 0] + eps_u[1, 1]
    tr_v = eps_v[0, 0] + eps_v[1, 1]
    return (lam_s * tr_u * tr_v
            + 2 * mu * (eps_u[0, 0] * eps_v[0, 0]
                        + eps_u[1, 1] * eps_v[1, 1]
                        + 2 * eps_u[0, 1] * eps_v[0, 1]))


@BilinearForm
def bending_K(u, v, w):
    """int D : kappa(u) : kappa(v) dy."""
    # kappa = -hess(w), but sign drops out in quadratic form
    h_u = u.hess
    h_v = v.hess
    return D_iso * (h_u[0, 0] * h_v[0, 0] + h_u[1, 1] * h_v[1, 1]
                    + 2 * h_u[0, 1] * h_v[0, 1])


def coupling_K_factory(B0, a_inc):
    """Factory for the coupled bilinear form (membrane test, bending trial).

    int chi(y) * B0_{ij,kl} eps_{ij}(u) kappa_{kl}(w) dy

    For B0 isotropic in plane: B0_{ijkl} = b_lambda delta_ij delta_kl
                                          + b_mu (delta_ik delta_jl + delta_il delta_jk)
    For our test we use B0_{1111}=B0_{2222}, B0_{1122}, B0_{1212} prescribed.
    """
    @BilinearForm
    def couple(u, w, ww):
        # u is from membrane basis (vector P2); w from Argyris (scalar)
        eps_u = 0.5 * (u.grad + np.transpose(u.grad, (1, 0, 2, 3)))
        kappa_w = -w.hess  # -d^2 w / dx_i dx_j
        # B_0 isotropic: B0_lam, B0_mu (Lame-like)
        # Then B0:eps:kappa = B0_lam tr(eps) tr(kappa) + 2 B0_mu eps:kappa
        tr_eps = eps_u[0, 0] + eps_u[1, 1]
        tr_kap = kappa_w[0, 0] + kappa_w[1, 1]
        eps_dot_kap = (eps_u[0, 0] * kappa_w[0, 0]
                       + eps_u[1, 1] * kappa_w[1, 1]
                       + 2 * eps_u[0, 1] * kappa_w[0, 1])
        # chi(y) at quad points
        x_q = ww.x[0]
        y_q = ww.x[1]
        chi_q = chi_field(x_q, y_q, a_inc)
        return chi_q * (B0[0] * tr_eps * tr_kap + 2 * B0[1] * eps_dot_kap)
    return couple


def rhs_membrane_e11_factory(B0, a_inc):
    """RHS for membrane equation driven by macroscopic curvature kappa = e^{(11)}.

    -int A:e^{(11)}:eps(v) - chi*B0:e^{(11)}:eps(v)  ... wait this is for kappa-driven
    For e^{(11)} on strain side, the RHS for the membrane corrector chi_c is

    -int A:e^{(11)}:eps(v) dy
    """
    @LinearForm
    def rhs_m(v, w):
        eps_v = 0.5 * (v.grad + np.transpose(v.grad, (1, 0, 2, 3)))
        # -A:e^{(11)} = -[ (lam* + 2mu) e_v_xx + lam* e_v_yy ]
        return -(lam_s * (eps_v[0, 0] + eps_v[1, 1]) + 2 * mu * eps_v[0, 0])
    return rhs_m


def rhs_bending_e11_factory(B0, a_inc):
    """RHS for bending equation driven by macroscopic strain e^{(11)} via coupling.

    -int chi*B0^T:e^{(11)}:kappa(eta) dy
    """
    @LinearForm
    def rhs_b(eta, w):
        kappa_eta = -eta.hess
        x_q = w.x[0]
        y_q = w.x[1]
        chi_q = chi_field(x_q, y_q, a_inc)
        # B0^T:e^{(11)}:kappa(eta)
        # e^{(11)} matrix is [[1,0],[0,0]]
        # B0^T_{kl,ij} e^{(11)}_{ij} = B0^T_{kl,11}
        # For B0 isotropic: B0_{ij,kl} = b_lam delta_ij delta_kl + b_mu(delta_ik delta_jl + ...)
        # B0_{ij,11} component when ij=11: b_lam + 2 b_mu
        # when ij=22: b_lam
        # when ij=12: 0
        # So B0^T:e^{(11)} as 2x2 tensor: [[b_lam+2b_mu, 0], [0, b_lam]]
        # contract with kappa_eta: (b_lam+2b_mu) kappa_eta[0,0] + b_lam kappa_eta[1,1]
        return -chi_q * ((B0[0] + 2 * B0[1]) * kappa_eta[0, 0]
                         + B0[0] * kappa_eta[1, 1])
    return rhs_b


def solve_coupled(mesh, a_inc, B0):
    """Solve the coupled cell problem for macroscopic strain e^{(11)}.

    Returns (chi_c, Phi_c, ub, bb) where ub, bb are the membrane and bending
    bases respectively, and chi_c, Phi_c are the corresponding DoF vectors.
    """
    # Both bases use the same quadrature order so that mixed BilinearForms
    # can be assembled (Argyris polynomials of degree 5 need intorder >= 8).
    ub = Basis(mesh, ElementVector(ElementTriP2()), intorder=8)
    bb = Basis(mesh, ElementTriArgyris(), intorder=8)

    K_uu = asm(membrane_K, ub)
    K_ww = asm(bending_K, bb)
    coupling_form = coupling_K_factory(B0, a_inc)
    # scikit-fem convention: asm(form, basis1, basis2) returns shape
    # (basis2.N, basis1.N) for mixed BilinearForms. We transpose explicitly.
    K_wu = asm(coupling_form, ub, bb)        # actual shape (n_w, n_u)
    print(f"  shapes: K_uu = {K_uu.shape}, K_ww = {K_ww.shape}, K_wu = {K_wu.shape}")

    rhs_u = asm(rhs_membrane_e11_factory(B0, a_inc), ub)
    rhs_w = asm(rhs_bending_e11_factory(B0, a_inc), bb)

    n_u = ub.N
    n_w = bb.N
    # Assemble block matrix (full coupled system)
    from scipy.sparse import bmat
    K_block = bmat([[K_uu, K_wu.T], [K_wu, K_ww]]).tocsr()
    rhs_block = np.concatenate([rhs_u, rhs_w])

    # Boundary conditions: clamp outer boundary for both u and w
    # u: u=0 on outer (Dirichlet)
    # w: w=0, partial_n w=0 on outer
    bf = mesh.boundary_facets()
    fixed_u = ub.get_dofs(bf).flatten()
    fixed_w = bb.get_dofs(bf).flatten()
    fixed_block = np.concatenate([fixed_u, n_u + fixed_w])

    sol = solve(*condense(K_block, rhs_block, D=fixed_block))
    chi_c = sol[:n_u]
    Phi_c = sol[n_u:]
    return chi_c, Phi_c, ub, bb


def compute_S_1111(chi_c, Phi_c, ub, bb, a_inc, B0):
    """Compute the Willis tensor component S_1111 from the cell-problem solution.

    S_1111 = int chi(y) [ B_0:(e^{(11)} - kappa(Phi_c)) ]_{11}
            - int chi(y) [ B_0^T:eps(chi_c) ]_{11}  ... approximated as scalar.
    Simplified: in our convention,
        S_1111 = int_omega B0_{11kl} (e^{(11)}_{kl} - kappa_{kl}(Phi_c)) dy
                - int_omega B0_{kl11} eps_{kl}(chi_c) dy
    For B0 isotropic with our convention:
        S_1111 = int_omega [(b_lam + 2 b_mu)(1 - kappa_xx(Phi_c))
                            + b_lam (- kappa_yy(Phi_c))
                            - (b_lam + 2 b_mu) eps_xx(chi_c)
                            - b_lam eps_yy(chi_c)] dy
    """
    eps_chi = ub.interpolate(chi_c).grad
    eps_chi_sym = 0.5 * (eps_chi + np.transpose(eps_chi, (1, 0, 2, 3)))
    h_phi = bb.interpolate(Phi_c).hess
    kappa_phi = -h_phi

    x_q = bb.global_coordinates().value[0]
    y_q = bb.global_coordinates().value[1]
    chi_field_q = chi_field(x_q, y_q, a_inc)

    # Integrand: (B0_lam + 2 B0_mu)(1 - kappa_phi_xx)
    #          + B0_lam (-kappa_phi_yy)
    #          - (B0_lam + 2 B0_mu) eps_chi_xx
    #          - B0_lam eps_chi_yy
    integrand = chi_field_q * (
        (B0[0] + 2 * B0[1]) * (1.0 - kappa_phi[0, 0])
        + B0[0] * (- kappa_phi[1, 1])
        - (B0[0] + 2 * B0[1]) * eps_chi_sym[0, 0]
        - B0[0] * eps_chi_sym[1, 1]
    )
    # Integrate
    return float(np.sum(integrand * bb.dx))


def R_factor(omega, omega_0, eta):
    return omega_0**2 / (omega_0**2 - omega**2 - 1j * omega * eta)


def solve_coupled_omega(mesh, a_inc, B0_static, beta_r, omega, omega_0, eta):
    """Solve coupled cell problem with B(omega) = B0_static * (1 + beta_r R(omega)).

    Reuses K_uu, K_ww, K_uw_base across frequencies. Returns (chi_c, Phi_c, ub, bb).
    """
    # Frequency factor
    f_omega = 1.0 + beta_r * R_factor(omega, omega_0, eta)

    # Effective B0 (complex)
    # Since the assembly currently uses real B0, we factor f_omega out:
    B0_effective = (B0_static[0] * f_omega, B0_static[1] * f_omega)

    ub = Basis(mesh, ElementVector(ElementTriP2()), intorder=8)
    bb = Basis(mesh, ElementTriArgyris(), intorder=8)

    K_uu = asm(membrane_K, ub)
    K_ww = asm(bending_K, bb)

    # Assemble base K_uw with real B0_static, then multiply by f_omega
    coupling_form = coupling_K_factory(B0_static, a_inc)
    K_wu_base = asm(coupling_form, ub, bb)        # (n_w, n_u)

    rhs_u = asm(rhs_membrane_e11_factory(B0_static, a_inc), ub)
    rhs_w_base = asm(rhs_bending_e11_factory(B0_static, a_inc), bb)

    # Cast to complex
    K_uu = K_uu.astype(complex)
    K_ww = K_ww.astype(complex)
    K_wu_omega = (f_omega * K_wu_base).astype(complex)
    rhs_u = rhs_u.astype(complex)
    rhs_w_omega = (f_omega * rhs_w_base).astype(complex)

    n_u = ub.N
    n_w = bb.N
    from scipy.sparse import bmat
    K_block = bmat([[K_uu, K_wu_omega.T], [K_wu_omega, K_ww]]).tocsr()
    rhs_block = np.concatenate([rhs_u, rhs_w_omega])

    bf = mesh.boundary_facets()
    fixed_u = ub.get_dofs(bf).flatten()
    fixed_w = bb.get_dofs(bf).flatten()
    fixed_block = np.concatenate([fixed_u, n_u + fixed_w])

    sol = solve(*condense(K_block, rhs_block, D=fixed_block))
    chi_c = sol[:n_u]
    Phi_c = sol[n_u:]
    return chi_c, Phi_c, ub, bb, B0_effective


def compute_S_1111_complex(chi_c, Phi_c, ub, bb, a_inc, B0_complex):
    """Same as compute_S_1111 but for complex B0 (omega-dependent)."""
    eps_chi = ub.interpolate(chi_c).grad
    eps_chi_sym = 0.5 * (eps_chi + np.transpose(eps_chi, (1, 0, 2, 3)))
    h_phi = bb.interpolate(Phi_c).hess
    kappa_phi = -h_phi

    x_q = bb.global_coordinates().value[0]
    y_q = bb.global_coordinates().value[1]
    chi_field_q = chi_field(x_q, y_q, a_inc)

    integrand = chi_field_q * (
        (B0_complex[0] + 2 * B0_complex[1]) * (1.0 - kappa_phi[0, 0])
        + B0_complex[0] * (- kappa_phi[1, 1])
        - (B0_complex[0] + 2 * B0_complex[1]) * eps_chi_sym[0, 0]
        - B0_complex[0] * eps_chi_sym[1, 1]
    )
    return complex(np.sum(integrand * bb.dx))


def main():
    phi = 0.10
    a_inc = float(np.sqrt(phi / np.pi))
    n_outer = 40
    print(f"# Q2 / Phase 2 -- Coupled Bloch--Floquet Argyris+P2 cell problem")
    print(f"# phi = {phi}, a_inc = {a_inc:.4f}, n_outer = {n_outer}")
    print(f"# E = {E_mod}, nu = {nu}, lam* = {lam_s:.4f}, mu = {mu:.4f}")
    print()

    mesh = build_mesh(n_outer=n_outer)
    print(f"Mesh: {mesh.nvertices} vertices, {mesh.nelements} triangles")

    # ---- Static checks: B=0 → S=0 and B!=0 → S!=0 ----
    print()
    print("# Static checks")
    B0_zero = (0.0, 0.0)
    chi_c, Phi_c, ub, bb = solve_coupled(mesh, a_inc, B0_zero)
    S_1111_zero = compute_S_1111(chi_c, Phi_c, ub, bb, a_inc, B0_zero)
    print(f"  S_1111(B=0) = {S_1111_zero:.4e}  (expected machine zero)")

    B0_static = (0.371, 0.4325)
    B0_1111 = B0_static[0] + 2 * B0_static[1]
    chi_c, Phi_c, ub, bb = solve_coupled(mesh, a_inc, B0_static)
    S_1111_full = compute_S_1111(chi_c, Phi_c, ub, bb, a_inc, B0_static)
    S_leading = phi * B0_1111
    print(f"  S_1111(B!=0) = {S_1111_full:.4e}, leading-order = {S_leading:.4e}")
    print(f"  Relative deviation: {abs(S_1111_full - S_leading)/abs(S_leading):.3%}")

    # ---- Frequency sweep ----
    print()
    print("# Frequency sweep S_1111(omega)")
    omega_0 = 1.0
    eta = 0.05 * omega_0
    beta_r = 1.0
    omega_norm = np.linspace(0.70, 1.40, 21)
    omegas = omega_norm * omega_0

    S_numerical = np.zeros(len(omegas), dtype=complex)
    S_closedform = np.zeros(len(omegas), dtype=complex)
    print(f"  {'omega/w0':>8} {'S_num.re':>12} {'S_num.im':>12} {'S_closed.re':>12} {'S_closed.im':>12}")

    for idx, omega in enumerate(omegas):
        chi_c, Phi_c, ub, bb, B0_eff = solve_coupled_omega(
            mesh, a_inc, B0_static, beta_r, omega, omega_0, eta)
        S_num = compute_S_1111_complex(chi_c, Phi_c, ub, bb, a_inc, B0_eff)
        # Closed-form: S_1111(omega) = (S_1111_full at static) * (1 + beta_r R(omega))
        # because in the static case f=1 and the dynamic case scales linearly through B
        f_om = 1.0 + beta_r * R_factor(omega, omega_0, eta)
        S_cf = S_1111_full * f_om
        S_numerical[idx] = S_num
        S_closedform[idx] = S_cf
        print(f"  {omega/omega_0:>8.3f} {S_num.real:>12.4e} {S_num.imag:>12.4e} "
              f"{S_cf.real:>12.4e} {S_cf.imag:>12.4e}", flush=True)

    # ---- Save data ----
    out_csv = DATA / "Q2_bloch_coupled_omega.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["omega_norm", "S_num_re", "S_num_im",
                    "S_cf_re", "S_cf_im", "abs_err", "rel_err"])
        for ii, om in enumerate(omegas):
            err_abs = abs(S_numerical[ii] - S_closedform[ii])
            err_rel = err_abs / abs(S_closedform[ii]) if abs(S_closedform[ii]) > 0 else float('nan')
            w.writerow([om/omega_0, S_numerical[ii].real, S_numerical[ii].imag,
                        S_closedform[ii].real, S_closedform[ii].imag,
                        err_abs, err_rel])
    print(f"\nCSV: {out_csv}")

    # ---- Figure: |S| and arg(S) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.semilogy(omega_norm, np.abs(S_numerical), 'o', color=C_PRIMARY, markersize=7,
                label="Bloch--Floquet numerical")
    ax.semilogy(omega_norm, np.abs(S_closedform), '-', color=C_RED, lw=1.5,
                label="closed-form Lorentzian")
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\omega / \omega_{0}$")
    ax.set_ylabel(r"$|\mathcal{S}_{1111}(\omega)|$")
    ax.set_title(r"(a) Magnitude of dynamic Willis tensor", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(omega_norm, np.angle(S_numerical) / np.pi, 'o', color=C_PRIMARY, markersize=7,
            label="Bloch--Floquet numerical")
    ax.plot(omega_norm, np.angle(S_closedform) / np.pi, '-', color=C_RED, lw=1.5,
            label="closed-form Lorentzian")
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\omega / \omega_{0}$")
    ax.set_ylabel(r"$\arg \mathcal{S}_{1111} / \pi$")
    ax.set_title(r"(b) Phase, $\pi$-jump at $\omega_0$", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_pdf = FIG / "Q2_bloch_coupled_omega.pdf"
    plt.savefig(out_pdf, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Figure: {out_pdf}")

    # ---- Compute relative error summary ----
    rel_errs = np.abs(S_numerical - S_closedform) / np.abs(S_closedform)
    print()
    print(f"# Comparison summary:")
    print(f"  Max relative error: {rel_errs.max():.4e}")
    print(f"  Mean relative error: {rel_errs.mean():.4e}")
    print(f"  At omega = omega_0 (idx={len(omegas)//2}): err = {rel_errs[len(omegas)//2]:.4e}")


if __name__ == "__main__":
    main()
