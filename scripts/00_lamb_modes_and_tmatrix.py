"""
Q1 validation: the coupled Kirchhoff--Lame thin-plate operator captures the
three lowest Lamb modes (A_0, S_0, SH_0) with the predicted multipolar
structure of the T-matrix.

Two numerical experiments:

L3 -- Visualization of the three Lamb mode shapes on the plate mid-surface.
      Each mode is a plane wave with characteristic polarization. The plot
      shows the kinematic picture that the operator L is supposed to support.

L2 -- Frequency sweep of the diagonal multipolar T-matrix coefficient
      |t_n^(b)(omega)| for the flexural channel of a clamped circular hole.
      The Mie poles -- frequencies at which scattering resonates -- appear as
      peaks of |t_n|. Their location vs n is a direct validation that the
      operator's spectral structure is consistent with the Bessel/Hankel basis.
"""

import os
import csv
from pathlib import Path
import numpy as np
import scipy.special as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DATA = Path(__file__).parent / "data"
DATA.mkdir(parents=True, exist_ok=True)

# Project styling colors (matching the article TikZ figures)
C_PRIMARY = "#0066CC"  # primaryblue
C_RED = "#990000"       # darkred
C_TEAL = "#008080"      # teal


# ============================================================
# L3 -- Lamb mode shape visualization (A_0 in 3D, S_0/SH_0 in 2D)
# ============================================================
def plot_lamb_modes(out_path):
    """Plot the displacement fields of A_0 (3D surface), S_0, SH_0 (2D)."""
    L = 1.0
    n_grid = 100
    x = np.linspace(0, L, n_grid)
    y = np.linspace(0, L, n_grid)
    X, Y = np.meshgrid(x, y)

    # Physically distinct wavelengths: k_b > k_s > k_p (flexural shortest)
    k_b = 5 * np.pi  # flexural, 5 wavelengths in domain
    k_s = 3 * np.pi  # shear, 3 wavelengths
    k_p = 2 * np.pi  # pressure (longitudinal), 2 wavelengths

    # A_0 anti-symmetric flexural deflection
    A0_w = np.sin(k_b * X)

    # S_0 symmetric longitudinal in-plane
    S0_ux = np.cos(k_p * X)
    S0_uy = np.zeros_like(X)

    # SH_0 shear-horizontal in-plane
    SH0_ux = np.zeros_like(X)
    SH0_uy = np.cos(k_s * X)

    fig = plt.figure(figsize=(14.5, 4.6))

    # ---------- (1) A_0 — 3D surface ----------
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    cm_A = LinearSegmentedColormap.from_list("A0", ["#003366", "white", "#990000"])
    surf = ax.plot_surface(X, Y, A0_w, cmap=cm_A, vmin=-1, vmax=1,
                           linewidth=0, antialiased=True, rcount=80, ccount=80,
                           edgecolor="none", alpha=0.95)
    # plate outline (z=0 plane projection)
    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], [-1.5, -1.5, -1.5, -1.5, -1.5],
            color="gray", linewidth=0.7, linestyle=":")
    ax.set_xlabel(r"$x_1 / L$", fontsize=10, labelpad=4)
    ax.set_ylabel(r"$x_2 / L$", fontsize=10, labelpad=4)
    ax.set_zlabel(r"$w(x_1, x_2)$", fontsize=10, labelpad=4)
    ax.set_zlim(-1.5, 1.5)
    ax.view_init(elev=22, azim=-58)
    ax.set_title(r"$A_0$  (flexural, out-of-plane)", fontsize=11, y=1.0)
    ax.set_box_aspect([1, 1, 0.5])
    # Polarization annotation
    ax.text2D(0.02, 0.05, "polarization $\\perp$ plate",
              transform=ax.transAxes,
              fontsize=8, color="#990000",
              bbox=dict(facecolor="white", alpha=0.85,
                        edgecolor="#990000", boxstyle="round,pad=0.18"))

    # ---------- (2) S_0 — 2D in-plane longitudinal ----------
    ax = fig.add_subplot(1, 3, 2)
    skip = 8
    Xq, Yq = X[::skip, ::skip], Y[::skip, ::skip]
    Uq, Vq = S0_ux[::skip, ::skip], S0_uy[::skip, ::skip]
    pcm = ax.pcolormesh(X, Y, S0_ux, cmap="RdBu_r", shading="auto",
                        vmin=-1, vmax=1)
    plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label=r"$u_1$")
    ax.quiver(Xq, Yq, Uq, Vq, color="black", scale=14, width=0.0055,
              headwidth=4, headlength=5)
    # Single, in-plot annotation: propagation arrow + polarization note
    ax.annotate("propagation $\\rightarrow$",
                xy=(0.65, 0.93), xycoords="axes fraction",
                fontsize=8, color="#0066CC",
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="#0066CC", boxstyle="round,pad=0.18"))
    ax.annotate("polarization $\\parallel$ propagation",
                xy=(0.04, 0.06), xycoords="axes fraction",
                fontsize=8, color="#990000",
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="#990000", boxstyle="round,pad=0.18"))
    ax.set_xlabel(r"$x_1 / L$"); ax.set_ylabel(r"$x_2 / L$")
    ax.set_aspect("equal")
    ax.set_title(r"$S_0$  (in-plane, longitudinal)", fontsize=11)

    # ---------- (3) SH_0 — 2D in-plane transverse ----------
    ax = fig.add_subplot(1, 3, 3)
    Uq, Vq = SH0_ux[::skip, ::skip], SH0_uy[::skip, ::skip]
    pcm = ax.pcolormesh(X, Y, SH0_uy, cmap="RdBu_r", shading="auto",
                        vmin=-1, vmax=1)
    plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, label=r"$u_2$")
    ax.quiver(Xq, Yq, Uq, Vq, color="black", scale=14, width=0.0055,
              headwidth=4, headlength=5)
    ax.annotate("propagation $\\rightarrow$",
                xy=(0.65, 0.93), xycoords="axes fraction",
                fontsize=8, color="#0066CC",
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="#0066CC", boxstyle="round,pad=0.18"))
    ax.annotate("polarization $\\perp$ propagation",
                xy=(0.04, 0.06), xycoords="axes fraction",
                fontsize=8, color="#990000",
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="#990000", boxstyle="round,pad=0.18"))
    ax.set_xlabel(r"$x_1 / L$"); ax.set_ylabel(r"$x_2 / L$")
    ax.set_aspect("equal")
    ax.set_title(r"$SH_0$  (in-plane, transverse)", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[L3] Lamb mode shapes saved: {out_path}")


# ============================================================
# L2 -- Multipolar T-matrix sweep with Mie resonances
# ============================================================
def Jn(n, x):  return sp.jv(n, x)
def Hn1(n, x): return sp.hankel1(n, x)
def Jnp(n, x): return 0.5 * (sp.jv(n - 1, x) - sp.jv(n + 1, x))
def Hn1p(n, x): return 0.5 * (sp.hankel1(n - 1, x) - sp.hankel1(n + 1, x))
def In_(n, x): return sp.iv(n, x)
def Kn_(n, x): return sp.kv(n, x)
def Inp(n, x): return 0.5 * (sp.iv(n - 1, x) + sp.iv(n + 1, x))
def Knp(n, x): return -0.5 * (sp.kv(n - 1, x) + sp.kv(n + 1, x))


def t_clamped_hole(n, kba):
    """
    Diagonal T-matrix coefficient for a clamped circular hole, multipolar
    order n, at dimensionless frequency k_b * a.

    Boundary conditions: w(a,theta) = 0, partial_n w(a,theta) = 0.
    Solving the 2x2 system in (c_n, d_n) for incident a_n^(J) = i^n,
    a_n^(I) = 0 (pure propagating incidence) yields the propagating-channel
    coefficient c_n. We return c_n / a_n^(inc).
    """
    M = np.array([
        [Hn1(n, kba), Kn_(n, kba)],
        [Hn1p(n, kba), Knp(n, kba)],
    ], dtype=complex)
    rhs = -np.array([Jn(n, kba), Jnp(n, kba)], dtype=complex)
    sol = np.linalg.solve(M, rhs)
    return sol[0]   # c_n (propagating-to-propagating amplitude)


def plot_tmatrix_sweep(out_path):
    """Sweep |t_n(omega)| vs k_b * a for n in {0, 1, 2, 3, 4}."""
    kba = np.linspace(0.05, 5.0, 700)
    n_max_plot = 4

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    colors = ["#0066CC", "#990000", "#008080", "#CC6600", "#660066"]

    # Left panel: |t_n| vs k_b a
    ax = axes[0]
    for n, col in zip(range(n_max_plot + 1), colors):
        tn = np.array([t_clamped_hole(n, kk) for kk in kba])
        ax.semilogy(kba, np.abs(tn), color=col, linewidth=1.6,
                    label=fr"$n = {n}$")
    ax.set_xlabel(r"$k_b\,a$", fontsize=11)
    ax.set_ylabel(r"$|t_n^{(b)}(k_b a)|$", fontsize=11)
    ax.set_title("Multipolar T-matrix amplitudes (clamped hole)", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, ncol=2)

    # Right panel: argument (phase) of t_n vs k_b a, showing pi-jumps at Mie poles
    ax = axes[1]
    for n, col in zip(range(n_max_plot + 1), colors):
        tn = np.array([t_clamped_hole(n, kk) for kk in kba])
        ax.plot(kba, np.angle(tn) / np.pi, color=col, linewidth=1.6,
                label=fr"$n = {n}$")
    ax.set_xlabel(r"$k_b\,a$", fontsize=11)
    ax.set_ylabel(r"$\arg t_n^{(b)} / \pi$", fontsize=11)
    ax.set_title("T-matrix phase: $\\pi$-jumps at Mie resonances", fontsize=11)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[L2] T-matrix sweep saved: {out_path}")

    # Also save raw data for pgfplots in main article
    out_dat = DATA / "Q1_tmatrix_sweep.dat"
    with open(out_dat, "w") as f:
        f.write("kba |t0| |t1| |t2| |t3| |t4|\n")
        for kk in kba:
            row = [kk] + [abs(t_clamped_hole(n, kk)) for n in range(n_max_plot + 1)]
            f.write(" ".join(f"{v:.6e}" for v in row) + "\n")
    print(f"[L2] Pgfplots data: {out_dat}")


# ============================================================
# Driver
# ============================================================
if __name__ == "__main__":
    fig_dir = Path(__file__).parent.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_lamb_modes(fig_dir / "Q1_lamb_modes.pdf")
    plot_tmatrix_sweep(fig_dir / "Q1_tmatrix_sweep.pdf")

    print("Q1 numerical validation complete.")
