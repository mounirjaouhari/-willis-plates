"""Re-analyze Bloch-Floquet coupled output against the LEADING-ORDER closed form. 

Reads Q2_bloch_coupled_omega.csv, computes S_LO(omega) = phi B^(1) (1 + R(omega))
with B^(1) = 1.236, phi = 0.10, eta/omega_0 = 0.05. Plots S_num and S_LO together
on log-magnitude and phase vs omega/omega_0.
"""

from pathlib import Path
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).parent / "data"
FIG = Path(__file__).parent.parent / "figures"

C_PRIMARY = "#0066CC"
C_RED = "#990000"
C_TEAL = "#008080"


def R_factor(omega, omega_0, eta):
    return omega_0 ** 2 / (omega_0 ** 2 - omega ** 2 - 1j * omega * eta)


def main():
    # Read data
    rows = []
    with open(DATA / "Q2_bloch_coupled_omega.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: float(v) for k, v in row.items()})

    omega_norm = np.array([r["omega_norm"] for r in rows])
    S_num = np.array([r["S_num_re"] + 1j * r["S_num_im"] for r in rows])

    # Leading-order closed form
    omega_0 = 1.0
    eta = 0.05 * omega_0
    phi = 0.10
    B_1111 = 1.236
    beta_r = 1.0
    omegas = omega_norm * omega_0
    S_LO = phi * B_1111 * (1 + beta_r * R_factor(omegas, omega_0, eta))

    # Errors vs leading-order
    err_LO = np.abs(S_num - S_LO) / np.abs(S_LO)
    print("# omega/w0  S_num.re  S_num.im  S_LO.re  S_LO.im  rel_err")
    for i, om in enumerate(omega_norm):
        print(f"  {om:.3f} {S_num[i].real:+.4e} {S_num[i].imag:+.4e} "
              f"{S_LO[i].real:+.4e} {S_LO[i].imag:+.4e} {err_LO[i]:.4f}")

    print()
    print(f"# Comparison vs leading-order S_LO = phi B^(1) (1 + R(omega)):")
    print(f"#   Max relative error: {err_LO.max():.4e}")
    print(f"#   Median relative error: {np.median(err_LO):.4e}")
    print(f"#   Mean relative error: {err_LO.mean():.4e}")
    print(f"# At omega = omega_0: rel_err = {err_LO[len(rows)//2]:.4e}")

    # Figure: |S| (log) and arg(S)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.semilogy(omega_norm, np.abs(S_num), 'o', color=C_PRIMARY, markersize=7,
                label="numerical (Argyris+$P_2$)")
    ax.semilogy(omega_norm, np.abs(S_LO), '-', color=C_RED, lw=1.5,
                label=r"leading-order $\phi B^{(1)}(1+\beta_r \mathcal{R})$")
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\omega / \omega_{0}$")
    ax.set_ylabel(r"$|\mathcal{S}_{1111}(\omega)|$")
    ax.set_title(r"(a) Magnitude of dynamic Willis tensor", fontsize=11)
    ax.legend(fontsize=9, loc="lower center")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(omega_norm, np.angle(S_num) / np.pi, 'o', color=C_PRIMARY, markersize=7,
            label="numerical")
    ax.plot(omega_norm, np.angle(S_LO) / np.pi, '-', color=C_RED, lw=1.5,
            label="leading-order")
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\omega / \omega_{0}$")
    ax.set_ylabel(r"$\arg \mathcal{S}_{1111} / \pi$")
    ax.set_title(r"(b) Phase, $\pi$-jump signature", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_pdf = FIG / "Q2_bloch_coupled_omega.pdf"
    plt.savefig(out_pdf, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nFigure: {out_pdf}")


if __name__ == "__main__":
    main()
