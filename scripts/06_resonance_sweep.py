""" 
Q2 -- L1: Frequency sweep of the dynamic effective tensors near the
internal-resonator pole.

Two physical observables are plotted across omega/omega_0:

    rho^eff(omega) = <rho>_Y + (m_r / |Y|) * R(omega),
    S_{1111}(omega) = phi * B^(1)_1111 * (1 + beta_r * R(omega)),

where R(omega) = omega_0^2 / (omega_0^2 - omega^2 - i omega eta) is the
mass-spring resonator amplification factor.

The figure shows:
    (a) |rho^eff(omega) / rho_h| -- Lorentzian peak at omega = omega_0
    (b) Re/Im of rho^eff -- negative-mass band, 90 degree phase relation
    (c) |S_{1111}(omega)| -- inherited Lorentzian
    (d) phase of S_{1111} -- pi-jump at resonance

Reference: Theorems 3.2 and 3.3 of the article. Resonator factor
eq.~(eq:resonator).
"""

from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).parent / "data"
DATA.mkdir(parents=True, exist_ok=True)
FIG = Path(__file__).parent.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Style
C_PRIMARY = "#0066CC"
C_RED = "#990000"
C_TEAL = "#008080"
C_NEG = "#CC6600"


def R_factor(omega, omega_0, eta):
    """Internal resonator amplification factor."""
    return omega_0**2 / (omega_0**2 - omega**2 - 1j * omega * eta)


def main():
    # ----- physical / model parameters -----
    omega_0 = 1.0
    eta = 0.05 * omega_0    # 5% damping
    rho_h = 1.0
    m_r_over_Y = 0.6        # resonator mass density relative to host
    phi = 0.10
    B_1111 = 1.236          # leading-order laminate moment from dilute_S_tensor.py
    beta_r = 1.0

    # ----- frequency grid -----
    omega = np.linspace(0.55, 1.55, 1000)
    R = R_factor(omega, omega_0, eta)
    rho_eff = rho_h + m_r_over_Y * R
    S_1111 = phi * B_1111 * (1 + beta_r * R)

    # baseline at omega_min (off-resonance reference)
    rho_ref = abs(rho_eff[0])
    S_ref = abs(S_1111[0])

    # ----- save pgfplots data -----
    out = DATA / "Q2_resonance_sweep.dat"
    with open(out, "w") as f:
        f.write("w_norm abs_rho re_rho im_rho abs_S re_S im_S phase_rho_pi phase_S_pi\n")
        for w, r, s in zip(omega, rho_eff, S_1111):
            f.write(f"{w/omega_0:.6f} "
                    f"{abs(r)/rho_h:.6e} {r.real/rho_h:.6e} {r.imag/rho_h:.6e} "
                    f"{abs(s)/S_ref:.6e} {s.real/S_ref:.6e} {s.imag/S_ref:.6e} "
                    f"{np.angle(r)/np.pi:.6f} {np.angle(s)/np.pi:.6f}\n")

    # ----- figure: 2x2 grid -----
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6))

    # (a) |rho^eff(omega)|
    ax = axes[0, 0]
    ax.semilogy(omega/omega_0, abs(rho_eff)/rho_h, color=C_PRIMARY, lw=2)
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\omega / \omega_0$")
    ax.set_ylabel(r"$|\rho^{\mathrm{eff}}(\omega)| / \rho_h$")
    ax.set_title("(a) Magnitude of dynamic effective mass density", fontsize=11)
    ax.grid(True, which="both", alpha=0.25)
    # Half-power FWHM annotation (FWHM of |R|^2 is eta; in normalized units eta/omega_0)
    fwhm = eta / omega_0
    ax.axvspan(1 - fwhm/2, 1 + fwhm/2, color=C_PRIMARY, alpha=0.10)
    ax.annotate(f"FWHM $= \\eta/\\omega_0 = {fwhm:.2f}$",
                xy=(1.0, abs(rho_eff[len(omega)//2])/rho_h * 0.5),
                xytext=(1.15, abs(rho_eff[len(omega)//2])/rho_h * 2),
                fontsize=9, color=C_PRIMARY,
                arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=0.8))

    # (b) Re/Im of rho^eff
    ax = axes[0, 1]
    ax.plot(omega/omega_0, rho_eff.real/rho_h, color=C_PRIMARY, lw=2,
            label=r"$\Re\,\rho^{\mathrm{eff}}/\rho_h$")
    ax.plot(omega/omega_0, rho_eff.imag/rho_h, color=C_RED, lw=2, ls="--",
            label=r"$\Im\,\rho^{\mathrm{eff}}/\rho_h$")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    # Highlight negative-mass band
    neg_band = rho_eff.real < 0
    if neg_band.any():
        omega_neg = omega[neg_band]/omega_0
        ax.axvspan(omega_neg.min(), omega_neg.max(),
                   color=C_NEG, alpha=0.15)
        ax.text((omega_neg.min()+omega_neg.max())/2, ax.get_ylim()[0]*0.6,
                "negative-mass\nband",
                ha="center", fontsize=9, color=C_NEG,
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor=C_NEG, boxstyle="round,pad=0.18"))
    ax.set_xlabel(r"$\omega / \omega_0$")
    ax.set_ylabel(r"$\rho^{\mathrm{eff}}(\omega)/\rho_h$")
    ax.set_title("(b) Real/imaginary parts: negative mass and dissipation", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)

    # (c) |S_{1111}(omega)|
    ax = axes[1, 0]
    ax.semilogy(omega/omega_0, abs(S_1111)/S_ref, color=C_RED, lw=2)
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\omega / \omega_0$")
    ax.set_ylabel(r"$|\mathcal{S}_{1111}(\omega)| / \mathcal{S}_{\mathrm{ref}}$")
    ax.set_title("(c) Willis coupling tensor: Lorentzian inheritance", fontsize=11)
    ax.grid(True, which="both", alpha=0.25)
    # Mark Lorentzian peak
    ax.axvspan(1 - fwhm/2, 1 + fwhm/2, color=C_RED, alpha=0.10)

    # (d) phase of S_1111
    ax = axes[1, 1]
    ax.plot(omega/omega_0, np.angle(S_1111)/np.pi, color=C_TEAL, lw=2,
            label=r"$\arg\,\mathcal{S}_{1111}/\pi$")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.axvline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\omega / \omega_0$")
    ax.set_ylabel(r"$\arg\,\mathcal{S}_{1111}\;/\;\pi$")
    ax.set_title("(d) Phase: $\\pi$-jump signature of the resonance", fontsize=11)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.25)
    # Annotate the jump
    ax.annotate(r"$\pi$-jump at $\omega = \omega_0$",
                xy=(1.0, 0.0),
                xytext=(0.65, 0.6),
                fontsize=9, color=C_TEAL,
                arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=0.9),
                bbox=dict(facecolor="white", alpha=0.9,
                          edgecolor=C_TEAL, boxstyle="round,pad=0.18"))

    plt.tight_layout()
    out_pdf = FIG / "Q2_resonance_sweep.pdf"
    plt.savefig(out_pdf, dpi=160, bbox_inches="tight")
    plt.close()

    # ----- print key numbers -----
    print(f"# Q2 / L1 -- Resonance sweep computed")
    print(f"# omega_0 = {omega_0}, eta = {eta} (FWHM = {fwhm:.4f})")
    print(f"# m_r/|Y| = {m_r_over_Y}, phi = {phi}, B^(1)_1111 = {B_1111}")
    print()
    idx_peak = np.argmin(abs(omega - omega_0))
    print(f"# At omega = omega_0 (peak):")
    print(f"   |rho^eff|/rho_h = {abs(rho_eff[idx_peak])/rho_h:.3f}")
    print(f"   |S_1111|/S_ref  = {abs(S_1111[idx_peak])/S_ref:.3f}")
    print()
    if neg_band.any():
        print(f"# Negative-mass band: omega/omega_0 in "
              f"[{omega[neg_band].min()/omega_0:.4f}, "
              f"{omega[neg_band].max()/omega_0:.4f}]  "
              f"(width = {(omega[neg_band].max() - omega[neg_band].min())/omega_0:.4f})")
    print()
    print(f"Figure: {out_pdf}")
    print(f"Data:   {out}")


if __name__ == "__main__":
    main()
