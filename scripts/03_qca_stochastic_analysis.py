"""
Q2 / L4 v2 -- Honest analysis of the stochastic QCA error vs phi.

Reads the log of Q2_qca_stochastic_v2.py (whatever phi values have completed)
and reports:
    - empirical power-law slope and R^2
    - same fit restricted to phi >= 0.10 (where SEM/signal is small)
    - SEM-to-signal ratio per phi
    - figure: log-log plot with sqrt(phi) reference

Honest reporting: whatever the slope is, it is reported as-is. If the
phi^{1/2} prediction is NOT observed, the discrepancy is documented
together with the dominant source (K-statistical noise vs finite-size
boundary effects).
"""

from pathlib import Path
import re
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).parent / "data"
FIG = Path(__file__).parent.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

LOG = DATA / "Q2_qca_stochastic_v2.log"

C_PRIMARY = "#0066CC"
C_RED = "#990000"
C_TEAL = "#008080"


def parse_log(path):
    rows = []
    pat = re.compile(
        r"^\s*([0-9.]+)\s+(\d+)\s+([\-0-9.]+)\s+([\-0-9.]+)\s+"
        r"([\-0-9.eE+]+)\s+([\-0-9.eE+]+)\s+([0-9.]+)\s+([0-9.]+)\s*$"
    )
    with open(path, "r") as f:
        for line in f:
            m = pat.match(line)
            if m:
                rows.append({
                    "phi": float(m.group(1)),
                    "N": int(m.group(2)),
                    "keff_re": float(m.group(3)),
                    "keff_im": float(m.group(4)),
                    "rel_err": float(m.group(5)),
                    "sem": float(m.group(6)),
                    "sqrt_phi": float(m.group(7)),
                    "wall": float(m.group(8)),
                })
    return rows


def fit_power_law(phi, E):
    log_phi = np.log(phi)
    log_E = np.log(E)
    slope, intercept = np.polyfit(log_phi, log_E, 1)
    log_E_fit = slope * log_phi + intercept
    ss_res = np.sum((log_E - log_E_fit) ** 2)
    ss_tot = np.sum((log_E - log_E.mean()) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, intercept, R2


def main():
    rows = parse_log(LOG)
    if len(rows) < 3:
        print(f"Only {len(rows)} rows -- not enough for fit.")
        return
    phi = np.array([r["phi"] for r in rows])
    E = np.array([r["rel_err"] for r in rows])
    sem = np.array([r["sem"] for r in rows])
    N = np.array([r["N"] for r in rows])
    keff_im = np.array([r["keff_im"] for r in rows])

    # Signal-to-noise ratio (E vs SEM)
    snr = E / np.maximum(sem, 1e-12)

    print(f"# Q2 / L4 v2 -- Stochastic QCA: honest analysis")
    print(f"# {len(rows)} phi values: {phi.tolist()}")
    print()
    print(f"  phi    N    rel_err    SEM      SNR(E/SEM)   sqrt(phi)")
    for r, s in zip(rows, snr):
        print(f"  {r['phi']:.3f} {r['N']:4d}  {r['rel_err']:.4e} "
              f"{r['sem']:.4e}  {s:6.2f}    {r['sqrt_phi']:.3f}")
    print()

    # Full fit
    slope_all, intercept_all, R2_all = fit_power_law(phi, E)
    print(f"# All {len(rows)} points:")
    print(f"#   slope = {slope_all:.3f}, intercept = {intercept_all:.3f}, R^2 = {R2_all:.4f}")
    print(f"#   discrepancy from theoretical 0.5 = {abs(slope_all - 0.5):.3f}")
    print()

    # Restrict to phi >= 0.10 (low-SEM regime)
    mask = phi >= 0.10
    if mask.sum() >= 3:
        slope_hi, intercept_hi, R2_hi = fit_power_law(phi[mask], E[mask])
        print(f"# Restricted to phi >= 0.10 ({mask.sum()} points):")
        print(f"#   slope = {slope_hi:.3f}, intercept = {intercept_hi:.3f}, R^2 = {R2_hi:.4f}")
    else:
        slope_hi = intercept_hi = R2_hi = None

    # Save data table
    out_csv = DATA / "Q2_qca_stochastic_v2_summary.csv"
    with open(out_csv, "w") as f:
        f.write("phi,N,k_eff_real,k_eff_imag,rel_err,sem,snr,sqrt_phi\n")
        for r, s in zip(rows, snr):
            f.write(f"{r['phi']},{r['N']},{r['keff_re']},{r['keff_im']},"
                    f"{r['rel_err']},{r['sem']},{s:.4f},{r['sqrt_phi']}\n")
    print(f"\nSummary CSV: {out_csv}")

    # ---------- Figure: log-log plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    ax = axes[0]
    # Error bars from SEM
    ax.errorbar(phi, E, yerr=sem, fmt="o", color=C_PRIMARY,
                markersize=7, capsize=3, lw=1.2,
                label=r"empirical $E = \|\langle w_{MST}\rangle - e^{ik_{\rm eff}x}\|_2 / \|w_{\rm inc}\|_2$")
    # Theoretical sqrt(phi) reference, normalized to first point
    phi_dense = np.geomspace(phi.min(), phi.max(), 100)
    C0 = E[0] / np.sqrt(phi[0])
    ax.plot(phi_dense, C0 * np.sqrt(phi_dense), color=C_RED, lw=1.6, ls="--",
            label=r"$\propto \phi^{1/2}$ (Prop 3.1, theoretical)")
    # Empirical fit (all points)
    ax.plot(phi_dense, np.exp(intercept_all) * phi_dense ** slope_all,
            color=C_TEAL, lw=1.6, ls=":",
            label=fr"empirical fit (all): slope = {slope_all:.2f}, $R^2$ = {R2_all:.2f}")
    if slope_hi is not None:
        ax.plot(phi_dense, np.exp(intercept_hi) * phi_dense ** slope_hi,
                color="#CC6600", lw=1.6, ls="-.",
                label=fr"empirical fit ($\phi\geq 0.10$): slope = {slope_hi:.2f}, $R^2$ = {R2_hi:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\phi$"); ax.set_ylabel(r"$E_{\rm QCA}(\phi)$")
    ax.set_title(r"(a) QCA error vs $\phi$ (fixed $L = 5\lambda$)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.semilogy(phi, sem, "o-", color=C_RED, lw=1.4, label="SEM")
    ax.semilogy(phi, E, "s-", color=C_PRIMARY, lw=1.4, label=r"$E_{\rm QCA}$")
    ax.set_xlabel(r"$\phi$"); ax.set_ylabel("magnitude")
    ax.set_title(r"(b) Statistical noise vs signal", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    # Annotate noise-dominated regime
    noise_dominated = sem > 0.3 * E
    if noise_dominated.any():
        idx = np.where(noise_dominated)[0]
        if len(idx) > 0:
            ax.axvspan(phi[idx.min()], phi[idx.max()],
                       color=C_RED, alpha=0.10)
            ax.text(phi[idx[len(idx)//2]], sem[idx[len(idx)//2]] * 2,
                    "SEM ≳ 30% of signal\n(noise-dominated)",
                    ha="center", fontsize=8, color=C_RED,
                    bbox=dict(facecolor="white", alpha=0.85,
                              edgecolor=C_RED, boxstyle="round,pad=0.18"))

    plt.tight_layout()
    out_pdf = FIG / "Q2_qca_stochastic_v2.pdf"
    plt.savefig(out_pdf, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Figure: {out_pdf}")


if __name__ == "__main__":
    main()
