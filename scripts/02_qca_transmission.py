""" 
Q2 / Phase 2 -- Stochastic QCA validation via coherent transmission coefficient.

Tests Proposition 3.1 (QCA asymptotic phi^{1/2} rate) using the metric
recommended by Mercier-Maurel for elastic plates and Tsang-Kong for EM:
the relative error of the ensemble-averaged coherent transmission

    E_QCA(phi) = | <T_MST>(phi) - T_QCA(phi) | / | T_QCA(phi) |

where
    <T_MST>(phi) = ensemble-averaged w(L_+, y) / w_inc(L_+, y), at the cluster
                   exit just past the rightmost inclusion
    T_QCA(phi)   = exp(i k_eff(phi) * L)
    k_eff(phi)   = k_b - i n0 f(0) / k_b   (Foldy formula)

Setup:
    a = 1, kb*a = 0.5, lambda = 4*pi*a ~ 12.57
    L = 5 * lambda ~ 62.83
    N(phi) = phi * L^2 / (pi * a^2)    (varies with phi)
    K_real = 50

Expected behavior:
    log E_QCA = constant + 0.5 * log(phi)   (Prop 3.1)

Reports honestly: empirical slope, R^2, bootstrap 95% CI on slope, comparison
to theoretical exponent 0.5.
"""

from pathlib import Path
import csv
import time
import numpy as np
import scipy.special as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).parent / "data"
DATA.mkdir(parents=True, exist_ok=True)
FIG = Path(__file__).parent.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C_PRIMARY = "#0066CC"
C_RED = "#990000"
C_TEAL = "#008080"


def Hn1(n, x):
    return sp.hankel1(n, x)


def t_matrix(n_max, kb_a):
    t = np.zeros(2 * n_max + 1, dtype=complex)
    for idx in range(2 * n_max + 1):
        n = idx - n_max
        t[idx] = -((1j) ** n) * sp.jv(n, kb_a) / sp.hankel1(n, kb_a)
    return t


def sample_hard_disks(N, L, a, d0_factor=2.5, max_iter=50000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    d0 = d0_factor * a
    pos = []
    tries = 0
    while len(pos) < N and tries < max_iter * N:
        x = rng.uniform(a, L - a)
        y = rng.uniform(a, L - a)
        ok = all((px - x) ** 2 + (py - y) ** 2 >= d0 ** 2 for (px, py) in pos)
        if ok:
            pos.append((x, y))
        tries += 1
    if len(pos) < N:
        raise RuntimeError(f"placement failed: N={N}, L={L:.1f}; got {len(pos)}/{N}")
    return np.array(pos)


def foldy_lax(positions, kb, n_max, t):
    N = len(positions)
    nrows = 2 * n_max + 1
    M = N * nrows
    A = np.eye(M, dtype=complex)
    rhs = np.zeros(M, dtype=complex)

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
                row = j * nrows + idx_n
                t_n = t[idx_n]
                for idx_m in range(nrows):
                    m = idx_m - n_max
                    col = k * nrows + idx_m
                    g = Hn1(n - m, kb * R_jk) * np.exp(1j * (n - m) * theta_jk)
                    A[row, col] -= t_n * g

    for j in range(N):
        phase = np.exp(1j * kb * positions[j, 0])
        for idx_n in range(nrows):
            n = idx_n - n_max
            t_n = t[idx_n]
            an_inc = (1j) ** n * phase
            row = j * nrows + idx_n
            rhs[row] = t_n * an_inc

    return A, rhs


def evaluate_field(positions, c_coeffs, kb, n_max, x_obs):
    N = len(positions)
    nrows = 2 * n_max + 1
    w = np.zeros(len(x_obs), dtype=complex)
    for i, x in enumerate(x_obs):
        for j in range(N):
            dx = x[0] - positions[j, 0]
            dy = x[1] - positions[j, 1]
            r = np.hypot(dx, dy)
            if r < 1e-10:
                continue
            theta = np.arctan2(dy, dx)
            for idx_n in range(nrows):
                n = idx_n - n_max
                w[i] += c_coeffs[j * nrows + idx_n] * Hn1(n, kb * r) * np.exp(1j * n * theta)
    return w


def k_eff_foldy(kb, n_max, t, n0):
    f0 = sum(t[idx] * ((-1j) ** (idx - n_max)) for idx in range(len(t)))
    return kb - 1j * n0 * f0 / kb


def main():
    a = 1.0
    kb_a = 0.5
    kb = kb_a / a
    lam = 2 * np.pi / kb
    L = 5 * lam   # fixed cluster size: 5 wavelengths
    n_max = 4
    K_real = 50
    phi_values = [0.05, 0.10, 0.15, 0.20, 0.25]

    print(f"# Q2 / Phase 2 -- Stochastic QCA via coherent transmission")
    print(f"# kb*a = {kb_a}, lambda = {lam:.3f}, L = {L:.3f}")
    print(f"# K_real = {K_real}, n_max = {n_max}")
    print()
    print(f"{'phi':>6} {'N':>5} {'keff_re':>10} {'keff_im':>10} "
          f"{'|T_QCA|':>10} {'|<T_MST>|':>10} {'E_QCA':>10} {'sem':>10} {'wall_s':>8}")
    print("-" * 100)

    rng = np.random.default_rng(seed=2025)
    t = t_matrix(n_max, kb_a)
    rows = []

    for phi in phi_values:
        N_inc = int(round(phi * L * L / (np.pi * a ** 2)))
        if N_inc < 5:
            print(f"  phi={phi}: N={N_inc} too small, skipping")
            continue
        n0 = phi / (np.pi * a ** 2)
        keff = k_eff_foldy(kb, n_max, t, n0)
        T_QCA = np.exp(1j * keff * L)

        # Observation: line at x = L + small, several y values, average over y
        eps_safe = 0.5  # 0.5 unit beyond rightmost inclusion (well beyond a=1)
        n_y_obs = 9
        ys_obs = np.linspace(L * 0.2, L * 0.8, n_y_obs)
        x_obs = np.array([(L + eps_safe, yo) for yo in ys_obs])

        t0 = time.time()
        T_MST_realizations = []
        for k_real in range(K_real):
            try:
                positions = sample_hard_disks(N_inc, L, a, d0_factor=2.5, rng=rng)
            except RuntimeError:
                continue
            A, rhs = foldy_lax(positions, kb, n_max, t)
            c_coeffs = np.linalg.solve(A, rhs)
            w_scat = evaluate_field(positions, c_coeffs, kb, n_max, x_obs)
            w_inc = np.exp(1j * kb * x_obs[:, 0])
            w_total = w_inc + w_scat
            # Transmission per realization = w_total / w_inc averaged over y
            T_real = (w_total / w_inc).mean()
            T_MST_realizations.append(T_real)

        T_MST_realizations = np.array(T_MST_realizations)
        T_MST_mean = T_MST_realizations.mean()
        # Bootstrap CI for mean
        nb = 1000
        T_MST_boot = np.array([
            T_MST_realizations[rng.integers(0, len(T_MST_realizations), len(T_MST_realizations))].mean()
            for _ in range(nb)
        ])

        E_QCA = abs(T_MST_mean - T_QCA) / abs(T_QCA)
        E_boot = np.abs(T_MST_boot - T_QCA) / abs(T_QCA)
        sem = float(E_boot.std(ddof=1))
        wall = time.time() - t0

        print(f"{phi:>6.3f} {N_inc:>5d} {keff.real:>10.4f} {keff.imag:>10.4f} "
              f"{abs(T_QCA):>10.4e} {abs(T_MST_mean):>10.4e} "
              f"{E_QCA:>10.4e} {sem:>10.4e} {wall:>8.1f}", flush=True)

        rows.append({
            'phi': phi, 'N': N_inc, 'L': L,
            'k_eff_real': keff.real, 'k_eff_imag': keff.imag,
            'T_QCA_re': T_QCA.real, 'T_QCA_im': T_QCA.imag,
            'T_MST_re': T_MST_mean.real, 'T_MST_im': T_MST_mean.imag,
            'E_QCA': E_QCA, 'sem': sem,
            'sqrt_phi': np.sqrt(phi),
            'realizations': len(T_MST_realizations),
            'E_boot': E_boot.tolist(),
        })

    out_csv = DATA / "Q2_qca_transmission.csv"
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'phi', 'N', 'L', 'k_eff_real', 'k_eff_imag',
            'T_QCA_re', 'T_QCA_im', 'T_MST_re', 'T_MST_im',
            'E_QCA', 'sem', 'sqrt_phi', 'realizations',
        ])
        writer.writeheader()
        for r in rows:
            r2 = {k: v for k, v in r.items() if k != 'E_boot'}
            writer.writerow(r2)
    print(f"\nData saved: {out_csv}")

    # Power-law fit
    if len(rows) >= 4:
        log_phi = np.log([r['phi'] for r in rows])
        log_E = np.log([r['E_QCA'] for r in rows])
        slope, intercept = np.polyfit(log_phi, log_E, 1)
        log_E_fit = slope * log_phi + intercept
        ss_res = np.sum((log_E - log_E_fit) ** 2)
        ss_tot = np.sum((log_E - log_E.mean()) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # Bootstrap on slope
        n_boot = 1000
        slopes_boot = []
        for _ in range(n_boot):
            E_b = np.array([np.random.choice(r['E_boot']) for r in rows])
            mask = E_b > 0
            if mask.sum() >= 3:
                s_b, _ = np.polyfit(log_phi[mask], np.log(E_b[mask]), 1)
                slopes_boot.append(s_b)
        slopes_boot = np.array(slopes_boot)
        ci_low = float(np.percentile(slopes_boot, 2.5))
        ci_high = float(np.percentile(slopes_boot, 97.5))

        print()
        print(f"# Empirical power-law fit:")
        print(f"#   log E_QCA = {intercept:.4f} + {slope:.4f} * log phi")
        print(f"#   R^2 = {R2:.4f}")
        print(f"#   bootstrap 95% CI on slope: [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"# Theoretical slope (Prop 3.1): 0.5")
        print(f"# Discrepancy from theoretical: {abs(slope - 0.5):.3f}")

        # Figure
        fig, ax = plt.subplots(figsize=(7, 5))
        phis = np.array([r['phi'] for r in rows])
        E_vals = np.array([r['E_QCA'] for r in rows])
        E_sem = np.array([r['sem'] for r in rows])
        ax.errorbar(phis, E_vals, yerr=E_sem, fmt='o', color=C_PRIMARY, markersize=8,
                    capsize=4, label=r"empirical $E_{\rm QCA}(\phi)$")
        phi_dense = np.geomspace(phis.min(), phis.max(), 100)
        ax.loglog(phi_dense, np.exp(intercept) * phi_dense ** slope,
                  '--', color=C_RED, lw=1.5,
                  label=fr"empirical fit, slope = {slope:.2f}, $R^2={R2:.3f}$")
        # Theoretical phi^0.5 line, normalized to first point
        C0 = E_vals[0] / np.sqrt(phis[0])
        ax.loglog(phi_dense, C0 * np.sqrt(phi_dense),
                  ':', color=C_TEAL, lw=1.5,
                  label=r"theoretical, slope $= 1/2$")
        ax.set_xlabel(r"surface filling fraction $\phi$")
        ax.set_ylabel(r"relative coherent transmission error $E_{\rm QCA}(\phi)$")
        ax.set_title("Asymptotic QCA rate: coherent transmission test", fontsize=11)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        out_pdf = FIG / "Q2_qca_transmission.pdf"
        plt.savefig(out_pdf, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Figure: {out_pdf}")

        # CI text file
        out_ci = DATA / "Q2_qca_transmission_summary.txt"
        with open(out_ci, "w") as f:
            f.write(f"# QCA transmission asymptotic rate\n")
            f.write(f"slope = {slope:.4f}\n")
            f.write(f"intercept = {intercept:.4f}\n")
            f.write(f"R^2 = {R2:.4f}\n")
            f.write(f"95% CI low = {ci_low:.4f}\n")
            f.write(f"95% CI high = {ci_high:.4f}\n")
            f.write(f"theoretical = 0.5\n")
            f.write(f"discrepancy = {abs(slope - 0.5):.4f}\n")
        print(f"Summary: {out_ci}")


if __name__ == "__main__":
    main()
