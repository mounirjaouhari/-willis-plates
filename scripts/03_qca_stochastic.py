"""
Q2 / L4 v2 -- Stochastic QCA with FIXED cluster size L = 5 wavelengths.

The previous run (Q2_qca_stochastic.py) kept N=50 fixed and let L shrink with
phi, so the cluster lost its bulk character at large phi (L/lambda ~ 1.8 at
phi=0.30). Here we fix L = 5 * lambda and let N vary with phi to maintain a
true bulk regime at every density.

Setup:
    a = 1, kb*a = 0.5, lambda = 4*pi*a ~ 12.57
    L = 5 * lambda ~ 62.83
    N(phi) = phi * L^2 / (pi * a^2)
    K_real = 100 (compromise: enough statistics, manageable runtime)

Observation:
    Strict-interior compact: a 5x5 grid of points in [L/2 - lambda, L/2 + lambda]
    centered in the cluster, well clear of the boundary layer.

The script reports:
    - Foldy-derived k_eff(omega; phi)
    - Ensemble-averaged coherent field <w>
    - Relative L^2 error |<w_MST> - exp(i*k_eff*x)| / |w_inc|
    - Empirical power-law fit: log E vs log phi
    - Comparison with predicted exponent 0.5 (Proposition 3.1)
"""

from pathlib import Path
import csv
import time
import numpy as np
import scipy.special as sp

DATA = Path(__file__).parent / "data"
DATA.mkdir(parents=True, exist_ok=True)


def Hn1(n, x): return sp.hankel1(n, x)


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
    K_real = 100
    phi_values = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    print(f"# Q2/L4 v2 -- Stochastic QCA, FIXED L = 5 lambda")
    print(f"# kb*a = {kb_a}, lambda = {lam:.3f}, L = {L:.3f}")
    print(f"# K_real = {K_real}, n_max = {n_max}")
    print()
    print(f"{'phi':>6} {'N':>5} {'keff_re':>10} {'keff_im':>10} "
          f"{'<w>err':>12} {'sem':>12} {'sqrt(phi)':>10} {'wall_s':>8}")
    print("-" * 90)

    rng = np.random.default_rng(seed=2024)
    t = t_matrix(n_max, kb_a)
    rows = []

    for phi in phi_values:
        N_inc = int(round(phi * L * L / (np.pi * a ** 2)))
        if N_inc < 5:
            print(f"  phi={phi}: N={N_inc} too small, skipping")
            continue
        n0 = phi / (np.pi * a ** 2)
        keff = k_eff_foldy(kb, n_max, t, n0)

        t0 = time.time()
        # Observation: 5x5 grid in central window of size 2 lambda
        n_obs = 5
        xs = np.linspace(L / 2 - lam, L / 2 + lam, n_obs)
        ys = np.linspace(L / 2 - lam, L / 2 + lam, n_obs)
        x_obs = np.array([(x, y) for x in xs for y in ys])

        all_w_total = []
        for k_real in range(K_real):
            try:
                positions = sample_hard_disks(N_inc, L, a, d0_factor=2.5, rng=rng)
            except RuntimeError as e:
                continue
            A, rhs = foldy_lax(positions, kb, n_max, t)
            c_coeffs = np.linalg.solve(A, rhs)
            w_scat = evaluate_field(positions, c_coeffs, kb, n_max, x_obs)
            w_inc = np.exp(1j * kb * x_obs[:, 0])
            w_total = w_inc + w_scat
            all_w_total.append(w_total)

        if len(all_w_total) == 0:
            print(f"  phi={phi}: no valid realizations")
            continue

        all_w_total = np.array(all_w_total)
        mean_total = all_w_total.mean(axis=0)
        std_total = all_w_total.std(axis=0, ddof=1)
        w_qca = np.exp(1j * keff * x_obs[:, 0])
        w_inc = np.exp(1j * kb * x_obs[:, 0])

        l2_inc = np.sqrt(np.mean(np.abs(w_inc) ** 2))
        l2_diff = np.sqrt(np.mean(np.abs(mean_total - w_qca) ** 2))
        l2_std = np.sqrt(np.mean(np.abs(std_total) ** 2))
        rel_err = l2_diff / l2_inc
        sem = l2_std / np.sqrt(len(all_w_total)) / l2_inc
        wall = time.time() - t0

        print(f"{phi:>6.3f} {N_inc:>5d} {keff.real:>10.4f} {keff.imag:>10.4f} "
              f"{rel_err:>12.4e} {sem:>12.4e} {np.sqrt(phi):>10.4f} {wall:>8.1f}", flush=True)
        rows.append({
            'phi': phi, 'N': N_inc, 'L': L,
            'k_eff_real': keff.real, 'k_eff_imag': keff.imag,
            'rel_err': rel_err, 'sem': sem, 'sqrt_phi': np.sqrt(phi),
            'realizations': len(all_w_total),
        })

    out_csv = DATA / "Q2_qca_stochastic_v2.csv"
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nData saved: {out_csv}")

    # Empirical power-law fit
    if len(rows) >= 4:
        log_phi = np.log([r['phi'] for r in rows])
        log_E = np.log([r['rel_err'] for r in rows])
        slope, intercept = np.polyfit(log_phi, log_E, 1)
        # R^2
        log_E_fit = slope * log_phi + intercept
        ss_res = np.sum((log_E - log_E_fit) ** 2)
        ss_tot = np.sum((log_E - log_E.mean()) ** 2)
        R2 = 1 - ss_res / ss_tot
        print(f"\n# Empirical power-law fit: log E_QCA = {intercept:.3f} + {slope:.3f} log phi")
        print(f"# R^2 = {R2:.4f}")
        print(f"# Theoretical exponent (Prop 3.1): 0.5")
        print(f"# Discrepancy: {abs(slope - 0.5):.3f}")


if __name__ == "__main__":
    main()
