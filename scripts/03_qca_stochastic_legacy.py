"""
Q2 / L4 -- Stochastic validation of QCA at K=200 realizations, N=50 inclusions.

Strategy:
    For each surface filling phi, generate K_real hard-disk configurations of
    N=50 rigid circular holes. Solve the multipolar Foldy-Lax system for each
    realization. The ensemble-mean field is compared, on a strict-interior
    observation set, to the QCA plane wave exp(i * k_eff(omega; phi) * x), where
    k_eff is the Foldy effective wavenumber computed from the multipolar
    T-matrix.

Honest reporting:
    Whatever the measured slope log E_QCA / log phi, it is reported as-is.
    If the empirical rate departs from the predicted phi^{1/2}, the deviation
    is documented; no curve fitting forces a 0.5 exponent on noisy data.

Observation set:
    To approach the bulk regime, observation points are placed at the center
    of each cluster within a window of size 2 wavelengths in x and y.
"""

from pathlib import Path
import csv
import numpy as np
import scipy.special as sp

DATA = Path(__file__).parent / "data"
DATA.mkdir(parents=True, exist_ok=True)


# ----- Bessel helpers -----
def Hn1(n, x): return sp.hankel1(n, x)


# ----- Single-hole T-matrix (clamped Dirichlet for tractability) -----
def t_matrix(n_max, kb_a):
    t = np.zeros(2 * n_max + 1, dtype=complex)
    for idx in range(2 * n_max + 1):
        n = idx - n_max
        # Reduced 1x1 BC: w(a) = 0  -> t_n = -i^n J_n(kb_a) / H_n^{(1)}(kb_a)
        t[idx] = -((1j) ** n) * sp.jv(n, kb_a) / sp.hankel1(n, kb_a)
    return t


# ----- Hard-disk Poisson sampling -----
def sample_hard_disks(N, phi, a, d0_factor=2.5, max_iter=20000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    L = np.sqrt(N * np.pi * a**2 / phi)
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
        raise RuntimeError(f"placement failed: phi={phi}, got {len(pos)}/{N}")
    return np.array(pos), L


# ----- Foldy-Lax assembly -----
def foldy_lax(positions, kb, n_max, t):
    N = len(positions)
    nrows = 2 * n_max + 1
    M = N * nrows
    A = np.eye(M, dtype=complex)  # identity diagonal
    rhs = np.zeros(M, dtype=complex)

    # Off-diagonal Graf-Watson kernel
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

    # RHS: incident plane wave amplitudes
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
    M = len(x_obs)
    w = np.zeros(M, dtype=complex)
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
    N = 50
    n_max = 4
    K_real = 200
    phi_values = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    print(f"# Q2/L4 stochastic QCA: N={N}, K={K_real}, n_max={n_max}, kb*a={kb_a}")
    print(f"# Wavelength lambda = 2 pi / kb = {2*np.pi/kb:.3f}")
    print()
    print(f"{'phi':>6} {'L':>7} {'lam/L':>7} {'keff_re':>10} {'keff_im':>10} "
          f"{'E_QCA':>12} {'std/sqrtK':>12} {'sqrt(phi)':>10}")
    print("-" * 90)

    rng = np.random.default_rng(seed=2024)
    t = t_matrix(n_max, kb_a)
    rows = []

    for phi in phi_values:
        n0 = phi / (np.pi * a ** 2)
        keff = k_eff_foldy(kb, n_max, t, n0)

        # Generate ensemble of realizations and accumulate fields
        # Observation: 5x5 grid in central region of size 2 wavelengths
        all_w_total = []
        all_w_inc = []
        L_used = None
        for k_real in range(K_real):
            try:
                positions, L = sample_hard_disks(N, phi, a, d0_factor=2.5, rng=rng)
            except RuntimeError:
                continue
            L_used = L
            # Observation: square of side 2*lambda centered at (L/2, L/2)
            lam = 2 * np.pi / kb
            half = lam  # half-side = 1 wavelength
            n_obs = 5
            xs = np.linspace(L / 2 - half, L / 2 + half, n_obs)
            ys = np.linspace(L / 2 - half, L / 2 + half, n_obs)
            x_obs = np.array([(x, y) for x in xs for y in ys])

            A, rhs = foldy_lax(positions, kb, n_max, t)
            c_coeffs = np.linalg.solve(A, rhs)
            w_scat = evaluate_field(positions, c_coeffs, kb, n_max, x_obs)
            w_inc = np.exp(1j * kb * x_obs[:, 0])
            w_total = w_inc + w_scat

            all_w_total.append(w_total)
            all_w_inc.append(w_inc)

            if (k_real + 1) % 50 == 0:
                print(f"  ... phi={phi:.2f}, realizations={k_real+1}/{K_real}", flush=True)

        if len(all_w_total) == 0:
            print(f"  phi={phi}: no valid realizations")
            continue

        # Mean field across realizations
        all_w_total = np.array(all_w_total)  # (K, n_obs**2)
        all_w_inc = np.array(all_w_inc)
        mean_total = all_w_total.mean(axis=0)
        std_total = all_w_total.std(axis=0, ddof=1)
        mean_inc = all_w_inc.mean(axis=0)

        # QCA prediction at observation points
        w_qca = np.exp(1j * keff * x_obs[:, 0])

        # Relative error and standard error of the mean
        l2_inc = np.sqrt(np.mean(np.abs(mean_inc) ** 2))
        l2_diff = np.sqrt(np.mean(np.abs(mean_total - w_qca) ** 2))
        l2_std = np.sqrt(np.mean(np.abs(std_total) ** 2))
        rel_err = l2_diff / l2_inc
        sem = l2_std / np.sqrt(len(all_w_total)) / l2_inc

        print(f"{phi:>6.3f} {L_used:>7.2f} {lam/L_used:>7.3f} "
              f"{keff.real:>10.4f} {keff.imag:>10.4f} "
              f"{rel_err:>12.4e} {sem:>12.4e} {np.sqrt(phi):>10.4f}")
        rows.append({
            'phi': phi,
            'L': L_used,
            'k_eff_real': keff.real,
            'k_eff_imag': keff.imag,
            'rel_err': rel_err,
            'sem': sem,
            'sqrt_phi': np.sqrt(phi),
            'realizations': len(all_w_total),
        })

    # Save data
    out_csv = DATA / "Q2_qca_stochastic.csv"
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"Data saved: {out_csv}")

    # Power-law fit log(E) = a + b log(phi)
    if len(rows) >= 4:
        log_phi = np.log([r['phi'] for r in rows])
        log_E = np.log([r['rel_err'] for r in rows])
        coeffs, residuals, rank, sv, _ = np.linalg.lstsq(
            np.vstack([log_phi, np.ones_like(log_phi)]).T, log_E, rcond=None
        )
        slope, intercept = coeffs
        print(f"\n# Empirical power-law fit: log E_QCA = {intercept:.3f} + {slope:.3f} log phi")
        print(f"# Theoretical exponent (Prop 3.1): 0.5")
        print(f"# Discrepancy: {abs(slope - 0.5):.3f}")


if __name__ == "__main__":
    main()
