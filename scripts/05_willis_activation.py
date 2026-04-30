""" 
Numerical demonstration of Theorem 3.3 (activation/vanishing of the Willis
coupling tensor under broken/preserved mirror symmetry).

Setup:
    Three-layer through-thickness profile of an inclusion of width 2 a in the
    plate of total thickness h. The local elasticity tensor is

        C^(1)(y, z) = c(y, z) * C0,

    where C0 is a fixed dimensionless rank-4 isotropic tensor and c(y, z) is
    a piecewise-constant scalar profile. The laminate moment tensor is

        B^(1)_{alpha beta gamma delta}(y) = int_{-h/2}^{h/2} z c(y, z) dz * C0_{...}

Three configurations are compared:
    A. Mirror-symmetric inclusion: c(y, z) = c_in for |z| <= h/2 inside hole region,
                                   c_out outside; c is even in z. -> B = 0.
    B. Asymmetric "stub" inclusion: extra mass layer on top (z in [h/2, h/2+h_s])
       while bottom is the matrix. The integral takes value over the asymmetric region. -> B != 0.
    C. Bilayer asymmetric: c_top != c_bottom. -> B != 0.

The script computes the diagonal component B_{1111} for each case, demonstrating
the activation criterion of Theorem 3.3 (i)-(ii).

This is a direct algebraic validation: no FEM, no scattering — pure analytical
verification of the laminate moment formula, sufficient for the binary
activation criterion.
"""

import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "data" / "willis_activation.csv"


def laminate_B(profile_c, z_min, z_max, n_quad=1000):
    """
    Compute B = int z * c(z) dz on [z_min, z_max] using composite Simpson.
    """
    z = np.linspace(z_min, z_max, n_quad)
    integrand = z * np.array([profile_c(zi) for zi in z])
    return np.trapezoid(integrand, z)


def main():
    h = 1.0
    h_s = 0.5 * h          # stub height (for asymmetric case)
    c_matrix = 1.0         # baseline elasticity coefficient
    c_stub = 3.0           # stub material (denser)
    c_top = 2.0
    c_bot = 0.5

    # ------------------------------------------------------------------
    # Case A: Mirror-symmetric inclusion (uniform fill of plate thickness)
    # ------------------------------------------------------------------
    def profile_A(z):
        # Symmetric: c(z) = c(-z); just c_matrix uniformly
        return c_matrix

    B_A = laminate_B(profile_A, -h / 2, h / 2)

    # ------------------------------------------------------------------
    # Case B: Asymmetric stub on TOP only
    # ------------------------------------------------------------------
    def profile_B(z):
        # Matrix in [-h/2, h/2], plus stub [h/2, h/2 + h_s]
        if -h / 2 <= z <= h / 2:
            return c_matrix
        elif h / 2 < z <= h / 2 + h_s:
            return c_stub
        else:
            return 0.0

    B_B = laminate_B(profile_B, -h / 2, h / 2 + h_s)

    # ------------------------------------------------------------------
    # Case C: Bilayer asymmetric (top half c_top, bottom half c_bot)
    # ------------------------------------------------------------------
    def profile_C(z):
        return c_top if z >= 0 else c_bot

    B_C = laminate_B(profile_C, -h / 2, h / 2)

    # ------------------------------------------------------------------
    # Analytical references
    # ------------------------------------------------------------------
    # Case A: int_{-h/2}^{h/2} z c_matrix dz = c_matrix * (h^2/8 - h^2/8) = 0
    B_A_ref = 0.0
    # Case B: int_{-h/2}^{h/2} z c_matrix dz + int_{h/2}^{h/2+h_s} z c_stub dz
    #       = 0 + c_stub * [(h/2 + h_s)^2 - (h/2)^2] / 2
    B_B_ref = c_stub * ((h / 2 + h_s) ** 2 - (h / 2) ** 2) / 2
    # Case C: int_{-h/2}^0 z c_bot dz + int_0^{h/2} z c_top dz
    #       = -c_bot h^2/8 + c_top h^2/8 = (c_top - c_bot) h^2 / 8
    B_C_ref = (c_top - c_bot) * h ** 2 / 8

    # ------------------------------------------------------------------
    # Print and save results
    # ------------------------------------------------------------------
    print("# Willis activation criterion (Theorem 3.3)")
    print(f"# Plate thickness h = {h}, stub height h_s = {h_s}")
    print()
    print(f"{'Case':<30} {'B_num':>12} {'B_ref':>12} {'rel_err':>12}")
    print("-" * 70)

    rows = [
        ("A. Mirror-symmetric (uniform)", B_A, B_A_ref),
        ("B. Asymmetric stub on top",      B_B, B_B_ref),
        ("C. Bilayer (c_top != c_bot)",     B_C, B_C_ref),
    ]
    for name, b_num, b_ref in rows:
        if b_ref == 0:
            rel = abs(b_num)
        else:
            rel = abs(b_num - b_ref) / abs(b_ref)
        print(f"{name:<30} {b_num:>12.6f} {b_ref:>12.6f} {rel:>12.2e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        f.write("case B_numerical B_analytical relative_error\n")
        for name, b_num, b_ref in rows:
            rel = abs(b_num - b_ref) / max(abs(b_ref), 1e-15) if b_ref != 0 else abs(b_num)
            tag = name.split('.')[0].strip()
            f.write(f"{tag} {b_num:.6e} {b_ref:.6e} {rel:.6e}\n")

    print()
    print("Conclusion (Theorem 3.3):")
    print(f"  A (mirror-symmetric): B = {B_A:.3e}    -> S vanishes (criterion (i))")
    print(f"  B (stub on top):      B = {B_B:.3f}    -> S != 0 (criterion (ii))")
    print(f"  C (bilayer):           B = {B_C:.3f}    -> S != 0 (criterion (ii))")


if __name__ == '__main__':
    main()
