"""Quick test that scikit-fem Argyris element works on a simple biharmonic problem.

Solves:  Delta^2 w = 1   on (0,1)^2,  w = 0, partial_n w = 0 on boundary.
Expected: w_max ~ 0.00126 (clamped square plate, uniform load, normalized).
"""

import numpy as np
from skfem import (MeshTri, Basis, BilinearForm, LinearForm, asm,
                   condense, solve)

try:
    from skfem.element import ElementTriArgyris
    print("ElementTriArgyris is available in skfem.element")
except ImportError:
    try:
        from skfem import ElementTriArgyris
        print("ElementTriArgyris is available in skfem (top-level)")
    except ImportError as e:
        print("ElementTriArgyris NOT available:", e)
        ElementTriArgyris = None

try:
    from skfem.element import ElementTriMorley
    print("ElementTriMorley is available")
except ImportError:
    try:
        from skfem import ElementTriMorley
        print("ElementTriMorley is available (top-level)")
    except ImportError:
        print("ElementTriMorley NOT available")
        ElementTriMorley = None

if ElementTriArgyris is None:
    print("Cannot proceed without Argyris/Morley element")
    raise SystemExit(1)


# Build mesh
mesh = MeshTri.init_tensor(np.linspace(0, 1, 11), np.linspace(0, 1, 11))

basis = Basis(mesh, ElementTriArgyris())
print(f"DOFs: {basis.N}")

# Bilinear form: integrate D2(w) : D2(v)  (biharmonic energy)
@BilinearForm
def biharmonic(u, v, w):
    # u.hess and v.hess give second derivatives
    h_u = u.hess
    h_v = v.hess
    # h is (n_dim, n_dim, n_tri, n_quad)
    return (h_u[0, 0] * h_v[0, 0] + h_u[1, 1] * h_v[1, 1]
            + 2 * h_u[0, 1] * h_v[0, 1])

# RHS: f = 1
@LinearForm
def rhs(v, w):
    return 1.0 * v

K = asm(biharmonic, basis)
b = asm(rhs, basis)

# Clamped: w = 0 and partial_n w = 0 on boundary
# For Argyris, we fix vertex DOFs (value + first derivs + second derivs)
# and edge mid-normal-derivative DOFs on the boundary.

# Get all boundary DOFs
fixed_dofs = basis.get_dofs().flatten()
print(f"Boundary DOFs: {len(fixed_dofs)}")

w_sol = solve(*condense(K, b, D=fixed_dofs))
print(f"Max |w|: {np.abs(w_sol).max():.6e}")

# Reference for clamped square plate, uniform load:
# w_max = 0.00126 * q L^4 / D where q=1, L=1, D=1 → ~0.00126
# Check the value at center ~ vertex (5,5):
center_idx = 5 * 11 + 5  # vertex index for (0.5, 0.5)
# In Argyris basis, vertex value DOF for vertex k is at position k*6 + 0
center_val_dof = center_idx * 6 + 0
if center_val_dof < len(w_sol):
    print(f"w(0.5, 0.5) ≈ {w_sol[center_val_dof]:.6e}")
print(f"Expected clamped-plate w_max ~ 1.26e-3 for D=1, q=1, L=1")
