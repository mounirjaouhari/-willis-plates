""" 
Generate pgfplots data + Voigt bound for the FEM cell-problem validation.
"""

import csv
from pathlib import Path

THIS = Path(__file__).parent
DATA_IN = THIS / "data" / "cell_membrane.csv"
DATA_OUT = THIS / "data" / "cell_membrane_pgf.dat"

# Material constants used in the FreeFEM script
E = 1.0
nu = 0.30
lambda_b = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
lambda_star = 2 * lambda_b * mu / (lambda_b + 2 * mu)

C1111_0 = lambda_star + 2 * mu  # = 1.0989...
C1212_0 = mu                     # = 0.3846...
C1122_0 = lambda_star            # = 0.3297...

print(f"Reference (no hole): C1111={C1111_0:.4f}, C1122={C1122_0:.4f}, C1212={C1212_0:.4f}")

rows = []
with open(DATA_IN, 'r') as f:
    # Custom parsing because separator is space
    header = f.readline().strip().split()
    for line in f:
        parts = line.strip().split()
        if not parts:
            continue
        rows.append({
            'phi':   float(parts[0]),
            'C1111': float(parts[1]),
            'C2222': float(parts[2]),
            'C1122': float(parts[3]),
            'C1212': float(parts[4]),
        })

# Output normalized values + Voigt bound
with open(DATA_OUT, 'w') as f:
    f.write("phi C1111_norm C1212_norm C1122_norm Voigt_norm\n")
    for r in rows:
        f.write(f"{r['phi']} "
                f"{r['C1111']/C1111_0:.6f} "
                f"{r['C1212']/C1212_0:.6f} "
                f"{r['C1122']/C1122_0:.6f} "
                f"{1 - r['phi']:.6f}\n")

print(f"Pgfplots data: {DATA_OUT}")
for r in rows:
    print(f"  phi={r['phi']:.2f}: C1111/C1111_0 = {r['C1111']/C1111_0:.4f},  Voigt = {1-r['phi']:.4f}")
