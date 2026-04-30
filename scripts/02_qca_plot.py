"""
Generate the QCA effective wavenumber figure from validation data.
Plots Re(k_eff)/k_b and Im(k_eff)/k_b vs phi, showing physical consistency
of the Foldy formula in the dilute regime.
"""

import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).parent
DATA = THIS / "data" / "qca_validation.csv"
FIG_DIR = THIS.parent / "figures"

rows = []
with open(DATA, 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({
            'phi': float(r['phi']),
            'k_eff_real': float(r['k_eff_real']),
            'k_eff_imag': float(r['k_eff_imag']),
        })

# Generate pgfplots .dat
out = THIS / "data" / "qca_keff.dat"
kb = 0.5  # from script: kb_a = 0.5, a = 1
with open(out, 'w') as f:
    f.write("phi keff_real_norm keff_imag_norm\n")
    for r in rows:
        f.write(f"{r['phi']} {r['k_eff_real']/kb:.6e} {r['k_eff_imag']/kb:.6e}\n")

print(f"Pgfplots data: {out}")
