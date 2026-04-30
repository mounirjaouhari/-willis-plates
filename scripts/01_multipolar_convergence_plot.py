"""
Generate the convergence figure from the data produced by multipolar_convergence.py.

Produces a TikZ-importable figure (via pgfplots-readable .dat format) AND a PDF
preview for sanity check.
"""

import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

THIS = Path(__file__).parent
DATA = THIS / "data" / "multipolar_convergence.csv"
FIG_DIR = THIS.parent / "figures"

rows = []
with open(DATA, 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append({
            'kb_a': float(r['kb_a']),
            'N': int(r['N']),
            'rel_err': float(r['rel_err']),
            'theory_rate': float(r['theory_rate']),
        })

kb_values = sorted(set(r['kb_a'] for r in rows))

# ----------------------------------------------------------------------------
# Generate per-frequency .dat files for pgfplots
# ----------------------------------------------------------------------------
data_dir = THIS / "data"
for kb in kb_values:
    sub = [r for r in rows if r['kb_a'] == kb]
    sub.sort(key=lambda r: r['N'])
    out = data_dir / f"convergence_kb_{kb:.2f}.dat"
    with open(out, 'w') as f:
        f.write("N rel_err theory\n")
        for r in sub:
            err = max(r['rel_err'], 1e-18)
            theory = min(r['theory_rate'], 1e18)
            f.write(f"{r['N']} {err:.6e} {theory:.6e}\n")

# ----------------------------------------------------------------------------
# Matplotlib preview (PDF) for inspection
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#0066CC', '#990000', '#008080', '#888800', '#660066']
for kb, col in zip([0.25, 0.5, 1.0, 1.5, 2.0], colors):
    sub = sorted([r for r in rows if r['kb_a'] == kb], key=lambda r: r['N'])
    Ns = [r['N'] for r in sub]
    errs = [max(r['rel_err'], 1e-18) for r in sub]
    theory = [min(r['theory_rate'], 1e18) for r in sub]
    ax.semilogy(Ns, errs, 'o-', color=col, label=f'$k_b a = {kb}$ (FEM)', linewidth=1.5, markersize=6)
    ax.semilogy(Ns, theory, '--', color=col, linewidth=0.9, alpha=0.6)

ax.axhline(0.02, color='gray', linestyle=':', linewidth=1)
ax.text(11.5, 0.025, '2% threshold', fontsize=8, color='gray', ha='right')
ax.set_xlabel('Multipolar order $N$', fontsize=11)
ax.set_ylabel('Relative error $\\mathcal{E}_N$', fontsize=11)
ax.set_xlim(0, 13)
ax.set_ylim(1e-18, 2)
ax.grid(True, which='both', alpha=0.25)
ax.legend(loc='upper right', fontsize=9, ncol=2)
ax.set_title('Multipolar truncation error: numerical vs Proposition 2.4 rate', fontsize=10)
plt.tight_layout()
out_pdf = FIG_DIR / "convergence_preview.pdf"
plt.savefig(out_pdf, dpi=150)
print(f"Preview saved: {out_pdf}")
print(f"Pgfplots data files: {data_dir}/convergence_kb_*.dat")
