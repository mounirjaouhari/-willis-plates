# Protocol mapping — Article §6 ↔ scripts ↔ data ↔ figures 

This document gives the bijection between the eight numerical protocols of
Section 6 of the paper, the reproducibility scripts of `scripts/`, the
output files of `data/`, and the figures of the manuscript.

## Mapping table

| §  | Article protocol | Script(s) | Output file(s) | Figure(s) | Theorem validated |
|----|------------------|-----------|----------------|-----------|-------------------|
| 6.1 | Numerical methods (preliminary T-matrix) | `00_lamb_modes_and_tmatrix.py` | `Q1_tmatrix_sweep.dat` | – | (setup) |
| 6.2 | Convergence of the multipolar truncation | `01_multipolar_convergence.py`, `01_multipolar_convergence_plot.py` | `multipolar_convergence.csv`, `convergence_kb_*.dat` | `fig:convergence`, `fig:tmatrix-sweep` | Prop 2.4 |
| 6.3 | Accuracy of the QCA (Foldy dilute regime) | `02_qca_validation.py`, `02_qca_transmission.py`, `02_qca_plot.py` | `qca_validation.csv`, `qca_keff.dat`, `Q2_qca_transmission.log` | `fig:qca-validation` | Prop 3.1 (qualitative) |
| 6.4 | Asymptotic QCA rate (stochastic campaign) | `03_qca_stochastic.py`, `03_qca_stochastic_analysis.py` | `Q2_qca_stochastic.csv`, `Q2_qca_stochastic_v2.log`, `Q2_qca_stochastic_v2_summary.csv` | – | Prop 3.1 (rate $\phi^{1/2}$) |
| 6.5 | Static membrane cell problem | `04_cell_membrane_static.py`, `04_cell_membrane_plot.py`, `freefem/cell_membrane.edp` | `cell_membrane.csv`, `cell_membrane_pgf.dat` | `fig:cell-membrane`, `fig:correctors-membrane` | Th 3.2 (partial: $C^{\mathrm{eff}}$) |
| 6.6 | Activation criterion for Willis | `05_willis_activation.py` | `willis_activation.csv`, `dilute_S.csv` | – | Th 3.3 (i) |
| 6.7 | Dynamic Willis tensor near resonance | `06_resonance_sweep.py` | `Q2_resonance_sweep.dat` | `fig:resonance-sweep` | Th 3.3 (iii), Lorentzian |
| 6.8 | Hadamard shape derivative (FD) | `07_hadamard_fd.py` | `Q3_hadamard_fd.csv`, `Q3_hadamard_fd.log` | `fig:hadamard-fd` | Th 5.2 |
| 6.9 | Coupled Bloch–Floquet evaluation of $\mathcal{S}$ | `08_bloch_coupled.py`, `08_bloch_coupled_postprocess.py` | `Q2_bloch_coupled_static.csv`, `Q2_bloch_coupled_omega.csv`, `Q2_bloch_coupled.log` | `fig:bloch-coupled` | Th 3.2 + Th 3.3 (full) |

## Auxiliary scripts

| Script | Role |
|--------|------|
| `01_multipolar_convergence_plot.py` | Plot helper for Sec 6.2 |
| `02_qca_plot.py` | Plot helper for Sec 6.3 |
| `04_cell_membrane_plot.py` | Plot helper for Sec 6.5 |
| `08_bloch_coupled_postprocess.py` | Post-processing of Bloch–Floquet output for Sec 6.9 |
| `03_qca_stochastic_legacy.py` | Earlier prototype of `03_qca_stochastic.py` (kept for traceability) |

## Library

The reusable Willis tensor builder lives in
`src/willis_plates/effective_tensors.py`; the dilute closed-form expression
$\mathcal{S}_{\alpha\beta\gamma\delta}^{\mathrm{LO}} = \phi B^{(1)}(1 + \beta_{r}\mathcal{R}(\omega))$
is the reference against which the numerical Bloch–Floquet evaluation of
Sec 6.9 is benchmarked.

## Tests

| Test | Role |
|------|------|
| `tests/test_argyris.py` | Smoke check for the Argyris element assembly used in Sec 6.9. |
| `tests/test_freefem.edp` | Smoke check for the FreeFEM++ environment used in Sec 6.5. |
