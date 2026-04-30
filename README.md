# willis-plates — Companion code

Companion repository for the article

> **Tensorial Effective Medium Theory for Thin Elastic Plates with Resonant
> Inclusions: Emergence of a Willis Coupling Tensor and Topology-Driven Design**
> Mounir JAOUHARI & Anass Rachid, 2026.

This repository reproduces the eight numerical protocols of Section 6 of the
paper, validates the three theorems on well-posedness, two-scale homogenization,
and Hadamard shape derivative, and provides a reference implementation of the
multipolar T-matrix, the quasi-crystalline closure, and the coupled
Argyris–$P_{2}$ Bloch–Floquet solver for the Willis tensor.

---

## Repository layout

```
code/
├── README.md                 (this file)
├── requirements.txt          Python dependencies
├── LICENSE                   MIT
│
├── src/willis_plates/        Reusable modules
│   ├── __init__.py
│   └── effective_tensors.py  Dilute Willis tensor builder
│
├── scripts/                  Reproducibility scripts (one per article protocol)
│   ├── 00_lamb_modes_and_tmatrix.py
│   ├── 01_multipolar_convergence.py
│   ├── 01_multipolar_convergence_plot.py
│   ├── 02_qca_validation.py
│   ├── 02_qca_transmission.py
│   ├── 02_qca_plot.py
│   ├── 03_qca_stochastic.py
│   ├── 03_qca_stochastic_analysis.py
│   ├── 03_qca_stochastic_legacy.py
│   ├── 04_cell_membrane_static.py
│   ├── 04_cell_membrane_plot.py
│   ├── 05_willis_activation.py
│   ├── 06_resonance_sweep.py
│   ├── 07_hadamard_fd.py
│   ├── 08_bloch_coupled.py
│   └── 08_bloch_coupled_postprocess.py
│
├── freefem/                  FreeFEM++ scripts
│   ├── cell_membrane.edp
│   └── cell_membrane_meshconv.edp
│
├── tests/                    Smoke tests
│   ├── test_argyris.py
│   └── test_freefem.edp
│
├── data/                     Pre-computed outputs (CSV / DAT / LOG)
│   └── README.md             Data dictionary
│
└── docs/
    └── protocol_mapping.md   Article §6.X ↔ scripts/0X_*.py
```

## Installation

```bash
pip install -r requirements.txt
```

The Argyris-based Willis solver of Section 6.9 additionally requires
[scikit-fem](https://github.com/kinnala/scikit-fem); the static cell problem
of Section 6.5 can also be run from FreeFEM++ via `freefem/cell_membrane.edp`.

## Reproducing the article figures

Each script in `scripts/` corresponds to one numerical protocol of Section 6 of
the paper. Run from the `code/` root:

```bash
python -m scripts.01_multipolar_convergence    # Sec 6.2  ↔ Prop 2.4
python -m scripts.02_qca_validation            # Sec 6.3  ↔ Foldy dilute
python -m scripts.03_qca_stochastic            # Sec 6.4  ↔ Prop 3.1 rate
python -m scripts.04_cell_membrane_static      # Sec 6.5  ↔ Th 3.2 / C^eff
python -m scripts.05_willis_activation         # Sec 6.6  ↔ Th 3.3(i) / σ_h
python -m scripts.06_resonance_sweep           # Sec 6.7  ↔ Th 3.3(iii) Lorentzian
python -m scripts.07_hadamard_fd               # Sec 6.8  ↔ Th 5.2 FD verification
python -m scripts.08_bloch_coupled             # Sec 6.9  ↔ Argyris+P2 coupled cell
```

The full mapping article ↔ script ↔ output ↔ figure is in
[`docs/protocol_mapping.md`](docs/protocol_mapping.md).

## Reproducibility

Each script is self-contained and reads its inputs and writes its outputs from
and to `data/`. Mesh convergence and ensemble size studies are performed at
finer resolutions than reported in the paper; numerical thresholds (relative
tolerances, mesh sizes, ensemble sizes) are documented in the head of each
script. Final outputs are tabulated in plain text (CSV / DAT / LOG) and are
read back by the plotting scripts (`*_plot.py`) producing the figures of the
article.

## License

MIT (see [LICENSE](LICENSE)).

## Citation

```bibtex
@article{JAOUHARI2026Willis,
  author  = {JAOUHARI Mounir,  and Anas, Rachid},
  title   = {Tensorial Effective Medium Theory for Thin Elastic Plates with
             Resonant Inclusions: Emergence of a Willis Coupling Tensor and  Topology-Driven Design},
  journal = {(submitted)},
  year    = {2026}
}
```
