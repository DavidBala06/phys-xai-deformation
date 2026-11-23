# Physically Grounded Explanations for Soft Tissue Deformation Tracking

Prototype and exploratory codebase for my BSc thesis (*Incorporating physical models into explainable AI solutions for real-time medical image analysis*), targeting real-time image-guided / endoscopic soft tissue deformation estimation ($\le 33\text{ ms}$ / $30\text{ FPS}$ budget).

## Current Status (Phase F0 / Initial Baseline Setup)
- Loaded the reference colon geometry (`full_model.obj` from C3VD, `1,269,488` vertices) and subsampled to `1,024` nodes with a `4-NN` elastic edge graph (`2,569` edges).
- Implemented a baseline `ColonDeformationPINN` trained with a combined supervised displacement loss and an edge-length elastic strain regularizer (`ColonElasticLoss`).
- Verified convergence on a synthetic smooth bending displacement (`u_y = 0.02 * z`), reducing mean Target Registration Error (TRE) from `1.3983 mm` (step 1) to `0.0121 mm` (step 80) on CPU.

> **Note:** This initial run uses a synthetic displacement field and placeholder feature input to validate the 3D mesh pipeline and elastic loss before connecting C3VD-DEFCOL video/depth sequences and evaluating intrinsic residual-based explanations vs. post-hoc XAI.

## Repository Layout
```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── test_pinn.py          # Standalone PINN training + elastic loss verification
├── src/
│   └── run_experiment.py # Mesh subsampling, elastic PINN training, and plot generation
└── results/              # Generated plots (gitignored)
    ├── training_curves.png
    └── deformation_comparison.png
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt
```

## Usage

Run the baseline experiment (plots are written to `results/`):

```bash
python src/run_experiment.py
```

Run the standalone PINN verification script (requires `full_model.obj` at the repo root):

```bash
python test_pinn.py
```

`full_model.obj` (~200 MB, C3VD reference colon mesh) is intentionally **not** tracked in Git (see `.gitignore`). Place it at the repo root or under `data/` before running.

## Roadmap
- Replace the synthetic visual feature vector with an image/depth-map encoder.
- Download and integrate the C3VD-DEFCOL video/depth sequences.
- Compare intrinsic residual-based explanations vs. post-hoc XAI methods.


