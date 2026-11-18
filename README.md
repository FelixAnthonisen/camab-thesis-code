# Experiments for the master's thesis on CAMAB mean shifts

Code for the simulated experiments in the Experiments chapter of the thesis. Every table and figure in that chapter is produced by one of the three scripts below.

## Setup

Python 3.13.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

Run the scripts from the repository root. Tables are printed to stdout and figures are written to `plots/`.

| Script                    | Thesis section                        | Produces                                                                                                                                         |
| ------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `threshold_comparison.py` | Pruning at the Mean-Shift Threshold   | Table of 2 e_mean(α) against ε(α), 20 000 instances per row                                                                                      |
| `family_profiles.py`      | Cost of Adaptivity, Instance Families | `plots/family_shift_profiles.png`, mean shifts of one instance per family                                                                        |
| `cost_comparison.py`      | Cost of Adaptivity, Results           | Cost table at γ = 1, net-saving table at γ = 0.2, the interval checks, `plots/adaptive_gain_gamma_1.png` and `plots/adaptive_gain_gamma_0.2.png` |
