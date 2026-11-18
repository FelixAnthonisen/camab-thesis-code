import os
import numpy as np
import matplotlib.pyplot as plt

from src.camab import (
    make_coarsening_camab,
    make_biased_camab,
    make_misaligned_camab,
    make_common_offset_camab,
    true_mean_shifts,
)

SIGMA = 0.25
N_CLUSTERS = 20
CLUSTER_SIZE = 5
EPS = 0.025
SEED = 0

FAMILIES = [
    ("faithful coarsening", make_coarsening_camab),
    ("biased abstract", make_biased_camab),
    ("misaligned ω", make_misaligned_camab),
    ("common offset", make_common_offset_camab),
]


def main():
    plt.rcParams.update({"font.size": 10})
    jitter = np.random.default_rng(1)
    fig, axes = plt.subplots(len(FAMILIES), 1, figsize=(6.2, 3.0), sharex=True)
    for ax, (name, factory) in zip(axes, FAMILIES):
        camab = factory(N_CLUSTERS, CLUSTER_SIZE, sigma=SIGMA, seed=SEED)
        shifts = true_mean_shifts(camab)
        m = shifts.max()
        spread = m - shifts.min()
        inside = int(np.sum(shifts >= m - EPS))
        ax.axvspan(m - EPS, m, color="tab:orange", alpha=0.25, lw=0)
        ax.axvline(m, color="tab:orange", lw=1.2)
        ax.plot(
            shifts,
            jitter.uniform(-1, 1, shifts.size),
            "o",
            ms=2.6,
            color="tab:blue",
            alpha=0.7,
        )
        ax.set_ylim(-1.7, 1.7)
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
        ax.text(
            1.01,
            0.5,
            f"S/ε = {spread / EPS:.1f}\n{inside} of {shifts.size} in window",
            transform=ax.transAxes,
            va="center",
            fontsize=9,
        )
        ax.grid(True, axis="x", alpha=0.3)
    axes[-1].set_xlabel("mean shift")
    axes[-1].set_xlim(left=0)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    path = "plots/family_shift_profiles.png"
    plt.savefig(path, dpi=200)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
