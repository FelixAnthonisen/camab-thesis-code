import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

from src.camab import (
    make_coarsening_camab,
    make_biased_camab,
    make_misaligned_camab,
    make_common_offset_camab,
    true_mean_shifts,
)
from src.explorers import OptimisticExplorer

DELTA = 0.1
SIGMA = 0.25
N_CLUSTERS = 20
CLUSTER_SIZE = 5
N_SEEDS = 10
EPS_BAR = 0.025
EPS_GRID = (0.10, 0.07, 0.05, 0.035, 0.025, 0.018, 0.012)

K = N_CLUSTERS * CLUSTER_SIZE

FAMILIES = [
    ("faithful coarsening", make_coarsening_camab),
    ("biased abstract", make_biased_camab),
    ("misaligned ω", make_misaligned_camab),
    ("common offset", make_common_offset_camab),
]


def abstract_ratio(gamma):
    return int(np.ceil(np.sqrt(CLUSTER_SIZE / gamma)))


def budget(epsilon, log_factor, gamma):
    return 4 * SIGMA**2 * log_factor * (1 + 1 / abstract_ratio(gamma)) / epsilon**2


def baseline_cost(epsilon, gamma):
    per_round = K + gamma * N_CLUSTERS * abstract_ratio(gamma)
    return np.ceil(budget(epsilon, 2 * np.log(2 * K / DELTA), gamma)) * per_round


def noise_seed(family_idx, eps_idx, s):
    return 1000 + 10_000 * family_idx + 100 * eps_idx + s


def run(camab, epsilon, seed, gamma):
    np.random.seed(seed)
    return OptimisticExplorer(camab, delta=DELTA, gamma=gamma, sigma=SIGMA).simulate(
        epsilon
    )


def run_experiment(gamma):

    print(f"\ngamma = {gamma:g}")
    print(
        "    %-22s %6s %7s %7s %11s %11s %7s %5s %6s"
        % (
            "family",
            "eps",
            "spread",
            "S/eps",
            "baseline",
            "optimistic",
            "net",
            "miss",
            "over",
        ),
        flush=True,
    )

    bar = {"baseline": [], "optimistic": [], "net": []}
    scatter = {}
    n_runs, miss_above, miss_below, overshoots = 0, 0, 0, []

    for f_idx, (name, factory) in enumerate(FAMILIES):
        scatter[name] = ([], [])
        for e_idx, epsilon in enumerate(EPS_GRID):
            cost_base = baseline_cost(epsilon, gamma)
            costs_opt, spreads, row_over = [], [], []
            row_miss = 0
            for s in range(N_SEEDS):
                camab = factory(N_CLUSTERS, CLUSTER_SIZE, sigma=SIGMA, seed=s)
                shifts = true_mean_shifts(camab)
                spread = shifts.max() - shifts.min()

                adapt = run(camab, epsilon, noise_seed(f_idx, e_idx, s), gamma)
                costs_opt.append(adapt["cost"])
                spreads.append(spread)
                scatter[name][0].append(spread / epsilon)
                scatter[name][1].append(cost_base / adapt["cost"])

                m_true = float(shifts.max())
                above = m_true > adapt["estimate"]
                below = m_true < adapt["lower"]
                miss_above += above
                miss_below += below
                row_miss += above or below
                row_over.append((adapt["estimate"] - m_true) / epsilon)
                n_runs += 1
            overshoots.extend(row_over)

            med_opt = np.median(costs_opt)
            med_spread = float(np.median(spreads))
            print(
                "    %-22s %6.3f %7.3f %7.2f %11.0f %11.0f %7.2f %5d %6.2f"
                % (
                    name,
                    epsilon,
                    med_spread,
                    med_spread / epsilon,
                    cost_base,
                    med_opt,
                    cost_base / med_opt,
                    row_miss,
                    np.median(row_over),
                ),
                flush=True,
            )
            if epsilon == EPS_BAR:
                bar["baseline"].append(cost_base)
                bar["optimistic"].append(med_opt)
                bar["net"].append(cost_base / med_opt)

    n_miss = miss_above + miss_below
    upper = beta.ppf(0.95, n_miss + 1, n_runs - n_miss) if n_miss < n_runs else 1.0
    print(f"\nInterval checks over {n_runs} runs:")
    print(f"    M above M_hat:                {miss_above}")
    print(f"    M below the largest lower end: {miss_below}")
    print(
        f"    one-sided 95% upper bound on the miss rate: {upper:.4f}"
        f"  (delta = {DELTA})"
    )
    print(
        f"    overshoot (M_hat - M) / eps: median {np.median(overshoots):.2f}, "
        f"range [{min(overshoots):.2f}, {max(overshoots):.2f}]",
        flush=True,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(FAMILIES))
    width = 0.36
    ax1.bar(
        x - width / 2,
        np.array(bar["baseline"]) / 1e3,
        width,
        label="non-adaptive baseline",
        color="0.65",
    )
    ax1.bar(
        x + width / 2,
        np.array(bar["optimistic"]) / 1e3,
        width,
        label="Algorithm 2",
        color="tab:blue",
    )
    for i, r in enumerate(bar["net"]):
        ax1.annotate(
            f"{r:.2f}×",
            (x[i], max(bar["baseline"][i], bar["optimistic"][i]) / 1e3),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=9,
            color="dimgray",
        )
    ax1.set_ylim(0, max(max(bar["baseline"]), max(bar["optimistic"])) / 1e3 * 1.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace(" ", "\n", 1) for n, _ in FAMILIES], fontsize=8)
    ax1.set_ylabel(f"cost to identify max shift to ε={EPS_BAR}  (thousands)")
    ax1.legend(loc="upper center", ncol=1, fontsize=9, framealpha=0.9)
    ax1.grid(True, axis="y", alpha=0.3)

    for name, (xs, ys) in scatter.items():
        ax2.plot(xs, ys, "o", ms=5, alpha=0.7, label=name)
    ax2.axvline(
        1.0, color="crimson", ls="-", lw=1.2, label="flat-case threshold  S = ε"
    )
    ax2.axhline(1.0, color="gray", ls="--", lw=1, label="Algorithm 2 breaks even")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("shift spread / ε      (S / ε)")
    ax2.set_ylabel("cost(non-adaptive) / cost(Algorithm 2)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    path = f"plots/adaptive_gain_gamma_{gamma:g}.png"
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    run_experiment(1)
    run_experiment(0.2)
