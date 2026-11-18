import numpy as np

from src.camab import coarsening_means, mean_shifts, pruning_at


def w2_normal(m1, s1, m2, s2):
    return np.sqrt((m1 - m2) ** 2 + (s1 - s2) ** 2)


def threshold_trial(Kp, csize, sigma_w, lam, equal_sd, rng, sd_range=(0.5, 1.5)):
    K = Kp * csize
    mu, omega, mu_p = coarsening_means(Kp, csize, sigma_w, rng)
    if equal_sd:
        sd, sd_p = np.ones(K), np.ones(Kp)
    else:
        sd, sd_p = rng.uniform(*sd_range, K), rng.uniform(*sd_range, Kp)

    e_mean = mean_shifts(mu, omega, mu_p)
    s_alpha = max(w2_normal(mu[a], sd[a], lam * mu[a], lam * sd[a]) for a in range(K))
    e_alpha = max(
        w2_normal(lam * mu[a], lam * sd[a], mu_p[omega[a]], sd_p[omega[a]])
        for a in range(K)
    )
    eps_alpha = 2 * (s_alpha + e_alpha)
    twice = 2 * e_mean.max()

    tol = 1e-9
    prune_twice = pruning_at(omega, mu_p, twice, tol=tol)
    prune_eps_alpha = pruning_at(omega, mu_p, eps_alpha, tol=tol)
    return {
        "twice": twice,
        "eps_alpha": eps_alpha,
        "s": s_alpha,
        "e": e_alpha,
        "prune_twice": prune_twice,
        "prune_eps_alpha": prune_eps_alpha,
    }


def run_experiment(n_trials=20_000, Kp=20, csize=5, seed=1):
    print(
        "    %-9s %5s %5s %6s %6s  %9s  %7s %7s"
        % ("variances", "sw", "lam", "s(a)", "e(a)", "2e/eps(a)", "pr eps", "pr 2e")
    )
    rng = np.random.default_rng(seed)
    for equal_sd, sd_range, label in (
        (True, None, "equal"),
        (False, (0.5, 1.5), "hetero"),
        (False, (0.9, 1.1), "mild"),
    ):
        for sigma_w in (0.02, 0.05, 0.10):
            for lam in (1.0, 0.9):
                rows = [
                    threshold_trial(
                        Kp, csize, sigma_w, lam, equal_sd, rng, sd_range or (0.5, 1.5)
                    )
                    for _ in range(n_trials)
                ]
                f_tw = np.array([r["twice"] / r["eps_alpha"] for r in rows])
                print(
                    "    %-9s %5.2f %5.1f %6.3f %6.3f  %9.3f  %7.3f %7.3f"
                    % (
                        label,
                        sigma_w,
                        lam,
                        np.median([r["s"] for r in rows]),
                        np.median([r["e"] for r in rows]),
                        np.median(f_tw),
                        np.mean([r["prune_eps_alpha"] for r in rows]),
                        np.mean([r["prune_twice"] for r in rows]),
                    )
                )


if __name__ == "__main__":
    run_experiment()
