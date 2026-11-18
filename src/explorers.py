import math
import numpy as np
from src.camab import CAMAB


class _BlockSampler:
    def __init__(self, dist, block: int = 4096):
        self.dist = dist
        self.block = block
        self._buf = None
        self._pos = 0

    def draw_sum(self, times: int) -> float:
        total = 0.0
        while times > 0:
            if self._buf is None or self._pos >= self._buf.size:
                self._buf = np.asarray(self.dist.rvs(size=max(self.block, times)))
                self._pos = 0
            take = min(times, self._buf.size - self._pos)
            total += self._buf[self._pos : self._pos + take].sum()
            self._pos += take
            times -= take
        return float(total)

    def draw_one(self) -> float:
        if self._buf is None or self._pos >= self._buf.size:
            self._buf = np.asarray(self.dist.rvs(size=self.block))
            self._pos = 0
        x = self._buf[self._pos]
        self._pos += 1
        return float(x)


class OptimisticExplorer:
    def __init__(
        self, camab: CAMAB, delta: float = 0.1, gamma: float = 1, sigma: float = 1
    ):
        self.camab = camab
        self.delta = delta
        self.gamma = gamma
        self.sigma = sigma
        self.omega = np.asarray(camab.omega, dtype=int)
        K, J = camab.base.n_arms, camab.abstract.n_arms
        self.n_base = np.zeros(K)
        self.s_base = np.zeros(K)
        self.n_abstract = np.zeros(J)
        self.s_abstract = np.zeros(J)
        self.cum_cost = 0.0
        self.members = [np.flatnonzero(self.omega == j) for j in range(J)]
        self._outside = [self.omega != j for j in range(J)]
        self._log_scale = 6 * K / delta

        self._base_sampler = [_BlockSampler(d) for d in camab.base.distributions]
        self._abstract_sampler = [
            _BlockSampler(d) for d in camab.abstract.distributions
        ]

    def _pull_base(self, a: int):
        self.s_base[a] += self._base_sampler[a].draw_one()
        self.n_base[a] += 1
        self.cum_cost += 1

    def _pull_abstract(self, ap: int, times: int):
        self.s_abstract[ap] += self._abstract_sampler[ap].draw_sum(times)
        self.n_abstract[ap] += times
        self.cum_cost += self.gamma * times

    def _width(self, n, n_abs):
        return self.sigma * math.sqrt(
            2 * (1 / n + 1 / n_abs) * math.log(self._log_scale * n * n * n_abs * n_abs)
        )

    def _interval(self, a: int, ap: int):
        n, n_abs = self.n_base[a], self.n_abstract[ap]
        d = abs(self.s_base[a] / n - self.s_abstract[ap] / n_abs)
        w = self._width(n, n_abs)
        return max(0.0, d - w), d + w

    def _bounds(self):
        n, n_abs = self.n_base, self.n_abstract[self.omega]
        d = np.abs(self.s_base / n - self.s_abstract[self.omega] / n_abs)
        w = self.sigma * np.sqrt(
            2 * (1 / n + 1 / n_abs) * np.log(self._log_scale * n**2 * n_abs**2)
        )
        return np.maximum(0.0, d - w), d + w

    def simulate(self, epsilon: float, max_steps: int = 10**9) -> dict:
        for a in range(self.camab.base.n_arms):
            self._pull_base(a)
        for ap in range(self.camab.abstract.n_arms):
            n = self.n_base[self.members[ap]]
            self._pull_abstract(ap, math.ceil(math.sqrt((n**2).sum() / self.gamma)))
        lower, upper = self._bounds()

        steps = 0
        while True:
            ap = int(self.omega[np.argmax(upper)])
            members = self.members[ap]
            u_out = upper[self._outside[ap]].max(initial=-np.inf)
            l_out = lower[self._outside[ap]].max(initial=-np.inf)
            lo, up = list(lower[members]), list(upper[members])
            while True:
                k = max(range(len(members)), key=up.__getitem__)
                if up[k] < u_out:
                    break
                top_l = max(l_out, max(lo))
                if up[k] - top_l <= epsilon:
                    return self._result(up[k], top_l)
                if steps >= max_steps:
                    raise RuntimeError(
                        f"no convergence to {epsilon} within {max_steps} steps"
                    )

                self._pull_base(int(members[k]))
                target = math.ceil(
                    math.sqrt((self.n_base[members] ** 2).sum() / self.gamma)
                )
                topped_up = target > self.n_abstract[ap]
                if topped_up:
                    self._pull_abstract(ap, int(target - self.n_abstract[ap]))
                for i in range(len(members)) if topped_up else (k,):
                    lo[i], up[i] = self._interval(int(members[i]), ap)
                steps += 1
            lower[members], upper[members] = lo, up

    def _result(self, m_hat, m_low) -> dict:
        return {
            "cost": float(self.cum_cost),
            "estimate": float(m_hat),
            "lower": float(m_low),
        }
