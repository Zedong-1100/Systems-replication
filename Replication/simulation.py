import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from network import normalize, net

N, T, B, H, K, SEED = 24, 550, 220, 10, 3, 20261022

SCENARIOS = ["factor_only", "lagged_only", "factor_plus_lagged"]

LABELS = dict(zip(SCENARIOS, "ABC"))


def sim_fit(Y):
    xx = np.column_stack((np.ones(len(Y) - 1), Y[:-1]))
    yy = Y[1:]
    b = np.linalg.lstsq(xx, yy, rcond=None)[0]
    A = b[1:].T
    e = yy - xx @ b
    S = e.T @ e / (len(e) - xx.shape[1])
    return A, S


def sim_gfevd(A, S, horizon=H):
    phi = np.eye(N)
    numer = np.zeros((N, N))
    den = np.zeros(N)
    for h in range(horizon):
        p = phi @ S
        numer += p * p / np.diag(S)[None, :]
        den += np.sum(p * phi, axis=1)
        if h == 0:
            g1 = numer / den[:, None]
        phi = phi @ A
    return g1, numer / den[:, None]


def draw(dgp, rng):
    perm = rng.permutation(N)
    sources = perm[:3]
    receivers = perm[3:]
    A = np.eye(N) * rng.uniform(0.07, 0.2, size=N)
    target_sets = []
    if dgp in ("lagged_only", "factor_plus_lagged"):
        for s in sources:
            targets = rng.choice(receivers, size=8, replace=False)
            A[targets, s] = rng.uniform(0.14, 0.23, size=8)
            target_sets.append(targets)
    fload = np.zeros(N)
    if dgp in ("factor_only", "factor_plus_lagged"):
        fload = rng.uniform(0.55, 1.35, size=N) * rng.choice(
            [-1, 1], size=N, p=[0.12, 0.88]
        )
    noise_sd = rng.uniform(0.65, 1.0, size=N)
    Y = np.zeros((T + 120, N))
    eps = rng.normal(size=(T + 120, N))
    factor = rng.normal(size=(T + 120, 1))
    U = eps * noise_sd + factor * fload[None, :]
    for t in range(1, T + 120):
        Y[t] = A @ Y[t - 1] + U[t]
    Sigma = np.outer(fload, fload) + np.diag(noise_sd**2)
    parameters = dict(
        perm=perm,
        A=A,
        S=Sigma,
        loading=fload,
        sd=noise_sd,
        eps=eps,
        factor=factor,
        targets=target_sets,
    )
    return Y[120:], sources, parameters


def simulate(A, sd, load, eps, factor, T):
    U = eps[: T + 120] * sd + factor[: T + 120] * load[None, :]
    Y = np.zeros_like(U)
    for t in range(1, T + 120):
        Y[t] = A @ Y[t - 1] + U[t]
    return Y[120:]


def monte_carlo(replications):
    basic = []
    rng = np.random.default_rng(SEED)
    draws = []
    # keep one random stream across A, B and C
    for dgp in SCENARIOS:
        for rep in range(B):
            Y, drawn, saved = draw(dgp, rng)
            A = saved["A"]
            assert np.max(np.abs(np.linalg.eigvals(A))) < 1
            ahat, shat = sim_fit(Y)
            if not (np.isfinite(ahat).all() and np.isfinite(shat).all()):
                raise ArithmeticError(f"nonfinite fitted VAR: {dgp}, rep={rep}")
            gf1, gf10 = sim_gfevd(ahat, shat)
            if dgp == "factor_only":
                known = set()
            else:
                known = set(drawn)
            entry = dict(
                dgp=dgp,
                rep=rep,
                true_n_sources=len(known),
                G_relative_increment=float(
                    np.linalg.norm(gf10 - gf1) / np.linalg.norm(gf10)
                ),
            )
            for nm in ["row", "scalar"]:
                one = net(normalize(gf1, nm))
                ten = net(normalize(gf10, nm))
                if np.std(one) >= 1e-12 and np.std(ten) >= 1e-12:
                    corr = spearmanr(one, ten)
                    rank_rho = float(corr.statistic)
                else:
                    rank_rho = np.nan
                if known:
                    top = set(np.argsort(-ten)[:3])
                    recall = float(len(top & known) / 3)
                else:
                    recall = np.nan
                entry.update(
                    {
                        f"{nm}_h1_mean_abs_NET": float(np.mean(np.abs(one))),
                        f"{nm}_h10_mean_abs_NET": float(np.mean(np.abs(ten))),
                        f"{nm}_horizon_NET_abs_delta": float(
                            np.mean(np.abs(ten - one))
                        ),
                        f"{nm}_horizon_NET_relL1": float(
                            np.sum(np.abs(ten - one)) / max(np.sum(np.abs(ten)), 1e-12)
                        ),
                        f"{nm}_h1_h10_rankrho": rank_rho,
                        f"{nm}_true_source_top3_recall": recall,
                    }
                )
            basic.append(entry)
            saved["fitted"] = ahat, shat, gf1, gf10
            draws.append(saved)
    df = pd.DataFrame(basic)
    df.to_csv(replications / "simulation.csv", index=False)
    metrics = []
    for c in df:
        if c not in ["dgp", "rep", "true_n_sources"]:
            metrics.append(c)
    rows = []
    for d, g in df.groupby("dgp", sort=False):
        for metric in metrics:
            x = g[metric]
            x = x.dropna()
            if len(x):
                result = {
                    "dgp": d,
                    "metric": metric,
                    "n": len(x),
                    "mean": x.mean(),
                    "median": x.median(),
                    "p05": x.quantile(0.05),
                    "p95": x.quantile(0.95),
                }
                rows.append(result)
    return pd.DataFrame(rows), draws
