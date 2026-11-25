import csv
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict

NODES = [
    "A",
    "B",
    "CA",
    "CB",
    "CC",
    "CD",
    "CE",
    "CF",
    "CG",
    "CH",
    "CI",
    "CJ",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "OTH",
    "J",
]

WINDOW, STEP, H = 250, 5, 10
BOOT, BLOCK = 2000, 10


def ci_block(series, seed, block=BLOCK, reps=BOOT):
    values = np.asarray(series, dtype=float)
    rng = np.random.default_rng(seed)
    N = len(values)
    means = np.empty(reps)
    for b in range(reps):
        starts = rng.integers(0, N, int(np.ceil(N / block)))
        ix = np.concatenate([np.arange(s, s + block) % N for s in starts])[:N]
        means[b] = values[ix].mean()
    return np.quantile(means, [0.025, 0.975])


def read_data(sector_path, market_path):
    with Path(sector_path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        date_col = "Trddt" if "Trddt" in reader.fieldnames else "date"
        entries = []
        for r in reader:
            values = []
            for s in NODES:
                values.append(float(r[s]))
            entries.append((r[date_col][:10], values))
    with Path(market_path).open(encoding="utf-8-sig", newline="") as f:
        market = {}
        for r in csv.DictReader(f):
            all_a = float(r["ret_all_a"])
            hs300 = float(r["ret_hs300"])
            market[r["date"]] = all_a, hs300
    dates, R, F = [], [], []
    for d, r in entries:
        if d in market:
            dates.append(d)
            R.append(r)
            F.append(market[d])
    R, F = np.asarray(R, float), np.asarray(F, float)
    if (
        len(dates) < WINDOW + 100
        or not np.isfinite(R).all()
        or not np.isfinite(F).all()
    ):
        raise ValueError("invalid return panel")
    return dates, R, F


def residualize(block, f):
    X = np.column_stack([np.ones(len(f)), f])
    return block - X @ np.linalg.lstsq(X, block, rcond=None)[0]


def normalized_metrics(G, scheme):
    if scheme == "row":
        M = G / G.sum(axis=1)[:, None]
    elif scheme == "scalar":
        M = G / G.sum(axis=1).max()
    else:
        raise ValueError(scheme)
    diag = np.diag(M)
    # rows receive; columns send
    FROM = (M.sum(axis=1) - diag) * 100
    TO = (M.sum(axis=0) - diag) * 100
    NET = TO - FROM
    TCI = float(np.mean(FROM))
    return TCI, NET


def raw_gfevd(block, horizon=H, both=False):
    n = block.shape[1]
    X = np.column_stack([np.ones(len(block) - 1), block[:-1]])
    Y = block[1:]
    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    A = B[1:].T
    E = Y - X @ B
    sigma = E.T @ E / (len(Y) - X.shape[1])
    if np.min(np.diag(sigma)) <= 0:
        raise ArithmeticError("invalid VAR/GFEVD")
    if max(abs(np.linalg.eigvals(A))) >= 1:
        raise ArithmeticError("invalid VAR/GFEVD")
    phi = np.eye(n)
    numer = np.zeros((n, n))
    denom = np.zeros(n)
    for h in range(horizon):
        ps = phi @ sigma
        numer += ps**2 / np.diag(sigma)[None, :]
        denom += np.sum(ps * phi, axis=1)
        if both and h == 0:
            g1 = numer / denom[:, None]
        phi = phi @ A
    if np.min(denom) <= 0:
        raise ArithmeticError("invalid VAR/GFEVD")
    G = numer / denom[:, None]
    if not np.isfinite(G).all() or np.min(G.sum(axis=1)) <= 0:
        raise ArithmeticError("invalid VAR/GFEVD")
    if both:
        return g1, G
    return G


def normalize(G, kind):
    if kind == "row":
        fenmu = G.sum(1, keepdims=True)
    else:
        fenmu = G.sum(1).max()
    return G / fenmu


def net(M):
    return 100 * (M.sum(axis=0) - M.sum(axis=1))


def estimate(base):
    dates, returns, factors = read_data(base / "returns.csv", base / "index.csv")
    sector_rows = []
    systems = {}
    # rolling 250-day VAR; keep H=1 and H=10 GFEVD
    for end in range(WINDOW - 1, len(dates), STEP):
        d = dates[end]
        block = returns[end - WINDOW + 1 : end + 1]
        fa = factors[end - WINDOW + 1 : end + 1, 0]
        fh = factors[end - WINDOW + 1 : end + 1, 1]
        Gs = {}
        Gs["raw"] = raw_gfevd(block, both=True)
        all_a = residualize(block, fa)
        Gs["allA"] = raw_gfevd(all_a, both=True)
        hs300 = residualize(block, fh)
        Gs["hs300"] = raw_gfevd(hs300, both=True)
        systems[d] = Gs
        c = {}
        for f, matrices in Gs.items():
            G = matrices[1]
            for scheme in ["row", "scalar"]:
                name = f + "_" + scheme
                c[name] = normalized_metrics(G, scheme)
        for k, name in enumerate(NODES):
            rr = {"date": d, "sector": name}
            for col, (_, net) in c.items():
                rr["NET_" + col] = float(net[k])
            sector_rows.append(rr)
        centered = block - block.mean(0)
        total = np.sum(centered * centered, axis=0)
        Gs["variance_removed"] = {
            "raw": 0.0,
            "allA": float(1 - np.mean(np.sum(all_a * all_a, axis=0) / total)),
            "hs300": float(1 - np.mean(np.sum(hs300 * hs300, axis=0) / total)),
        }
    with (base / "net.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sector_rows[0]))
        writer.writeheader()
        writer.writerows(sector_rows)
    return systems


def horizons(base, systems):
    specs = [
        "raw_row",
        "raw_scalar",
        "allA_row",
        "allA_scalar",
        "hs300_row",
        "hs300_scalar",
    ]

    def rank(x):
        return pd.Series(x).rank(method="average").to_numpy(float)

    dynamic = []
    for d in systems:
        for factor in ["raw", "allA", "hs300"]:
            G1, G10 = systems[d][factor]
            for mode in ["row", "scalar"]:
                n1 = net(normalize(G1, mode))
                n10 = net(normalize(G10, mode))
                name = factor + "_" + mode
                delta = n10 - n1
                if np.std(n1) > 1e-10:
                    if np.std(n10) > 1e-10:
                        rank1 = rank(n1)
                        rank10 = rank(n10)
                        corr = np.corrcoef(rank1, rank10)
                        rank_rho = float(corr[0, 1])
                    else:
                        rank_rho = np.nan
                    sign_change = int(np.sum(np.sign(n1) != np.sign(n10)))
                    top1 = set(np.argsort(-n1)[:5])
                    top10 = set(np.argsort(-n10)[:5])
                    top_overlap = len(top1 & top10)
                else:
                    rank_rho = np.nan
                    sign_change = np.nan
                    top_overlap = np.nan
                dynamic.append(
                    {
                        "date": d,
                        "year": int(d[:4]),
                        "spec": name,
                        "H1_NET_L1_mean": float(np.mean(np.abs(n1))),
                        "H10_NET_L1_mean": float(np.mean(np.abs(n10))),
                        "NET_change_mean_abs": float(np.mean(np.abs(delta))),
                        "NET_change_rel_L1": float(
                            np.sum(np.abs(delta)) / max(np.sum(np.abs(n10)), 1e-12)
                        ),
                        "NET_change_rel_L2": float(
                            np.linalg.norm(delta) / max(np.linalg.norm(n10), 1e-12)
                        ),
                        "NET_H1_H10_rankrho": rank_rho,
                        "NET_sign_change_H1_H10": sign_change,
                        "H1_H10_top5_overlap": top_overlap,
                        "G_increment_relative_Fnorm": float(
                            np.linalg.norm(G10 - G1) / np.linalg.norm(G10)
                        ),
                        "G10_asymmetry_Fnorm": float(
                            np.linalg.norm(G10 - G10.T) / np.linalg.norm(G10)
                        ),
                    }
                )
    dyndict = defaultdict(list)
    for r in dynamic:
        dyndict[r["spec"]].append(r)
    fields = [
        "H1_NET_L1_mean",
        "H10_NET_L1_mean",
        "NET_change_mean_abs",
        "NET_change_rel_L1",
        "NET_change_rel_L2",
        "NET_H1_H10_rankrho",
        "NET_sign_change_H1_H10",
        "H1_H10_top5_overlap",
        "G_increment_relative_Fnorm",
        "G10_asymmetry_Fnorm",
    ]
    summary = []
    for s in specs:
        earlier = []
        for r in dyndict[s]:
            if r["year"] <= 2023:
                earlier.append(r)
        for era, rr in [("full", dyndict[s]), ("pre2024", earlier)]:
            row = {"spec": s, "period": era, "n": len(rr)}
            for key in fields:
                vals = []
                for x in rr:
                    vals.append(float(x[key]))
                vals = np.array(vals)
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    row[key] = float(vals.mean())
                else:
                    row[key] = np.nan
            if s.endswith("_row"):
                vals = []
                for x in rr:
                    vals.append(x["NET_change_rel_L1"])
                vals = np.array(vals)
                (
                    row["NET_change_rel_L1_ci95_low"],
                    row["NET_change_rel_L1_ci95_high"],
                ) = ci_block(vals, seed=4401, block=50, reps=900)
            summary.append(row)
    with (base / "horizon_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)


def grid(base, systems):
    days, R, F = read_data(base / "returns.csv", base / "index.csv")
    tci = []
    for end in range(WINDOW - 1, len(days), STEP):
        for n in [24, 23]:
            x = R[end - WINDOW + 1 : end + 1, :n]
            f = F[end - WINDOW + 1 : end + 1]
            data1 = x
            if n == 23:
                data2 = residualize(x, f[:, 0])
                data3 = residualize(x, f[:, 1])
            else:
                data2 = data3 = None
            for rep, b in [("raw", data1), ("allA", data2), ("hs300", data3)]:
                ts = {}
                if n == 24:
                    g1, g10 = systems[days[end]][rep]
                else:
                    g1, g10 = raw_gfevd(b, both=True)
                for h, G in [(1, g1), (10, g10)]:
                    for norm in ["row", "scalar"]:
                        if norm == "row":
                            fenmu = G.sum(1)[:, None]
                        else:
                            fenmu = G.sum(1).max()
                        M = G / fenmu
                        diag = M.diagonal()
                        fr = 100 * (M.sum(1) - diag)
                        ts[f"{norm}_h{h}"] = fr.mean()
                row = {"date": days[end], "N": n, "representation": rep}
                row.update(ts)
                tci.append(row)
    pd.DataFrame(tci).to_csv(base / "tci.csv", index=False)
