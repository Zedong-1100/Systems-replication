import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import combinations
from network import NODES, WINDOW, ci_block


def linear_ridge(train, test, feat, alpha=10):
    X = []
    for r in train:
        values = []
        for s in feat:
            values.append(r[s])
        X.append(values)
    X = np.array(X, float)
    Z = []
    for r in test:
        values = []
        for s in feat:
            values.append(r[s])
        Z.append(values)
    Z = np.array(Z, float)
    y = np.array([r["future_tail_breadth"] for r in train])
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.maximum(sd, 1e-09)
    X = (X - mu) / sd
    Z = (Z - mu) / sd
    xm = X.mean(0)
    ym = y.mean()
    X0 = X - xm
    beta = np.linalg.solve(X0.T @ X0 + alpha * np.eye(len(feat)), X0.T @ (y - ym))
    return np.clip(ym + (Z - xm) @ beta, 0, 1)


def ridge_predictions(tr, te, cols, alpha=10):
    X = tr[cols].to_numpy(float)
    Z = te[cols].to_numpy(float)
    y = tr["y"].to_numpy(float)
    mu = X.mean(0)
    sd = np.maximum(X.std(0), 1e-08)
    xs = (X - mu) / sd
    zs = (Z - mu) / sd
    co = np.linalg.solve(xs.T @ xs + alpha * np.eye(len(cols)), xs.T @ (y - y.mean()))
    return np.maximum(0, y.mean() + zs @ co)


def original(base):
    S = NODES
    FACTORS = ["raw", "allA", "hs300"]
    NORMS = ["row", "scalar"]
    SPECS = []
    for f in FACTORS:
        for n in NORMS:
            SPECS.append(f + "_" + n)
    NET = pd.read_csv(base / "net.csv")
    DATES = sorted(NET.date.unique())
    rets = pd.read_csv(base / "returns.csv")
    market = pd.read_csv(base / "index.csv")
    rets.Trddt = rets.Trddt.astype(str).str[:10]
    M = market.set_index("date")
    M = M.reindex(rets.Trddt)
    ret = rets[S].to_numpy(float)
    m = M["ret_all_a"].to_numpy(float)
    ds = rets.Trddt.to_list()
    inds = {}
    for i, d in enumerate(ds):
        inds[d] = i
    if not np.isfinite(ret).all():
        raise ValueError(f"nonfinite sector returns: {base / 'returns.csv'}")
    MATS = {}
    for spec in SPECS:
        df1 = NET.pivot(index="date", columns="sector", values="NET_" + spec)
        df2 = df1.reindex(index=DATES, columns=S)
        MATS[spec] = df2.to_numpy(float)
    for spec, values in MATS.items():
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite NET values: {base / 'net.csv'}, {spec}")

    out = []
    pairs = []
    for f in FACTORS:
        pairs.append(("normalization", f, f + "_row", f + "_scalar"))
    for n in NORMS:
        for f1, f2 in combinations(FACTORS, 2):
            pairs.append(("representation", n, f1 + "_" + n, f2 + "_" + n))
    ranks = {}
    for s in SPECS:
        temp = []
        for row in MATS[s]:
            rank = pd.Series(row).rank(method="average")
            temp.append(rank.to_numpy(float))
        ranks[s] = np.array(temp)
    rhos = {}
    for family, conditional, s1, s2 in pairs:
        rho = []
        for a, b in zip(ranks[s1], ranks[s2]):
            corr = np.corrcoef(a, b)
            rho.append(corr[0, 1])
        rhos[s1, s2] = np.array(rho)
    for k in [3, 5, 10]:
        top = {}
        for s in SPECS:
            top[s] = []
            for row in MATS[s]:
                top[s].append(set(np.argsort(-row)[:k]))
        for family, conditional, s1, s2 in pairs:
            overlap = []
            for a, b in zip(top[s1], top[s2]):
                overlap.append(len(a & b))
            overlap = np.array(overlap)
            rho = rhos[s1, s2]
            values = overlap[np.isfinite(overlap)]
            ci = ci_block(values, seed=4500 + k, block=50, reps=800)
            pairsrow = {
                "k": k,
                "family": family,
                "held_constant": conditional,
                "spec1": s1,
                "spec2": s2,
                "mean_topk_overlap": float(overlap.mean()),
                "random_independent_topk_expected": k * k / 24,
                "mean_rho": float(rho.mean()),
                "overlap_ci95_low": float(ci[0]),
                "overlap_ci95_high": float(ci[1]),
                "n_dates": len(DATES),
            }
            out.append(pairsrow)
    pd.DataFrame(out).to_csv(base / "overlap.csv", index=False)
    TOP = {}
    for s in SPECS:
        temp = []
        for row in MATS[s]:
            order = np.argsort(-row)
            temp.append(order[:5])
        TOP[s] = np.array(temp, int)
    CONS = []
    for ti in range(len(DATES)):
        top_lists = []
        for s in SPECS:
            top_lists.append(TOP[s][ti])
        vote = np.bincount(np.concatenate(top_lists), minlength=24)
        ranklist = []
        for s in SPECS:
            ranklist.append(ranks[s][ti])
        rankmean = np.mean(ranklist, axis=0)
        order = []
        for j in range(24):
            order.append((-vote[j], -rankmean[j], j))
        order.sort()
        idx = []
        for row in order[:5]:
            idx.append(row[2])
        CONS.append(idx)
    TOP["consensus"] = np.array(CONS)
    SPEC_MON = SPECS + ["consensus", "volatility_top5"]
    rec = []
    for ti, d in enumerate(DATES):
        end = inds[d]
        if end + 21 >= len(ds):
            continue
        window = ret[end - WINDOW + 1 : end + 1, :]
        marketwin = m[end - WINDOW + 1 : end + 1]
        X = np.column_stack([np.ones(WINDOW), marketwin])
        coef = np.linalg.lstsq(X, window, rcond=None)[0]
        sig = (window - X @ coef).std(axis=0, ddof=2).clip(0.0001)
        surprise = (ret[end + 1] - np.r_[1, m[end + 1]] @ coef) / sig
        negative = np.clip(-surprise, 0, 6)
        mean_shock = float(np.mean(negative))
        past_market = float(np.prod(1 + m[end - 19 : end + 1]) - 1)
        past_marketvol = float(np.std(m[end - 19 : end + 1], ddof=1))
        common = {
            "date": d,
            "year": int(d[:4]),
            "idx": end,
            "mean_negative_source_shock": mean_shock,
            "market_shock_nextday": float(m[end + 1]),
            "past_market20": past_market,
            "past_marketvol20": past_marketvol,
            "down_breadth_nextday": float(np.mean(negative > 0)),
        }
        future = {}
        for h in [5, 20]:
            rf = ret[end + 2 : end + 2 + h, :]
            exret = rf - np.column_stack([np.ones(h), m[end + 2 : end + 2 + h]]) @ coef
            future[h] = np.maximum(0, -np.sum(exret, axis=0))
        volc = np.argsort(-window[-20:].std(axis=0, ddof=1))[:5]
        for s in SPEC_MON:
            if s == "volatility_top5":
                top = volc
            else:
                top = TOP[s][ti]
            other = []
            selected = set(top)
            for j in range(24):
                if j not in selected:
                    other.append(j)
            other = np.array(other)
            top_shock = float(np.mean(negative[top]))
            non_shock = float(np.mean(negative[other]))
            record = common.copy()
            record["spec"] = s
            record["top_negative_shock"] = top_shock
            record["nontop_negative_shock"] = non_shock
            record["top_minus_nontop_shock"] = top_shock - non_shock
            for h in [5, 20]:
                record[f"other_industry_future_{h}d_loss"] = float(
                    future[h][other].mean()
                )
            # keep training labels before 2019; purge the crossing windows
            if ds[end + 21] <= "2018-12-31":
                record["split"] = "train"
            elif d >= "2019-01-01":
                record["split"] = "test"
            else:
                record["split"] = "purge"
            rec.append(record)
    recdf = pd.DataFrame(rec)
    CTL = [
        "past_market20",
        "past_marketvol20",
        "market_shock_nextday",
        "mean_negative_source_shock",
    ]
    NEW = "top_minus_nontop_shock"

    result = []
    group_models = {}
    for s in SPEC_MON:
        df1 = recdf[recdf.spec == s]
        allr = df1[df1.split != "purge"]
        train = allr[allr.split == "train"]
        test = allr[allr.split == "test"]
        for h in [5, 20]:
            ycol = f"other_industry_future_{h}d_loss"
            tr = train.copy()
            te = test.copy()
            tr["y"] = tr[ycol]
            te["y"] = te[ycol]
            X = tr[CTL + [NEW]].copy()
            mean = X.mean()
            std = X.std(ddof=0)
            std = std.clip(lower=1e-08)
            X = (X - mean) / std
            X = sm.add_constant(X)
            # HAC lags: 2 for 5d, 5 for 20d
            reg = sm.OLS(tr.y, X).fit(
                cov_type="HAC", cov_kwds={"maxlags": 2 if h == 5 else 5}
            )
            p0 = ridge_predictions(tr, te, CTL)
            p1 = ridge_predictions(tr, te, CTL + [NEW])
            y = te["y"].to_numpy(float)
            group_models[s, h] = tr, te, y, p0, p1
            diff = (y - p0) ** 2 - (y - p1) ** 2
            ci = ci_block(diff, seed=2026 + h, block=2 if h == 5 else 5, reps=700)
            result.append(
                {
                    "spec": s,
                    "horizon": h,
                    "n_train": len(tr),
                    "n_test": len(te),
                    "beta_std_top_minus_nontop_train": float(reg.params[NEW]),
                    "se_HAC": float(reg.bse[NEW]),
                    "p_HAC": float(reg.pvalues[NEW]),
                    "test_mse_controls": float(np.mean((y - p0) ** 2)),
                    "test_mse_plus_critical_node_shock": float(np.mean((y - p1) ** 2)),
                    "test_mse_improvement": float(diff.mean()),
                    "ci95_improvement_low": float(ci[0]),
                    "ci95_improvement_high": float(ci[1]),
                }
            )
    pd.DataFrame(result).to_csv(base / "shock_models.csv", index=False)
    return group_models
