import json
from pathlib import Path
import numpy as np
import pandas as pd
from network import (
    NODES,
    WINDOW,
    STEP,
    H,
    read_data,
    residualize,
    normalized_metrics,
    normalize,
    net,
)
from simulation import (
    N,
    B,
    K as SOURCE_K,
    SCENARIOS,
    LABELS,
    sim_fit,
    sim_gfevd,
    simulate,
)
from forecast import linear_ridge, ridge_predictions

KEY = ["cell", "level", "normalization", "metric", "H", "K"]
REP = "raw", "allA", "hs300", "pca1"
SCHEMES = {"row": "Row", "scalar": "Scalar"}
K, PCA_BLOCK, PCA_DRAWS, PCA_SEED = 5, 50, 800, 2026111609
ROLES = "typical", "strong", "weak"
BOOT_DRAWS, BOOT_BLOCK, BOOT_SEED = 400, 5, 2026111608
RET = "returns.csv"
MKT = "index.csv"
NET = "net.csv"
SIX = ["raw_row", "raw_scalar", "allA_row", "allA_scalar", "hs300_row", "hs300_scalar"]
RULES = SIX + ["consensus", "volatility_top5"]
FIXED = ["past_market20", "past_marketvol20", "market_shock_nextday"]
COMMON_CTL = ["past_market20", "past_market_vol20", "past_breadth_down"]
GROUP_MODELS = {
    "M0_fixed": FIXED,
    "M1_outcome": FIXED + ["nontop_negative_shock"],
    "M2_outcome_nominee": FIXED + ["nontop_negative_shock", "top_negative_shock"],
}
ACTIVE = ["controls"] + RULES
TRAIN_N, TEST_N = 238, 336  # complete windows in the 2018/2019 train/test split
EVENT_Q = 0.9  # event cutoff: training outcome q90
ALERT_RATE = 0.1  # alert cutoff: training score q90
ALERT_BLOCK, ALERT_DRAWS, ALERT_SEED = 8, 1600, 44
# FP = 1, FN = 5 for this example
FP_COST, FN_COST = 1.0, 5.0


def moments(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0, mean=np.nan, mcse=np.nan, ci95_low=np.nan, ci95_high=np.nan)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    return dict(
        n=len(x),
        mean=mean,
        mcse=se,
        ci95_low=mean - 1.96 * se,
        ci95_high=mean + 1.96 * se,
    )


def rankcorr(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return rankrho(x, y)


def ids(x):
    values = []
    for i in x:
        values.append(str(int(i)))
    return ";".join(values)


def source_recovery(draws, replications):
    rep_rows = []
    for i, saved in enumerate(draws):
        dgp = SCENARIOS[i // B]
        rep = i % B
        A, S = saved["A"], saved["S"]
        ahat, _, _, gf10 = saved["fitted"]
        _, gp10 = sim_gfevd(A, S)
        if dgp == "factor_only":
            known = set()
        else:
            known = set(saved["perm"][:3])
        mask = []
        for j in range(N):
            mask.append(j in known)
        mask = np.array(mask)
        current = []
        scores = {}
        for level, a, g10 in [("population", A, gp10), ("fitted", ahat, gf10)]:
            lag = a.copy()
            np.fill_diagonal(lag, 0.0)
            l2 = np.sqrt(np.sum(lag**2, axis=0))
            lagorder = np.argsort(-l2)
            scores[level, "none", "lag_L2"] = l2
            current.append(
                dict(
                    scenario=LABELS[dgp],
                    dgp=dgp,
                    rep=rep,
                    true_source_ids=ids(sorted(known)),
                    level=level,
                    normalization="none",
                    metric="lag_L2",
                    H=H,
                    K=SOURCE_K,
                    top3_ids=ids(lagorder[:SOURCE_K]),
                    recall=len(set(lagorder[:SOURCE_K]) & known) / 3
                    if known
                    else np.nan,
                    mean_source_score=float(l2[mask].mean()) if known else np.nan,
                    mean_nonsource_score=float(l2[~mask].mean()),
                    all_scores_numerically_zero=bool(np.max(np.abs(l2)) < 1e-12),
                )
            )
            for nm in ["row", "scalar"]:
                m = normalize(g10, nm)
                to = 100 * (m.sum(0) - np.diag(m))
                nn = net(m)
                oto = np.argsort(-to)
                onet = np.argsort(-nn)
                for metric, score, order in [("TO", to, oto), ("NET", nn, onet)]:
                    scores[level, nm, metric] = score
                    current.append(
                        dict(
                            scenario=LABELS[dgp],
                            dgp=dgp,
                            rep=rep,
                            true_source_ids=ids(sorted(known)),
                            level=level,
                            normalization=nm,
                            metric=metric,
                            H=H,
                            K=SOURCE_K,
                            top3_ids=ids(order[:SOURCE_K]),
                            recall=len(set(order[:SOURCE_K]) & known) / 3
                            if known
                            else np.nan,
                            mean_source_score=float(score[mask].mean())
                            if known
                            else np.nan,
                            mean_nonsource_score=float(score[~mask].mean()),
                            all_scores_numerically_zero=bool(
                                np.max(np.abs(score)) < 1e-12
                            ),
                        )
                    )
        for row in current:
            for q in current:
                if q["level"] == row["level"]:
                    continue
                if q["normalization"] != row["normalization"]:
                    continue
                if q["metric"] != row["metric"]:
                    continue
                peer = q
                break
            else:
                raise StopIteration
            row["population_fitted_top3_overlap"] = len(
                set(row["top3_ids"].split(";")) & set(peer["top3_ids"].split(";"))
            )
            values = scores[row["level"], row["normalization"], row["metric"]]
            peer_scores = scores[peer["level"], peer["normalization"], peer["metric"]]
            row["population_fitted_score_pearson"] = (
                float(np.corrcoef(values, peer_scores)[0, 1]) if known else np.nan
            )
            row["population_fitted_rank_rho"] = (
                rankcorr(values, peer_scores) if known else np.nan
            )
        rep_rows.extend(current)
    repdf = pd.DataFrame(rep_rows)
    repdf.to_csv(replications / "source.csv", index=False)
    summaries = []
    for (scenario, level, nm, metric), g in repdf.groupby(
        ["scenario", "level", "normalization", "metric"], sort=False
    ):
        row = dict(
            scenario=scenario,
            analysis="recall",
            level=level,
            normalization=nm,
            metric=metric,
            comparator="",
        )
        stats = moments(g.recall)
        row.update(stats)
        summaries.append(row)
    for scenario in "BC":
        g = repdf[repdf.scenario == scenario]
        pivot = g.pivot(
            index="rep", columns=["level", "normalization", "metric"], values="recall"
        )
        contrasts = [(("fitted", "none", "lag_L2"), ("population", "none", "lag_L2"))]
        for nm in ["row", "scalar"]:
            for metric in ["TO", "NET"]:
                contrasts.append((("fitted", nm, metric), ("population", nm, metric)))
        for level in ["population", "fitted"]:
            for metric in ["TO", "NET"]:
                contrasts.append(((level, "scalar", metric), (level, "row", metric)))
            for nm in ["row", "scalar"]:
                contrasts.append(((level, nm, "NET"), (level, nm, "TO")))
                contrasts.append(((level, "none", "lag_L2"), (level, nm, "NET")))
        for lhs, rhs in contrasts:
            diff = pivot[lhs] - pivot[rhs]
            row = dict(
                scenario=scenario,
                analysis="paired_difference",
                level=lhs[0],
                normalization=lhs[1],
                metric=lhs[2],
                comparator="/".join(rhs),
            )
            stats = moments(diff)
            row.update(stats)
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    return summary


def system(cell, rep, draws, cfg):
    if cell["id"] == "B_original":
        typ = "B"
    elif cell["id"] == "A_original":
        typ = "A"
    else:
        typ = "C"
    i = "ABC".index(typ) * 220 + rep
    d = draws[i]
    A = d["A"].copy()
    sd = d["sd"]
    load = d["loading"].copy()
    m = cell["sources"]
    count = cell["receivers"]
    if m:
        sources = d["perm"][:m]
    else:
        sources = np.array([], dtype=int)
    if cell["group"] == "sources":
        A = np.diag(np.diag(A))
        rng = np.random.default_rng(
            np.random.SeedSequence([cfg["new_seed"], rep, 10 + m])
        )
        for source in sources:
            targets = rng.choice(d["perm"][m:], size=count, replace=False)
            A[targets, source] = rng.uniform(0.14, 0.23, size=count)
    if cell["group"] == "receivers":
        A = np.diag(np.diag(A))
        rng = np.random.default_rng(np.random.SeedSequence([cfg["new_seed"], rep, 2]))
        for source, oldtargets in zip(sources, d["targets"]):
            if count == 4:
                targets = oldtargets[:4]
                A[targets, source] = d["A"][targets, source]
            else:
                A[oldtargets, source] = d["A"][oldtargets, source]
                candidates = []
                for j in d["perm"][3:]:
                    if j not in oldtargets:
                        candidates.append(j)
                candidates = np.array(candidates)
                targets = rng.choice(candidates, size=4, replace=False)
                A[targets, source] = rng.uniform(0.14, 0.23, size=4)
    A = np.diag(np.diag(A)) + cell["link_scale"] * (A - np.diag(np.diag(A)))
    q = cell["q"]
    if q == "pc1_target":
        target = cfg["calibration_pc1_target"]
        low, high = 0.0, 64.0  # PC1-share search range

        def share(x):
            S = x * np.outer(load, load) + np.diag(sd**2)
            return np.linalg.eigvalsh(S)[-1] / np.trace(S)

        assert share(low) < target < share(high)
        for _ in range(60):
            mid = (low + high) / 2
            if share(mid) < target:
                low = mid
            else:
                high = mid
        q = (low + high) / 2
    load = load * np.sqrt(q)
    S = np.outer(load, load) + np.diag(sd**2)
    eps, factor = d["eps"], d["factor"]
    if cell["T"] > 550:
        rng = np.random.default_rng(np.random.SeedSequence([cfg["new_seed"], rep, 1]))
        # draw the longest series first so T settings use the same random numbers
        eps = np.concatenate((eps, rng.normal(size=(1950, 24))))
        factor = np.concatenate((factor, rng.normal(size=(1950, 1))))
    return A, S, load, sd, sources, eps, factor, i, float(q)


def summarise(df):
    rows = []
    table = {}
    grouped = df.groupby(KEY, sort=False)
    for key, g in grouped:
        table[key] = g.sort_values("rep")
    for key, g in table.items():
        meta = {}
        for name, value in zip(KEY, key):
            meta[name] = value
        base = {}
        for name in [
            "cell_group",
            "T",
            "sources",
            "receivers",
            "link_scale",
            "q_requested",
        ]:
            base[name] = g.iloc[0][name]
        row = meta.copy()
        row.update(base)
        row["analysis"] = "mean"
        row["comparator"] = ""
        row["contrast"] = ""
        stats = moments(g.recall)
        row.update(stats)
        row["mean_hits"] = g.hits.mean()
        row["mean_precision"] = g.precision.mean()
        row["recall_ceiling"] = g.recall_ceiling.iloc[0]
        row["random_recall"] = g.random_recall.iloc[0]
        row["mean_q"] = g.q_actual.mean()
        row["mean_factor_trace_share"] = g.factor_trace_share.mean()
        row["mean_pc1_share"] = g.pc1_share.mean()
        rows.append(row)
        cell, level, nm, metric, H, K = key
        comparisons = []
        if level == "fitted":
            comparisons.append(
                ("fitted_minus_population", (cell, "population", nm, metric, H, K))
            )
        if nm == "scalar":
            comparisons.append(("scalar_minus_row", (cell, level, "row", metric, H, K)))
        if metric == "NET":
            comparisons.append(("NET_minus_TO", (cell, level, nm, "TO", H, K)))
            comparisons.append(("L2_minus_NET", (cell, level, "none", "lag_L2", 0, K)))
        if cell not in ["C_base", "A_original", "B_original"] and (H, K) in [
            (10, 3),
            (0, 3),
        ]:
            comparisons.append(
                ("cell_minus_C_base", ("C_base", level, nm, metric, H, K))
            )
        if cell in ["C_base", "A_original", "B_original"]:
            if H in [20, 50]:
                comparisons.append(("H_minus_10", (cell, level, nm, metric, 10, K)))
            if K in [1, 5]:
                comparisons.append(("K_minus_3", (cell, level, nm, metric, H, 3)))
        for label, other in comparisons:
            p = table[other]
            assert np.array_equal(g.rep.to_numpy(), p.rep.to_numpy())
            a = g.recall.to_numpy()
            b = p.recall.to_numpy()
            if label == "L2_minus_NET":
                delta = b - a
            else:
                delta = a - b
            row = meta.copy()
            row.update(base)
            row["analysis"] = "paired"
            row["comparator"] = "/".join(map(str, other))
            row["contrast"] = label
            stats = moments(delta)
            row.update(stats)
            rows.append(row)
    return pd.DataFrame(rows)


def sensitivity(draws, replications):
    cfg = json.loads((Path(__file__).parent / "config/sim.json").read_text())
    rows = []
    for cell in cfg["cells"]:
        for rep in range(220):
            A, S, loading, sd, sources, eps, factor, i, q = system(
                cell, rep, draws, cfg
            )
            mask = []
            for j in range(24):
                mask.append(j in sources)
            mask = np.array(mask)
            true_radius = float(np.max(np.abs(np.linalg.eigvals(A))))
            assert true_radius < 1
            if cell["id"] in ["C_base", "A_original", "B_original"]:
                ahat, shat = draws[i]["fitted"][:2]
            else:
                Y = simulate(A, sd, loading, eps, factor, cell["T"])
                ahat, shat = sim_fit(Y)
                if not (np.isfinite(ahat).all() and np.isfinite(shat).all()):
                    raise ArithmeticError(
                        f"nonfinite fitted VAR: {cell['id']}, rep={rep}"
                    )
            fit_radius = float(np.max(np.abs(np.linalg.eigvals(ahat))))
            if not np.linalg.eigvalsh(shat).min() > 0:
                raise ArithmeticError(
                    f"nonpositive VAR covariance: {cell['id']}, rep={rep}"
                )
            factor_diag = float(np.mean(loading**2 / np.diag(S)))
            factor_trace = float(np.sum(loading**2) / np.trace(S))
            pc1 = float(np.linalg.eigvalsh(S)[-1] / np.trace(S))
            base = dict(
                cell=cell["id"],
                cell_group=cell["group"],
                rep=rep,
                T=cell["T"],
                sources=cell["sources"],
                receivers=cell["receivers"],
                link_scale=cell["link_scale"],
                q_requested=str(cell["q"]),
                q_actual=q,
                true_source_ids=";".join(map(str, sorted(sources))),
                factor_diag_share=factor_diag,
                factor_trace_share=factor_trace,
                pc1_share=pc1,
                true_spectral_radius=true_radius,
                fitted_spectral_radius=fit_radius,
            )
            if cell["group"] in ["anchor", "control", "diagnostic"]:
                combos = [(10, 3), (20, 3), (50, 3), (10, 1), (10, 5)]
            else:
                combos = [(10, 3)]
            for level, a, s in [("population", A, S), ("fitted", ahat, shat)]:
                lag = a.copy()
                np.fill_diagonal(lag, 0.0)
                l2 = np.sqrt((lag**2).sum(0))

                def record(nm, metric, H, K, score):
                    order = np.argsort(-score)
                    top = order[:K]
                    hits = int(mask[top].sum())
                    m = len(sources)
                    row = base.copy()
                    row["level"] = level
                    row["normalization"] = nm
                    row["metric"] = metric
                    row["H"] = H
                    row["K"] = K
                    row["topk_ids"] = ";".join(map(str, top))
                    row["hits"] = hits
                    if m:
                        row["recall"] = hits / m
                    else:
                        row["recall"] = np.nan
                    row["precision"] = hits / K
                    row["random_hits"] = K * m / 24
                    if m:
                        row["random_recall"] = K / 24
                    else:
                        row["random_recall"] = np.nan
                    row["random_precision"] = m / 24
                    if m:
                        row["recall_ceiling"] = min(K, m) / m
                    else:
                        row["recall_ceiling"] = np.nan
                    row["mean_abs_score"] = float(np.mean(np.abs(score)))
                    row["numerically_zero"] = bool(np.max(np.abs(score)) < 1e-12)
                    rows.append(row)

                k_values = set()
                for h, k in combos:
                    k_values.add(k)
                for K in sorted(k_values):
                    record("none", "lag_L2", 0, K, l2)
                horizons = []
                for h, k in combos:
                    if h not in horizons:
                        horizons.append(h)
                cache = {}
                for h in horizons:
                    matrices = sim_gfevd(a, s, h)
                    cache[h] = matrices[1]
                for H, K in combos:
                    G = cache[H]
                    for nm in ["row", "scalar"]:
                        M = normalize(G, nm)
                        to = 100 * (M.sum(0) - np.diag(M))
                        nn = net(M)
                        record(nm, "TO", H, K, to)
                        record(nm, "NET", H, K, nn)
    df = pd.DataFrame(rows)
    path = replications / "sensitivity.csv"
    df.to_csv(path, index=False)
    return summarise(df)


def fit_pca(y):
    X = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    response = y[1:]
    beta, _, rank, _ = np.linalg.lstsq(X, response, rcond=None)
    A = beta[1:].T
    E = response - X @ beta
    fenmu = len(response) - X.shape[1]
    S = E.T @ E / fenmu
    return dict(
        A=A,
        S=S,
        rho=float(np.max(np.abs(np.linalg.eigvals(A)))),
        sigma_min_diag=float(np.min(np.diag(S))),
        rank_Y=int(np.linalg.matrix_rank(y)),
        rank_X=int(rank),
        rank_Sigma=int(np.linalg.matrix_rank(S)),
        noise_divisor=fenmu,
        effective_df=len(response) - int(rank),
    )


def pca_gfevd(fit):
    A, S = fit["A"], fit["S"]
    phi = np.eye(24)
    num = np.zeros((24, 24))
    den = np.zeros(24)
    for _ in range(H):
        ps = phi @ S
        num += ps**2 / np.diag(S)[None, :]
        den += np.sum(ps * phi, axis=1)
        phi = phi @ A
    if not np.min(den) > 0:
        raise ArithmeticError("nonpositive PCA forecast variance")
    G = num / den[:, None]
    if not (np.isfinite(G).all() and np.min(G.sum(1)) > 0):
        raise ArithmeticError("invalid PCA GFEVD")
    return G


def overlap(a, b):
    return len(set(a.split(";")) & set(b.split(";")))


def shortlist(net):
    ix = np.argsort(-net)
    names = []
    for j in ix[:K]:
        names.append(NODES[j])
    s = ";".join(names)
    return s, float(net[ix[4]] - net[ix[5]])


def rankrho(a, b):
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    return float(np.corrcoef(ar, br)[0, 1])


def pca_metrics(G):
    _, nr = normalized_metrics(G, "row")
    _, ns = normalized_metrics(G, "scalar")
    row, _ = shortlist(nr)
    scalar, _ = shortlist(ns)
    out = dict(
        financial_J_Row_NET=float(nr[23]),
        financial_J_Scalar_NET=float(ns[23]),
        J_sign_row=int(np.sign(nr[23])),
        J_sign_scalar=int(np.sign(ns[23])),
        J_top5_row=int("J" in row.split(";")),
        J_top5_scalar=int("J" in scalar.split(";")),
        row_top5=row,
        scalar_top5=scalar,
        row_scalar_overlap=overlap(row, scalar),
        row_scalar_rankrho=rankrho(nr, ns),
    )
    return out


def pca_window(date, rep, y):
    fit = fit_pca(y)
    row = dict(date=date, representation=rep)
    if fit["rho"] >= 1 or fit["sigma_min_diag"] <= 0:
        raise ArithmeticError(f"invalid PCA VAR: {date}, {rep}")
    G = pca_gfevd(fit)
    row.update(pca_metrics(G))
    for name in [
        "rank_Y",
        "rank_X",
        "rank_Sigma",
        "rho",
        "sigma_min_diag",
        "noise_divisor",
        "effective_df",
    ]:
        row[name] = fit[name]
    return row


def pca_adjust(raw):
    std = raw.std(axis=0, ddof=1)
    if not np.all(std > 0):
        bad = []
        for j in range(len(std)):
            if not std[j] > 0:
                bad.append(NODES[j])
        raise ValueError("nonpositive PCA input standard deviation: " + ",".join(bad))
    Z = (raw - raw.mean(0)) / std
    C = Z.T @ Z / (WINDOW - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    v = eigenvectors[:, -1]
    factor = Z @ v
    residual = residualize(raw, factor)
    share = float(eigenvalues[-1] / eigenvalues.sum())
    meta = dict(PCA_PC1_explained_share=share)
    return residual, meta


def add_comparisons(frame):
    byrep = {}
    for r in REP:
        temp = frame[frame.representation == r]
        temp = temp.set_index("date")
        temp = temp.sort_index()
        byrep[r] = temp
    frame = frame.copy()
    for idx, row in frame.iterrows():
        date = row["date"]
        if row["representation"] != "pca1":
            continue
        for refrep in ["raw", "allA", "hs300"]:
            ref = byrep[refrep].loc[date]
            for scheme, title in SCHEMES.items():
                rn = row[f"financial_J_{title}_NET"]
                bn = ref[f"financial_J_{title}_NET"]
                frame.loc[idx, "J_NET_delta_" + refrep + "_" + scheme] = rn - bn
                frame.loc[idx, "overlap_" + refrep + "_" + scheme] = overlap(
                    row[scheme + "_top5"], ref[scheme + "_top5"]
                )
    return frame


def time_statistics(frame):
    pca = frame[frame.representation == "pca1"]
    pca = pca.sort_values("date")
    columns = []
    for ref in ["raw", "allA", "hs300"]:
        for nm in SCHEMES:
            columns.append(f"J_NET_delta_{ref}_{nm}")
    columns.append("row_scalar_overlap")
    for ref in ["raw", "allA", "hs300"]:
        for nm in SCHEMES:
            columns.append(f"overlap_{ref}_{nm}")
    values = pca[columns].to_numpy(float)
    assert np.isfinite(values).all()
    # resample shared dates for the paired comparisons
    indices = cbb_indices(582, PCA_BLOCK, PCA_DRAWS, PCA_SEED)
    means = []
    for ix in indices:
        mean = values[ix].mean(axis=0)
        means.append(mean)
    means = np.array(means)
    intervals = np.quantile(means, [0.025, 0.975], axis=0)
    out = {}
    for j, c in enumerate(columns):
        out[c] = dict(
            mean=float(values[:, j].mean()),
            low=float(intervals[0, j]),
            high=float(intervals[1, j]),
            bootstrap_mean_mcse=float(means[:, j].std(ddof=1) / np.sqrt(PCA_DRAWS)),
        )
    return out


def pca(base, systems):
    dates, R, _ = read_data(base / RET, base / MKT)
    rows = []
    for end in range(WINDOW - 1, len(dates), STEP):
        date = dates[end]
        for rep in ["raw", "allA", "hs300"]:
            row = dict(date=date, representation=rep)
            row.update(pca_metrics(systems[date][rep][1]))
            row["standardized_variance_removed"] = systems[date]["variance_removed"][
                rep
            ]
            rows.append(row)
    for end in range(WINDOW - 1, len(dates), 5):
        raw = R[end - WINDOW + 1 : end + 1]
        # remove the first PCA factor in each window
        residual, meta = pca_adjust(raw)
        row = pca_window(dates[end], "pca1", residual)
        row.update(meta)
        centered = raw - raw.mean(0)
        row["standardized_variance_removed"] = float(
            1
            - np.mean(
                np.sum(residual * residual, axis=0)
                / np.sum(centered * centered, axis=0)
            )
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame = add_comparisons(frame)
    stats = time_statistics(frame)
    return frame, stats


def pca_summary(frame):
    raw = frame[frame.representation == "raw"].set_index("date").sort_index()
    pca = frame[frame.representation == "pca1"].set_index("date").sort_index()
    assert raw.index.equals(pca.index)
    before = raw.financial_J_Scalar_NET
    after = pca.financial_J_Scalar_NET
    rows = [
        ("n_windows", len(pca)),
        ("pca_row_scalar_rankrho_mean", pca.row_scalar_rankrho.mean()),
        (
            "finance_scalar_negative_to_positive",
            int(((before < 0) & (after > 0)).sum()),
        ),
        (
            "finance_scalar_positive_to_negative",
            int(((before > 0) & (after < 0)).sum()),
        ),
        ("pca_allA_row_top5_overlap_mean", pca.overlap_allA_row.mean()),
        ("pca_allA_scalar_top5_overlap_mean", pca.overlap_allA_scalar.mean()),
        ("pca_explained_variation_mean", pca.PCA_PC1_explained_share.mean()),
    ]
    for rep in ["allA", "hs300"]:
        values = frame.loc[frame.representation == rep, "standardized_variance_removed"]
        rows.append((rep + "_removed_variation_mean", values.mean()))
    for name in [
        "rank_Y",
        "rank_X",
        "rank_Sigma",
        "rho",
        "sigma_min_diag",
        "noise_divisor",
        "effective_df",
    ]:
        rows.append(("pca_" + name + "_min", pca[name].min()))
        rows.append(("pca_" + name + "_max", pca[name].max()))
    return pd.DataFrame(rows, columns=["statistic", "value"])


def fit_boot(y, p=1):
    n = y.shape[1]
    columns = [np.ones(len(y) - p)]
    for len_lag in range(1, p + 1):
        block = y[p - len_lag : len(y) - len_lag]
        columns.append(block)
    X = np.column_stack(columns)
    response = y[p:]
    beta, _, rank, _ = np.linalg.lstsq(X, response, rcond=None)
    E = response - X @ beta
    df = len(response) - X.shape[1]
    S = E.T @ E / df
    A = []
    for l in range(p):
        coef = beta[1 + l * n : 1 + (l + 1) * n].T
        A.append(coef)
    A = np.array(A)
    companion = np.zeros((p * n, p * n))
    companion[:n] = np.hstack(A)
    if p > 1:
        companion[n:, :-n] = np.eye((p - 1) * n)
    return {
        "c": beta[0],
        "A": A,
        "S": S,
        "E": E,
        "companion": companion,
        "rho": float(np.max(np.abs(np.linalg.eigvals(companion)))),
        "df": df,
        "nobs": len(response),
        "regressors": X.shape[1],
        "rank": rank,
        "cond": float(np.linalg.cond(X)) if p == 2 else np.nan,
        "sigma_min_eig": float(np.linalg.eigvalsh(S)[0]),
    }


def boot_gfevd(fit):
    n = len(NODES)
    phi = np.eye(len(fit["companion"]))
    fenzi = np.zeros((n, n))
    fenmu = np.zeros(n)
    for _ in range(H):
        response = phi[:n, :n]
        ps = response @ fit["S"]
        fenzi += ps**2 / np.diag(fit["S"])[None, :]
        fenmu += np.sum(ps * response, axis=1)
        phi = phi @ fit["companion"]
    if np.min(fenmu) <= 0:
        raise ArithmeticError("nonpositive forecast variance")
    G = fenzi / fenmu[:, None]
    if not np.isfinite(G).all() or np.min(G.sum(1)) <= 0:
        raise ArithmeticError("invalid GFEVD")
    return G


def boot_metrics(G):
    tci_r, net_r = normalized_metrics(G, "row")
    tci_s, net_s = normalized_metrics(G, "scalar")
    sr, gr = shortlist(net_r)
    ss, gs = shortlist(net_s)
    return dict(
        top5_row=sr,
        top5_scalar=ss,
        overlap_rs=overlap(sr, ss),
        row_net=";".join(format(x, ".17g") for x in net_r),
        scalar_net=";".join(format(x, ".17g") for x in net_s),
        row_tci=tci_r,
        scalar_tci=tci_s,
        row_cutoff_gap=gr,
        scalar_cutoff_gap=gs,
    )


def base_record(date, role, start, end_index, fit, kind, rep=-1, p=1):
    return dict(
        kind=kind,
        date=date,
        role=role,
        rep=rep,
        p=p,
        window=WINDOW,
        horizon=H,
        K=K,
        window_start=start,
        end_index=end_index,
        nobs=fit["nobs"],
        df=fit["df"],
        regressors=fit["regressors"],
        design_rank=fit["rank"],
        design_condition=fit["cond"],
        rho=fit["rho"],
        sigma_min_eig=fit["sigma_min_eig"],
        status="ok",
        B=BOOT_DRAWS if kind == "bootstrap" else 0,
        block_len=BOOT_BLOCK if kind == "bootstrap" else 0,
        master_seed=BOOT_SEED if kind == "bootstrap" else "",
    )


def attach_reference(row, ref):
    if row["status"] == "ok":
        row["overlap_row0"] = overlap(row["top5_row"], ref["top5_row"])
        row["overlap_scalar0"] = overlap(row["top5_scalar"], ref["top5_scalar"])
        row["D_RS"] = K - row["overlap_rs"]
        row["D_R"] = K - row["overlap_row0"]
        row["D_S"] = K - row["overlap_scalar0"]
        row["delta_vs_row"] = row["D_RS"] - row["D_R"]
        row["delta_vs_scalar"] = row["D_RS"] - row["D_S"]


def selected_roles(rows):
    values = np.array([r["overlap_rs"] for r in rows])
    used, selection = set(), {}
    for role, target in (
        ("typical", np.median(values)),
        ("strong", values.min()),
        ("weak", values.max()),
    ):
        eligible = []
        for r in rows:
            if r["date"] not in used:
                eligible.append(r)
        eligible.sort(key=lambda r: (abs(r["overlap_rs"] - target), r["date"]))
        picked = eligible[0]
        used.add(picked["date"])
        selection[picked["date"]] = role
    return selection


def boot_base(base, systems):
    dates, R, _ = read_data(base / RET, base / MKT)
    rows, models = [], {}
    for end in range(WINDOW - 1, len(dates), STEP):
        d = dates[end]
        y = R[end - WINDOW + 1 : end + 1]
        fit = fit_boot(y)
        assert fit["rank"] == 25
        if not (fit["rho"] < 1 and fit["sigma_min_eig"] > 0):
            raise ArithmeticError(f"invalid baseline VAR: {d}")
        G = boot_gfevd(fit)
        row = base_record(d, "", dates[end - WINDOW + 1], end, fit, "baseline")
        row.update(boot_metrics(G))
        row["denominator_cv"] = float(
            systems[d]["raw"][0].sum(1).std(ddof=0) / systems[d]["raw"][0].sum(1).mean()
        )
        A = fit["A"][0]
        off = np.abs(A[~np.eye(24, dtype=bool)])
        row["offdiag_abs_median"] = float(np.median(off))
        row["pc1_covariance_share"] = float(
            np.linalg.eigvalsh(fit["S"])[-1] / np.trace(fit["S"])
        )
        attach_reference(row, row)
        rows.append(row)
        models[d] = y, fit
    roles = selected_roles(rows)
    for row in rows:
        row["role"] = roles.get(row["date"], "")
    return rows, models


def residual_diagnostics(fit):
    E = fit["E"] - fit["E"].mean(0)
    result = {}
    for power in (1, 2):
        for lag in (1, 5):
            correlations = []
            for j in range(24):
                a = E[:-lag, j] ** power
                b = E[lag:, j] ** power
                corr = np.corrcoef(a, b)
                correlations.append(corr[0, 1])
            result[f"resid_power{power}_lag{lag}_absmedian"] = float(
                np.median(np.abs(correlations))
            )
            result[f"resid_power{power}_lag{lag}_absmax"] = float(
                np.max(np.abs(correlations))
            )
    return result


def pseudo_sample(y, fit, date, rep):
    rng = np.random.default_rng(
        np.random.SeedSequence([BOOT_SEED, int(date.replace("-", "")), rep])
    )
    residuals = fit["E"] - fit["E"].mean(0)
    residuals = residuals * np.sqrt(fit["nobs"] / fit["df"])
    starts = rng.integers(
        0, len(residuals), size=int(np.ceil((WINDOW - 1) / BOOT_BLOCK))
    )
    indices = ((starts[:, None] + np.arange(BOOT_BLOCK)) % len(residuals)).ravel()[
        : WINDOW - 1
    ]
    innovations = residuals[indices]
    sample = np.empty_like(y)
    sample[0] = y[0]
    for t in range(1, WINDOW):
        sample[t] = fit["c"] + fit["A"][0] @ sample[t - 1] + innovations[t - 1]
    return sample


def run_bootstrap(rows, models):
    output = list(rows)
    selected = []
    for role in ROLES:
        for r in rows:
            if r["role"] == role:
                selected.append(r)
    for ref in selected:
        date = ref["date"]
        y, fitted = models[date]
        ref.update(residual_diagnostics(fitted))
        for rep in range(BOOT_DRAWS):
            sample = pseudo_sample(y, fitted, date, rep)
            try:
                fit = fit_boot(sample)
            except (np.linalg.LinAlgError, ArithmeticError, FloatingPointError):
                row = dict(
                    kind="bootstrap",
                    date=date,
                    role=ref["role"],
                    rep=rep,
                    p=1,
                    window=WINDOW,
                    horizon=H,
                    K=K,
                    window_start=ref["window_start"],
                    end_index=ref["end_index"],
                    nobs=249,
                    df=224,
                    regressors=25,
                    status="fit_failure",
                    B=BOOT_DRAWS,
                    block_len=BOOT_BLOCK,
                    master_seed=BOOT_SEED,
                )
                attach_reference(row, ref)
                output.append(row)
                continue
            row = base_record(
                date,
                ref["role"],
                ref["window_start"],
                ref["end_index"],
                fit,
                "bootstrap",
                rep,
            )
            if not np.isfinite(sample).all() or fit["rank"] != 25:
                row.update(status="rank_or_nonfinite")
            elif fit["rho"] >= 1:
                row.update(status="unstable")
            elif fit["sigma_min_eig"] <= 0:
                row.update(status="covariance_not_pd")
            else:
                try:
                    row.update(boot_metrics(boot_gfevd(fit)))
                except (ArithmeticError, FloatingPointError, np.linalg.LinAlgError):
                    row.update(status="numerical_failure")
            attach_reference(row, ref)
            output.append(row)
    for ref in selected:
        y, _ = models[ref["date"]]
        fit = fit_boot(y, p=2)
        row = base_record(
            ref["date"],
            ref["role"],
            ref["window_start"],
            ref["end_index"],
            fit,
            "lag_check",
            rep=-2,
            p=2,
        )
        if fit["rho"] >= 1 or fit["rank"] != 49 or fit["sigma_min_eig"] <= 0:
            row.update(status="unstable_or_invalid")
        else:
            row.update(boot_metrics(boot_gfevd(fit)))
        attach_reference(row, ref)
        output.append(row)
    return pd.DataFrame(output)


def distribution(values):
    x = np.asarray(values, dtype=float)
    lo, hi = np.quantile(x, [0.025, 0.975], method="inverted_cdf")
    return dict(
        mean=float(x.mean()),
        median=float(np.median(x)),
        lo=lo,
        hi=hi,
        mcse=float(np.std(x, ddof=1) / np.sqrt(len(x))),
    )


def cbb_indices(n, L, B, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(B):
        starts = rng.integers(0, n, size=(n + L - 1) // L)
        blocks = []
        for z in starts:
            block = np.arange(z, z + L) % n
            blocks.append(block)
        indices = np.concatenate(blocks)[:n]
        rows.append(indices)
    return np.array(rows)


def loss_stats(actual, base, aug, indices):
    d = (actual - base) ** 2 - (actual - aug) ** 2
    lo, hi = np.quantile(d[indices].mean(axis=1), [0.025, 0.975])
    return dict(gain=float(d.mean()), low=float(lo), high=float(hi))


def load_data(base):
    dates, R, F = read_data(base / RET, base / MKT)
    market = F[:, 0]
    net = pd.read_csv(base / NET)
    endpoints = sorted(net.date.unique())
    mats = {}
    for s in SIX:
        temp = net.pivot(index="date", columns="sector", values="NET_" + s)
        temp = temp.reindex(index=endpoints, columns=NODES)
        mats[s] = temp.to_numpy(float)
    tops_a = {}
    index = {}
    for i, d in enumerate(dates):
        index[d] = i
    for t, d in enumerate(endpoints):
        six = {}
        for s in SIX:
            order = np.argsort(-mats[s][t])
            six[s] = order[:5]
        vote = np.bincount(np.concatenate(list(six.values())), minlength=24)
        ranks = []
        for s in SIX:
            order = np.argsort(mats[s][t])
            rank = np.argsort(order)
            ranks.append(rank.astype(float))
        ordinal = np.array(ranks).mean(0)
        order = []
        for j in range(24):
            order.append((-vote[j], -ordinal[j], NODES[j], j))
        order.sort()
        ca = []
        for row in order[:5]:
            ca.append(row[3])
        ca = np.array(ca)
        end = index[d]
        va = np.argsort(-R[end - 19 : end + 1].std(axis=0, ddof=1))[:5]
        tops_a[d] = six.copy()
        tops_a[d]["consensus"] = ca
        tops_a[d]["volatility_top5"] = va
    return dates, R, market, endpoints, index, tops_a


def common_forecasts(dates, R, market, endpoints, index, tops):
    records = []
    for d in endpoints:
        end = index[d]
        p20 = np.prod(1 + R[end - 19 : end + 1], axis=0) - 1
        trailing = np.maximum(0, -(np.prod(1 + R[end - 4 : end + 1], axis=0) - 1))
        r = dict(
            date=d,
            idx_end=end,
            past_market20=float(np.prod(1 + market[end - 19 : end + 1]) - 1),
            past_market_vol20=float(market[end - 19 : end + 1].std(ddof=1)),
            past_breadth_down=float(np.mean(p20 < 0)),
        )
        for rule in RULES:
            r["shock_" + rule] = float(trailing[tops[d][rule]].mean())
        if end + 20 >= len(dates):
            continue
        r["future20"] = np.prod(1 + R[end + 1 : end + 21], axis=0) - 1
        r["outcome_start"] = dates[end + 1]
        r["outcome_end"] = dates[end + 20]
        if dates[end + 20] <= "2018-12-31":
            r["split"] = "train"
        elif d >= "2019-01-01":
            r["split"] = "test"
        else:
            r["split"] = "purge"
        records.append(r)
    train = []
    for r in records:
        if r["split"] == "train":
            train.append(r)
    values = []
    for r in train:
        values.append(r["future20"])
    values = np.array(values)
    threshold = np.quantile(values, 0.1, axis=0)
    for r in records:
        r["future_tail_breadth"] = float((r["future20"] <= threshold).mean())
    test = []
    for r in records:
        if r["split"] == "test":
            test.append(r)
    y = []
    for r in test:
        y.append(r["future_tail_breadth"])
    y = np.array(y)
    combined = train + test
    train_y = []
    for r in train:
        train_y.append(r["future_tail_breadth"])
    mean = np.mean(train_y)
    scores = {}
    scores["historical_mean"] = np.full(len(combined), mean)
    scores["controls"] = linear_ridge(train, combined, COMMON_CTL)
    for rule in RULES:
        scores[rule] = linear_ridge(train, combined, COMMON_CTL + ["shock_" + rule])
    predictions = {}
    for name, values in scores.items():
        predictions[name] = values[len(train) :]
    return train, test, y, predictions, scores


def predict(base, old):
    dates, R, m, endpoints, index, ta = load_data(base)
    train, test, ya, pa, scores = common_forecasts(dates, R, m, endpoints, index, ta)
    assert len(train) == TRAIN_N and len(test) == TEST_N
    output = []
    trstart = train[0]["date"]
    trend = train[-1]["date"]
    train_label_end = train[-1]["outcome_end"]
    for model in ["historical_mean", "controls"]:
        for i, r in enumerate(test):
            output.append(
                dict(
                    task="common",
                    date=r["date"],
                    forecast_origin=r["date"],
                    outcome_start=r["outcome_start"],
                    outcome_end=r["outcome_end"],
                    horizon=20,
                    nomination_rule="shared",
                    model=model,
                    actual_outcome=float(ya[i]),
                    predicted_outcome=float(pa[model][i]),
                    training_start_date=trstart,
                    training_end_date=trend,
                    training_label_end_date=train_label_end,
                )
            )
    for rule in RULES:
        for i, r in enumerate(test):
            output.append(
                dict(
                    task="common",
                    date=r["date"],
                    forecast_origin=r["date"],
                    outcome_start=r["outcome_start"],
                    outcome_end=r["outcome_end"],
                    horizon=20,
                    nomination_rule=rule,
                    model="plus_nominee",
                    actual_outcome=float(ya[i]),
                    predicted_outcome=float(pa[rule][i]),
                    training_start_date=trstart,
                    training_end_date=trend,
                    training_label_end_date=train_label_end,
                )
            )
    for rule in RULES:
        for h in (5, 20):
            tr, te, y, p0, p1 = old[rule, h]
            forecasts = {"original_aggregate": p0, "original_aggregate_location": p1}
            for model, cols in GROUP_MODELS.items():
                forecasts[model] = ridge_predictions(tr, te, cols)
            label_end = dates[int(tr.iloc[-1]["idx"]) + h + 1]
            for model, pred in forecasts.items():
                for k, r in enumerate(te.to_dict("records")):
                    end = int(r["idx"])
                    record = dict(
                        task="group-specific",
                        date=r["date"],
                        forecast_origin=dates[end + 1],
                        outcome_start=dates[end + 2],
                        outcome_end=dates[end + h + 1],
                        horizon=h,
                        nomination_rule=rule,
                        model=model,
                        actual_outcome=float(y[k]),
                        predicted_outcome=float(pred[k]),
                        training_start_date=tr.date.iloc[0],
                        training_end_date=tr.date.iloc[-1],
                        training_label_end_date=label_end,
                    )
                    output.append(record)
    frame = pd.DataFrame(output)
    return frame, train, test, scores


def summaries(frame):
    common = frame[frame.task == "common"]
    groups = frame[frame.task == "group-specific"]
    grouped = common.groupby("date")
    ca_y = grouped.actual_outcome.first()
    ca_y = ca_y.sort_index()
    ca_y = ca_y.to_numpy()
    ca_pred = {}
    for (rule, model), g in common.groupby(["nomination_rule", "model"]):
        if rule == "shared":
            name = model
        else:
            name = rule
        temp = g.sort_values("date")
        ca_pred[name] = temp.predicted_outcome.to_numpy()
    ia = cbb_indices(TEST_N, 8, 1600, 44)
    common_stats = {}
    for key, p in ca_pred.items():
        s = loss_stats(ca_y, ca_pred["controls"], p, ia)
        common_stats[key] = s.copy()
        common_stats[key]["mse"] = float(np.mean((ca_y - p) ** 2))
    pairs = []
    for rule in RULES:
        if rule != "consensus":
            pairs.append((rule, "consensus"))
    for rep in ["raw", "allA", "hs300"]:
        pairs.append((rep + "_row", rep + "_scalar"))
    pair_stats = {}
    for a, b in pairs:
        pair_stats[a, b] = loss_stats(ca_y, ca_pred[a], ca_pred[b], ia)
    group_stats = {}
    for (rule, h), g in groups.groupby(["nomination_rule", "horizon"]):
        pivot = g.pivot(index="date", columns="model", values="predicted_outcome")
        pivot = pivot.sort_index()
        grouped = g.groupby("date")
        y = grouped.actual_outcome.first()
        y = y.sort_index()
        y = y.to_numpy()
        p = {}
        for c in pivot:
            p[c] = pivot[c].to_numpy()
        ix = cbb_indices(TEST_N, 2 if h == 5 else 5, 700, 2026 + int(h))
        mse = {}
        for c, v in p.items():
            error = (y - v) ** 2
            mse[c] = float(np.mean(error))
        result = {}
        result["mse"] = mse
        result["outcome"] = loss_stats(y, p["M0_fixed"], p["M1_outcome"], ix)
        result["nominee"] = loss_stats(y, p["M1_outcome"], p["M2_outcome_nominee"], ix)
        result["original"] = loss_stats(
            y, p["original_aggregate"], p["original_aggregate_location"], ix
        )
        group_stats[rule, int(h)] = result
    return common_stats, pair_stats, group_stats


def alerts(train, test, scores):
    train_y = np.array([r["future_tail_breadth"] for r in train])
    event_threshold = float(np.quantile(train_y, EVENT_Q, method="linear"))
    combined = train + test
    output = []
    for model in ACTIVE:
        score = scores[model]
        threshold = float(np.quantile(score[:TRAIN_N], 1 - ALERT_RATE, method="linear"))
        if np.unique(score[:TRAIN_N]).size <= 1:
            raise ValueError(f"constant training alert scores: {model}")
        for i, r in enumerate(combined):
            event = int(r["future_tail_breadth"] >= event_threshold)
            alert = int(score[i] >= threshold)
            rec = dict(
                date=r["date"],
                split=r["split"],
                model=model,
                actual_common_breadth=r["future_tail_breadth"],
                severe_event=event,
                risk_score=score[i],
                severe_event_threshold=event_threshold,
                alert_score_threshold=threshold,
                alert=alert,
                forecast_origin=r["date"],
                outcome_start=r["outcome_start"],
                outcome_end=r["outcome_end"],
                forecast_loss=float((r["future_tail_breadth"] - score[i]) ** 2),
            )
            output.append(rec)
    frame = pd.DataFrame(output)
    return frame


def alert_summary(frame):
    test = frame[(frame.split == "test") & frame.model.isin(ACTIVE)]
    pivot = test.pivot(index="date", columns="model", values="alert")
    pivot = pivot.sort_index()
    grouped = test.groupby("date")
    event = grouped.severe_event.first()
    event = event.sort_index()
    event = event.to_numpy(int)
    ids = cbb_indices(len(event), ALERT_BLOCK, ALERT_DRAWS, ALERT_SEED)
    eb = event[ids]
    positives, negatives = eb.sum(1), (1 - eb).sum(1)
    stats, boot = {}, {}
    for model in ACTIVE:
        alert = pivot[model].to_numpy(int)
        tp = int(np.sum((alert == 1) & (event == 1)))
        fp = int(np.sum((alert == 1) & (event == 0)))
        fn = int(np.sum((alert == 0) & (event == 1)))
        tn = int(np.sum((alert == 0) & (event == 0)))
        if tp + fn:
            recall = float(tp / (tp + fn))
        else:
            recall = np.nan
        if fp + tn:
            fpr = float(fp / (fp + tn))
        else:
            fpr = np.nan
        if tp + fp:
            fdr = float(fp / (tp + fp))
            precision = float(tp / (tp + fp))
        else:
            fdr = np.nan
            precision = np.nan
        stats[model] = dict(
            TP=tp,
            FP=fp,
            FN=fn,
            TN=tn,
            alerts=tp + fp,
            alert_rate=(tp + fp) / TEST_N,
            recall=recall,
            fpr=fpr,
            fdr=fdr,
            precision=precision,
            cost=FP_COST * fp + FN_COST * fn,
            cost100=(FP_COST * fp + FN_COST * fn) / TEST_N * 100,
            mse=float(test[test.model == model].forecast_loss.mean()),
        )
        ab = alert[ids]
        tpb = ((ab == 1) & (eb == 1)).sum(1)
        fpb = ((ab == 1) & (eb == 0)).sum(1)
        fnb = ((ab == 0) & (eb == 1)).sum(1)
        rb = np.full(ALERT_DRAWS, np.nan)
        fb = np.full(ALERT_DRAWS, np.nan)
        np.divide(tpb, positives, out=rb, where=positives > 0)
        np.divide(fpb, negatives, out=fb, where=negatives > 0)
        boot[model] = dict(
            recall=rb, fpr=fb, cost100=(FP_COST * fpb + FN_COST * fnb) / TEST_N * 100
        )
    comparison = {}
    for model in RULES:
        comp = {}
        for key in ["recall", "fpr", "cost100"]:
            difference = boot[model][key] - boot["controls"][key]
            valid = np.isfinite(difference)
            if valid.any():
                lo, hi = np.quantile(difference[valid], [0.025, 0.975])
            else:
                lo, hi = np.nan, np.nan
            comp[key] = dict(
                point=stats[model][key] - stats["controls"][key],
                low=float(lo),
                high=float(hi),
                valid=int(valid.sum()),
            )
        comparison[model] = comp
    return stats, comparison


def bootstrap_tables(f):
    boot = f.query('kind=="bootstrap" and status=="ok"')
    rows = []
    frequencies = []
    for date, g in boot.groupby("date", sort=False):
        for col in ["D_RS", "D_R", "D_S", "delta_vs_row", "delta_vs_scalar"]:
            row = dict(date=date, metric=col)
            stats = distribution(g[col])
            row.update(stats)
            row["positive"] = float((g[col] > 0).mean())
            row["zero"] = float((g[col] == 0).mean())
            row["negative"] = float((g[col] < 0).mean())
            rows.append(row)
        for scheme in ["row", "scalar"]:
            lists = []
            for x in g["top5_" + scheme]:
                lists.append(set(x.split(";")))
            for node in [
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
            ]:
                count = 0
                for v in lists:
                    count += node in v
                frequency = count / len(lists)
                frequencies.append(
                    dict(date=date, scheme=scheme, node=node, frequency=frequency)
                )
    panel = pd.DataFrame(rows)
    return panel, pd.DataFrame(frequencies)
