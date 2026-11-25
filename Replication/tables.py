import pandas as pd
from extra import summaries, alert_summary, bootstrap_tables, pca_summary
from network import NODES, BOOT, BLOCK, ci_block


def export(
    base,
    original_simulation,
    source,
    sensitivity,
    paired,
    pca,
    forecasts,
    alerts,
    out,
    pca_stats,
):
    out.mkdir(parents=True, exist_ok=False)

    def save(name, df):
        df.to_csv(out / (name + ".csv"), index=False, float_format="%.17g")

    horizons = pd.read_csv(base / "horizon_summary.csv")
    temp = horizons.query('period=="full"')
    save("Table_3", temp)

    reliability = pd.read_csv(base / "overlap.csv")
    temp = reliability.query("k==5")
    save("Table_4", temp)
    temp = reliability.query("k!=5")
    save("Table_B1", temp)

    grid = pd.read_csv(base / "tci.csv")
    grouped = grid.groupby(["N", "representation"], sort=False)
    tci = grouped[["row_h1", "row_h10", "scalar_h10"]].mean()
    tci = tci.reset_index()
    save("Table_7", tci)
    paired_tci = grid[grid.N == 24].pivot(
        index="date", columns="representation", values="row_h10"
    )
    raw = paired_tci["raw"].to_numpy(float)
    all_a = paired_tci["allA"].to_numpy(float)
    values = raw - all_a
    seed = 20261021 + len(values) + len("raw24_minus_allA")
    ci = ci_block(values, seed)
    interval = pd.DataFrame(
        [
            dict(
                n=len(values),
                seed=seed,
                block_len=BLOCK,
                B=BOOT,
                original_mean=raw.mean(),
                all_share_mean=all_a.mean(),
                mean_difference=values.mean(),
                ci_low=ci[0],
                ci_high=ci[1],
            )
        ]
    )
    interval.to_csv(out.parent / "TCI_interval.csv", index=False, float_format="%.17g")
    save("Table_8A", original_simulation)
    save("Table_8B", source)

    shock = pd.read_csv(base / "shock_models.csv")
    save("Table_9", shock)
    temp = shock.query("horizon==5")
    save("Table_C1", temp)
    temp = shock.query("horizon==20")
    save("Table_C2", temp)

    temp = sensitivity.query('analysis=="mean"')
    save("Table_E1", temp)
    temp = sensitivity.query('analysis=="mean" and level=="fitted"')
    save("Table_E2", temp)
    temp = sensitivity.query('analysis=="paired"')
    save("Table_E3", temp)

    rows = []
    for rep, label in [
        ("raw", "Original"),
        ("allA", "All Share adjusted"),
        ("hs300", "CSI 300 adjusted"),
        ("pca1", "PCA one-factor"),
    ]:
        g = pca[pca.representation == rep]
        rows.append(
            dict(
                representation=label,
                row_mean_NET=g.financial_J_Row_NET.mean(),
                row_positive_percent=100 * (g.J_sign_row > 0).mean(),
                row_top5_windows=int(g.J_top5_row.sum()),
                scalar_mean_NET=g.financial_J_Scalar_NET.mean(),
                scalar_positive_percent=100 * (g.J_sign_scalar > 0).mean(),
                scalar_top5_windows=int(g.J_top5_scalar.sum()),
            )
        )
    save("Table_5", pd.DataFrame(rows))
    pca_summary(pca).to_csv(
        out.parent / "PCA_summary.csv", index=False, float_format="%.17g"
    )

    panel = pd.DataFrame(pca_stats).T
    panel = panel.rename_axis("metric")
    panel = panel.reset_index()
    save("Table_F1", panel)

    panel, frequencies = bootstrap_tables(paired)
    baseline = paired[paired.kind == "baseline"].set_index("date")
    rows = []
    for date, g in panel.groupby("date", sort=False):
        stats = g.set_index("metric")
        rows.append(
            dict(
                date=date,
                original_overlap=int(baseline.loc[date, "overlap_rs"]),
                cross_rule_mean=stats.loc["D_RS", "mean"],
                row_own_mean=stats.loc["D_R", "mean"],
                scalar_own_mean=stats.loc["D_S", "mean"],
                cross_minus_row_mean=stats.loc["delta_vs_row", "mean"],
                cross_minus_row_empirical_lower=stats.loc["delta_vs_row", "lo"],
                cross_minus_row_empirical_upper=stats.loc["delta_vs_row", "hi"],
                cross_minus_scalar_mean=stats.loc["delta_vs_scalar", "mean"],
                cross_minus_scalar_empirical_lower=stats.loc["delta_vs_scalar", "lo"],
                cross_minus_scalar_empirical_upper=stats.loc["delta_vs_scalar", "hi"],
            )
        )
    save("Table_6", pd.DataFrame(rows))
    save("Table_F2", panel)
    save("Table_F3", frequencies)
    temp = paired.drop(
        columns=["denominator_cv", "offdiag_abs_median", "pc1_covariance_share"]
    )
    temp = temp.query('kind=="lag_check"').copy()
    baseline = paired[paired.kind == "baseline"]
    baseline = baseline.set_index("date")
    pos = temp.columns.get_loc("window_start")
    temp.insert(pos, "n_sectors", len(NODES))
    temp.insert(pos + 1, "representation", "raw")
    temp.insert(pos + 2, "intercept", True)
    for name, field in [
        ("original_top5_row", "top5_row"),
        ("original_top5_scalar", "top5_scalar"),
        ("baseline_overlap_rs", "overlap_rs"),
    ]:
        values = []
        for date in temp.date:
            values.append(baseline.loc[date, field])
        pos = temp.columns.get_loc("overlap_row0")
        temp.insert(pos, name, values)
    save("Table_F4", temp)

    common, pairs, groups = summaries(forecasts)
    panel = pd.DataFrame(common).T
    panel = panel.rename_axis("model")
    panel = panel.reset_index()
    save("Table_10", panel)

    rows = []
    for (a, b), stats in pairs.items():
        row = dict(rule1=a, rule2=b)
        row.update(stats)
        rows.append(row)
    panel = pd.DataFrame(rows)
    save("Table_G1", panel)

    rows = []
    nested = []
    for (rule, h), stats in groups.items():
        row = dict(rule=rule, horizon=h)
        row.update(stats["nominee"])
        rows.append(row)
        row = dict(rule=rule, horizon=h)
        row.update(stats["mse"])
        for key, value in stats["outcome"].items():
            row["outcome_" + key] = value
        nested.append(row)
    save("Table_11", pd.DataFrame(rows))
    save("Table_G2", pd.DataFrame(nested))

    stats, comparison = alert_summary(alerts)
    panel = pd.DataFrame(stats).T
    panel = panel.rename_axis("model")
    panel = panel.reset_index()
    save("Table_12", panel)
    grouped = alerts.groupby("model", sort=False)
    thresholds = grouped[["severe_event_threshold", "alert_score_threshold"]].first()
    thresholds = thresholds.reset_index()
    h1 = panel.merge(thresholds, on="model")
    train = alerts[alerts.split == "train"]
    train_alerts = train.groupby("model", sort=False).alert.sum()
    h1["train_alerts"] = h1.model.map(train_alerts).astype(int)
    h1 = h1.rename(
        columns={"alerts": "test_alerts", "alert_score_threshold": "alert_threshold"}
    )
    h1 = h1[
        [
            "model",
            "alert_threshold",
            "train_alerts",
            "test_alerts",
            "TP",
            "FP",
            "FN",
            "TN",
            "severe_event_threshold",
        ]
    ]
    save("Table_H1", h1)

    rows = []
    for model, values in comparison.items():
        for metric, stats in values.items():
            row = dict(model=model, metric=metric)
            row.update(stats)
            rows.append(row)
    save("Table_H2", pd.DataFrame(rows))

    plots = {}
    plots["figure4"] = grid
    plots["figureD1"] = horizons
    plots["figure2"] = reliability
    plots["figure5"] = original_simulation
    plots["finance_net"] = pd.read_csv(base / "net.csv")
    return plots
