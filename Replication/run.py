import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csmar-dir", type=Path)
    parser.add_argument("--returns-csv", type=Path)
    parser.add_argument("--market-csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("RESULTS"))
    args = parser.parse_args()
    if args.csmar_dir:
        if args.returns_csv or args.market_csv:
            parser.error("choose CSMAR or paired CSV inputs")
    elif not (args.returns_csv and args.market_csv):
        parser.error("provide --csmar-dir or both CSV inputs")
    for path in [args.csmar_dir, args.returns_csv, args.market_csv]:
        if path is not None and not path.exists():
            raise FileNotFoundError(path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    work = output.parent / (output.name + "_work")
    work.mkdir(exist_ok=False)
    try:
        import data
        import network
        import simulation
        import extra
        import forecast
        import tables
        import figures

        if args.csmar_dir:
            data.convert(args.csmar_dir, work)
            import duckdb

            files = json.loads(
                (Path(__file__).parent / "config/inputs.json").read_text()
            )["files"]
            paths = [
                str(work / "stock_daily" / Path(f["name"]).with_suffix(".parquet"))
                for f in files
                if f["name"].startswith("TRD_")
            ]
            with duckdb.connect() as con:
                con.from_parquet(paths).create_view("stocks")
                con.execute(
                    "SELECT DISTINCT Trddt AS date FROM stocks WHERE Trddt BETWEEN DATE '2013-01-01' AND DATE '2025-12-31' ORDER BY date"
                ).df().to_csv(work / "trading_calendar.csv", index=False)
            records = data.index_candidates(args.csmar_dir, work)
            data.index_extract(records, work)
            data.index_merge(work)
            data.portfolios(args.csmar_dir, work)
        else:
            shutil.copyfile(args.returns_csv, work / "returns.csv")
            shutil.copyfile(args.market_csv, work / "index.csv")
        import numpy as np
        import pandas as pd

        returns = pd.read_csv(work / "returns.csv")
        market = pd.read_csv(work / "index.csv")
        dates = pd.to_datetime(returns.Trddt.astype(str).str[:10], format="%Y-%m-%d")
        market_dates = pd.to_datetime(market.date, format="%Y-%m-%d")
        if (
            dates.isna().any()
            or not dates.is_unique
            or not dates.is_monotonic_increasing
        ):
            raise ValueError(
                "returns.csv: dates must be unique and strictly increasing"
            )
        if market_dates.isna().any() or not market_dates.is_unique:
            raise ValueError("index.csv: dates must be unique")
        if dates.min() < pd.Timestamp("2013-01-01") or dates.max() > pd.Timestamp(
            "2025-12-31"
        ):
            raise ValueError("returns.csv: dates must be within 2013-2025")
        if not dates.isin(market_dates).all():
            raise ValueError("index.csv: missing industry trading dates")
        if not np.isfinite(returns[network.NODES].to_numpy(float)).all():
            raise ValueError("returns.csv: nonfinite industry returns")
        if not np.isfinite(market[["ret_all_a", "ret_hs300"]].to_numpy(float)).all():
            raise ValueError("index.csv: nonfinite market returns")
        systems = network.estimate(work)
        network.horizons(work, systems)
        group_models = forecast.original(work)
        network.grid(work, systems)
        replications = output / "replications"
        replications.mkdir()
        original_simulation, draws = simulation.monte_carlo(replications)
        source = extra.source_recovery(draws, replications)
        sensitivity = extra.sensitivity(draws, replications)
        rows, models = extra.boot_base(work, systems)
        bootstrap = extra.run_bootstrap(rows, models)
        bootstrap.to_csv(
            replications / "bootstrap.csv", index=False, float_format="%.17g"
        )
        pca, pca_stats = extra.pca(work, systems)
        predictions, train, test, scores = extra.predict(work, group_models)
        alerts = extra.alerts(train, test, scores)
        plots = tables.export(
            work,
            original_simulation,
            source,
            sensitivity,
            bootstrap,
            pca,
            predictions,
            alerts,
            output / "tables",
            pca_stats,
        )
        figures.draw(plots, output / "figures")
    finally:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
