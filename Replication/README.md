# Systems Replication Code

## Requirements

Python 3.11.9.

```sh
python -m pip install -r requirements.txt
```

## Data

The sample is 2013-01-04–2025-12-31. `INPUTS` contains the 30 CSMAR Excel exports listed in `config/inputs.json`, including stock trading, industry classification and market index data, with their original headers and metadata rows.

Alternatively, provide both CSV inputs:

- `RETURNS.csv`: `Trddt` and A,B,CA,CB,CC,CD,CE,CF,CG,CH,CI,CJ,D,E,F,G,H,I,K,L,M,N,OTH,J.
- `INDEX.csv`: `date,ret_all_a,ret_hs300,close_all_a,close_hs300`.

Returns are decimals. Licensed CSMAR access is required; original data are not included. Database restrictions also apply to sharing derived data.

## Run

Use a new output directory:

```sh
python run.py --csmar-dir INPUTS --output RESULTS
```

Or use the paired CSV inputs:

```sh
python run.py --returns-csv RETURNS.csv --market-csv INDEX.csv --output RESULTS
```

## Results

- `RESULTS/tables/`: 25 numerical tables.
- `RESULTS/figures/`: Figures 2–5 and D1, PNG at 700 dpi.
- `RESULTS/replications/`: Monte Carlo, source recovery, sensitivity and Bootstrap replication data.
- `RESULTS/TCI_interval.csv`: paired Original minus All Share Row TCI mean and 95% interval (24 sectors, H=10).
- `RESULTS/PCA_summary.csv`: PCA comparisons, variation shares and rank diagnostics reported in the text and Appendix F.
