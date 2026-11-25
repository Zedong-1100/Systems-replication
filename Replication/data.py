from network import NODES
import csv
import re
import numpy as np
import json
import pyarrow as pa
import pyarrow.parquet as pq
import math
import os
import duckdb
from bisect import bisect_right
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from openpyxl.utils.datetime import from_excel
from openpyxl import load_workbook


def clean(x):
    s = "" if x is None else str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def parse_date(x):
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, (int, float)):
        try:
            d = from_excel(x)
            return d.date() if isinstance(d, datetime) else d
        except (ValueError, TypeError, OverflowError):
            return None
    s = clean(x).replace("/", "-")
    if len(s) == 8 and s.isdigit():
        s = f"{s[:4]}-{s[4:6]}-{s[6:]}"
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def rows_xlsx(path, columns):
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb.active
        ws.reset_dimensions()
        it = ws.iter_rows(values_only=True)
        head = [clean(x) for x in next(it)]
        ix = [head.index(c) for c in columns]
        for row in it:
            if row:
                yield tuple(row[i] if i < len(row) else None for i in ix)
    finally:
        wb.close()


def group24(code):
    m = re.fullmatch("([A-Z])(\\d{2})", clean(code))
    if not m:
        return None
    letter, n = m.group(1), int(m.group(2))
    if letter == "C":
        mapping = [
            (13, 16, "CA"),
            (17, 19, "CB"),
            (20, 21, "CC"),
            (22, 24, "CD"),
            (25, 29, "CE"),
            (30, 33, "CF"),
            (34, 37, "CG"),
            (38, 39, "CH"),
            (40, 40, "CI"),
            (41, 43, "CJ"),
        ]
        for lo, hi, name in mapping:
            if lo <= n <= hi:
                return name
        return None
    if letter in "OPQRS":
        return "OTH"
    if letter == "J":
        return "J"
    return letter if letter in "ABDEFGHIKLMN" else None


def convert(raw, work):
    OUT_DIR = work / "stock_daily"
    BATCH_SIZE = 25000
    FIELDS = [
        "Stkcd",
        "Trddt",
        "Clsprc",
        "Dnshrtrd",
        "Dnvaltrd",
        "Dsmvosd",
        "Dsmvtll",
        "Dretwd",
        "Dretnd",
        "Trdsta",
    ]
    NUMERIC_FIELDS = [
        "Clsprc",
        "Dnshrtrd",
        "Dnvaltrd",
        "Dsmvosd",
        "Dsmvtll",
        "Dretwd",
        "Dretnd",
    ]
    SCHEMA = pa.schema(
        [
            pa.field("Stkcd", pa.string()),
            pa.field("Trddt", pa.date32()),
            *[pa.field(x, pa.float64()) for x in NUMERIC_FIELDS],
            pa.field("Trdsta", pa.string()),
        ]
    )

    def clean_code(value):
        if value is None:
            return None
        s = str(value).strip()
        if s.endswith(".0") and s[:-2].isdigit():
            s = s[:-2]
        if not s:
            return None
        return s.zfill(6) if s.isdigit() else s

    def clean_date(value):
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if value is None:
            return None
        # stock dates can be YYYYMMDD numbers or Excel serial dates
        if isinstance(value, (int, float)):
            s = str(int(value))
            if len(s) == 8 and s.isdigit():
                try:
                    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
                except ValueError:
                    return None
            try:
                result = from_excel(value)
                return result.date() if isinstance(result, datetime) else None
            except (ValueError, TypeError, OverflowError):
                return None
        s = str(value).strip().replace("/", "-")
        if len(s) == 8 and s.isdigit():
            s = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        try:
            return date.fromisoformat(s[:10])
        except ValueError:
            return None

    def clean_number(value):
        if value is None:
            return None
        try:
            x = float(str(value).replace(",", "").strip())
            return x if math.isfinite(x) else None
        except (ValueError, TypeError):
            return None

    def convert_file(path):
        filename = f"{path.stem}.parquet"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        target = OUT_DIR / filename
        temp = OUT_DIR / (filename + ".tmp")
        wb = load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        ws.reset_dimensions()
        writer = None
        try:
            reader = ws.iter_rows(values_only=True)
            headers = [str(v).strip() for v in next(reader)]
            next(reader, None)
            next(reader, None)
            positions = {}
            buffer = {}
            for name in FIELDS:
                positions[name] = headers.index(name)
            for name in FIELDS:
                buffer[name] = []
            writer = pq.ParquetWriter(str(temp), SCHEMA, compression="zstd")
            for row in reader:
                if not row or all(v is None for v in row):
                    continue
                raw = {}
                for name in FIELDS:
                    pos = positions[name]
                    if pos < len(row):
                        raw[name] = row[pos]
                    else:
                        raw[name] = None
                record = {
                    "Stkcd": clean_code(raw["Stkcd"]),
                    "Trddt": clean_date(raw["Trddt"]),
                    "Trdsta": str(raw["Trdsta"]).strip()
                    if raw["Trdsta"] is not None
                    else None,
                }
                for name in NUMERIC_FIELDS:
                    record[name] = clean_number(raw[name])
                for name in FIELDS:
                    buffer[name].append(record[name])
                if len(buffer["Stkcd"]) >= BATCH_SIZE:
                    table = pa.Table.from_pydict(buffer, schema=SCHEMA)
                    writer.write_table(table)
                    for name in FIELDS:
                        buffer[name].clear()
            if buffer["Stkcd"]:
                table = pa.Table.from_pydict(buffer, schema=SCHEMA)
                writer.write_table(table)
                for name in FIELDS:
                    buffer[name].clear()
        finally:
            if writer is not None:
                writer.close()
            wb.close()
        os.replace(temp, target)

    entries = json.loads((Path(__file__).parent / "config/inputs.json").read_text())[
        "files"
    ]
    files = []
    for e in entries:
        if e["name"].startswith("TRD_"):
            name = Path(e["name"]).with_suffix(".xlsx").name
            files.append(raw / name)
    for file in files:
        convert_file(file)


def portfolios(raw, work):
    SNAPSHOT_XLSX = raw / "STK_LISTEDCOINFOANL.xlsx"
    INDEX_CSV = work / "index.csv"
    STOCK_DIR = work / "stock_daily"
    OUT_DAILY = work / "portfolios.parquet"
    OUT_MATRIX = work / "returns.csv"
    work.mkdir(parents=True, exist_ok=True)
    by_stock = defaultdict(dict)
    for stock, day, code in rows_xlsx(
        SNAPSHOT_XLSX, ["Symbol", "EndDate", "IndustryCode"]
    ):
        d = parse_date(day)
        s = clean(stock).zfill(6)
        if d is None or d.year < 2012 or d.year > 2023:
            continue
        if not re.fullmatch("[036][0-9]{5}", s):
            continue
        g = group24(code)
        if g is None:
            continue
        prev = by_stock[s].get(d.year)
        if prev is not None and prev[0] >= d:
            continue
        by_stock[s][d.year] = d, g, clean(code)
    maprows = []
    for stock, years in by_stock.items():
        available = sorted(years)
        for calyear in range(2013, 2026):
            for phase in (0, 1):
                # Jan-Apr: use t-2; May-Dec: use t-1; start with the 2012 snapshot
                latest_allowed = min(
                    2023, max(2012, calyear - (2 if phase == 0 else 1))
                )
                j = bisect_right(available, latest_allowed) - 1
                if j >= 0:
                    source_year = available[j]
                    _, sector, code = years[source_year]
                    maprows.append((stock, calyear, phase, sector, source_year, code))
    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=4")
    tmp = work / "duckdb_temp41"
    tmp.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory='{str(tmp).replace(chr(39), chr(39) + chr(39))}'")
    try:
        entries = json.loads(
            (Path(__file__).parent / "config/inputs.json").read_text()
        )["files"]
        parquets = []
        for e in entries:
            if e["name"].startswith("TRD_"):
                name = Path(e["name"]).with_suffix(".parquet")
                path = str(STOCK_DIR / name)
                parquets.append(path)
        con.from_parquet(parquets).create_view("stock_input")
        if con.execute(
            "SELECT count(*) FROM (SELECT Stkcd,Trddt FROM stock_input GROUP BY 1,2 HAVING count(*)>1)"
        ).fetchone()[0]:
            raise ValueError("duplicate stock-date keys")
        con.execute(
            "CREATE TABLE classmap(stock VARCHAR, calyear INTEGER, phase INTEGER, sector VARCHAR, source_year INTEGER, code VARCHAR)"
        )
        con.executemany("INSERT INTO classmap VALUES (?, ?, ?, ?, ?, ?)", maprows)
        con.from_parquet(parquets, union_by_name=True).create_view(
            "raw_stocks", replace=True
        )
        # keep returns between -1 and 5
        sql = """
        COPY (
           WITH parsed AS (
             SELECT
               LPAD(CAST(TRY_CAST(Stkcd AS BIGINT) AS VARCHAR),6,'0') AS stock,
               COALESCE(TRY_CAST(Trddt AS DATE),
                   TRY_STRPTIME(CAST(Trddt AS VARCHAR),'%Y%m%d')::DATE) AS d,
               TRY_CAST(Dretwd AS DOUBLE) AS ret,
               TRY_CAST(Dsmvosd AS DOUBLE) AS cap
             FROM raw_stocks
           ), eligible AS (
             SELECT stock, d, ret, cap,
               LAG(cap) OVER (PARTITION BY stock ORDER BY d) AS lag_cap,
               LAG(d) OVER (PARTITION BY stock ORDER BY d) AS previous_day
             FROM parsed
             WHERE d >= DATE '2012-01-01'
               AND d < DATE '2026-01-01'
               AND REGEXP_MATCHES(stock, '^[036][0-9]{5}$')
           ), attached AS (
             SELECT e.d, m.sector, m.source_year,
                    e.ret, e.lag_cap,
                    DATE_DIFF('day',e.previous_day,e.d) AS gap_days
             FROM eligible e
             INNER JOIN classmap m
             ON e.stock=m.stock AND YEAR(e.d)=m.calyear
               AND (CASE WHEN MONTH(e.d)>=5 THEN 1 ELSE 0 END)=m.phase
             WHERE e.d >= DATE '2013-01-01'
               AND e.ret IS NOT NULL AND ISFINITE(e.ret)
               AND e.ret >= -1 AND e.ret <= 5
               AND e.lag_cap > 0 AND ISFINITE(e.lag_cap)
           )
           SELECT d AS Trddt, sector,
                  COUNT(*) AS stock_count,
                  SUM(lag_cap) AS total_lag_cap,
                  SUM(ret*lag_cap)/SUM(lag_cap) AS ret_vw,
                  SUM(CASE WHEN gap_days<=30 THEN ret*lag_cap ELSE 0 END)
                   / NULLIF(SUM(CASE WHEN gap_days<=30 THEN lag_cap ELSE 0 END),0)
                     AS ret_gap30,
                  COUNT(*) FILTER (WHERE gap_days>30) AS long_gap_count,
                  MIN(source_year) AS oldest_source_year,
                  MAX(source_year) AS newest_source_year
           FROM attached
           GROUP BY 1,2
           ORDER BY 1,2
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        dest = str(OUT_DAILY).replace("'", "''")
        con.execute(sql.replace("TO ? (FORMAT", f"TO '{dest}' (FORMAT"))
        mat = defaultdict(dict)
        datarows = con.execute(
            "SELECT Trddt,sector,ret_vw FROM read_parquet(?) ORDER BY Trddt,sector",
            [str(OUT_DAILY)],
        ).fetchall()
        for d, sector, ret in datarows:
            s = d.isoformat() if isinstance(d, date) else str(d)[:10]
            mat[s][sector] = float(ret)
        with INDEX_CSV.open(encoding="utf-8-sig", newline="") as f:
            market = {}
            for r in csv.DictReader(f):
                market[r["date"]] = r
        days = []
        R24 = []
        for d in sorted(market):
            if d not in mat:
                continue
            complete = True
            for s in NODES:
                if s not in mat[d]:
                    complete = False
                    break
            if not complete:
                continue
            vals = [mat[d][s] for s in NODES]
            factors = [float(market[d]["ret_all_a"]), float(market[d]["ret_hs300"])]
            if not np.isfinite(vals).all() or not np.isfinite(factors).all():
                continue
            days.append(d)
            R24.append(vals)
        # require at least 2500 complete trading days
        if len(days) < 2500:
            raise ValueError("insufficient industry sample")
        R24 = np.asarray(R24)
        with OUT_MATRIX.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Trddt"] + NODES)
            for d, r in zip(days, R24):
                w.writerow([d] + r.tolist())
    finally:
        con.close()


TARGETS = "000985", "000300"

SUPPLEMENT = {f"IDX_Idxtrd{n}.xlsx" for n in [111, 222, 333, 444, 555, 666, 777, 888]}


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def index_candidates(raw, work):
    files = json.loads((Path(__file__).parent / "config/inputs.json").read_text())[
        "files"
    ]
    daily = {}
    buchong = {}
    for code in TARGETS:
        daily[code] = {}
    for code in TARGETS:
        buchong[code] = {}
    for entry in files:
        if not entry["name"].startswith("IDX_Idxtrd"):
            continue
        path = raw / entry["name"]
        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            ws = wb.active
            ws.reset_dimensions()
            rows = ws.iter_rows(values_only=True)
            header = [clean(x) for x in next(rows)]
            pos = {}
            for name in ["Indexcd", "Idxtrd01", "Idxtrd05", "Idxtrd08"]:
                pos[name] = header.index(name)
            for row in rows:
                if not row:
                    continue
                code = clean(row[pos["Indexcd"]]).zfill(6)
                if code not in TARGETS:
                    continue
                day = parse_date(row[pos["Idxtrd01"]])
                if day is None or not 2013 <= day.year <= 2025:
                    continue
                ret = number(row[pos["Idxtrd08"]])
                close = number(row[pos["Idxtrd05"]])
                daily[code].setdefault(day, (ret, close))
                if path.name in SUPPLEMENT:
                    buchong[code].setdefault(
                        day, (ret / 100 if ret is not None else None, close)
                    )
        finally:
            wb.close()
    path = work / "index_candidates.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Indexcd", "date", "return", "close"])
        for code in sorted(daily):
            for day, values in sorted(daily[code].items()):
                writer.writerow([code, str(day), *values])
    return buchong


def index_extract(data, work):
    calendar = work / "trading_calendar.csv"
    with calendar.open(encoding="utf-8", newline="") as f:
        dates = [date.fromisoformat(r["date"]) for r in csv.DictReader(f)]
    with (work / "index_extract.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["date", "ret_all_a", "ret_hs300", "close_all_a", "close_hs300"]
        )
        for day in dates:
            ar, ac = data["000985"].get(day, (None, None))
            hr, hc = data["000300"].get(day, (None, None))
            writer.writerow([str(day), ar, hr, ac, hc])


def index_merge(work):
    codes = {
        "000985": ("ret_all_a", "close_all_a"),
        "000300": ("ret_hs300", "close_hs300"),
    }
    data = {}
    for code in codes:
        data[code] = {}

    def num(value):
        if value is None or str(value).strip() == "":
            return None
        x = float(value)
        if not math.isfinite(x):
            raise ValueError("nonfinite index value")
        return x

    def add(code, day, ret, close):
        if ret is None and close is None:
            return
        if day in data[code]:
            pr, pc = data[code][day]
            if not (
                pr is not None
                and ret is not None
                and abs(pr - ret) < 1e-06
                and pc is not None
                and close is not None
                and abs(pc - close) < 0.01
            ):
                raise ValueError("conflicting index observations")
            return
        data[code][day] = ret, close

    dates = []
    with (work / "index_extract.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            day = row["date"]
            dates.append(day)
            for code, (ret, close) in codes.items():
                add(code, day, num(row[ret]), num(row[close]))
    if len(dates) != len(set(dates)):
        raise ValueError("duplicate index dates")
    dates.sort()
    date_set = set(dates)
    with (work / "index_candidates.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            code, day = row["Indexcd"].strip(), row["date"].strip()
            if code in codes and day in date_set:
                ret = num(row["return"])
                add(
                    code, day, ret / 100 if ret is not None else None, num(row["close"])
                )
    with (work / "index.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["date", "ret_all_a", "ret_hs300", "close_all_a", "close_hs300"]
        )
        for day in dates:
            ar, ac = data["000985"][day]
            hr, hc = data["000300"][day]
            if any(x is None for x in (ar, hr, ac, hc)):
                raise ValueError("incomplete index observations")
            writer.writerow([day, ar, hr, ac, hc])
