"""
Stock Market Intelligence Pipeline — Airflow DAG
-------------------------------------------------
Orchestrates daily ingestion, processing, validation,
reporting and cloud storage for 8 NSE stocks.

Schedule : Weekdays at 09:00 IST
Pipeline :
    [ingest_live_quotes]     ─┐
                               ├──> [pyspark_processing]
    [ingest_historical_data] ─┘         │
                                   [validate_data]
                                         │
                                   [generate_report]
                                         │
                                     [s3_upload]
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, '/home/deepak/stock_pipeline')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE       = '/home/deepak/stock_pipeline'
RAW_DIR    = f'{BASE}/data/raw'
VIZ_DIR    = f'{BASE}/data/viz'
PROC_DIR   = f'{BASE}/data/processed'
REPORT_DIR = f'{BASE}/data/reports'

STOCKS_NSE = [
    'TCS', 'RELIANCE', 'INFY', 'HDFCBANK',
    'WIPRO', 'LT', 'SBIN', 'BAJFINANCE',
]
STOCKS_YF = [
    'TCS.NS', 'RELIANCE.NS', 'INFY.NS', 'HDFCBANK.NS',
    'WIPRO.NS', 'LT.NS', 'SBIN.NS', 'BAJFINANCE.NS',
]

# ---------------------------------------------------------------------------
# Default Arguments
# ---------------------------------------------------------------------------
default_args = {
    'owner':            'deepak',
    'depends_on_past':  False,
    'retries':          2,
    'retry_delay':      timedelta(minutes=5),
    'start_date':       datetime(2026, 3, 1),
    'email_on_failure': False,
    'email_on_retry':   False,
}


# ---------------------------------------------------------------------------
# Task 1 — Ingest Live Quotes
# ---------------------------------------------------------------------------
def task_ingest_live_quotes(**context):
    """
    Fetch real-time NSE stock quotes via nsetools
    and persist them to data/raw/live_quotes.csv.
    """
    import pandas as pd
    from datetime import datetime as dt
    from nsetools import Nse

    nse     = Nse()
    records = []

    for stock in STOCKS_NSE:
        try:
            q     = nse.get_quote(stock)
            intra = q.get('intraDayHighLow', {})
            records.append({
                'Symbol':     stock,
                'Last_Price': q.get('lastPrice', 0),
                'Open':       q.get('open', 0),
                'High':       intra.get('max', 0),
                'Low':        intra.get('min', 0),
                'Close':      q.get('previousClose', 0),
                'Change_Pct': q.get('pChange', 0),
                'Volume':     q.get('totalTradedVolume', 0),
                'Timestamp':  dt.now().isoformat(),
            })
        except Exception as exc:
            print(f"  WARNING [{stock}]: {exc}")

    os.makedirs(RAW_DIR, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(f'{RAW_DIR}/live_quotes.csv', index=False)

    print(f"Task 1 complete — {len(df)} quotes saved.")
    return len(df)


# ---------------------------------------------------------------------------
# Task 2 — Ingest Historical Data
# ---------------------------------------------------------------------------
def task_ingest_historical(**context):
    """
    Download 5-year daily OHLCV data from Yahoo Finance
    (one ticker at a time to avoid column duplication)
    and persist to data/raw/historical_data.csv.
    """
    import pandas as pd
    import yfinance as yf
    from datetime import datetime as dt

    all_data = []

    for name, ticker in zip(STOCKS_NSE, STOCKS_YF):
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(
                start='2020-01-01',
                end=dt.now().strftime('%Y-%m-%d'),
            )
            df = df.reset_index()
            df = df[['Date', 'Open', 'High',
                     'Low', 'Close', 'Volume']]
            df['Date']   = (
                pd.to_datetime(df['Date'])
                  .dt.strftime('%Y-%m-%d')
            )
            df['Symbol'] = name
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = df[col].round(2)
            df['Volume'] = (
                df['Volume'].fillna(0).astype(int)
            )
            all_data.append(df)
            print(f"  {name}: {len(df):,} rows")
        except Exception as exc:
            print(f"  WARNING [{name}]: {exc}")

    final = pd.concat(all_data, ignore_index=True)

    # Guard against column duplication
    assert len(final.columns) == 7, (
        f"Schema error: expected 7 columns, "
        f"got {len(final.columns)} — {final.columns.tolist()}"
    )

    os.makedirs(RAW_DIR, exist_ok=True)
    final.to_csv(f'{RAW_DIR}/historical_data.csv', index=False)

    print(f"Task 2 complete — {len(final):,} records saved.")
    return len(final)


# ---------------------------------------------------------------------------
# Task 3 — PySpark Processing
# ---------------------------------------------------------------------------
def task_pyspark_processing(**context):
    """
    Invoke the standalone PySpark preprocessing script.
    Computes MA7, MA30, RSI, Volatility, Daily Return,
    MA Signal and saves Parquet + CSV outputs.
    """
    import subprocess

    result = subprocess.run(
        ['python3', f'{BASE}/pyspark_preprocessing.py'],
        capture_output=True,
        text=True,
    )

    # Always surface stdout for Airflow logs
    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            f"PySpark preprocessing failed — "
            f"exit code {result.returncode}"
        )

    print("Task 3 complete.")


# ---------------------------------------------------------------------------
# Task 4 — Validate Data
# ---------------------------------------------------------------------------
def task_validate_data(**context):
    """
    Run data-quality assertions on the processed feature CSV.
    Fails the pipeline immediately if any check is violated.
    """
    import pandas as pd

    df = pd.read_csv(f'{VIZ_DIR}/stock_features.csv')

    checks = {
        'Total Records': len(df),
        'Stock Count':   df['Symbol'].nunique(),
        'Null Values':   int(df.isnull().sum().sum()),
        'Bullish Days':  len(df[df['MA_Signal'] == 'Bullish']),
        'Bearish Days':  len(df[df['MA_Signal'] == 'Bearish']),
        'Neutral Days':  len(df[df['MA_Signal'] == 'Neutral']),
    }

    print("Data Quality Report")
    print("-" * 35)
    for key, val in checks.items():
        print(f"  {key:<18}: {val:,}")
    print("-" * 35)

    assert checks['Total Records'] > 1000, \
        "Validation failed: record count below threshold."
    assert checks['Stock Count'] == 8, \
        "Validation failed: one or more stocks missing."
    assert checks['Null Values'] == 0, \
        "Validation failed: null values detected."

    print("Task 4 complete — all quality checks passed.")
    return checks


# ---------------------------------------------------------------------------
# Task 5 — Generate Daily Report
# ---------------------------------------------------------------------------
def task_generate_report(**context):
    """
    Join live quotes with the latest MA signals to produce
    a plain-text daily summary and write it to data/reports/.
    """
    import pandas as pd
    from datetime import datetime as dt

    df   = pd.read_csv(f'{VIZ_DIR}/stock_features.csv')
    live = pd.read_csv(f'{RAW_DIR}/live_quotes.csv')

    today = dt.now().strftime('%Y-%m-%d')
    lines = [
        f"DAILY STOCK REPORT — {today}",
        "=" * 55,
        f"  {'Symbol':<12} {'Price (Rs.)':>12} "
        f"{'Change %':>10}  {'Signal':<10}",
        "-" * 55,
    ]

    for _, row in live.iterrows():
        subset = df[df['Symbol'] == row['Symbol']]
        signal = (
            subset['MA_Signal'].iloc[-1]
            if len(subset) > 0 else 'N/A'
        )
        lines.append(
            f"  {row['Symbol']:<12} "
            f"{row['Last_Price']:>12,.2f} "
            f"{row['Change_Pct']:>+9.2f}%  "
            f"{signal:<10}"
        )

    lines.append("=" * 55)
    report_text = "\n".join(lines)
    print(report_text)

    os.makedirs(REPORT_DIR, exist_ok=True)
    fname = dt.now().strftime("%Y%m%d")
    with open(f'{REPORT_DIR}/report_{fname}.txt', 'w') as fh:
        fh.write(report_text)

    print("Task 5 complete — daily report saved.")
    return report_text


# ---------------------------------------------------------------------------
# Task 6 — AWS S3 Upload
# ---------------------------------------------------------------------------
def task_s3_upload(**context):
    """
    Sync today's viz/ and reports/ outputs to AWS S3
    using the project's aws_storage module.
    """
    from aws_storage import daily_sync
    daily_sync()
    print("Task 6 complete — S3 sync done.")


# ---------------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id='stock_market_pipeline',
    default_args=default_args,
    description=(
        'Daily NSE stock market ingestion, '
        'processing and storage pipeline.'
    ),
    schedule_interval='0 9 * * 1-5',
    catchup=False,
    tags=['stocks', 'bigdata', 'pyspark'],
) as dag:

    t1 = PythonOperator(
        task_id='ingest_live_quotes',
        python_callable=task_ingest_live_quotes,
        provide_context=True,
    )
    t2 = PythonOperator(
        task_id='ingest_historical_data',
        python_callable=task_ingest_historical,
        provide_context=True,
    )
    t3 = PythonOperator(
        task_id='pyspark_processing',
        python_callable=task_pyspark_processing,
        provide_context=True,
    )
    t4 = PythonOperator(
        task_id='validate_data',
        python_callable=task_validate_data,
        provide_context=True,
    )
    t5 = PythonOperator(
        task_id='generate_report',
        python_callable=task_generate_report,
        provide_context=True,
    )
    t6 = PythonOperator(
        task_id='s3_upload',
        python_callable=task_s3_upload,
        provide_context=True,
    )

    # t1 and t2 run in parallel, then sequential
    [t1, t2] >> t3 >> t4 >> t5 >> t6
