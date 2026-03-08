from nsetools import Nse
import yfinance as yf
import pandas as pd
import os
from datetime import datetime

STOCKS_NSE = ['TCS', 'RELIANCE', 'INFY',
              'HDFCBANK', 'WIPRO',
              'LT', 'SBIN', 'BAJFINANCE']

STOCKS_YF  = ['TCS.NS', 'RELIANCE.NS',
              'INFY.NS', 'HDFCBANK.NS',
              'WIPRO.NS', 'LT.NS',
              'SBIN.NS', 'BAJFINANCE.NS']


def get_live_quotes():
    print("\n Fetching live NSE quotes...")
    nse = Nse()
    records = []

    for stock in STOCKS_NSE:
        try:
            q = nse.get_quote(stock)
            intra = q.get('intraDayHighLow', {})
            records.append({
                'Symbol':     stock,
                'Last_Price': q.get('lastPrice', 0),
                'Open':       q.get('open', 0),
                'High':       intra.get('max', 0),
                'Low':        intra.get('min', 0),
                'Close':      q.get('previousClose', 0),
                'Change_Pct': q.get('pChange', 0),
                'Volume':     q.get(
                    'totalTradedVolume', 0),
                'Timestamp':  datetime.now()
                                      .isoformat()
            })
            print(f"   {stock}: "
                  f"₹{q.get('lastPrice',0):,.2f}")
        except Exception as e:
            print(f"   {stock}: {e}")

    os.makedirs('data/raw', exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(
        'data/raw/live_quotes.csv', index=False
    )
    print(f"✅ Live quotes saved: {len(df)} stocks")
    return df


def get_historical_data():
    print("\n📥 Fetching historical data...")

    all_data = []

    # ── Key Fix: Download ONE stock at a time ──
    for stock_nse, stock_yf in zip(
        STOCKS_NSE, STOCKS_YF
    ):
        print(f"  Fetching {stock_nse}...")
        try:
            # Download single stock only
            raw = yf.Ticker(stock_yf)
            df  = raw.history(
                start='2020-01-01',
                end=datetime.now()
                    .strftime('%Y-%m-%d')
            )

            if df.empty:
                print(f"   {stock_nse}: empty!")
                continue

            # Reset index — Date becomes column
            df = df.reset_index()

            # Keep only clean columns
            df = df[['Date', 'Open', 'High',
                     'Low', 'Close', 'Volume']]

            # Clean Date — remove timezone
            df['Date'] = pd.to_datetime(
                df['Date']
            ).dt.date.astype(str)

            # Add symbol
            df['Symbol'] = stock_nse

            # Round prices to 2 decimal
            for c in ['Open','High',
                      'Low','Close']:
                df[c] = df[c].round(2)

            # Reset volume to int
            df['Volume'] = df['Volume'] \
                             .fillna(0) \
                             .astype(int)

            all_data.append(df)
            print(f"   {stock_nse}: "
                  f"{len(df):,} rows | "
                  f"Cols: {df.columns.tolist()}")

        except Exception as e:
            print(f"  {stock_nse}: {e}")

    if not all_data:
        print(" No data fetched!")
        return pd.DataFrame()

    # Stack all stocks vertically (long format)
    final_df = pd.concat(
        all_data, ignore_index=True
    )

    # Final verification
    print(f"\n   Final columns: "
          f"{final_df.columns.tolist()}")
    print(f"   Shape: {final_df.shape}")
    print(f"   Stocks: "
          f"{final_df['Symbol'].unique()}")
    print(f"\n  Sample data:")
    print(final_df.head(3).to_string())

    # Save clean CSV
    os.makedirs('data/raw', exist_ok=True)
    final_df.to_csv(
        'data/raw/historical_data.csv',
        index=False
    )
    print(f"\n Historical data saved: "
          f"{len(final_df):,} records")
    return final_df


# ── Main ───────────────────────────────────────────
if __name__ == "__main__":
    print(" Starting Stock Data Ingestion...")
    print("=" * 50)

    live_df = get_live_quotes()
    hist_df = get_historical_data()

    print("\n" + "=" * 50)
    print("🎉 Ingestion Complete!")
    print(f"   Live quotes:     {len(live_df)}")
    print(f"   Historical rows: {len(hist_df):,}")
    print(f"\n Files saved:")
    print(f"   data/raw/live_quotes.csv")
    print(f"   data/raw/historical_data.csv")