import os
import pandas as pd
import numpy as np

def fetch_and_process_historical_data(
    series_ids=["SP500", "NASDAQCOM"],
    output_path="data/historical_sp500_10yr.csv"
):
    print(f"=== Downloading 10-Year Historical Market Data from FRED (2016-2026) ===")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    dfs = []
    for s in series_ids:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"
        d = pd.read_csv(url, index_col="observation_date", parse_dates=True)
        d[s] = pd.to_numeric(d[s], errors='coerce')
        dfs.append(d)
        
    df = pd.concat(dfs, axis=1).ffill().bfill().dropna()
    
    # Calculate daily percentage returns
    returns_df = df.pct_change().dropna()
    
    df.to_csv(output_path)
    returns_path = output_path.replace(".csv", "_returns.csv")
    returns_df.to_csv(returns_path)
    
    print(f"[Success] Downloaded {len(df)} trading days of historical market data.")
    print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Prices saved to: {output_path}")
    print(f"Returns saved to: {returns_path}\n")
    return df, returns_df

if __name__ == "__main__":
    fetch_and_process_historical_data()
