import pandas as pd
import os
import json
import pandas as pd
import yfinance as yf
import numpy as np


def read_snp500_data(folder_path="data/snp500_data/yfinance"):
    """
    Reads the hierarchy JSON and historical CSV files for S&P 500 constituents (and index),
    as well as precomputed industry- and sector-level CSVs, from `folder_path` and returns:
      - hierarchy: the dict loaded from "hierarchy_SP500.json"
      - series:    a dict mapping each symbol/industry/sector → list of its “Close” values
                   (in chronological order) from the CSVs in folder_path.

    Expects this folder structure:
      folder_path/
        ├─ hierarchy_SP500.json
        ├─ GSPC_history.csv           (the index)
        ├─ historical_data/           (all "<TICKER>_history.csv" or "<TICKER>.csv")
        ├─ industry/                  (all "<INDUSTRY>_series.csv")
        └─ sector/                    (all "<SECTOR>_series.csv")
    """
    # 1. Load hierarchy from JSON
    json_path = os.path.join(folder_path, "hierarchy_SP500.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Hierarchy file not found: {json_path}")
    with open(json_path, "r") as f:
        hierarchy = json.load(f)

    # 2. Prepare the series dict
    series = {}

    # 2.a. Load the index (e.g. ^GSPC) from "GSPC_history.csv"
    index_file = os.path.join(folder_path, "GSPC_history.csv")
    if os.path.isfile(index_file):
        df_index = pd.read_csv(index_file, parse_dates=["Date"], index_col="Date")
        if "Close" not in df_index.columns:
            raise KeyError(f"'Close' column not found in {index_file}")
        series["^GSPC"] = np.array(df_index["Close"].tolist())
    else:
        raise FileNotFoundError(f"Index file not found: {index_file}")

    # 2.b. Load all individual‐ticker CSVs from "historical_data/"
    hist_folder = os.path.join(folder_path, "historical_data")
    if not os.path.isdir(hist_folder):
        raise FileNotFoundError(f"Folder not found: {hist_folder}")

    for fname in os.listdir(hist_folder):
        if not fname.lower().endswith(".csv"):
            continue
        base = os.path.splitext(fname)[0]
        # Remove "_history" suffix if present
        if base.endswith("_history"):
            symbol = base[:-len("_history")]
        else:
            symbol = base

        file_path = os.path.join(hist_folder, fname)
        df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
        if "Close" not in df.columns:
            raise KeyError(f"'Close' column not found in {file_path}")
        series[symbol] = np.array(df["Close"].tolist())

    # 2.c. Load all industry‐level CSVs from "industry/"
    industry_folder = os.path.join(folder_path, "industry")
    if os.path.isdir(industry_folder):
        for fname in os.listdir(industry_folder):
            if not fname.lower().endswith(".csv"):
                continue
            base = os.path.splitext(fname)[0]
            # Remove "_series" suffix if present
            if base.endswith("_series"):
                industry = base[:-len("_series")]
            else:
                industry = base

            file_path = os.path.join(industry_folder, fname)
            df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
            if "Close" not in df.columns:
                raise KeyError(f"'Close' column not found in {file_path}")
            series[industry] = np.array(df["Close"].tolist())
    else:
        # If the folder doesn't exist, just skip without error
        pass

    # 2.d. Load all sector‐level CSVs from "sector/"
    sector_folder = os.path.join(folder_path, "sector")
    if os.path.isdir(sector_folder):
        for fname in os.listdir(sector_folder):
            if not fname.lower().endswith(".csv"):
                continue
            base = os.path.splitext(fname)[0]
            # Remove "_series" suffix if present
            if base.endswith("_series"):
                sector = base[:-len("_series")]
            else:
                sector = base

            file_path = os.path.join(sector_folder, fname)
            df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
            if "Close" not in df.columns:
                raise KeyError(f"'Close' column not found in {file_path}")
            
            # As numpy array
            series[sector] = np.array(df["Close"].tolist())

    else:
        # If the folder doesn't exist, just skip without error
        pass

    # Clean hierarchy to remove empty lists
    hierarchy = {k: v for k, v in hierarchy.items() if v}

    return hierarchy, series




def download_snp_data(folder_path = "data/snp500_data"):

    # 1. Fetch S&P 500 constituents (symbol, sector, industry) from Wikipedia
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(wiki_url)
    # The first table on this page is the list of constituents
    sp500_table = tables[0]

    # Keep only the columns we need: Symbol, Security (company name), GICS Sector, GICS Sub-Industry
    sp500 = sp500_table[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    sp500.rename(columns={
        "Symbol": "Ticker",
        "Security": "Company",
        "GICS Sector": "Sector",
        "GICS Sub-Industry": "Industry"
    }, inplace=True)

    # 2. Build the hierarchy dictionary
    hierarchy = {}

    # Top level: the index name
    index_name = "S&P 500"
    hierarchy[index_name] = []

    # Collect unique sectors
    sectors = sorted(sp500["Sector"].unique())
    hierarchy[index_name] = sectors

    # For each sector, collect its industries
    for sector in sectors:
        sector_industries = sorted(sp500.loc[sp500["Sector"] == sector, "Industry"].unique())
        hierarchy[sector] = sector_industries

        # For each industry in this sector, collect its tickers
        for industry in sector_industries:
            tickers = sorted(
                sp500.loc[
                    (sp500["Sector"] == sector) & (sp500["Industry"] == industry), 
                    "Ticker"
                ].tolist()
            )
            hierarchy[industry] = tickers

            # For completeness, give each stock an empty list of children
            for t in tickers:
                hierarchy[t] = []

    # 3. Download historical data for index (^GSPC) and all tickers over the last two years
    # Define the “last two years” relative to today:
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=2)

    # Create a folder to save CSVs if it doesn’t exist
    os.makedirs("historical_data", exist_ok=True)

    # 3.a Download S&P 500 index history
    index_ticker = "^GSPC"
    idx = yf.Ticker(index_ticker)
    hist_index = idx.history(start=start_date, end=end_date)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    index_csv_path = os.path.join(folder_path, f"{index_ticker.replace('^','')}_history.csv")
    hist_index.to_csv(index_csv_path)
    print(f"Saved index history to {index_csv_path}")

    # 3.b Download history for each S&P 500 stock
    # (This may take 5–10 minutes to run, depending on your connection.)
    for ticker in sp500["Ticker"].tolist():
        try:
            tkr = yf.Ticker(ticker)
            hist = tkr.history(start=start_date, end=end_date)
            if not hist.empty:
                file_name = f"{ticker}_history.csv"
                file_path = os.path.join(folder_path, "historical_data", file_name)
                hist.to_csv(file_path)
                print(f"  └─ {ticker}: saved to {file_path}")
            else:
                print(f"  └─ {ticker}: no data returned, skipped.")
        except Exception as e:
            print(f"  └─ {ticker}: ERROR: {e}")

    # 4. Save the hierarchy dictionary as JSON
    json_path = os.path.join(folder_path, "hierarchy_SP500.json")
    with open(json_path, "w") as f:
        json.dump(hierarchy, f, indent=2)
    print(f"Hierarchy dictionary saved to {json_path}")

    create_intermediate_level_series(hierarchy, folder_path)



def create_intermediate_level_series(hierarchy, folder_path):
    """
    Builds cap-weighted close-price series for each sector and each industry,
    based on individual stock CSVs in folder_path + "/historical_data". Saves
    the resulting time series as CSVs under:
        folder_path + "/sector/<sector>_series.csv"
        folder_path + "/industry/<industry>_series.csv"
    Uses each stock’s current market cap (via yfinance) to compute static weights.

    Parameters
    ----------
    hierarchy : dict
        A mapping where:
          • hierarchy["S&P 500"] = [sector1, sector2, ...]
          • hierarchy[sector] = [industry1, industry2, ...]
          • hierarchy[industry] = [ticker1, ticker2, ...]
          • hierarchy[ticker] = []
    folder_path : str
        Root folder containing:
          • "historical_data/"  (folder with "<TICKER>_history.csv" or "<TICKER>.csv")
          • (this function will create) "sector/"  and "industry/" subfolders for outputs.

    Notes
    -----
    • Assumes each CSV in "historical_data/" has columns: "Date", "Open", "High", "Low",
      "Close", "Volume", etc.  We only read "Date" & "Close".  
    • Uses yfinance to fetch each ticker’s current marketCap (float-adjusted).
    • The weight for ticker i in a group is:
           weight_i = marketCap_i / sum(marketCap_j for j in group)
      and does not vary over time.
    • The cap-weighted series for a group on date d is:
           sum_i[ weight_i * Close_i(d) ]
      missing values on a given date are dropped before summing.
    • The output CSV for group G is:
           Date, Close
      where "Close" is the cap-weighted index value for G.
    """

    # Paths
    hist_folder = os.path.join(folder_path, "historical_data")
    sector_folder = os.path.join(folder_path, "sector")
    industry_folder = os.path.join(folder_path, "industry")

    # Ensure subfolders exist
    os.makedirs(sector_folder, exist_ok=True)
    os.makedirs(industry_folder, exist_ok=True)

    # 1) Process sectors
    sectors = hierarchy.get("^GSPC", [])
    for sector in sectors:
        # Gather all tickers under this sector (from every industry)
        tickers = []
        for industry in hierarchy.get(sector, []):
            tickers.extend(hierarchy.get(industry, []))

        # Fetch market caps via yfinance
        market_caps = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                cap = info.get("marketCap", None)
                if cap is None:
                    raise KeyError(f"No 'marketCap' for {ticker}")
                market_caps[ticker] = float(cap)
            except Exception as e:
                print(f"Warning: Could not fetch marketCap for {ticker}: {e}")
        # Remove any tickers lacking market cap
        market_caps = {t: mc for t, mc in market_caps.items() if mc > 0}
        if not market_caps:
            print(f"  ⚠️ No valid market caps for sector '{sector}', skipping.")
            continue

        # Compute static weights
        total_cap = sum(market_caps.values())
        weights = {t: mc / total_cap for t, mc in market_caps.items()}

        # Read each ticker’s Close series into a DataFrame
        dfs = []
        for ticker in weights:
            # Possible file names: "<ticker>_history.csv" or "<ticker>.csv"
            fname1 = f"{ticker}_history.csv"
            fname2 = f"{ticker}.csv"
            path1 = os.path.join(hist_folder, fname1)
            path2 = os.path.join(hist_folder, fname2)

            csv_path = None
            if os.path.isfile(path1):
                csv_path = path1
            elif os.path.isfile(path2):
                csv_path = path2
            else:
                print(f"  ⚠️ Historical CSV not found for {ticker}, skipping.")
                continue

            df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date", usecols=["Date", "Close"])
            df = df.rename(columns={"Close": ticker})
            dfs.append(df)

        if not dfs:
            print(f"  ⚠️ No valid CSV data for sector '{sector}', skipping.")
            continue

        # Merge on Date (inner join to drop dates where no data at all)
        merged = pd.concat(dfs, axis=1).sort_index()

        # Multiply each column by its weight, then sum across columns row-wise
        weight_series_df = merged.multiply(pd.Series(weights), axis=1)
        sector_close = weight_series_df.sum(axis=1).to_frame(name="Close")

        # Save to CSV
        sector_csv = os.path.join(sector_folder, f"{sector}_series.csv")
        sector_close.to_csv(sector_csv, index_label="Date")
        print(f"Saved sector series for '{sector}' → {sector_csv}")

    # 2) Process industries
    # Iterate over every sector to enumerate its industries
    for sector in sectors:
        for industry in hierarchy.get(sector, []):
            tickers = hierarchy.get(industry, [])

            # Fetch market caps via yfinance
            market_caps = {}
            for ticker in tickers:
                try:
                    info = yf.Ticker(ticker).info
                    cap = info.get("marketCap", None)
                    if cap is None:
                        raise KeyError(f"No 'marketCap' for {ticker}")
                    market_caps[ticker] = float(cap)
                except Exception as e:
                    print(f"Warning: Could not fetch marketCap for {ticker}: {e}")
            # Remove any tickers lacking market cap
            market_caps = {t: mc for t, mc in market_caps.items() if mc > 0}
            if not market_caps:
                print(f"  ⚠️ No valid market caps for industry '{industry}', skipping.")
                continue

            # Compute static weights
            total_cap = sum(market_caps.values())
            weights = {t: mc / total_cap for t, mc in market_caps.items()}

            # Read each ticker’s Close series into a DataFrame
            dfs = []
            for ticker in weights:
                fname1 = f"{ticker}_history.csv"
                fname2 = f"{ticker}.csv"
                path1 = os.path.join(hist_folder, fname1)
                path2 = os.path.join(hist_folder, fname2)

                csv_path = None
                if os.path.isfile(path1):
                    csv_path = path1
                elif os.path.isfile(path2):
                    csv_path = path2
                else:
                    print(f"  ⚠️ Historical CSV not found for {ticker}, skipping.")
                    continue

                df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date", usecols=["Date", "Close"])
                df = df.rename(columns={"Close": ticker})
                dfs.append(df)

            if not dfs:
                print(f"  ⚠️ No valid CSV data for industry '{industry}', skipping.")
                continue

            # Merge on Date and compute weighted sum
            merged = pd.concat(dfs, axis=1).sort_index()
            weight_series_df = merged.multiply(pd.Series(weights), axis=1)
            industry_close = weight_series_df.sum(axis=1).to_frame(name="Close")

            # Save to CSV
            industry_csv = os.path.join(industry_folder, f"{industry}_series.csv")
            industry_close.to_csv(industry_csv, index_label="Date")
            print(f"Saved industry series for '{industry}' → {industry_csv}")

    

if __name__ == "__main__":
    download_snp_data(folder_path="data/snp500_data/yfinance")
    print("S&P 500 data download complete.")

    hierarchy, series = read_snp500_data(folder_path="data/snp500_data/yfinance")

