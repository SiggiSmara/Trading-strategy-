import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time

def save_stock_data(stocks:list, start_date:datetime, end_date:datetime, theo_shape:tuple):
        data_path = Path(__file__).parent.parent / "stock_data" / "curr_df.sav"

        def get_yf(stock:str, start_date:datetime, end_date:datetime):
            df = get_yfinance_data(stock=stock, start_date=start_date, end_date=end_date)
            time.sleep(2)
            return df
        
        def save_yf(df1, df2):
            df2 = pd.concat([df2, df1], axis=0).drop_duplicates(
                subset = ['date', 'stock'], 
                keep = 'last').reset_index(drop=True)
            df2.to_pickle(data_path)
            return df2

        if data_path.is_file():
            curr_df = pd.read_pickle(data_path)
            print(f"pickle read with shape {curr_df.shape}")
        else:
            curr_df = pd.DataFrame()
        # print(curr_df.shape)
        # print(curr_df)
        for stock in stocks:
            if curr_df.shape[1] == 0 or "date" not in curr_df.columns or "stock" not in curr_df.columns:
                print(f"Getting new finance data for {stock}@{start_date}-{end_date}")
                df = get_yf(stock=stock, start_date=start_date, end_date=end_date)
                curr_df = save_yf(df, curr_df)
                if df.shape != theo_shape:
                    print(f"unexpected shape returned: {df.shape}")
                    print(f"min date: {df['date'].min()}, max date: {df['date'].max()}")
            elif not all((curr_df["date"][curr_df["stock"]== stock].max() >= end_date-timedelta(1), 
                 curr_df["date"][curr_df["stock"]== stock].min() <= start_date)):
                print(f"Updating finance data for {stock}@{start_date}-{end_date}")
                # print(curr_df["date"][curr_df["stock"]== stock])
                # print((curr_df["date"][curr_df["stock"]== stock].max(skipna=True) >= end_date-timedelta(1)))
                # print((curr_df["date"][curr_df["stock"]== stock].min(skipna=True), start_date))
                # print(f"Getting missing finance data for {stock}@{start_date}-{end_date}")
                df = get_yf(stock=stock, start_date=start_date, end_date=end_date)
                curr_df = save_yf(df, curr_df)
                if df.shape != theo_shape:
                    print(f"unexpected shape returned: {df.shape}")
                    print(f"min date: {df['date'].min()}, max date: {df['date'].max()}")
                
            


def get_yfinance_data(stock:str, start_date:datetime, end_date:datetime) -> pd.DataFrame:
    
    if start_date is None:
        start_date = datetime(2007, 1, 1)
    
    # get the ticker
    ticker = yf.Ticker(stock)
    df = ticker.history(start = start_date, end = end_date, interval = "1d")

    #rename the index and reset it, also remove timezone
    df = df.rename_axis("date").reset_index()
    df["date"] = [x.replace(tzinfo=None) for x in df["date"]]

    #make columns lowercase
    df.columns = [x.lower() for x in df.columns]
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df["stock"] = stock
    return df



if __name__ == "__main__":
    dow = ["SPY", 'AXP', 'AMGN', 'AAPL', 'BA',  'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC',
        'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH',
        'CRM', 'VZ', 'WBA', 'WMT', 'DIS']
    other = ['AMD', 'MU', 'ABT', 'AAL',    'BAC', 'PNC', 'C', 'EBAY', 'AMZN', 'GOOG',   'UAL', 'DAL', 'V',
        'FDX', 'MCD', 'PEP', 'SNAP', 'ATVIX', 'ANTMX',  'META',]  # 
    stocks = list(set(dow + other))

    start_date = datetime(2007, 1, 1)
    end_date = datetime(2022, 1, 1)
    base_stock = "SPY"

    spy = get_yfinance_data(stock=base_stock, start_date=start_date, end_date=end_date)
    theo_shape = spy.shape
    print(theo_shape)
    save_stock_data(stocks=stocks, start_date=start_date, end_date=end_date, theo_shape=theo_shape)

    "V", "DAL", 