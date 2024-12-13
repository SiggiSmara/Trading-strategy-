import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time
import math

from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
   pass

session = CachedLimiterSession(
   limiter=Limiter(RequestRate(2, Duration.SECOND*5)),  # max 2 requests per 5 seconds
   bucket_class=MemoryQueueBucket,
   backend=SQLiteCache("yfinance.cache"),
)

def save_stock_data(stocks:list, start_date:datetime, end_date:datetime, interval:str="1d", add_data:bool = True):
        data_path = Path(__file__).parent.parent / "stock_data" / f"yf_df_{interval}.sav"

        def get_yf(stock:str, start_date:datetime, end_date:datetime):
            df = get_yfinance_data(stock=stock, start_date=start_date, end_date=end_date, interval=interval)
            return df
        
        def save_yf(df1, df2):
            df2 = pd.concat([df2, df1], axis=0).drop_duplicates(
                subset = ['date', 'stock'], 
                keep = 'last').reset_index(drop=True)
            df2.to_pickle(data_path)
            return df2

        if add_data and data_path.is_file():
            curr_df = pd.read_pickle(data_path)
            print(f"pickle read with shape {curr_df.shape}")
        else:
            curr_df = pd.DataFrame()

        for stock in stocks:
            print(f"Updating finance data for {stock}@{start_date}-{end_date}")
            df = get_yf(stock=stock, start_date=start_date, end_date=end_date)
            curr_df = save_yf(df, curr_df)
                

def get_yfinance_data(stock:str, start_date:datetime, end_date:datetime, interval:str = "1d") -> pd.DataFrame:
        
    # get the ticker
    ticker = yf.Ticker(stock, session=session)
    # print(f"interval: {interval}")
    if interval == "1h":
        # make sure the day limit is not reached
        today = datetime.now()
        if (today - start_date).days >= 730:
            start_date = today - timedelta(729)
            # print(f"start_date changed to: {start_date}")
        else:
            pass
            # print(f"start date within {(start_date - today).days} of today")
        if (today - end_date).days >= 730:
            end_date = today
            # print(f"end_date changed to: {end_date}")
        else:
            pass
            # print(f"end_date date within {(today - end_date).days} of today")
        
    df = ticker.history(start = start_date, end = end_date, interval = interval)
    
    # convert date index to a date column
    df = df.rename_axis("date").reset_index()
    # remove timezone
    df["date"] = [x.replace(tzinfo=None) for x in df["date"]]

    # make columns lowercase
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

    spy = get_yfinance_data(
        stock=base_stock, 
        start_date=start_date, 
        end_date=end_date, 
        interval="1h"
    )
    theo_shape = spy.shape
    print(theo_shape)
    save_stock_data(stocks=stocks, start_date=start_date, end_date=end_date, interval="1d")

    # "V", "DAL", 