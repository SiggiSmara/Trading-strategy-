# from td.client import TDClient
import requests, time, re, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from pathlib import Path

"""
author - Kaneel Senevirathne
date - 1/8/2022
stock utils for preparing training data.
"""

#TD API - 
TD_API = 'XXXXX' ### your TD ameritrade api key

data_path = Path(os.getcwd()) / "stock_data" / "curr_df"
if data_path.is_file():
    curr_df = pd.read_pickle(data_path)
else:
    curr_df = pd.DataFrame()

def timestamp(dt):
    epoch = datetime.utcfromtimestamp(0)
    return int((dt - epoch).total_seconds() * 1000)


def linear_regression(x, y):
    """
    performs linear regression given x and y. outputs regression coefficient
    """
    #fit linear regression
    lr = LinearRegression()
    lr.fit(x, y)
    
    return lr.coef_[0][0]

def n_day_regression(n, df, idxs):
    """
    n day regression.
    """
    #variable
    _varname_ = f'{n}_reg'
    df[_varname_] = np.nan

    for idx in idxs:
        if idx > n:
            
            y = df['close'][idx - n: idx].to_numpy()
            x = np.arange(0, n)
            #reshape
            y = y.reshape(y.shape[0], 1)
            x = x.reshape(x.shape[0], 1)
            #calculate regression coefficient 
            coef = linear_regression(x, y)
            df.loc[idx, _varname_] = coef #add the new value
            
    return df

def normalized_values(high, low, close):
    """
    normalize the price between 0 and 1.
    """
    #epsilon to avoid deletion by 0
    epsilon = 10e-10
    
    #subtract the lows
    high = high - low
    close = close - low
    return close/(high + epsilon)

# def get_stock_price(stock, date):
#     """
#     returns the stock price given a date
#     """
#     start_date = date - timedelta(days = 10)
#     end_date = date
    
#     #enter url of database
#     url = f'https://api.tdameritrade.com/v1/marketdata/{stock}/pricehistory'

#     query = {'apikey': str(TD_API), 'startDate': timestamp(start_date), \
#             'endDate': timestamp(end_date), 'periodType': 'year', 'frequencyType': \
#             'daily', 'frequency': '1', 'needExtendedHoursData': 'False'}

#     #request
#     results = requests.get(url, params = query)
#     data = results.json()
    
#     try:
#         #change the data from ms to datetime format
#         data = pd.DataFrame(data['candles'])
#         data['date'] = pd.to_datetime(data['datetime'], unit = 'ms')
#         return data['close'].values[-1]
#     except:
#         pass
    
# def get_data(sym, start_date = None, end_date = None, n = 10):

#     #enter url
#     url = f'https://api.tdameritrade.com/v1/marketdata/{sym}/pricehistory'
    
#     if start_date:
#         payload = {'apikey': str(TD_API), 'startDate': timestamp(start_date), \
#             'endDate': timestamp(end_date), 'periodType': 'year', 'frequencyType': \
#             'daily', 'frequency': '1', 'needExtendedHoursData': 'False'}
#     else:
#         payload = {'apikey': str(TD_API), 'startDate': timestamp(datetime(2007, 1, 1)), \
#             'endDate': timestamp(datetime(2020, 12, 31)), 'periodType': 'year', 'frequencyType': \
#             'daily', 'frequency': '1', 'needExtendedHoursData': 'False'}
            
#     #request
#     results = requests.get(url, params = payload)
#     data = results.json()
    
#     #change the data from ms to datetime format
#     data = pd.DataFrame(data['candles'])
#     data['date'] = pd.to_datetime(data['datetime'], unit = 'ms')

#     #add the noramlzied value function and create a new column
#     data['normalized_value'] = data.apply(lambda x: normalized_values(x.high, x.low, x.close), axis = 1)
    
#     #column with local minima and maxima
#     data['loc_min'] = data.iloc[argrelextrema(data.close.values, np.less_equal, order = n)[0]]['close']
#     data['loc_max'] = data.iloc[argrelextrema(data.close.values, np.greater_equal, order = n)[0]]['close']

#     #idx with mins and max
#     idx_with_mins = np.where(data['loc_min'] > 0)[0]
#     idx_with_maxs = np.where(data['loc_max'] > 0)[0]
    
#     return data, idx_with_mins, idx_with_maxs

def save_stock_data(stocks:list, start_date:datetime, end_date:datetime):
        data_path = Path(__file__).parent.parent / "stock_data" / "curr_df.sav"

        def get_yf(stock:str, start_date:datetime, end_date:datetime):
            df = get_yfinance_data(stock=stock, start_date=start_date, end_date=end_date)
            df["stock"] = stock
            time.sleep(1)
            return df
        def save_yf(df1, df2):
            df2 = pd.concat([df2, df1], axis=0).drop_duplicates().reset_index(drop=True)
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
            elif not all((curr_df["date"][curr_df["stock"]== stock].max() >= end_date-timedelta(1), 
                 curr_df["date"][curr_df["stock"]== stock].min() <= start_date)):
                print(f"Getting updating finance data for {stock}@{start_date}-{end_date}")
                # print(curr_df["date"][curr_df["stock"]== stock])
                # print((curr_df["date"][curr_df["stock"]== stock].max(skipna=True) >= end_date-timedelta(1)))
                # print((curr_df["date"][curr_df["stock"]== stock].min(skipna=True), start_date))
                # print(f"Getting missing finance data for {stock}@{start_date}-{end_date}")
                df = get_yf(stock=stock, start_date=start_date, end_date=end_date)
                curr_df = save_yf(df, curr_df)
                
            


def get_yfinance_data(stock:str, start_date:datetime = None, end_date:datetime = None, delta_days:int = None) -> pd.DataFrame:
    
    if start_date is None:
        start_date = datetime(2007, 1, 1)
    
    if end_date is None:
        end_date = datetime.now()
    
    if delta_days is not None:
        start_date = end_date - timedelta(delta_days)
    
    # get the ticker
    ticker = yf.Ticker(stock)
    df = ticker.history(start = start_date, end = end_date, interval = "1d")

    #rename the index and reset it, also remove timezone
    df = df.rename_axis("date").reset_index()
    df["date"] = [x.replace(tzinfo=None) for x in df["date"]]

    #make columns lowercase
    df.columns = [x.lower() for x in df.columns]
    return df

def get_stored_data(stock:str, end_date:datetime, delta_days:int = 0 )->pd.DataFrame:
    data_path = Path(__file__).parent.parent / "stock_data" / "curr_df.sav"
    if data_path.is_file():
        curr_df = pd.read_pickle(data_path)
    else:
        raise ValueError("No curr_df found")
    df = curr_df[curr_df["stock"] == stock]
    if df.shape[0] == 0:
        raise ValueError(f"Stock {stock} not found")
    
    if delta_days == 0:
        df = df[df["date"] == end_date]
    else:
        end = np.where(df['date'] == end_date)[0][0]
        start = end - delta_days
        df = df.iloc[start:end]
    return df.reset_index(drop=True)
    
def get_stock_price(stock, date):
    """
    returns the stock price given a date
    """
    data = get_stored_data(stock=stock, end_date=date)
    # print(data)
    # print(date)
    # print(data['date'] == date)
    # data = data[data['date'] == date]
    # print(len(data['close']))
    if len(data['close']) == 1:
        return data['close'][0]
    else:
        # print(data)
        raise ValueError(f"{stock} has no value for date {date}")
    
def get_trading_days(start_date:datetime, end_date:datetime) -> pd.Series:
    data_path = Path(__file__).parent.parent / "stock_data" / "curr_df.sav"
    if data_path.is_file():
        curr_df:pd.DataFrame = pd.read_pickle(data_path)
    else:
        raise ValueError("No curr_df found")
    ser = curr_df["date"].drop_duplicates().sort_values(ascending=True).reset_index(drop=True)
    
    my_end = np.where(ser <= end_date)
    my_start = np.where(ser >= start_date)
    # print(start_date)
    # print(my_start[0][0])
    # print(end_date)
    # print(my_end[0][-1])
    ser = [ x.to_pydatetime() for x in ser[my_start[0][0]:my_end[0][-1]]]
    
    return ser

def get_data(sym:str, end_date:datetime, delta_days:int, n:int = 10):
    data = get_stored_data(stock=sym, end_date=end_date, delta_days=delta_days)
    if data.shape[0] == 0:
        raise ValueError("No data found")
    #add the noramlzied value function and create a new column
    data['normalized_value'] = data.apply(lambda x: normalized_values(x.high, x.low, x.close), axis = 1)
    
    #column with local minima and maxima
    data['loc_min'] = data.iloc[argrelextrema(data.close.values, np.less_equal, order = n)[0]]['close']
    data['loc_max'] = data.iloc[argrelextrema(data.close.values, np.greater_equal, order = n)[0]]['close']

    #idx with mins and max
    idx_with_mins = np.where(data['loc_min'] > 0)[0]
    idx_with_maxs = np.where(data['loc_max'] > 0)[0]
    
    return data, idx_with_mins, idx_with_maxs

def create_train_data(stock, end_date, delta_days:int, n = 10):

    #get data to a dataframe
    data, idxs_with_mins, idxs_with_maxs = get_data(stock, end_date=end_date, delta_days=delta_days, n=n)
    
    #create regressions for 3, 5 and 10 days
    data = n_day_regression(3, data, list(idxs_with_mins) + list(idxs_with_maxs))
    data = n_day_regression(5, data, list(idxs_with_mins) + list(idxs_with_maxs))
    data = n_day_regression(10, data, list(idxs_with_mins) + list(idxs_with_maxs))
    data = n_day_regression(20, data, list(idxs_with_mins) + list(idxs_with_maxs))
  
    _data_ = data[(data['loc_min'] > 0) | (data['loc_max'] > 0)].reset_index(drop = True)
    
    #create a dummy variable for local_min (0) and max (1)
    _data_['target'] = [1 if x > 0 else 0 for x in _data_.loc_max]
    
    #columns of interest
    cols_of_interest = ['volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg', 'target']
    _data_ = _data_[cols_of_interest]
    
    return _data_.dropna(axis = 0)

def create_test_data_lr(stock, end_date, delta_days:int, n = 10):
    """
    this function create test data sample for logistic regression model
    """
    #get data to a dataframe
    data, _, _ = get_data(stock, end_date=end_date, delta_days=delta_days, n=n)
    
    idxs = np.arange(0, len(data))
    # print(idxs)
    #create regressions for 3, 5 and 10 days
    data = n_day_regression(3, data, idxs)
    data = n_day_regression(5, data, idxs)
    data = n_day_regression(10, data, idxs)
    data = n_day_regression(20, data, idxs)
    
    cols = ['close', 'volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg']
    data = data[cols]
    # print(data)
    data = data.dropna(axis = 0)
    # print(data.shape)
    return data

def predict_trend(stock, _model_, start_date = None, end_date = None, n = 10):

    #get data to a dataframe
    data, _, _ = get_data(stock, start_date, end_date, n)
    
    idxs = np.arange(0, len(data))
    #create regressions for 3, 5 and 10 days
    data = n_day_regression(3, data, idxs)
    data = n_day_regression(5, data, idxs)
    data = n_day_regression(10, data, idxs)
    data = n_day_regression(20, data, idxs)
        
    #create a column for predicted value
    data['pred'] = np.nan

    #get data
    cols = ['volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg']
    x = data[cols]

    #scale the x data
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    for i in range(x.shape[0]):
        
        try:
            data['pred'][i] = _model_.predict(x[i, :])

        except:
            data['pred'][i] = np.nan

    return data


