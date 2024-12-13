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

regressions = [3, 5, 10, 20, 40]
reg_cols = [f"{reg}_reg" for reg in regressions]
reg_col="normalized_value"
# reg_col="close"

data_path = Path(os.getcwd()) / "stock_data" / "yf_df_1d"
if data_path.is_file():
    yf_df_1d = pd.read_pickle(data_path)
else:
    yf_df_1d = pd.DataFrame()

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

def n_day_regression(n, df, idxs, reg_col:str = "close"):
    """
    n day regression.
    """
    #variable
    _varname_ = f'{n}_reg'
    df[_varname_] = np.nan

    for idx in idxs:
        if idx> n:
            
            # y = df[reg_col][idx - n +1: idx+1]
            # print(idx)
            # print(y)
            y = df[reg_col][idx - n+1: idx+1].to_numpy()
            
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
        data_path = Path(__file__).parent.parent / "stock_data" / "yf_df_1d.sav"

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
            yf_df_1d = pd.read_pickle(data_path)
            print(f"pickle read with shape {yf_df_1d.shape}")
        else:
            yf_df_1d = pd.DataFrame()
        # print(yf_df_1d.shape)
        # print(yf_df_1d)
        for stock in stocks:
            if yf_df_1d.shape[1] == 0 or "date" not in yf_df_1d.columns or "stock" not in yf_df_1d.columns:
                print(f"Getting new finance data for {stock}@{start_date}-{end_date}")
                df = get_yf(stock=stock, start_date=start_date, end_date=end_date)
                yf_df_1d = save_yf(df, yf_df_1d)
            elif not all((yf_df_1d["date"][yf_df_1d["stock"]== stock].max() >= end_date-timedelta(1), 
                 yf_df_1d["date"][yf_df_1d["stock"]== stock].min() <= start_date)):
                print(f"Getting updating finance data for {stock}@{start_date}-{end_date}")
                # print(yf_df_1d["date"][yf_df_1d["stock"]== stock])
                # print((yf_df_1d["date"][yf_df_1d["stock"]== stock].max(skipna=True) >= end_date-timedelta(1)))
                # print((yf_df_1d["date"][yf_df_1d["stock"]== stock].min(skipna=True), start_date))
                # print(f"Getting missing finance data for {stock}@{start_date}-{end_date}")
                df = get_yf(stock=stock, start_date=start_date, end_date=end_date)
                yf_df_1d = save_yf(df, yf_df_1d)
                
            


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
    data_path = Path(__file__).parent.parent / "stock_data" / "yf_df_1d.sav"
    if data_path.is_file():
        yf_df_1d = pd.read_pickle(data_path)
    else:
        raise ValueError("No yf_df_1d found")
    df = yf_df_1d[yf_df_1d["stock"] == stock]
    if df.shape[0] == 0:
        raise ValueError(f"Stock {stock} not found")
    else:
        # print(f"stock found for {stock}, df shape is: {df.shape}")
        pass
    
    if delta_days == 0:
        df = df[df["date"] == end_date]
    else:
        # end = np.where(df['date'] >= end_date)
        # print(f"end_date {end_date} looked for: {end[0]}")
        # print(f"min date: {df['date'].min()}, max date: {df['date'].max()}")
        end = np.where(df['date'] >= end_date)[0][0]
        start = end - delta_days
        # print((start,end))
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
    data_path = Path(__file__).parent.parent / "stock_data" / "yf_df_1d.sav"
    if data_path.is_file():
        yf_df_1d:pd.DataFrame = pd.read_pickle(data_path)
    else:
        raise ValueError("No yf_df_1d found")
    ser = yf_df_1d["date"].drop_duplicates().sort_values(ascending=True).reset_index(drop=True)
    
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
    # print(data.shape)
    if data.shape[0] == 0:
        raise ValueError("No data found")
    #add the noramlzied value function and create a new column
    data['normalized_value'] = data.apply(lambda x: normalized_values(x.high, x.low, x.close), axis = 1)
    
    #column with local minima and maxima
    idx_with_mins = argrelextrema(data.close.values, np.less_equal, order = n)[0]
    idx_with_maxs = argrelextrema(data.close.values, np.greater_equal, order = n)[0]
    data['loc_min'] = data.iloc[idx_with_mins]['close']
    # data['loc_min'] = data.iloc[argrelextrema(data.close.values, np.less_equal, order = n)[0]]['close']
    data['loc_max'] = data.iloc[idx_with_maxs]['close']

    #idx with mins and max
    # idx_with_mins = np.where(data['loc_min'] > 0)[0]
    # idx_with_maxs = np.where(data['loc_max'] > 0)[0]
    # data = data.reset_index(drop=True)
    idx_no_min_max = [x for x in data.index if x not in idx_with_mins and x not in idx_with_maxs]
    idx_no_min_max = np.random.choice(idx_no_min_max, int(len(idx_no_min_max) * 0.035), replace=False)
    data["loc_no_mm"] = data.iloc[idx_no_min_max]['close']

    return data, idx_with_mins, idx_with_maxs, idx_no_min_max

def create_train_data(stock, end_date, delta_days:int, n = 10):

    #get data to a dataframe
    data, idxs_with_mins, idxs_with_maxs, idx_no_min_max = get_data(stock, end_date=end_date, delta_days=delta_days, n=n)
    #create regressions for 3, 5 and 10 days

    
    for reg in regressions:
        data = n_day_regression(n=reg, df=data, idxs=list(idxs_with_mins) + list(idxs_with_maxs), reg_col=reg_col)

    # print(data.shape)
    _data_ = data[(data['loc_min'] > 0) | (data['loc_max'] > 0) ].reset_index(drop = True) #| (data["loc_no_mm"] > 0)
    mins = _data_['loc_min']
    maxs = _data_['loc_max']
    tweens = _data_['loc_no_mm']
    # List of conditions
    conditions = [
        #(tweens > 0)
        #, 
        (mins > 0)
        , (maxs > 0)
    ]
    # List of values to return
    choices  = [
        #0
        #, 
        0
        , 1
    ]
    _data_['target']  = np.select(conditions, choices)
    #create a dummy variable for local_min (0) and max (1)
    # _data_['target'] = [1 if x > 0 else 0 for x in _data_.loc_max]
    
    #columns of interest
    cols_of_interest = ['volume', 'normalized_value', 'target'] + reg_cols 
    _data_ = _data_[cols_of_interest]
    # print(_data_.shape)
    return _data_.dropna(axis = 0)

def create_test_data_lr(stock, end_date, delta_days:int, n = 10):
    """
    this function create test data sample for logistic regression model
    """
    #get data to a dataframe
    data, _, _, _ = get_data(stock, end_date=end_date, delta_days=delta_days, n=n)
    
    idxs = [len(data)-1, ]
    # print(idxs)
    #create regressions for 3, 5 and 10 days
    for reg in regressions:
        data = n_day_regression(n=reg, df=data, idxs=idxs, reg_col=reg_col)
    
    
    cols = ['close', 'volume', 'normalized_value'] + reg_cols 
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
    for reg in regressions:
        data = n_day_regression(n=reg, df=data, idxs=idxs, reg_col=reg_col)
        
    #create a column for predicted value
    data['pred'] = np.nan

    #get data
    cols = ['volume', 'normalized_value'] + reg_cols 
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


