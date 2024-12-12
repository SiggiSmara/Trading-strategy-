"""
stock backtester to test the model given a dataset. 
author - Kaneel Senevirathne
date - 1/13/2022
"""

from doctest import OutputChecker
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from stock_utils.stock_utils import timestamp, create_train_data, get_data, create_test_data_lr, get_stock_price
from datetime import timedelta
import time
from pathlib import Path
import os

base_path = Path(os.getcwd())

def load_LR(model_version):
    #add path to the saved models folder and then model version 'eg: f'//saved_models//lr_{model_version}.sav'
    file = base_path / "saved_models" / f"lr_{model_version}.sav"
    loaded_model = pickle.load(open(file, 'rb'))

    return loaded_model

def load_scaler(model_version):

    # add path to //saved_models//scaler_{model_version}.sav'
    file = base_path / "saved_models" / f"scaler_{model_version}.sav"
    loaded_model = pickle.load(open(file, 'rb'))

    return loaded_model

def _threshold(probs, threshold):
    """
    Inputs the probability and returns 1 or 0 based on the threshold
    """
    prob_thresholded = [0 if x > threshold else 1 for x in probs[:, 0]]

    return np.array(prob_thresholded)

#create model and scaler instances
scaler = load_scaler('v3')
lr = load_LR('v3')

def LR_v1_predict(stock, start_date, end_date, threshold = 0.98):
    """
    this function predict given the data
    """
    
    #create input
    # print(f"LR_v1_predict, getting LR stock values for {stock}@{start_date}-{end_date}")
    data = create_test_data_lr(stock, end_date=end_date, delta_days=(end_date-start_date).days)
    # time.sleep(0.2)
    #get close price of final date
    # close_price = data['close'].values[-1]
    # print(close_price)
    close_price = get_stock_price(stock, end_date)
    #get input data to model
    input_data = data[['volume', 'normalized_value', '3_reg', '5_reg', '10_reg', '20_reg']]
    input_data = input_data.to_numpy()[-1].reshape(1, -1)
    #scale input data
    input_data_scaled = scaler.transform(input_data)
    prediction = lr._predict_proba_lr(input_data_scaled)
    prediction_thresholded = _threshold(prediction, threshold)
   
    return prediction[:, 0], prediction_thresholded[0], close_price

def LR_v1_sell(stock, buy_date, buy_price, todays_date, sell_perc = 0.1, hold_till = 3, stop_perc = 0.05):
    """
    gets stock price. recommnd to sell if the stock price is higher sell_perc * buy_price + buy_price
    stock - stock ticker symbol
    buy_date - the date the stock was bought
    todays_date - date today
    sell_perc - sell percentage 
    hold_till - how many days to hold from today
    """
    # print(f"LR_v1_sell, getting stock value for {stock}@{todays_date}")
    current_price = get_stock_price(stock, todays_date) #current stock value

    for i in range(1,6):
        last_trading_day = todays_date - timedelta(i)
        try:
            last_price = get_stock_price(stock, last_trading_day) #current stock value
            break
        except ValueError:
            pass
    sell_price = buy_price + buy_price * sell_perc
    stop_price = buy_price - buy_price * stop_perc
    last_perc = 1 - (current_price / last_price) 
    sell_date = buy_date + timedelta(days = hold_till) #the day to sell 
    # time.sleep(0.2) #to make sure the requested transactions per seconds is not exeeded.
    #some times it returns current price as none
    # if (current_price is not None) and ((current_price < stop_price) or (current_price >= sell_price) or (todays_date >= sell_date)):
    if (current_price < stop_price) or (last_perc > stop_perc) or (current_price >= sell_price) or (todays_date >= sell_date):
        return "SELL", current_price #if criteria is met recommend to sell
    else:
        return "HOLD", current_price #if crieteria is not met hold the stock





