"""
stock backtester to test the model given a dataset. 
author - Kaneel Senevirathne
date - 1/13/2022
"""

import numpy as np
from stock_utils.simulator import simulator
from stock_utils.stock_utils import get_stock_price
from models import lr_inference
from datetime import datetime
from datetime import timedelta
import time
from pathlib import Path
# from td.client import TDClient
import pandas as pd
from models.lr_inference import LR_v1_predict, LR_v1_sell
from stock_utils.stock_utils import save_stock_data, get_trading_days
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore")
import os
import pickle
from tqdm import tqdm

class backtester(simulator):

    def __init__(self, stocks_list, model, capital, start_date, end_date, threshold = 0.99, sell_perc = 0.04, hold_till = 5,\
         stop_perc = 0.005):
        
        super().__init__(capital) #initialize simulator

        self.stocks = stocks_list
        self.model = model
        self.start_date = start_date
        self.day:datetime = start_date
        self.end_date = end_date  
        self.status = 'buy' #the status says if the backtester is in buy mode or sell mode
        self.threshold = threshold
        self.sell_perc = sell_perc
        self.hold_till = hold_till
        self.stop_perc = stop_perc

        #current directory
        self.current_dir = Path(os.getcwd())
        results_dir = self.current_dir / 'results'
        folder_name = f'{str(self.model.__name__)}_{self.threshold}_{self.hold_till}'
        self.folder_dir = results_dir / folder_name
        if not self.folder_dir.exists():
            #create a new folder
            self.folder_dir.mkdir(parents=True, exist_ok=True)
      
    def backtest(self):
        """
        start backtesting
        """
        delta = timedelta(days = 1)
        
        #progress bar to track progress
        trading_days = get_trading_days(start_date=self.start_date, end_date=self.end_date)
        total_days = len(trading_days)
        # total_days = (self.end_date - self.start_date).days
        d = 0
        pbar = tqdm(desc = 'Progress', total = total_days)
        self.day = trading_days[0]
        for self.day in trading_days:
            # print(self.day.weekday())
            # print(self.day)
            # if self.day.weekday() < 5:
            #daily scanner dict
            self.daily_scanner = {}  
            if self.status == 'buy':
                #scan stocks for the day
                self.scanner()
                if list(self.daily_scanner.keys()) != []:
                    recommended_stock = list(self.daily_scanner.keys())[0]
                    recommended_price = list(self.daily_scanner.values())[0][2]
                    self.buy(recommended_stock, recommended_price, self.day) #buy stock
                    # print(f'Bought {recommended_stock} for {recommended_price} on the {self.day}')
                    self.status = 'sell' #change the status to sell
                else:
                    # print('No recommendations')
                    pass
            else: #if the status is sell
                #get stock price on the day
                stocks = [key for key in self.buy_orders.keys()]
                # print(f"stocks in play:{stocks}")
                for s in stocks:
                    recommended_action, current_price = LR_v1_sell(s, self.buy_orders[s][3], self.buy_orders[s][0], self.day, \
                        self.sell_perc, self.hold_till, self.stop_perc)
                    if recommended_action == "SELL":
                        # print(f'Sold {s} for {current_price} on {self.day}')
                        self.sell(s, current_price, self.buy_orders[s][1], self.day)
                        self.status = 'buy'              
            #go to next day
            self.day += delta
            d += 1
            # print(d)
            pbar.update(1)
        pbar.close()
        #sell the final stock and print final capital also print stock history 
        self.print_bag()
        self.print_summary() 
        self.save_results()      
        return

    def get_stock_data(self, stock, back_to = 40):
        """
        this function queries to td database and get data of a particular stock on a given day back to certain amount of days
        (default is 30). 
        """
        #get start and end dates
        end = self.day
        start = self.day - timedelta(days = back_to)     
        # prediction, prediction_thresholded, close_price = LR_v1_predict(stock, start, end, threshold = 0.5)
        prediction, prediction_thresholded, close_price = self.model(stock, start, end, self.threshold)
        return prediction[0], prediction_thresholded, close_price

    def scanner(self):
        """
        scan the stocks to find good stocks
        """
        for stock in self.stocks:
            # try:#to ignore the stock if no data is available. #for staturdays or sundays etc
            prediction, prediction_thresholded, close_price = self.get_stock_data(stock)
            #if prediction greater than
            if prediction_thresholded < 1: #if prediction is zero
                self.daily_scanner[stock] = (prediction, prediction_thresholded, close_price)
            
            # except Exception as e:
            #     print(e)
            #     pass

        def take_first(elem):
            return elem[1]

        self.daily_scanner = OrderedDict(sorted(self.daily_scanner.items(), key = take_first, reverse = True))

    def save_results(self):
        """
        save history dataframe create figures and save
        """
        #save csv file
        self.folder_dir.mkdir(parents=True,  exist_ok=True)
        results_df_path = self.folder_dir / 'history_df.csv'

        self.history_df.to_csv(results_df_path, index = False)
        
        #save params and results summary
        results_summary_path = self.folder_dir /  'results_summary'
        results_summary = [self.initial_capital, self.total_gain]
        params_path = self.folder_dir /  'params'
        params = [self.threshold, self.hold_till, self.sell_perc, self.stop_perc, self.start_date, self.end_date]
        
        with open(results_summary_path, 'wb') as fp:
            pickle.dump(results_summary, fp)
        with open(params_path, 'wb') as fp:
            pickle.dump(params, fp)
    
    
            

if __name__ == "__main__":
    #stocks list
    # dow = ['AXP', 'AMGN', 'AAPL', 'BA', 'META', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC',
    #     'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH',
    #     'CRM', 'VZ', 'WBA', 'WMT', 'DIS']
    # other = ['AMD', 'MU', 'ABT', 'AAL',    'BAC', 'PNC', 'C', 'EBAY', 'AMZN', 'GOOG',  'ANTMX', 'ATVIX', 'SNAP', 'V', 'UAL', 'DAL', 
    #     'FDX', 'MCD', 'PEP', ] #'ANTM', 'ATVI', 'FB', 'SNAP', 'TWTR', 'V', 'UAL', 'DAL', 

    # stocks = list(np.unique(dow + other))

    dow = ["SPY", 'AXP', 'AMGN', 'AAPL', 'BA',  'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC',
        'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH',
        'CRM', 'VZ', 'WBA', 'WMT', 'DIS']
    other = ['AMD', 'MU', 'ABT', 'AAL',    'BAC', 'PNC', 'C', 'EBAY', 'AMZN', 'GOOG',   'UAL', 'DAL', 'V',
        'FDX', 'MCD', 'PEP', ]  # 'SNAP', 'ATVIX', 'ANTMX',  'META',
    stocks = list(set(dow + other))
    # save_stock_data(stocks=stocks, start_date=datetime(2006, 1, 3), end_date=datetime(2022, 1, 1))
    current_dir = Path(os.getcwd())
    origs_dir =current_dir / 'results' / "origs"
    model = "LR_v1_predict_0.9_10"
    # model = "LR_v1_predict_0.95_21"
    # model = "LR_v1_predict_0.7_3"
    print(model)
    try:
        threshold, hold_till, sell_perc, stop_perc, start_date, end_date = pickle.load(open(origs_dir / model / "params", "r"))
    except UnicodeDecodeError:
        try:
            threshold, hold_till, sell_perc, stop_perc, start_date, end_date = pickle.load(open(origs_dir / model / "params", "rb"), encoding='windows-1252')
        except Exception as e:
            print(e)
            exit(1)
    # threshold, hold_till, sell_perc, stop_perc, start_date, end_date = [0.9, 10, 0.05, 0.05, datetime(2021, 1, 1), datetime(2021, 12, 31)]
    print([threshold, hold_till, sell_perc, stop_perc, start_date, end_date])
    back = backtester(
        dow, 
        LR_v1_predict, 3000, 
        start_date, 
        end_date, 
        threshold = threshold, 
        sell_perc = sell_perc, 
        hold_till = hold_till,
        stop_perc = stop_perc
    )
    back.backtest()

    


    
