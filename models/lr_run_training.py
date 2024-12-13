"""
LR training 
author - Kaneel Senevirathne
date - 1/13/2022
"""

# from td.client import TDClient
import requests, time, re, os
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import numpy as np
import datetime
plt.style.use('grayscale')

from scipy import linalg
import math
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import time
from datetime import datetime
import os
import sys
import pickle

from pathlib import Path

#append path
current_dir = os.getcwd()
sys.path.append(current_dir)

from stock_utils import stock_utils
from stock_utils.stock_utils import reg_cols
from stock_data import get_data
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

class LR_training:

    def __init__(self, model_version, threshold = 0.9, start_date = None, end_date = None):

        self.model_version = model_version
        self.threshold = threshold
        
        self.my_path:Path = Path(os.getcwd())
        if start_date:
            self.start_date = start_date
        if end_date:
            self.end_date = end_date

        #get stock ticker symbols
        dow = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC',\
        'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH',\
        'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS']
        sp500 = pd.read_csv("companies.csv")
        sp = list(sp500['Ticker'])
        stocks = dow + sp[:20]
        self.stocks = list(np.unique(stocks))

        #main dataframe
        self.main_df = pd.DataFrame(columns = ['volume', 'normalized_value', 'target', "stock"] + reg_cols)

        #init models
        self.scaler = MinMaxScaler()
        self.lr = LogisticRegression()

        #run logistic regresion
        self.fetch_data()
        self.create_train_test()
        self.fit_model()
        self.confusion_matrix()
        self.save_model()

    def fetch_data(self):
        """
        get train and test data
        """ 
        picklep_path = self.my_path / "models" / "finance_data.sav"
        if picklep_path.is_file():
            self.main_df = pd.read_pickle(picklep_path)
        else:
            start_date=datetime(year=2006, month=1, day=1)
            end_date=datetime(year=2023, month=1, day=1) 
            get_data.save_stock_data(stocks=self.stocks, start_date=start_date, end_date=end_date, interval="1d")
            end_date=datetime(year=2021, month=1, day=1)
            for stock in self.stocks:
                try: 
                    print(f"Creating training data for {stock}")
                    df = stock_utils.create_train_data(
                        stock, 
                        n = 3, 
                        end_date = end_date, 
                        delta_days=(end_date - start_date).days
                    )
                    df["stock"] = stock
                    self.main_df = pd.concat([self.main_df, df], axis=0)
                except Exception as e:
                    print(e)
                    # exit()
                    pass
            self.main_df.to_pickle(self.my_path / "models" / "finance_data.sav") 
        self.main_df = self.main_df.drop("stock", axis=1)

        print(f'{self.main_df.shape[0]} samples were fetched from the database..')

    def create_train_test(self):
        """
        create train and test data
        """
        # , random_state = 3
        # self.main_df = self.main_df.sample(frac = 1). reset_index(drop = True)
        self.main_df['target'] = self.main_df['target'].astype('category')
        
        y = self.main_df.pop('target').to_numpy()
        y = y.reshape(y.shape[0], 1)
        x = self.scaler.fit_transform(self.main_df)

        #test train split
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(x, y, \
            test_size = 0.1,  shuffle = True) #random_state = 50,

        print(f'Created test and train data... {len(self.train_y)} vs {len(self.test_y)}')

    def fit_model(self):

        print('Training model...')
        self.lr.fit(self.train_x, self.train_y)
        
        #predict the test data
        self.predictions = self.lr.predict(self.test_x)
        self.score = self.lr.score(self.test_x, self.test_y)
        # print(self.predictions )
        print(f'Logistic regression model score: {self.score}')

        #preds with threshold
        self.predictions_proba = self.lr._predict_proba_lr(self.test_x)
        # print(self.predictions_proba)
        self.predictions_proba_thresholded = self._threshold(self.predictions_proba, self.threshold)
      
    def confusion_matrix(self):
        cm = confusion_matrix(self.test_y, self.predictions)
        self.cmd = ConfusionMatrixDisplay(cm)
        
        # print(self.predictions_proba_thresholded)
        cm_thresholded = confusion_matrix(self.test_y, self.predictions_proba_thresholded)
        self.cmd_thresholded = ConfusionMatrixDisplay(cm_thresholded)

        
    def _threshold(self, predictions, threshold):

        def one_thres(pred, thres):
            retval = [1 if x > thres else 0 for x in pred]
            # print(prob_thresholded)
            if sum(retval) == 0:
                retval = 0
            elif sum(retval) > 1:
                retval = 0 #np.where(pred == np.max(pred))[0][0]
            else:
                retval = np.where(np.array(retval) > 0)[0][0]
            return int(retval)
        prob_thresholded = [one_thres(x,thres=threshold) for x in predictions]
        return prob_thresholded
        # prob_thresholded = [0 if x > threshold else 1 for x in predictions[:, 0]]

        # return np.array(prob_thresholded)

    def save_model(self):

        #save models
        saved_models_dir = self.my_path / 'saved_models'
        model_file = f'lr_{self.model_version}.sav'
        model_dir = saved_models_dir /  model_file
        pickle.dump(self.lr, open(model_dir, 'wb'))

        scaler_file = f'scaler_{self.model_version}.sav'
        scaler_dir = saved_models_dir /  scaler_file
        pickle.dump(self.scaler, open(scaler_dir, 'wb'))

        print(f'Saved the model and scaler in {saved_models_dir}')
        cm_path = self.my_path /  'results' / 'Confusion Matrices'
        
        #save cms
        plt.figure()
        self.cmd.plot()
        plt.savefig( cm_path / f'cm_{self.model_version}.jpg')

        plt.figure()
        self.cmd_thresholded.plot()
        plt.savefig(cm_path / f'cm_thresholded_{self.model_version}.jpg')
        print(f'Figures saved in {cm_path}')

import argparse

if __name__ == "__main__":
    run_lr = LR_training('v4')
    
   

