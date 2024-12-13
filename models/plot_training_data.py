import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import sys, os
import warnings
warnings.filterwarnings("ignore")

#append path
current_dir = os.getcwd()
sys.path.append(current_dir)

from stock_utils import stock_utils
from stock_utils.stock_utils import reg_cols, regressions
from models.lr_inference import LR_v1_predict

my_path = Path(__file__).parent

my_peak_data:pd.DataFrame = pd.read_pickle(my_path / "finance_data.sav")

stocks = list(set(my_peak_data["stock"].to_list()))

start_date=datetime(year=2021, month=1, day=1)
end_date=datetime(year=2022, month=1, day=1)

def get_prediction(stock, end:datetime, back_to:int = 80):
    """
    this function queries to td database and get data of a particular stock on a given day back to certain amount of days
    (default is 30). 
    """
    #get start and end dates
    # print(back_to)
    # print(type(back_to))
    start = end - timedelta(days=back_to)     
    # prediction, prediction_thresholded, close_price = LR_v1_predict(stock, start, end, threshold = 0.5)
    prediction, prediction_thresholded, close_price = LR_v1_predict(stock, start, end, 0.9)
    return prediction, prediction_thresholded, close_price

for stock in stocks:
    print(stock)
    hilo_df = my_peak_data[my_peak_data["stock"] == stock]
    
    # try:
    # print(end_date)
    end_date=datetime(year=2022, month=1, day=1)
    all_data, idx_with_mins, idx_with_maxs, idx_tweens = stock_utils.get_data(
        sym=stock,
        end_date=end_date,
        delta_days=(end_date-start_date).days
    )
    min_data = all_data.iloc[idx_with_mins]
    max_data = all_data.iloc[idx_with_maxs]

    cutoff = 0.5
    # if stock == "XOM":
    max_pred = {"date":[], "close":[]}
    max_false_pred = {"date":[], "close":[]}
    pre_pred1 = {"date":[], "close":[]}
    pre_pred2 = {"date":[], "close":[]}
    min_pred = {"date":[], "close":[]}
    min_false_pred = {"date":[], "close":[]}
    for end_date in max_data["date"]:
        predic, thres, close = get_prediction(stock=stock, end=end_date)
        # print(predic)
        max_predic = np.where(predic == np.max(predic))[0][0]
        categ = max_predic
        categ = thres
        # print(thres)
        if categ == 1:
            max_pred["date"].append(end_date)
            max_pred["close"].append(max_data["close"][max_data["date"]==end_date])
        elif categ == 0:
            max_false_pred["date"].append(end_date)
            max_false_pred["close"].append(max_data["close"][max_data["date"]==end_date])
        # print(f"max_pred: {ret[0]}, {ret[1]} {datetime.strftime(end_date, '%Y-%m-%d')}")
    for end_date in min_data["date"]:
        predic, thres, close = get_prediction(stock=stock, end=end_date)
        max_predic = np.where(predic == np.max(predic))[0][0]
        categ = max_predic
        categ = thres
        # print(ret[0])
        if categ == 0:
            min_pred["date"].append(end_date)
            min_pred["close"].append(min_data["close"][min_data["date"]==end_date])
        elif categ == 1:
            min_false_pred["date"].append(end_date)
            min_false_pred["close"].append(min_data["close"][min_data["date"]==end_date])
        # print(f"min_pred: {ret[0]}, {ret[1]} {datetime.strftime(end_date, '%Y-%m-%d')}")
    for end_date in all_data["date"]:
        predic, thres, close = get_prediction(stock=stock, end=end_date)
        max_predic = np.where(predic == np.max(predic))[0][0]
        categ = max_predic
        categ = thres
        if categ == 0:
            pre_pred1["date"].append(end_date)
            pre_pred1["close"].append(all_data["close"][all_data["date"]==end_date])
        elif categ == 1:
            pre_pred2["date"].append(end_date)
            pre_pred2["close"].append(all_data["close"][all_data["date"]==end_date])

    plt.plot(all_data["date"], all_data["close"])
    plt.scatter(min_data["date"], min_data["close"], c="green", alpha=0.5)
    plt.scatter(min_pred["date"], min_pred["close"], c="green", alpha=1)
    plt.scatter(min_false_pred["date"], min_false_pred["close"], c="red", marker="x", alpha=1)
    plt.scatter(max_data["date"], max_data["close"], c="red", alpha=0.5)
    plt.scatter(max_pred["date"], max_pred["close"], c="red", alpha=1)
    plt.scatter(max_false_pred["date"], max_false_pred["close"], c="green", marker="x", alpha=1)
    plt.savefig(my_path / "plots" / "training"  / f"{stock}.png")
    plt.clf()
    plt.plot(all_data["date"], all_data["close"])
    plt.scatter(pre_pred2["date"], pre_pred2["close"], c="red", alpha=0.5)
    plt.scatter(pre_pred1["date"], pre_pred1["close"], c="green", alpha=0.5)
    plt.savefig(my_path / "plots" / "prediction"  / f"{stock}.png")
    plt.clf()
    # except ValueError:
    #     print(f"no data found for {stock}")

