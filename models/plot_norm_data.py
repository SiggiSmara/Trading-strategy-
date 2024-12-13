import pandas as pd
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
from stock_utils.stock_utils import reg_cols
from models.lr_inference import LR_v1_predict

my_path = Path(__file__).parent

my_peak_data:pd.DataFrame = pd.read_pickle(my_path / "finance_data.sav")

stocks = list(set(my_peak_data["stock"].to_list()))

start_date=datetime(year=2021, month=1, day=1)
end_date=datetime(year=2022, month=1, day=1)

for stock in stocks:
    print(stock)
    end_date=datetime(year=2022, month=1, day=1)
    all_data, idx_with_mins, idx_with_maxs, idx_tweens = stock_utils.get_data(
        sym=stock,
        end_date=end_date,
        delta_days=(end_date-start_date).days
    )

    # plt.plot(all_data["date"], all_data["normalized_value"], label="raw")
    # plt.plot(all_data["date"], all_data["normalized_value"].rolling(window=3).mean(), label="3")
    fig, axs = plt.subplots(2)
    axs[1].sharex(axs[0])
    axs[0].plot(all_data["date"], all_data["close"])
    # axs[1].plot(all_data["date"], all_data["normalized_value"].rolling(window=5).mean(), label=5)
    axs[1].plot(all_data["date"], all_data["normalized_value"].rolling(window=10).mean(), label=10)
    axs[1].plot(all_data["date"], all_data["normalized_value"].rolling(window=20).mean(), label=20)
    plt.legend()
    plt.savefig(my_path / "plots" / "norm"  / f"{stock}.png")
    plt.clf()
    

