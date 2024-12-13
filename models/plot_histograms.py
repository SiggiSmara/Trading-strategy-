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
target0 = my_peak_data[my_peak_data["target"] == 0]
target1 = my_peak_data[my_peak_data["target"] == 1]
target2 = my_peak_data[my_peak_data["target"] == 2]
cols = ["normalized_value",] + reg_cols
for col in cols:
    plt.hist([target0[col], target1[col], target2[col]], bins=10, density=True, histtype='step', label=["target 0","target 1", "target 2"])
    plt.legend()
    plt.savefig(my_path / "plots" /  f"hist_{col}.png")
    plt.clf()
    

