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
from models.lr_inference import LR_v1_predict

my_path = Path(__file__).parent

my_peak_data:pd.DataFrame = pd.read_pickle(my_path / "finance_data.sav")

my_peak_data.to_excel(my_path / "finance_data.xls")