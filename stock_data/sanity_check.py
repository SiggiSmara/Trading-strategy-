from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

data_path = Path(__file__).parent.parent / "stock_data" / "curr_df.sav"
if data_path.is_file():
    curr_df:pd.DataFrame = pd.read_pickle(data_path)
else:
    raise ValueError("No curr_df found")

df = curr_df.groupby(["date"], as_index=False)["close"].count()

spy = curr_df[curr_df["stock"] == "SPY"]
spy = spy[spy["date"] >= "2021-01-01"].reset_index()
perf = round((( spy.iloc[-1]["close"] / spy.iloc[0]["close"] ) - 1 ) * 100, 1)
print(spy["date"])
print(f"SPY performance for 2021: {perf}")

plt.savefig("hist.png")