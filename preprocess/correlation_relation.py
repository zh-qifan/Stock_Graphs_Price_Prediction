import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta

def date_converter(date_str):
    date_str = date_str.decode("utf-8")
    date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")  # Adjust the format if needed
    return date_obj.strftime("%Y-%m-%d")

def correlation_relation(index, window, start_date, end_date, threshold):

    data_folder = "data/google_finance/"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    date_path = f"data/{index}_aver_line_dates.csv"
    tickers_path = f"data/{index}_tickers_qualify_dr-0.98_min-5_smooth.csv"
    date_list = np.loadtxt(date_path, dtype='datetime64[s]', delimiter=',', converters={0: date_converter})
    tickers_list = np.loadtxt(tickers_path, dtype=str, delimiter=',')
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    date_list_before_start = date_list[(date_list < start_date)]
    start_date_window_adjusted = date_list_before_start[-min(window, len(date_list_before_start) - 1)]
    date_list_full = date_list[(date_list >= start_date_window_adjusted) & (date_list <= end_date)] 
    date_list = date_list[(date_list <= end_date) & (date_list >= start_date)]

    full_df = pd.DataFrame({"Date": date_list_full})

    for j, ticker in enumerate(tickers_list):    
        price_data = pd.read_csv(f"{data_folder}/{index}_{ticker}_30Y.csv")
        price_data.rename({"Unnamed: 0":"Date"}, axis=1, inplace=True)
        price_data["Date"] = pd.to_datetime(price_data["Date"].str[:10], format='%Y-%m-%d')
        price_data = price_data.loc[(price_data["Date"] >= start_date_window_adjusted) & (price_data["Date"] <= end_date), ["Date", "Close"]]
        price_data = price_data.rename({"Close": ticker}, axis=1)
        price_data = price_data.reset_index(drop=True)
        price_data = price_data.set_index("Date")
        price_data = price_data.fillna(method="ffill")
        price_data = price_data.reset_index(drop=False)
        full_df = pd.merge(full_df, price_data, how="left", on="Date")

    relation = np.zeros((len(tickers_list), len(tickers_list), len(date_list)), dtype=bool)

    for t in range(len(date_list)):
        d = date_list[t]
        idx = full_df.loc[full_df["Date"] == d].index[0]
        d_df = full_df.loc[idx-window+1:idx, :]
        d_df = d_df.drop(["Date"], axis=1)
        d_df.iloc[1:, :] = d_df.iloc[1:, :].values / d_df.iloc[:-1, :].values - 1
        d_df = d_df.iloc[1:, :]
        corr_mat = d_df.corr().values

        corr_mat[(~np.isfinite(corr_mat))] = 0
        corr_mat = (np.fabs(corr_mat) >= threshold)
        for i in range(len(tickers_list)):
            corr_mat[i, i] = 0

        relation[:, :, t] = corr_mat
        print(f"{index} - COMPLETED: {t} / {len(date_list)}")
    np.save(f"processed_data/{index}_corr_relation.npy", relation)
            
if __name__ == "__main__":
    threshold = 0.7
    # correlation_relation("NASDAQ", 30, "2013-01-02", "2017-11-25", threshold)
    correlation_relation("NYSE", 30, "2013-01-02", "2017-11-25", threshold)
