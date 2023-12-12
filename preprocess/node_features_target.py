import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def date_converter(date_str):
    date_str = date_str.decode("utf-8")
    date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")  # Adjust the format if needed
    return date_obj.strftime("%Y-%m-%d")

def get_node_features(index, start_date, end_date, windows, feature_list=["Close", "Volume"]):
    '''
    Following the work of "Temporal Relational Ranking for Stock Prediction" (https://arxiv.org/abs/1809.09441), we 
    only consider moving-average-type node features. The original paper uses 5-day, 10-day, 20-day, 30-day MA of Close.
    To make our model more robust, we also consider 5-day, 10-day, 20-day, 30-day MA of Volume.

    Here 5-day refers to 5 trading days.

    Args:
        index: str, index name, e.g. "NASDAQ"
        start_date: str, start date of the time period we consider
        end_date: str, end date of the time period we consider
        windows: list of int, windows of moving average
        feature_list: list of str, features we consider, e.g. ["Close", "Volume"]
    
    Returns:
       feat_mat: list of np.array, each np.array is a matrix of shape (num_dates, num_tickers, num_features), 
                    num_features is the number of features we consider 
    '''

    assert index in ["NASDAQ", "NYSE"], "index must be one of NASDAQ, NYSE, AMEX"
    assert isinstance(start_date, str), "start_date must be a string"
    assert isinstance(end_date, str), "end_date must be a string"
    assert isinstance(windows, list), "windows must be a list"
    assert isinstance(feature_list, list), "feature_list must be a list"

    data_folder = "data/google_finance/"
    date_path = f"data/{index}_aver_line_dates.csv"
    tickers_path = f"data/{index}_tickers_qualify_dr-0.98_min-5_smooth.csv"
    date_list = np.loadtxt(date_path, dtype='datetime64[s]', delimiter=',', converters={0: date_converter})
    tickers_list = np.loadtxt(tickers_path, dtype=str, delimiter=',')
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    date_list_before_start = date_list[(date_list < start_date)]
    date_list = date_list[(date_list <= end_date) & (date_list >= start_date)]

    start_date_window_adjusted = date_list_before_start[-min(max(windows), len(date_list_before_start) - 1)]

    feat_mat = []
    dates_df = pd.DataFrame({"Date": date_list})
    for feature in feature_list:
        feat_mat.append(np.zeros((len(date_list), len(tickers_list))))
        for window in windows:
            feat_mat.append(np.zeros((len(date_list), len(tickers_list))))
    
    for j, ticker in enumerate(tickers_list):    
        price_data = pd.read_csv(f"{data_folder}/{index}_{ticker}_30Y.csv")
        price_data.rename({"Unnamed: 0":"Date"}, axis=1, inplace=True)
        price_data["Date"] = pd.to_datetime(price_data["Date"].str[:10], format='%Y-%m-%d')
        price_data = price_data.loc[(price_data["Date"] >= start_date_window_adjusted) & (price_data["Date"] <= end_date), ["Date"] + feature_list]
        price_data = price_data.reset_index(drop=True)
        price_data = price_data.set_index("Date")
        price_data = price_data.fillna(method="ffill")

        i = 0
        for feature in feature_list:
            feat = price_data[feature]
            feat = pd.merge(dates_df, feat, how="left", on="Date")
            feat = feat.loc[(feat["Date"] >= start_date) & (feat["Date"] <= end_date), feature]
            feat = feat.values
            feat_mat[i][:, j] = feat
            i += 1
            for window in windows:
                # ith features and jth ticker
                feat = price_data[feature].rolling(window, min_periods=1).mean().reset_index()
                feat = pd.merge(dates_df, feat, how="left", on="Date")
                feat = feat.loc[(feat["Date"] >= start_date) & (feat["Date"] <= end_date), feature]
                feat = feat.values
                feat_mat[i][:, j] = feat
                i += 1

    feat_mat = [feat.reshape(len(date_list), len(tickers_list), 1) for feat in feat_mat]
    feat_mat = np.concatenate(feat_mat, axis=-1)

    return feat_mat

def get_target(index, start_date, end_date, window=5):
    '''
    In this project, we use the 5-day return as the target. The return is defined as the ratio of the price at the end of
    the 5-day period to the price at the beginning of the 5-day period. The return is calculated for each ticker and each
    5-day period. 

    Here 5-day refers to 5 trading days. 

    Args:
        index: str, index name, e.g. "NASDAQ"
        start_date: str, start date of the time period we consider
        end_date: str, end date of the time period we consider
        window: int, window of moving average
    
    Returns:
         target_mat: np.array of shape (num_dates, num_tickers, 1)
    '''

    assert index in ["NASDAQ", "NYSE"], "index must be one of NASDAQ, NYSE, AMEX"
    assert isinstance(start_date, str), "start_date must be a string"
    assert isinstance(end_date, str), "end_date must be a string"
    assert isinstance(window, int), "window must be an int"

    data_folder = "data/google_finance/"
    date_path = f"data/{index}_aver_line_dates.csv"
    tickers_path = f"data/{index}_tickers_qualify_dr-0.98_min-5_smooth.csv"
    date_list = np.loadtxt(date_path, dtype='datetime64[s]', delimiter=',', converters={0: date_converter})
    tickers_list = np.loadtxt(tickers_path, dtype=str, delimiter=',')
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    date_list_after_end = date_list[(date_list > end_date)]
    date_list = date_list[(date_list <= end_date) & (date_list >= start_date)]

    end_date_window_adjusted = date_list_after_end[min(window, len(date_list_after_end) - 1)]

    target_mat = np.zeros((len(date_list), len(tickers_list)))
    dates_df = pd.DataFrame({"Date": date_list})
    for j, ticker in enumerate(tickers_list):
        price_data = pd.read_csv(f"{data_folder}/{index}_{ticker}_30Y.csv")
        price_data.rename({"Unnamed: 0":"Date"}, axis=1, inplace=True)
        price_data["Date"] = pd.to_datetime(price_data["Date"].str[:10], format='%Y-%m-%d')
        price_data = price_data.loc[(price_data["Date"] >= start_date) & (price_data["Date"] <= end_date_window_adjusted), ["Date", "Close"]]
        price_data = price_data.reset_index(drop=True)
        price_data = price_data.set_index("Date")
        price_data = price_data.pct_change(periods=window).reset_index()
        price_data.loc[:, "Close"] = price_data.loc[:, "Close"].shift(-window)
        price_data = pd.merge(dates_df, price_data, how="left", on="Date")
        price_data = price_data.loc[(price_data["Date"] >= start_date) & (price_data["Date"] <= end_date), "Close"]

        target_mat[:, j] = price_data

    target_mat = target_mat.reshape(len(date_list), len(tickers_list), 1)

    return target_mat

def get_price_target(index, start_date, end_date, window=1):

    assert index in ["NASDAQ", "NYSE"], "index must be one of NASDAQ, NYSE, AMEX"
    assert isinstance(start_date, str), "start_date must be a string"
    assert isinstance(end_date, str), "end_date must be a string"
    assert isinstance(window, int), "window must be an int"

    data_folder = "data/google_finance/"
    date_path = f"data/{index}_aver_line_dates.csv"
    tickers_path = f"data/{index}_tickers_qualify_dr-0.98_min-5_smooth.csv"
    date_list = np.loadtxt(date_path, dtype='datetime64[s]', delimiter=',', converters={0: date_converter})
    tickers_list = np.loadtxt(tickers_path, dtype=str, delimiter=',')
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    date_list_after_end = date_list[(date_list > end_date)]
    date_list = date_list[(date_list <= end_date) & (date_list >= start_date)]

    end_date_window_adjusted = date_list_after_end[min(window, len(date_list_after_end) - 1)]

    target_mat = np.zeros((len(date_list), len(tickers_list)))
    dates_df = pd.DataFrame({"Date": date_list})
    for j, ticker in enumerate(tickers_list):
        price_data = pd.read_csv(f"{data_folder}/{index}_{ticker}_30Y.csv")
        price_data.rename({"Unnamed: 0":"Date"}, axis=1, inplace=True)
        price_data["Date"] = pd.to_datetime(price_data["Date"].str[:10], format='%Y-%m-%d')
        price_data = price_data.loc[(price_data["Date"] >= start_date) & (price_data["Date"] <= end_date_window_adjusted), ["Date", "Close"]]
        price_data.loc[:, "Close"] = price_data.loc[:, "Close"].shift(-window)
        price_data = pd.merge(dates_df, price_data, how="left", on="Date")
        price_data = price_data.loc[(price_data["Date"] >= start_date) & (price_data["Date"] <= end_date), "Close"]

        target_mat[:, j] = price_data

    target_mat = target_mat.reshape(len(date_list), len(tickers_list), 1)

    return target_mat



def main():
    dates_dict = {
        "train": ["2013-01-02", "2015-12-31"],
        "val": ["2016-01-04", "2016-12-30"],
        "test": ["2017-01-03", "2017-11-25"]
    }
    for index in ["NASDAQ", "NYSE"]:
        windows = [5, 10, 20, 30]
        # windows = [5]
        feature_list = ["Close", "Volume"]
        for key, dates in dates_dict.items():
            start_date = dates[0]
            end_date = dates[1]
            # feat_mat = get_node_features(index, start_date, end_date, windows, feature_list)
            # target_mat = get_target(index, start_date, end_date, 5)
            price_target_mat = get_price_target(index, start_date, end_date, 5)
            # np.save(f"processed_data/{index}_{key}_feat_mat.npy", feat_mat)
            # np.save(f"processed_data/{index}_{key}_target_mat.npy", target_mat)
            np.save(f"processed_data/{index}_{key}_price_target_mat.npy", price_target_mat)

if __name__ == "__main__":
    main()
    

    
