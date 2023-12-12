import numpy as np
import pandas as pd
import os
import json

def get_sector_industry_relation(index):
    '''
    This function returns the sector-industry relation matrix. The matrix is of shape (num_sectors, num_tickers, num_tickers).
    The matrix[i, j, k] = 1 if the jth ticker is in the ith sector and the kth ticker is in the ith sector. Otherwise, the
    matrix[i, j, k] = 0.

    Args:
        index: str, index name, e.g. "NASDAQ"
    
    Returns:
        relation: np.array of shape (num_sectors, num_tickers, num_tickers)
    '''
    data_folder = "data/relation/sector_industry"
    tickers_path = f"data/{index}_tickers_qualify_dr-0.98_min-5_smooth.csv"
    tickers_list = np.loadtxt(tickers_path, dtype=str, delimiter=',')

    tickers_dict = {}
    for i, ticker in enumerate(tickers_list):
        tickers_dict[ticker] = i

    relation = []    
    with open(f"{data_folder}/{index}_industry_ticker.json", 'r') as f:
        data = json.load(f)
    
    for sector, stock_list in data.items():
        if len(stock_list) <= 1 or sector == "n/a":
            continue
        relation.append(np.zeros((len(tickers_list), len(tickers_list), 1)))
        for s in stock_list:
            for t in stock_list:
                relation[-1][tickers_dict[s], tickers_dict[t], 0] = 1
    
    relation = np.concatenate(relation, axis=-1)
    print("The ratio of non-zero entries in the relation matrix is: ", np.sum(relation) / (relation.shape[0] * relation.shape[1] * relation.shape[2]))
    np.save(f"processed_data/{index}_sector_industry_relation.npy", relation)
    return relation

if  __name__ == "__main__":
    get_sector_industry_relation("NASDAQ")
    get_sector_industry_relation("NYSE")
    
