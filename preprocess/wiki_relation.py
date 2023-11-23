import numpy as np
import pandas as pd
import os
import json

def get_wiki_relation(index):
    
    data_folder = "data/relation/wikidata"
    tickers_path = f"data/{index}_tickers_qualify_dr-0.98_min-5_smooth.csv"
    tickers_list = np.loadtxt(tickers_path, dtype=str, delimiter=',')
    tickers_wiki_path = f"data/{index}_wiki.csv"
    tickers_wiki_list = np.loadtxt(tickers_wiki_path, dtype=str, delimiter=',')
    sel_wiki_path = "data/relation/wikidata/selected_wiki_connections.csv"
    sel_paths = np.genfromtxt(sel_wiki_path, dtype=str, delimiter=' ',
                              skip_header=False)
    sel_paths = set(sel_paths[:, 0])
    relation_path = f"{data_folder}/{index}_connections.json"
    with open(relation_path, 'r') as f:
        connections = json.load(f)
    
    tickers_dict = {}
    for i, ticker in enumerate(tickers_list):
        tickers_dict[ticker] = i

    wikiid_tickers_dict = {}
    for i, wt in enumerate(tickers_wiki_list):
        if wt[-1] != "unknown":
            wikiid_tickers_dict[wt[-1]] = tickers_dict[wt[0]]
    
    occur_paths = set()
    for sou_item, conns in connections.items():
        for tar_item, paths in conns.items():
            for p in paths:
                path_key = '_'.join(p)
                if path_key in sel_paths:
                    occur_paths.add(path_key)
   
    valid_path_index = {}
    for ind, path in enumerate(occur_paths):
        valid_path_index[path] = ind
    
    relation = np.zeros(
        [tickers_list.shape[0], tickers_list.shape[0], len(valid_path_index) + 1],
        dtype=int
    )

    for sou_item, conns in connections.items():
        for tar_item, paths in conns.items():
            for p in paths:
                path_key = '_'.join(p)
                if path_key in valid_path_index.keys():
                    relation[wikiid_tickers_dict[sou_item]][wikiid_tickers_dict[tar_item]][valid_path_index[path_key]] = 1

    print("The ratio of non-zero entries in the relation matrix is: ", np.sum(relation) / (relation.shape[0] * relation.shape[1] * relation.shape[2]))

    np.save(f"processed_data/{index}_wiki_relation.npy", relation)

    return relation

if  __name__ == "__main__":
    get_wiki_relation("NASDAQ")
    get_wiki_relation("NYSE")