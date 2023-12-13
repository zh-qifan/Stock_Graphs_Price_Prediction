import os
import sys
import logging
import numpy as np
import torch
import gc
import matplotlib.pyplot as plt


from preprocess.correlation_relation import correlation_relation
from preprocess.node_features_target import get_node_features, get_price_target
from preprocess.sector_industry_relation import get_sector_industry_relation
from preprocess.wiki_relation import get_wiki_relation

from model.train_test_lstm import train_test_lstm
from model.train_test_lstm_gat import train_test_lstm_gat
from model.utils.augment_adj import augment_adj

from model.modules.gat import LSTMGAT

torch.manual_seed(1234)
np.random.seed(1234)

def main(index, trial):

    # Creare folders
    if not os.path.exists("loggings"):
        os.makedirs("loggings")
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    if not os.path.exists("processed_data"):
        os.makedirs("processed_data")

    # Set up logger
    logger = logging.getLogger('CPSC 583')
    logger.setLevel(logging.DEBUG)
    if os.path.exists(f"loggings/{index}_{trial}.log"):
        os.remove(f"loggings/{index}_{trial}.log")
    file_handler = logging.FileHandler(f'loggings/{index}_{trial}.log')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s' + f' - {index}')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Set up parameters
    tau = 5
    corr_window = 30
    corr_threshold = 0.7
    seq_window = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 10
    lstm_hidden_size = 64
    EPOCHS = 5000
    loss_record = []
    model_record = []

    # Preprocess data and if processed data exists, pass.

    if not os.path.exists(f"processed_data/{index}_wiki_relation.npy"):
        logger.info("Preprocessing wiki relation...")
        get_wiki_relation(index)
        logger.info("Preprocessing wiki relation done.")
    else:
        logger.info("Wiki relation already preprocessed.")
    
    if not os.path.exists(f"processed_data/{index}_sector_industry_relation.npy"):
        logger.info("Preprocessing sector industry relation...")
        get_sector_industry_relation(index)
        logger.info("Preprocessing sector industry relation done.")
    else:
        logger.info("Sector industry relation already preprocessed.")
    
    if not os.path.exists(f"processed_data/{index}_train_feat_mat.npy"):
        logger.info("Preprocessing node features and target...")
        dates_dict = {
            "train": ["2013-01-02", "2015-12-31"],
            "val": ["2016-01-04", "2016-12-30"],
            "test": ["2017-01-03", "2017-11-25"]
        }
        windows = [5, 10, 20, 30]
        feature_list = ["Close", "Volume"]
        for key, dates in dates_dict.items():
            start_date = dates[0]
            end_date = dates[1]
            feat_mat = get_node_features(index, start_date, end_date, windows, feature_list)
            price_target_mat = get_price_target(index, start_date, end_date, tau)
            np.save(f"processed_data/{index}_{key}_feat_mat.npy", feat_mat)
            np.save(f"processed_data/{index}_{key}_price_target_mat.npy", price_target_mat)
        logger.info("Preprocessing node features and target done.")
    else:
        logger.info("Node features and target already preprocessed.")

    if not os.path.exists(f"processed_data/{index}_corr_relation.npy"):
        logger.info("Preprocessing correlation relation...")
        correlation_relation(index, corr_window, "2013-01-02", "2017-11-25", corr_threshold)
        logger.info("Preprocessing correlation relation done.")
    else:
        logger.info("Correlation relation already preprocessed.")

    # Experiments
    ## Load data and preprocess
    logger.info("Loading features data...")
    features_train = np.load(f"processed_data/{index}_train_feat_mat.npy")
    target_train = np.load(f"processed_data/{index}_train_price_target_mat.npy")

    features_val = np.load(f"processed_data/{index}_val_feat_mat.npy")
    target_val = np.load(f"processed_data/{index}_val_price_target_mat.npy")

    features_test = np.load(f"processed_data/{index}_test_feat_mat.npy")
    target_test = np.load(f"processed_data/{index}_test_price_target_mat.npy")

    features_mean = np.nanmean(features_train, axis=0)
    features_std = np.nanstd(features_train, axis=0)
    features_train = (features_train - features_mean) / features_std
    features_val = (features_val - features_mean) / features_std
    features_test = (features_test - features_mean) / features_std
    
    features_val = np.concatenate((features_train[-seq_window + 1:, :, :], features_val), axis=0)
    features_test = np.concatenate((features_val[-seq_window + 1:, :, :], features_test), axis=0)

    features_train = np.nan_to_num(features_train)
    features_val = np.nan_to_num(features_val)
    features_test = np.nan_to_num(features_test)

    target_mean = np.nanmean(target_train, axis=0)
    target_std = np.nanstd(target_train, axis=0)
    target_train = (target_train - target_mean) / target_std
    target_val = (target_val - target_mean) / target_std
    target_test = (target_test - target_mean) / target_std

    x_train = []
    for i in range(features_train.shape[0] - seq_window + 1):
        x_train.append(np.transpose(features_train[i:i + seq_window, :, :], (1, 0, 2)))

    x_train = torch.tensor(np.array(x_train), dtype=torch.float32).to(device)
    y_train = torch.tensor(target_train[seq_window - 1:, :], dtype=torch.float32).to(device)

    x_val = []
    for i in range(features_val.shape[0] - seq_window + 1):
        x_val.append(np.transpose(features_val[i:i + seq_window, :, :], (1, 0, 2)))

    x_val = torch.tensor(np.array(x_val), dtype=torch.float32).to(device)
    y_val = torch.tensor(target_val, dtype=torch.float32).to(device)

    x_test = []
    for i in range(features_test.shape[0] - seq_window + 1):
        x_test.append(np.transpose(features_test[i:i + seq_window, :, :], (1, 0, 2)))

    x_test = torch.tensor(np.array(x_test), dtype=torch.float32).to(device)
    y_test = torch.tensor(target_test, dtype=torch.float32).to(device)
    logger.info("Loading features data done.")

    # x_train = x_train[:10]
    # y_train = y_train[:10]
    # x_val = x_val[:10]
    # y_val = y_val[:10]
    # x_test = x_test[:10]
    # y_test = y_test[:10]

    # LSTM trained on each stock separately
    logger.info("Training LSTM on each stock separately...")
    test_losses = []
    for i in range(x_train.shape[1]):
        data = {
            "x_train": x_train[:, i, :, :],
            "y_train": y_train[:, i, :],
            "x_val": x_val[:, i, :, :],
            "y_val": y_val[:, i, :],
            "x_test": x_test[:, i, :, :],
            "y_test": y_test[:, i, :]
        }
        total_test_loss = train_test_lstm(data, input_size, seq_window, lstm_hidden_size, device, EPOCHS, 0.01, logger)
        test_losses.append(total_test_loss)
    logger.info("Training LSTM on each stock separately done.")
    logger.info(f"Average test loss: {np.sum(test_losses) / (len(test_losses) * y_test.shape[0])}")
    loss_record.append(np.sum(test_losses) / (len(test_losses) * y_test.shape[0]))
    model_record.append("lstm_sep")

    ## LSTM trained on all stocks
    # logger.info("Training LSTM on all stocks...")
    # data = {
    #     "x_train": x_train.view(-1, seq_window, input_size),
    #     "y_train": y_train.view(-1, 1),
    #     "x_val": x_val.view(-1, seq_window, input_size),
    #     "y_val": y_val.view(-1, 1),
    #     "x_test": x_test.view(-1, seq_window, input_size),
    #     "y_test": y_test.view(-1, 1)
    # }
    # total_test_loss = train_test_lstm(data, input_size, seq_window, lstm_hidden_size, device, EPOCHS, 0.01, logger)
    # logger.info("Training LSTM on all stocks done.")
    # logger.info(f"Average test loss: {total_test_loss / (y_test.shape[0] * y_test.shape[1])}")
    # loss_record.append(total_test_loss / (y_test.shape[0] * y_test.shape[1]))
    # model_record.append("lstm_all")

    ## LSTM + GAT trained on all stocks with relation (aggregated)
    logger.info("Training LSTM + GAT on all stocks with relation (aggregated)...")
    ### Load relation
    logger.info("Loading relation...")
    wiki_relation = np.load(f"processed_data/{index}_wiki_relation.npy")
    sector_industry_relation = np.load(f"processed_data/{index}_sector_industry_relation.npy")
    corr_relation = np.load(f"processed_data/{index}_corr_relation.npy")
    metadata = [["stock"], [("stock", "wiki", "stock"), ("stock", "sector_industry", "stock")]]
    edge_index = {}
    wiki_relation_agg = np.sum(wiki_relation, axis=-1)
    sector_industry_relation_agg = np.sum(sector_industry_relation, axis=-1)
    edge_index[("stock", "wiki", "stock")] = torch.tensor(np.array(np.where(wiki_relation_agg)), dtype=torch.int32).to(device)
    edge_index[("stock", "sector_industry", "stock")] = torch.tensor(np.array(np.where(sector_industry_relation_agg)), dtype=torch.int32).to(device)

    data = {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test
    }
    total_test_loss = train_test_lstm_gat(data, edge_index, corr_relation, input_size, seq_window,lstm_hidden_size, metadata, device, EPOCHS, 0.01, logger, f"lstm_gat_agg_{index}")
    logger.info("Training LSTM + GAT on all stocks with relation (aggregated) done.")
    logger.info(f"Average test loss: {total_test_loss / (y_test.shape[0] * y_test.shape[1])}")
    loss_record.append(total_test_loss / (y_test.shape[0] * y_test.shape[1]))
    model_record.append("lstm_gat_agg")

    ## LSTM + GAT trained on all stocks with relation (not aggregated)
    logger.info("Training LSTM + GAT on all stocks with relation (not aggregated)...")
    metadata = [["stock"], []]
    edge_index = {}
    for i in range(wiki_relation.shape[-1]):
        if np.sum(wiki_relation[:, :, i]) > 1:
            idx = np.where(np.sum(wiki_relation[:, :, i], axis=1) > 0)[0]
            wiki_relation[idx, idx, i] = 1
            metadata[1].append(("stock", f"wiki_{i}", "stock"))
            ei = torch.tensor(np.array(np.where(wiki_relation[:, :, i])), dtype=torch.long).to(device)
            edge_index[("stock", f"wiki_{i}", "stock")] = ei
    for i in range(sector_industry_relation.shape[-1]):
        if np.sum(sector_industry_relation[:, :, i]) > 1:
            idx = np.where(np.sum(sector_industry_relation[:, :, i], axis=1) > 0)[0]
            sector_industry_relation[idx, idx, i] = 1
            metadata[1].append(("stock", f"sector_industry_{i}", "stock"))
            ei = torch.tensor(np.array(np.where(sector_industry_relation[:, :, i])), dtype=torch.long).to(device)
            edge_index[("stock", f"sector_industry_{i}", "stock")] = ei
    total_test_loss = train_test_lstm_gat(data, edge_index, corr_relation, input_size, seq_window, lstm_hidden_size, metadata, device, EPOCHS, 0.01, logger, f"lstm_gat_not_agg_{index}")
    logger.info("Training LSTM + GAT on all stocks with relation (not aggregated) done.")
    logger.info(f"Average test loss: {total_test_loss / (y_test.shape[0] * y_test.shape[1])}")
    loss_record.append(total_test_loss / (y_test.shape[0] * y_test.shape[1]))
    model_record.append("lstm_gat_not_agg")

    ## LSTM + GAT trained on all stocks with relation (not aggregated) with bases
    logger.info("Training LSTM + GAT on all stocks with relation (not aggregated) with bases 5...")
    num_bases = 5
    total_test_loss = train_test_lstm_gat(data, edge_index, corr_relation, input_size, seq_window, lstm_hidden_size, metadata, device, EPOCHS, 0.01, logger, f"lstm_gat_not_agg_bases_{index}_{num_bases}", num_bases=num_bases)
    logger.info("Training LSTM + GAT on all stocks with relation (not aggregated) with bases 5 done.")
    logger.info(f"Average test loss: {total_test_loss / (y_test.shape[0] * y_test.shape[1])}")
    loss_record.append(total_test_loss / (y_test.shape[0] * y_test.shape[1]))
    model_record.append("lstm_gat_not_agg_bases_5")
    logger.info(f"Loss record: {loss_record}")
    logger.info(f"Model record: {model_record}")

    # ## LSTM + GAT trained on all stocks with relation (not aggregated) with bases
    logger.info("Training LSTM + GAT on all stocks with relation (not aggregated) with bases 30...")
    num_bases = 30
    total_test_loss = train_test_lstm_gat(data, edge_index, corr_relation, input_size, seq_window, lstm_hidden_size, metadata, device, EPOCHS, 0.01, logger, f"lstm_gat_not_agg_bases_{index}_{num_bases}", num_bases=num_bases)
    logger.info("Training LSTM + GAT on all stocks with relation (not aggregated) with bases 30 done.")
    logger.info(f"Average test loss: {total_test_loss / (y_test.shape[0] * y_test.shape[1])}")
    loss_record.append(total_test_loss / (y_test.shape[0] * y_test.shape[1]))
    model_record.append("lstm_gat_not_agg_bases_30")
    logger.info(f"Loss record: {loss_record}")
    logger.info(f"Model record: {model_record}")

    ## LSTM + GAT trained on all stocks with relation (aggregated) with augmented adj
    logger.info("Training LSTM + GAT on all stocks with relation (aggregated) with augmented adj...")
    wiki_relation = np.load(f"processed_data/{index}_wiki_relation.npy")
    sector_industry_relation = np.load(f"processed_data/{index}_sector_industry_relation.npy")
    metadata = [["stock"], [("stock", "wiki", "stock"), ("stock", "sector_industry", "stock")]]
    edge_index = {}
    wiki_relation_agg = np.sum(wiki_relation, axis=-1)
    wiki_relation_agg = torch.tensor(wiki_relation_agg, dtype=torch.int32).to(device)
    sector_industry_relation_agg = np.sum(sector_industry_relation, axis=-1)
    sector_industry_relation_agg = torch.tensor(sector_industry_relation_agg, dtype=torch.int32).to(device)
    edge_index[("stock", "wiki", "stock")] = augment_adj(wiki_relation_agg, seq_window)
    edge_index[("stock", "sector_industry", "stock")] = augment_adj(sector_industry_relation_agg, seq_window)

    total_test_loss = train_test_lstm_gat(data, edge_index, corr_relation, input_size, seq_window, lstm_hidden_size, metadata, device, EPOCHS, 0.01, logger, f"lstm_gat_agg_aug_{index}", aug=True)
    logger.info("Training LSTM + GAT on all stocks with relation (aggregated) with augmented adj done.")
    logger.info(f"Average test loss: {total_test_loss / (y_test.shape[0] * y_test.shape[1])}")
    loss_record.append(total_test_loss / (y_test.shape[0] * y_test.shape[1]))
    model_record.append("lstm_gat_agg_aug")

    # Sensitivity analysis
    ## num_bases
    ### LSTM + GAT trained on all stocks with relation (not aggregated) with bases
    logger.info("Training LSTM + GAT on all stocks with relation (not aggregated) with bases...")
    metadata = [["stock"], []]
    edge_index = {}
    for i in range(wiki_relation.shape[-1]):
        if np.sum(wiki_relation[:, :, i]) > 1:
            idx = np.where(np.sum(wiki_relation[:, :, i], axis=1) > 0)[0]
            wiki_relation[idx, idx, i] = 1
            metadata[1].append(("stock", f"wiki_{i}", "stock"))
            ei = torch.tensor(np.array(np.where(wiki_relation[:, :, i])), dtype=torch.long).to(device)
            edge_index[("stock", f"wiki_{i}", "stock")] = ei
    for i in range(sector_industry_relation.shape[-1]):
        if np.sum(sector_industry_relation[:, :, i]) > 1:
            idx = np.where(np.sum(sector_industry_relation[:, :, i], axis=1) > 0)[0]
            sector_industry_relation[idx, idx, i] = 1
            metadata[1].append(("stock", f"sector_industry_{i}", "stock"))
            ei = torch.tensor(np.array(np.where(sector_industry_relation[:, :, i])), dtype=torch.long).to(device)
            edge_index[("stock", f"sector_industry_{i}", "stock")] = ei
    
    for num_bases in [10, 20, 50]:
        total_test_loss = train_test_lstm_gat(data, edge_index, corr_relation, input_size, seq_window, lstm_hidden_size, metadata, device, EPOCHS, 0.01, logger, f"lstm_gat_not_agg_bases_{index}_{num_bases}", num_bases=num_bases)
        logger.info(f"Training LSTM + GAT on all stocks with relation (not aggregated) with bases {num_bases} done.")
        logger.info(f"Average test loss: {total_test_loss / (y_test.shape[0] * y_test.shape[1])}")
        loss_record.append(total_test_loss / (y_test.shape[0] * y_test.shape[1]))
        model_record.append(f"lstm_gat_not_agg_bases_{num_bases}")
        logger.info(f"Loss record: {loss_record}")
        logger.info(f"Model record: {model_record}")
    
    ## num_heads
    ### LSTM + GAT trained on all stocks with relation (aggregated)
    logger.info("Training LSTM + GAT on all stocks with relation (aggregated)...")
    metadata = [["stock"], [("stock", "wiki", "stock"), ("stock", "sector_industry", "stock")]]
    edge_index = {}
    wiki_relation_agg = np.sum(wiki_relation, axis=-1)
    sector_industry_relation_agg = np.sum(sector_industry_relation, axis=-1)
    edge_index[("stock", "wiki", "stock")] = torch.tensor(np.array(np.where(wiki_relation_agg)), dtype=torch.int32).to(device)
    edge_index[("stock", "sector_industry", "stock")] = torch.tensor(np.array(np.where(sector_industry_relation_agg)), dtype=torch.int32).to(device)

    for num_heads in [3, 5, 7]:
        total_test_loss = train_test_lstm_gat(data, edge_index, corr_relation, input_size, seq_window,lstm_hidden_size, metadata, device, EPOCHS, 0.01, logger, f"lstm_gat_agg_{index}_{num_heads}")
        logger.info(f"Training LSTM + GAT on all stocks with relation (aggregated) with {num_heads} heads done.")
        logger.info(f"Average test loss: {total_test_loss / (y_test.shape[0] * y_test.shape[1])}")
        loss_record.append(total_test_loss / (y_test.shape[0] * y_test.shape[1]))
        model_record.append(f"lstm_gat_agg_{num_heads}")

    # Analysis on attention
    logger.info("Analysis on attention...")
    num_heads = 3
    wiki_relation = np.load(f"processed_data/{index}_wiki_relation.npy")
    sector_industry_relation = np.load(f"processed_data/{index}_sector_industry_relation.npy")

    metadata = [["stock"], [("stock", "wiki", "stock"), ("stock", "sector_industry", "stock")]]
    edge_index = {}
    wiki_relation_agg = np.sum(wiki_relation, axis=-1)
    sector_industry_relation_agg = np.sum(sector_industry_relation, axis=-1)
    edge_index[("stock", "wiki", "stock")] = torch.tensor(np.array(np.where(wiki_relation_agg)), dtype=torch.int32).to(device)
    edge_index[("stock", "sector_industry", "stock")] = torch.tensor(np.array(np.where(sector_industry_relation_agg)), dtype=torch.int32).to(device)

    model = LSTMGAT(input_size, seq_window, lstm_hidden_size, metadata, num_heads=num_heads).to(device)
    model.load_state_dict(torch.load(f"saved_models/lstm_gat_agg_{index}.pt", map_location=device))
    wiki_alphas_list = []
    sector_industry_alphas_list = []
    wiki_edges_index_list = None
    sector_industry_edges_index_list = None
    model.eval()
    with torch.no_grad():
        for i in range(x_test.shape[0]):
            alphas, self_loop_edge_indexes = model.get_att_mat(x_test[i], edge_index)
            w_values = alphas["stock__wiki__stock"]
            s_values = alphas["stock__sector_industry__stock"]
            wiki_alphas_list.append(w_values)
            sector_industry_alphas_list.append(s_values)
            if i == 0:
                wiki_edges_index_list = self_loop_edge_indexes["stock__wiki__stock"]
                sector_industry_edges_index_list = self_loop_edge_indexes["stock__sector_industry__stock"]

    if index == "NASDAQ":
        src = 2
    elif index == "NYSE":
        src = 147
    tickers_path = f"data/{index}_tickers_qualify_dr-0.98_min-5_smooth.csv"
    tickers_list = np.loadtxt(tickers_path, dtype=str, delimiter=',')
    logger.info(f"Stock: {tickers_list[src]}")
    wiki_neighbors = wiki_edges_index_list[0][wiki_edges_index_list[1] == src]
    sector_industry_neighbors = sector_industry_edges_index_list[0][sector_industry_edges_index_list[1] == src]
    logger.info(f"Wiki neighbors: {tickers_list[wiki_neighbors]}")
    logger.info(f"Sector industry neighbors: {tickers_list[sector_industry_neighbors]}")
    wiki_alphas_list = np.array(wiki_alphas_list)
    sector_industry_alphas_list = np.array(sector_industry_alphas_list)
    wiki_alpha_values = np.array([a[wiki_edges_index_list[1] == src] for a in wiki_alphas_list[:, :, 1]])
    sector_industry_alpha_values = np.array([a[sector_industry_edges_index_list[1] == src] for a in sector_industry_alphas_list[:, :, 1]])
    np.save(f"loggings/{index}_wiki_alpha_values.npy", wiki_alpha_values)
    np.save(f"loggings/{index}_sector_industry_alpha_values.npy", sector_industry_alpha_values)

    # Ticker labels for each edge type
    sector_industry_tickers = tickers_list[sector_industry_neighbors]
    wiki_tickers = tickers_list[wiki_neighbors]

    # Create heatmaps
    plt.figure(figsize=(18, 8))

    # Heatmap for sector-industry edge type
    plt.subplot(1, 2, 1)
    heatmap1 = plt.imshow(sector_industry_alpha_values.T, aspect='auto', cmap='viridis')
    plt.colorbar(heatmap1)
    plt.xticks([])
    plt.yticks(range(len(sector_industry_tickers)), sector_industry_tickers)
    plt.title(f'Sector-Industry Neighbor for {tickers_list[src]}')
    plt.xlabel('Time')

    # Heatmap for wiki edge type
    plt.subplot(1, 2, 2)
    heatmap2 = plt.imshow(wiki_alpha_values.T, aspect='auto', cmap='viridis')
    plt.colorbar(heatmap2)
    plt.xticks([])
    plt.yticks(range(len(wiki_tickers)), wiki_tickers)
    plt.title(f'Supplier-Customer Neighbor for {tickers_list[src]}')
    plt.xlabel('Time')

    plt.tight_layout()
    plt.savefig(f"loggings/{index}_attention.png", dpi=300)
    logger.info(f"Analysis on attention done.")

    logger.info(f"Loss record: {loss_record}")
    logger.info(f"Model record: {model_record}")

    # Close logger
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    
    return loss_record, model_record

if __name__ == "__main__":
    trials = 3
    for i in range(trials):
        loss_record, model_record = main("NASDAQ", i)
        gc.collect()
        torch.cuda.empty_cache()

    for i in range(trials):
        loss_record, model_record = main("NYSE", i)
        gc.collect()
        torch.cuda.empty_cache()

    

