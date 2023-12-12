from .modules.gat import LSTMGAT, LSTMGATAug
from .modules.gat_cat import LSTMGATBasis
from .utils.early_stop import EarlyStopper
import torch
import torch.nn as nn
import numpy as np

def train_test_lstm_gat(
        data, 
        edge_index, 
        corr_edge_index, 
        input_size, 
        sequence_length, 
        lstm_hidden_size, 
        metadata, 
        device, 
        epochs, 
        lr, 
        logger, 
        model_name, 
        num_bases=None,
        aug=False
    ):

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    if corr_edge_index is not None:
        corr_relation_train = corr_edge_index[:, :, :x_train.shape[0]]
        corr_relation_val = corr_edge_index[:, :, x_train.shape[0]:x_train.shape[0] + x_val.shape[0]]
        corr_relation_test = corr_edge_index[:, :, x_train.shape[0] + x_val.shape[0]:]

        corr_relation_train = torch.tensor(corr_relation_train, dtype=torch.int8).to(device)
        corr_relation_val = torch.tensor(corr_relation_val, dtype=torch.int8).to(device)
        corr_relation_test = torch.tensor(corr_relation_test, dtype=torch.int8).to(device)

    # Create model
    assert not aug or num_bases is None

    if not aug and num_bases is None:
        model = LSTMGAT(input_size, sequence_length, lstm_hidden_size, metadata, num_heads=3).to(device)
    elif aug:
        model = LSTMGATAug(input_size, sequence_length, lstm_hidden_size, metadata, num_heads=3).to(device)
    else:
        model = LSTMGATBasis(input_size, sequence_length, lstm_hidden_size, metadata, num_bases, num_heads=3).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1, verbose=False)
    criterion = nn.MSELoss()
    early_stopper = EarlyStopper(patience=30, min_delta=0.01)

    data_index = np.arange(x_train.size(0))
    np.random.shuffle(data_index)

    for epoch in range(epochs):
        train_losses = 0
        val_losses = 0
        np.random.shuffle(data_index)
        model.train()
        for i in data_index:
            edge_index_corr = torch.stack(torch.where(corr_relation_train[:, :, i]), dim=0)
            optimizer.zero_grad()
            output = model(x_train[i], (edge_index, edge_index_corr))
            mask = torch.isnan(y_train[i])
            loss = criterion(output[~mask], y_train[i][~mask])
            loss.backward()
            optimizer.step()
            train_losses += loss.item()
        train_losses /= len(data_index)
        
        model.eval()
        with torch.no_grad():
            for i in range(x_val.shape[0]):
                edge_index_corr = torch.stack(torch.where(corr_relation_val[:, :, i]), dim=0)
                output = model(x_val[i], (edge_index, edge_index_corr))
                mask = torch.isnan(y_val[i])
                val_losses += criterion(output[~mask], y_val[i][~mask]).item()
        val_losses /= x_val.shape[0]
        if val_losses < early_stopper.min_validation_loss:
            torch.save(model.state_dict(), f"saved_models/{model_name}.pt")
        if early_stopper.early_stop(val_losses):
            break
        scheduler.step()
    
    model.load_state_dict(torch.load(f"saved_models/{model_name}.pt", map_location=device))
    model.eval()
    test_losses = []
    with torch.no_grad():
        for i in range(x_test.shape[0]):
            edge_index_corr = torch.stack(torch.where(corr_relation_test[:, :, i]), dim=0)
            output = model(x_test[i], (edge_index, edge_index_corr))
            mask = torch.isnan(y_test[i])
            test_losses.append(criterion(output[~mask], y_test[i][~mask]).item() * y_test[i][~mask].shape[0])
    
    return np.sum(test_losses)



