import gc
from .modules.lstm import LSTMPred
from .utils.early_stop import EarlyStopper
from .utils.data_loader import StockDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

def train_test_lstm(data, input_size, sequence_length, lstm_hidden_size, device, epochs, lr, logger, batch_size=None):

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    # Create model
    model = LSTMPred(input_size, sequence_length, lstm_hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1, verbose=False)
    criterion = nn.MSELoss()
    early_stopper = EarlyStopper(patience=30, min_delta=0.01)
    if batch_size is None:
        batch_size = x_train.shape[0]
    
    train_dataset = StockDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = StockDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = StockDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(epochs):
        model.train()
        train_losses = 0
        val_losses = 0
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            output = model(x_train)
            mask = torch.isnan(y_train)
            loss = criterion(output[~mask], y_train[~mask])
            loss.backward()
            optimizer.step()
            train_losses += loss.item()

        model.eval()
        with torch.no_grad():
            for x_val, y_val in val_loader:
                output = model(x_val)
                mask = torch.isnan(y_val)
                val_losses += criterion(output[~mask], y_val[~mask]).item()
        # if epoch % 50 == 0:
        #     logger.info("LSTM - Epoch: {} - Avg Train Loss: {}".format(epoch, train_losses))
        #     logger.info("LSTM - Epoch: {} - Avg Val Loss: {}".format(epoch, val_losses))
        if early_stopper.early_stop(val_losses):
            break
        scheduler.step()
    
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            output = model(x_test)
            mask = torch.isnan(y_test)
            total_test_loss += criterion(output[~mask], y_test[~mask]).item() * y_test[~mask].shape[0]
    
    del model
    del optimizer
    del scheduler
    del criterion
    del early_stopper
    gc.collect()
    torch.cuda.empty_cache()

    return total_test_loss

