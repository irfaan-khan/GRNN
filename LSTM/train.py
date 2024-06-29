import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, f1_score

import numpy as np
import torch.nn as nn
from .data_preparation import load_and_preprocess_data
from .model import LSTMModel


def train_and_evaluate(ticker, data_file, start_date, end_date, input_size, hidden_size, num_layers, output_size,
                       seq_length, dropout, learning_rate, num_epochs, batch_size, patience, k_folds, test_size, run):
    """
    Train and evaluate the LSTM model.

    Args:
        ticker (str): Stock ticker symbol.
        data_file (str): Path to the data file.
        start_date (str): Start date for data fetching.
        end_date (str): End date for data fetching.
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units.
        num_layers (int): Number of RNN layers.
        output_size (int): Number of output features.
        seq_length (int): Sequence length.
        dropout (float): Dropout rate.
        learning_rate (float): Learning rate.
        num_epochs (int): Number of epochs.
        batch_size (int): Batch size.
        patience (int): Patience for early stopping.
        k_folds (int): Number of folds for cross-validation.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        float: Average accuracy of the model.
        np.array: Predicted values.
        np.array: Actual values.
        float: F1 score.
        np.array: Predicted probabilities.
        list: Training losses.
        list: Validation losses.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_full, X_test, y_train_full, y_test, close_scaler = load_and_preprocess_data(ticker, data_file, start_date,
                                                                                        end_date, seq_length, test_size)

    kf = KFold(n_splits=k_folds, shuffle=False)
    fold = 1
    all_accuracies = []
    all_train_losses = []
    all_val_losses = []

    for train_index, val_index in kf.split(X_train_full):
        print(f'Fold {fold}')
        X_train, X_val = X_train_full[train_index], X_train_full[val_index]
        y_train, y_val = y_train_full[train_index], y_train_full[val_index]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout, seq_length, device).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_loss = float('inf')
        counter = 0
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            train_losses.append(epoch_train_loss / len(train_loader))

            model.eval()
            with torch.no_grad():
                outputs = model(X_val_tensor)
                val_loss = criterion(outputs, y_val_tensor.unsqueeze(1)).item()
                val_losses.append(val_loss)
                scheduler.step(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    counter = 0
                else:
                    counter += 1

            print(f'Run {run}, Fold {fold}, LSTM Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss}')
            #if counter >= patience:
            #    print('Early stopping...')
            #    break

        fold += 1

    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X_test, dtype=torch.float32).to(device))
        predicted = outputs.cpu().numpy()

    predicted = predicted.reshape(-1, output_size)
    predicted = close_scaler.inverse_transform(predicted)
    y_test = y_test.reshape(-1, 1)
    y_test = close_scaler.inverse_transform(y_test)

    mae = mean_absolute_error(y_test, predicted)
    rmse = np.sqrt(mean_squared_error(y_test, predicted))

    threshold = 1.0
    accurate_predictions = np.abs(predicted - y_test) < threshold
    accuracy = np.mean(accurate_predictions)

    y_test_binary = (y_test > y_test.mean()).astype(int)
    predicted_binary = (predicted > y_test.mean()).astype(int)

    f1 = f1_score(y_test_binary, predicted_binary)
    roc_auc = roc_auc_score(y_test_binary, predicted)  # Use continuous predictions for ROC-AUC

    print(f'MAE: {mae}, RMSE: {rmse}, Accuracy: {accuracy * 100}%, F1 Score: {f1}, ROC-AUC: {roc_auc}')
    return accuracy, predicted, y_test, f1, roc_auc, train_losses, val_losses


