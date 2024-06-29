import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, f1_score
import numpy as np
import torch.nn as nn
from .data_preparation import load_and_preprocess_data
from .model import RefinedRNN
from .optimizer import GSGD

def guided_training(network, optimizer, loss_function, x_batches, y_batches, verset_x, verset_response, rho, versetnum, revisitNum, device, epochs):
    prev_error = float('inf')
    psi = []
    dataset_X, dataset_y = [], []
    avgBatchLosses = []
    revisit = False
    is_guided = False
    loopCount = -1
    batch_nums = len(x_batches)

    # Shuffle batches
    shuffled_batch_indxs = np.random.permutation(batch_nums)
    new_X = np.copy(x_batches[shuffled_batch_indxs])
    new_y = np.copy(y_batches[shuffled_batch_indxs])

    for epoch in range(epochs):
        getVerificationData = True
        for et in range(batch_nums):
            if getVerificationData:
                versetindxs = shuffled_batch_indxs[:versetnum]
                verset_x = np.array(new_X[versetindxs])
                verset_response = np.array(new_y[versetindxs])
                batch_nums -= versetnum
                new_X = np.delete(new_X, versetindxs, axis=0)
                new_y = np.delete(new_y, versetindxs, axis=0)
                getVerificationData = False

            iteration = et + 1
            loopCount += 1
            x_inst = new_X[et]
            y_inst = new_y[et]
            dataset_X.append(x_inst)
            dataset_y.append(y_inst)

            # Train network
            train_network(network, x_inst.to(device=device), y_inst.to(device=device), loss_function, optimizer)

            # Get verification data loss
            veridxperms = np.random.permutation(versetnum)
            veridxperm = veridxperms[0]
            verloss = getErrorMSE(veridxperm, verset_x, verset_response, network, loss_function)
            pos = -1 if verloss < prev_error else 1

            # Revisit previous batches of data and recalculate their losses only
            if revisit:
                revisit_dataX = np.array(dataset_X)
                revisit_dataY = np.array(dataset_y)
                loopend = min(loopCount, revisitNum - 1)
                currentBatchNumber = loopCount - 1
                for i in range(loopend, loopCount, -1):
                    currentBatchNumber -= 1
                    lossofrevisit = getErrorMSE(currentBatchNumber, revisit_dataX, revisit_dataY, network, loss_function)
                    psi[currentBatchNumber] = np.append(psi[currentBatchNumber], (-1 * pos) * (prev_error - lossofrevisit))

            current_batch_error = prev_error - verloss
            psi.append(current_batch_error)
            prev_error = verloss
            revisit = True

            # Check to see if it's time for GSGD
            if iteration % rho == 0:
                avgBatchLosses = [np.mean(p) for p in psi]
                is_guided = True
                for k in range(loopCount):
                    avgBatchLosses = np.append(avgBatchLosses, np.mean(psi[k]))

                this_dataX = np.array(dataset_X)
                this_dataY = np.array(dataset_y)
                avgBatchLosses_idxs = np.argsort(avgBatchLosses)[::-1]
                min_repeat = min(rho // 2, len(avgBatchLosses))
                for r in range(int(min_repeat)):
                    if avgBatchLosses[r] > 0:
                        guidedIdx = avgBatchLosses_idxs[r]
                        x_inst = this_dataX[guidedIdx]
                        y_inst = this_dataY[guidedIdx]
                        train_network(network, x_inst.to(device=device), y_inst.to(device=device), loss_function, optimizer)
                        verIDX = np.random.permutation(versetnum)[0]
                        verLoss = getErrorMSE(verIDX, verset_x, verset_response, network, loss_function)
                        prev_error = verLoss
                avgBatchLosses, psi, dataset_X, dataset_y = [], [], [], []
                loopCount, revisit, is_guided = -1, False, False

def train_and_evaluate(ticker, data_file, start_date, end_date, input_size, hidden_size, num_layers, output_size,
                       seq_length, dropout, learning_rate, num_epochs, batch_size, patience, k_folds, test_size, run):
    """
    Train and evaluate the GSGD RNN model.

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
        float: ROC-AUC score.
        list: Training losses.
        list: Validation losses.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_full, X_test, y_train_full, y_test, close_scaler = load_and_preprocess_data(ticker, data_file, start_date,
                                                                                        end_date, seq_length, test_size)

    kf = KFold(n_splits=k_folds, shuffle=False)
    fold = 1
    all_accuracies = []
    all_f1_scores = []
    all_roc_aucs = []
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

        model = RefinedRNN(input_size, hidden_size, num_layers, output_size, dropout, seq_length, device).to(device)
        criterion = nn.MSELoss()
        optimizer = GSGD(model.parameters(), lr=learning_rate, alpha=0.99, beta=0.999, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_loss = float('inf')
        counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                outputs = model(X_val_tensor)
                val_loss = criterion(outputs, y_val_tensor.unsqueeze(1)).item()
                scheduler.step(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    counter = 0
                else:
                    counter += 1

                #if counter >= patience:
                #    print('Early stopping...')
                #    break

            train_losses.append(loss.item())
            val_losses.append(val_loss)

            print(f'Run {run}, Fold {fold}, GSGD Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss}')

        fold += 1
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

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

    accuracy = np.mean(np.abs(predicted - y_test) < 1.0)

    y_test_binary = (y_test > y_test.mean()).astype(int)
    predicted_binary = (predicted > y_test.mean()).astype(int)

    f1 = f1_score(y_test_binary, predicted_binary)
    roc_auc = roc_auc_score(y_test_binary, predicted)  # Use continuous predictions for ROC-AUC

    print(f'MAE: {mae}, RMSE: {rmse}, Accuracy: {accuracy * 100}%, F1 Score: {f1}, ROC-AUC: {roc_auc}')
    return accuracy, predicted, y_test, f1, roc_auc, all_train_losses, all_val_losses

