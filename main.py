import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
import os
import sys

from GSGD_RNN.train import train_and_evaluate as train_gsgd_rnn
from RNN.train import train_and_evaluate as train_rnn
from LSTM.train import train_and_evaluate as train_lstm

class DualLogger:
    def __init__(self, filepath):
        self.console = sys.stdout
        self.file = open(filepath, 'w')

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()

def plot_roc_curve(y_test, predicted_probabilities, label):
    fpr, tpr, _ = roc_curve(y_test, predicted_probabilities)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    return roc_auc

def save_results_to_csv(results, filepath):
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)

def save_results_to_html(results, filepath):
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_html(filepath, index=False)

def plot_results_table(results, filepath):
    import pandas as pd
    import matplotlib.pyplot as plt
    from pandas.plotting import table

    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    tbl = table(ax, df, loc='center', cellLoc='center', colWidths=[0.1] * len(df.columns))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)
    plt.savefig(filepath)

def pad_sequences(sequences, maxlen):
    padded_sequences = np.zeros((len(sequences), maxlen), dtype=np.float32)
    for i, seq in enumerate(sequences):
        if isinstance(seq, list):
            seq = np.concatenate([np.array(subseq, dtype=np.float32).flatten() for subseq in seq])
        else:
            seq = np.array(seq, dtype=np.float32).flatten()
        padded_sequences[i, :min(len(seq), maxlen)] = seq[:maxlen]
    return padded_sequences

def main():
    input_size = 8
    hidden_size = 128
    num_layers = 2
    output_size = 1
    seq_length = 5
    learning_rate = 0.001
    num_epochs = 200
    batch_size = 64
    dropout = 0.2
    patience = 5
    k_folds = 10
    test_size = 0.1

    data_file = 'data/GOOGL_data.csv'
    start_date = '2006-01-01'
    end_date = '2021-01-01'
    ticker = 'GOOGL'

    total_runs = 30

    accuracies_gsgd_rnn = []
    f1_scores_gsgd_rnn = []
    roc_aucs_gsgd_rnn = []
    predictions_gsgd_rnn = []

    accuracies_rnn = []
    f1_scores_rnn = []
    roc_aucs_rnn = []
    predictions_rnn = []

    accuracies_lstm = []
    f1_scores_lstm = []
    roc_aucs_lstm = []
    predictions_lstm = []

    actual_values = None
    all_train_losses = []
    all_val_losses = []

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    log_file_path = os.path.join(results_dir, 'logs.txt')
    sys.stdout = DualLogger(log_file_path)

    for run in range(1, total_runs + 1):
        message = f'Run {run}/{total_runs}\n'
        print(message)

        accuracy_gsgd_rnn, predicted_gsgd_rnn, y_test, f1_gsgd_rnn, roc_auc_gsgd_rnn, train_losses, val_losses = train_gsgd_rnn(
            ticker, data_file, start_date, end_date, input_size, hidden_size, num_layers, output_size, seq_length,
            dropout,
            learning_rate, num_epochs, batch_size, patience, k_folds, test_size, run
        )
        accuracies_gsgd_rnn.append(accuracy_gsgd_rnn)
        f1_scores_gsgd_rnn.append(f1_gsgd_rnn)
        roc_aucs_gsgd_rnn.append(roc_auc_gsgd_rnn)
        predictions_gsgd_rnn.append(predicted_gsgd_rnn)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        accuracy_rnn, predicted_rnn, y_test, f1_rnn, roc_auc_rnn, train_losses, val_losses = train_rnn(
            ticker, data_file, start_date, end_date, input_size, hidden_size, num_layers, output_size, seq_length,
            dropout,
            learning_rate, num_epochs, batch_size, patience, k_folds, test_size, run
        )
        accuracies_rnn.append(accuracy_rnn)
        f1_scores_rnn.append(f1_rnn)
        roc_aucs_rnn.append(roc_auc_rnn)
        predictions_rnn.append(predicted_rnn)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        accuracy_lstm, predicted_lstm, y_test, f1_lstm, roc_auc_lstm, train_losses, val_losses = train_lstm(
            ticker, data_file, start_date, end_date, input_size, hidden_size, num_layers, output_size, seq_length,
            dropout,
            learning_rate, num_epochs, batch_size, patience, k_folds, test_size, run
        )
        accuracies_lstm.append(accuracy_lstm)
        f1_scores_lstm.append(f1_lstm)
        roc_aucs_lstm.append(roc_auc_lstm)
        predictions_lstm.append(predicted_lstm)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        if actual_values is None:
            actual_values = y_test

    max_train_length = max(len(seq) for seq in all_train_losses)
    max_val_length = max(len(seq) for seq in all_val_losses)

    padded_train_losses = pad_sequences(all_train_losses, max_train_length)
    padded_val_losses = pad_sequences(all_val_losses, max_val_length)

    avg_accuracy_gsgd_rnn = np.mean(accuracies_gsgd_rnn)
    best_accuracy_gsgd_rnn = np.max(accuracies_gsgd_rnn)
    worst_accuracy_gsgd_rnn = np.min(accuracies_gsgd_rnn)

    avg_accuracy_rnn = np.mean(accuracies_rnn)
    best_accuracy_rnn = np.max(accuracies_rnn)
    worst_accuracy_rnn = np.min(accuracies_rnn)

    avg_accuracy_lstm = np.mean(accuracies_lstm)
    best_accuracy_lstm = np.max(accuracies_lstm)
    worst_accuracy_lstm = np.min(accuracies_lstm)

    avg_f1_gsgd_rnn = np.mean(f1_scores_gsgd_rnn)
    best_f1_gsgd_rnn = np.max(f1_scores_gsgd_rnn)
    worst_f1_gsgd_rnn = np.min(f1_scores_gsgd_rnn)

    avg_f1_rnn = np.mean(f1_scores_rnn)
    best_f1_rnn = np.max(f1_scores_rnn)
    worst_f1_rnn = np.min(f1_scores_rnn)

    avg_f1_lstm = np.mean(f1_scores_lstm)
    best_f1_lstm = np.max(f1_scores_lstm)
    worst_f1_lstm = np.min(f1_scores_lstm)

    avg_roc_auc_gsgd_rnn = np.mean(roc_aucs_gsgd_rnn)
    best_roc_auc_gsgd_rnn = np.max(roc_aucs_gsgd_rnn)
    worst_roc_auc_gsgd_rnn = np.min(roc_aucs_gsgd_rnn)

    avg_roc_auc_rnn = np.mean(roc_aucs_rnn)
    best_roc_auc_rnn = np.max(roc_aucs_rnn)
    worst_roc_auc_rnn = np.min(roc_aucs_rnn)

    avg_roc_auc_lstm = np.mean(roc_aucs_lstm)
    best_roc_auc_lstm = np.max(roc_aucs_lstm)
    worst_roc_auc_lstm = np.min(roc_aucs_lstm)

    message = f'Average Accuracy for GSGD RNN: {avg_accuracy_gsgd_rnn * 100}%\n'
    print(message)

    message = f'Best Accuracy for GSGD RNN: {best_accuracy_gsgd_rnn * 100}%\n'
    print(message)

    message = f'Worst Accuracy for GSGD RNN: {worst_accuracy_gsgd_rnn * 100}%\n'
    print(message)

    message = f'Average Accuracy for RNN: {avg_accuracy_rnn * 100}%\n'
    print(message)

    message = f'Best Accuracy for RNN: {best_accuracy_rnn * 100}%\n'
    print(message)

    message = f'Worst Accuracy for RNN: {worst_accuracy_rnn * 100}%\n'
    print(message)

    message = f'Average Accuracy for LSTM: {avg_accuracy_lstm * 100}%\n'
    print(message)

    message = f'Best Accuracy for LSTM: {best_accuracy_lstm * 100}%\n'
    print(message)

    message = f'Worst Accuracy for LSTM: {worst_accuracy_lstm * 100}%\n'
    print(message)

    message = f'Average F1 Score for GSGD RNN: {avg_f1_gsgd_rnn}\n'
    print(message)

    message = f'Best F1 Score for GSGD RNN: {best_f1_gsgd_rnn}\n'
    print(message)

    message = f'Worst F1 Score for GSGD RNN: {worst_f1_gsgd_rnn}\n'
    print(message)

    message = f'Average F1 Score for RNN: {avg_f1_rnn}\n'
    print(message)

    message = f'Best F1 Score for RNN: {best_f1_rnn}\n'
    print(message)

    message = f'Worst F1 Score for RNN: {worst_f1_rnn}\n'
    print(message)

    message = f'Average F1 Score for LSTM: {avg_f1_lstm}\n'
    print(message)

    message = f'Best F1 Score for LSTM: {best_f1_lstm}\n'
    print(message)

    message = f'Worst F1 Score for LSTM: {worst_f1_lstm}\n'
    print(message)

    message = f'Average ROC-AUC Score for GSGD RNN: {avg_roc_auc_gsgd_rnn}\n'
    print(message)

    message = f'Best ROC-AUC Score for GSGD RNN: {best_roc_auc_gsgd_rnn}\n'
    print(message)

    message = f'Worst ROC-AUC Score for GSGD RNN: {worst_roc_auc_gsgd_rnn}\n'
    print(message)

    message = f'Average ROC-AUC Score for RNN: {avg_roc_auc_rnn}\n'
    print(message)

    message = f'Best ROC-AUC Score for RNN: {best_roc_auc_rnn}\n'
    print(message)

    message = f'Worst ROC-AUC Score for RNN: {worst_roc_auc_rnn}\n'
    print(message)

    message = f'Average ROC-AUC Score for LSTM: {avg_roc_auc_lstm}\n'
    print(message)

    message = f'Best ROC-AUC Score for LSTM: {best_roc_auc_lstm}\n'
    print(message)

    message = f'Worst ROC-AUC Score for LSTM: {worst_roc_auc_lstm}\n'
    print(message)

    results = {
        'Model': ['GSGD RNN', 'RNN', 'LSTM'],
        'Average Accuracy': [avg_accuracy_gsgd_rnn, avg_accuracy_rnn, avg_accuracy_lstm],
        'Best Accuracy': [best_accuracy_gsgd_rnn, best_accuracy_rnn, best_accuracy_lstm],
        'Worst Accuracy': [worst_accuracy_gsgd_rnn, worst_accuracy_rnn, worst_accuracy_lstm],
        'Average F1 Score': [avg_f1_gsgd_rnn, avg_f1_rnn, avg_f1_lstm],
        'Best F1 Score': [best_f1_gsgd_rnn, best_f1_rnn, best_f1_lstm],
        'Worst F1 Score': [worst_f1_gsgd_rnn, worst_f1_rnn, worst_f1_lstm],
        'Average ROC-AUC': [avg_roc_auc_gsgd_rnn, avg_roc_auc_rnn, avg_roc_auc_lstm],
        'Best ROC-AUC': [best_roc_auc_gsgd_rnn, best_roc_auc_rnn, best_roc_auc_lstm],
        'Worst ROC-AUC': [worst_roc_auc_gsgd_rnn, worst_roc_auc_rnn, worst_roc_auc_lstm]
    }

    save_results_to_csv(results, os.path.join(results_dir, 'results.csv'))
    save_results_to_html(results, os.path.join(results_dir, 'results.html'))
    plot_results_table(results, os.path.join(results_dir, 'results_table.png'))

    plt.figure(figsize=(12, 6))
    plt.plot(actual_values, label='Actual', color='black')
    plt.plot(np.mean(predictions_gsgd_rnn, axis=0), label='GSGD RNN', linestyle='dashed')
    plt.plot(np.mean(predictions_rnn, axis=0), label='RNN', linestyle='dotted')
    plt.plot(np.mean(predictions_lstm, axis=0), label='LSTM', linestyle='dashdot')
    plt.legend()
    plt.title('Actual vs. Average Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.savefig(os.path.join(results_dir, 'actual_vs_predicted.png'))
    plt.show()

    avg_train_losses = np.mean(padded_train_losses, axis=0)
    avg_val_losses = np.mean(padded_val_losses, axis=0)
    plt.figure(figsize=(12, 6))
    plt.plot(avg_train_losses, label='Training Loss')
    plt.plot(avg_val_losses, label='Validation Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'))
    plt.show()

    y_test_binary = (actual_values > actual_values.mean()).astype(int)
    pred_gsgd_rnn_binary = (np.mean(predictions_gsgd_rnn, axis=0) > actual_values.mean()).astype(int)
    pred_rnn_binary = (np.mean(predictions_rnn, axis=0) > actual_values.mean()).astype(int)
    pred_lstm_binary = (np.mean(predictions_lstm, axis=0) > actual_values.mean()).astype(int)

    plt.figure(figsize=(12, 6))
    roc_auc_gsgd_rnn = plot_roc_curve(y_test_binary, np.mean(predictions_gsgd_rnn, axis=0), label='GSGD RNN')
    roc_auc_rnn = plot_roc_curve(y_test_binary, np.mean(predictions_rnn, axis=0), label='RNN')
    roc_auc_lstm = plot_roc_curve(y_test_binary, np.mean(predictions_lstm, axis=0), label='LSTM')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
    plt.show()


if __name__ == '__main__':
    main()
