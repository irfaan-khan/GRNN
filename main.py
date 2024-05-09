import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Parameters
input_size = 5  # Number of input features
hidden_size = 128  # Number of hidden units
num_layers = 2  # Number of RNN layers
output_size = 1  # Number of output features
seq_length = 5  # Sequence length
learning_rate = 0.001
num_epochs = 1000
batch_size = 64
dropout = 0.2
patience = 5

# Download and preprocess data
data = yf.download('GOOGL', start='2015-01-01', end='2018-01-01')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Fit a separate StandardScaler for 'Close' prices
close_scaler = StandardScaler()
close_prices = data[['Close']]
close_scaler.fit(close_prices)

# Prepare data for training
X = []
y = []
for i in range(len(data_normalized) - seq_length):
    X.append(data_normalized[i:i+seq_length])
    y.append(data_normalized[i+seq_length][3])  # Close price is at index 3
X = np.array(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define RNN model with dropout
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# Define GSGD optimizer
class GSGD(torch.optim.Optimizer):
    def __init__(self, params, lr, alpha, beta):
        defaults = dict(lr=lr, alpha=alpha, beta=beta)
        super(GSGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data      # Compute the gradient of the loss function
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['v'] = torch.zeros_like(p.data) # Running average of the Gradient
                    state['g'] = torch.zeros_like(p.data) # Squared Gradient

                v = state['v']
                g = state['g']

                state['step'] += 1
                lr = group['lr']
                alpha = group['alpha'] # Decay rates
                beta = group['beta']   # Decay rates

                v = alpha * v + (1 - alpha) * grad
                g = beta * g + (1 - beta) * grad**2

                p.data -= lr * v / (torch.sqrt(g) + 1e-8)

# Initialize model, loss function, and optimizer
model = RNN(input_size, hidden_size, num_layers, output_size, dropout).cuda()
criterion = nn.MSELoss()
optimizer = GSGD(model.parameters(), lr=learning_rate, alpha=0.99, beta=0.999)  # Adjust alpha and beta as needed


# Training loop with early stopping
best_loss = float('inf')
counter = 0
for epoch in range(num_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor.cuda())
        loss = criterion(outputs, y_test_tensor.unsqueeze(1).cuda()).item()
        if loss < best_loss:
            best_loss = loss
            counter = 0
        else:
            counter += 1

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss}')

    # Early stopping
    #if counter >= patience:
    #    print('Early stopping...')
    #    break

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor.cuda())
    predicted = outputs.cpu().numpy()

# Inverse transform to get actual values
predicted = predicted.reshape(-1, output_size)
predicted = close_scaler.inverse_transform(predicted)
y_test = y_test.reshape(-1, 1)
y_test = close_scaler.inverse_transform(y_test)

# Calculate MAE and RMSE
mae = mean_absolute_error(y_test, predicted)
rmse = np.sqrt(mean_squared_error(y_test, predicted))

# Calculate custom accuracy: percentage of predictions close to actual value
threshold = 1.0  # Define your own threshold
accurate_predictions = np.abs(predicted - y_test) < threshold
accuracy = np.mean(accurate_predictions)

print(f'MAE: {mae}, RMSE: {rmse}, Accuracy: {accuracy * 100}%')

# Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(predicted, label='Predicted')
plt.legend()
plt.show()