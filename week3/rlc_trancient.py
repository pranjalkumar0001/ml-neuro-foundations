import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def get_circuit_transient_data(seq_length=50, total_samples=1000):
    """
    Generates deterministic, physical voltage data for an underdamped RLC circuit.
    Splits into 80% Train and 20% Test for validation.
    """
    # 1. Simulate the physics (Decaying sine wave)
    t = np.linspace(0, 15, total_samples)
    voltage = np.exp(-0.2 * t) * np.cos(2 * np.pi * 0.75 * t)
    
    # 2. Create the rolling windows
    X, Y = [], []
    for i in range(len(voltage) - seq_length - 1):
        X.append(voltage[i : i + seq_length])     # The past 50 steps
        Y.append(voltage[i + seq_length])         # The 51st step (Target)
        
    # --- THE MEMORY FIX IS HERE ---
    # We bind the list into a NumPy array first, then convert to a Tensor.
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1) 
    Y = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(-1) 
    # ------------------------------
    
    # 3. TIME-SERIES SPLIT (80% Train, 20% Test)
    split_idx = int(0.8 * len(X))
    
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_test, Y_test = X[split_idx:], Y[split_idx:]
    
    return X_train, Y_train, X_test, Y_test
X_train, Y_train, X_test, Y_test = get_circuit_transient_data()

print(f"Training Data: {X_train.shape} -> Targets: {Y_train.shape}")
print(f"Testing Data:  {X_test.shape} -> Targets: {Y_test.shape}")

rnn_layer = nn.RNN(1, 64, batch_first=True)
head = nn.Linear(64, 1)
loss_fn = nn.MSELoss()
optimiser = optim.Adam(list(rnn_layer.parameters())+list(head.parameters()), lr=0.01)
for epoches in range(150):
    optimiser.zero_grad()
    _, final_hidden_state = rnn_layer(X_train)
    prediction = head(final_hidden_state.squeeze(0))
    loss = loss_fn(prediction, Y_train)
    loss.backward()
    optimiser.step()
    if epoches % 30 == 0:
        print(f"at epoch = {epoches}loss is {loss.item()}")
print(f"loss after 150 epoches is {loss.item()}")

# EVALUATION ON TEST SET
print("evaluating in test data set")
with torch.no_grad():
    _ , final_hidden_state_test = rnn_layer(X_test)
    prediction_test = head(final_hidden_state_test.squeeze(0))
    loss_test = loss_fn(prediction_test, Y_test)

print(f"loss in the text data is {loss_test.item():.7f}")