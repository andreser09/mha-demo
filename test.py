import torch
import torch.nn as nn
from attn_lstm import MultiHeadAttentionLSTM
from CMAPSSDataset import CMAPSSDataset
import numpy as np
import pandas as pd

# 1. Instantiate the model with the same parameters used during training
model = MultiHeadAttentionLSTM(
    cell='lstm',
    sequence_len=30,
    feature_num=24,
    hidden_dim=100,
    fc_layer_dim=100,
    rnn_num_layers=3,
    output_dim=1,
    fc_activation='relu',
    attention_order=[],
    feature_head_num=4,
    sequence_head_num=4,
    fc_dropout=0.9,
    rnn_dropout=0.2,
    bidirectional=False,
    return_attention_weights=False
)

checkpoint_path = 'checkpoints/lightning_logs/version_11/checkpoints/checkpoint-epoch=18-val_rmse=15.5607.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Use 'cuda' if on GPU

# Adjust the state_dict by removing the "net." prefix
state_dict = checkpoint['state_dict']  # Adjust key if different
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("net.", "")  # Remove "net." prefix
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)  # Adjust key if your checkpoint has a different structure
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

test_df = pd.read_csv('CMAPSSData/test_FD001.txt', sep=' ', header=None)
test_df = test_df.iloc[:, :26]  # Drop extra columns if present
test_df.columns = CMAPSSDataset.DATASET_COLS

# Load RUL ground truth (optional, for evaluation)
rul_df = pd.read_csv('CMAPSSData/RUL_FD001.txt', header=None)
final_rul = rul_df.values.squeeze()

# Training parameters (must match what was used during training)
sequence_len = 30  # Example value
norm_type = 'z-score'  # Example: match training
cluster_operations = False  # Example: FD001/FD003 don’t need clustering
norm_by_operations = False
max_rul = 125  # Example: common value for CMAPSS

# Load normalization parameters from training (assume saved)
# Example: norm_params shape (24, 2) for z-score (mean, std per feature)
norm_params = np.load('FD001_params.npy')  # Replace with actual path

# Initialize the dataset
test_dataset = CMAPSSDataset(
    data_df=test_df,
    sequence_len=sequence_len,
    final_rul=final_rul,  # Include if you have it, else None
    norm_params=norm_params,  # Use training params
    norm_type=norm_type,
    max_rul=max_rul,
    only_final=True,  # Common for test sets (last sequence only)
    init=False,  # Manual initialization
    cluster_operations=cluster_operations,
    norm_by_operations=norm_by_operations
)

# Apply preprocessing steps manually
if cluster_operations:
    CMAPSSDataset.clustering_operations([test_dataset])

if norm_type:
    test_dataset.normalization()

test_dataset.gen_sequence()

# Access the prepared data
sequence_array = test_dataset.sequence_array  # Shape: (num_sequences, sequence_len, feature_num)
label_array = test_dataset.label_array       # Shape: (num_sequences,) or (num_sequences, sequence_len)

# Convert to PyTorch tensor
input_data = torch.tensor(sequence_array, dtype=torch.float32).to(device)

# Make predictions
with torch.no_grad():
    predictions = model(input_data)

# Output predictions
print("Predictions shape:", predictions.shape)  # (num_sequences, 1)
print("Predictions:", predictions.cpu().numpy())

# Compare with ground truth (if final_rul provided)
if final_rul is not None:
    print("Ground truth RUL:", label_array)
    print("Mean Absolute Error:", np.mean(np.abs(predictions.cpu().numpy().flatten() - label_array)))
