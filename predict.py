import pandas as pd
import numpy as np
import torch
from collections import defaultdict, deque
from attn_lstm import MultiHeadAttentionLSTM
from CMAPSSDataset import CMAPSSDataset

# Load the trained model
model = MultiHeadAttentionLSTM(
    cell='lstm',
    sequence_len=30, #Time window
    feature_num=24,   # op1, op2, op3 + 21 sensors
    hidden_dim=100,
    fc_layer_dim=100,
    rnn_num_layers=3,
    output_dim=1,     # RUL regression
    fc_activation='relu',
    attention_order=[],
    feature_head_num=4,
    sequence_head_num=4,
    fc_dropout=0.9,
    rnn_dropout=0.2,
    bidirectional=False,
    return_attention_weights=False
)

checkpoint_path = 'pretrained_model.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
state_dict = {key.replace("net.", ""): value for key, value in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load normalization parameters from training
norm_params = np.load('FD001_params.npy')  # Shape: (24, 2) for z-score
norm_type = 'z-score'  # Match training
sequence_len = 30
feature_cols = CMAPSSDataset.OPERATION_COLS + CMAPSSDataset.SENSOR_COLS  # 24 features

# Buffer to store sequences per engine id
sequence_buffer = defaultdict(lambda: deque(maxlen=sequence_len))

# Function to normalize a single row
def normalize_row(row, norm_params, norm_type):
    mean, std = norm_params[:, 0], norm_params[:, 1]
    if np.any(std == 0):
        std = np.where(std == 0, 1.0, std)  # Avoid division by zero
    return (row - mean) / std

# Function to process a new data row and predict RUL
def predict_rul(new_row):
    # new_row: array-like with [engine_id, timestamp, 'alt', 'tra', 'mach', 't2', 't24', 't30', 't50', \
    #				'p2', 'p15', 'p30', 'nf', 'nc', 'epr', 'ps30', 'phi', 'nrf', 'nrc', 'bpr', \
    #				 'farb', 'htbleed', 'nf_dmd', 'pcnfr_dmd', 'w31', 'w32']
    engine_id = int(new_row[0])
    features = new_row[2:26]  # Extract op1, op2, op3, s1, ..., s21 (24 features)

    # Normalize the new data point
    normalized_features = normalize_row(features, norm_params, norm_type)

    #The time window for attention is 30 so a prediction can be done only if we have the past 30 periods before
    # Update the sequence buffer for this engine
    sequence_buffer[engine_id].append(normalized_features)

    # Check if we have enough data for a prediction
    if len(sequence_buffer[engine_id]) < sequence_len:
        # Pad with the first value if sequence is incomplete
        current_seq = list(sequence_buffer[engine_id])
        padded_seq = np.pad(
            current_seq,
            ((sequence_len - len(current_seq), 0), (0, 0)),
            mode='edge'
        )
    else:
        padded_seq = np.array(sequence_buffer[engine_id])

    # Prepare input tensor: shape (1, sequence_len, feature_num)
    input_data = torch.tensor(padded_seq, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict RUL
    with torch.no_grad():
        rul_pred = model(input_data)

    return rul_pred.item()

# Example
# Generate 50 rows of data for one engine (engine_id=0001)
stream_data = []

for t in range(1, 51):
    row = [
        1,  # id
        t,  # timestamp -> convert to integer. The dataset is hourly so the RUL is the hours remaining
        0.5 + np.random.randn() * 0.01,  # op1
        0.6 + np.random.randn() * 0.01,  # op2
        0.7 + np.random.randn() * 0.01,  # op3
    ]
    # 21 sensor values with slight upward trend and noise
    sensors = [np.random.randn() * 0.1 + (i * 0.05 + t * 0.001) for i in range(1, 22)]
    row.extend(sensors)
    stream_data.append(row)

# Real-time prediction loop
for row in stream_data:
    rul = predict_rul(row)
    print(f"Engine ID: {int(row[0])}, Time: {int(row[1])}, Predicted RUL: {rul:.2f}")
