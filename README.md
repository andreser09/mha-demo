# Real-Time RUL Prediction with MultiHeadAttentionLSTM

This repository provides a solution for predicting Remaining Useful Life (RUL) in real time using a trained MultiHeadAttentionLSTM model on streaming sensor data, originally designed for the CMAPSS dataset. This README guides you through setup, data preparation, real-time prediction, and creating an API endpoint to serve predictions.

## Overview

The code:
- Uses a pre-trained MultiHeadAttentionLSTM model to predict RUL from sensor data streams.
- Processes incoming data incrementally, maintaining a sliding window of sequence_len time steps.
- Normalizes data on-the-fly using precomputed parameters from training.
- Includes error handling for NaN predictions and debugging tools.

## Prerequisites

- Python: 3.8+
- Dependencies (install via pip):
  pip install pytorch_lightning
- Hardware: CPU or GPU (CUDA-enabled if using GPU).
- Files:
  - Trained model checkpoint (e.g., checkpoint.ckpt).
  - Normalization parameters (e.g., FD001_params.npy) from training.

Setup

1. Clone the Repository:

2. Build with Dockerfile inside Omniparse-hai folder.
   There are some difficulties with flash_attn and getting the dependencies right
   I managed to get it to run using this Dockerfile configuration from other project using a T4 gpu
   with nvidia-driver-535-server which has cuda 12.4, but in the docker using cuda 11.8.
   

3. Inside the docker run predict.py

## Code Structure

Main Components

- MultiHeadAttentionLSTM: The PyTorch model class defined in attn_lstm.py.
- Normalization: Function to preprocess incoming sensor data (use z-core for this example)
- Prediction: Real-time RUL prediction with a sliding window.

Key Files
- attn_lstm.py: Contains the MultiHeadAttentionLSTM class (if separate).
- predict.py: Main script for real-time prediction (below).
- checkpoint.pth: Trained model weights.
- norm_params.npy: Normalization parameters (mean/std for z-score).

## Usage

Data Format
The model expects data in the CMAPSS format:
- Columns: ['engine_id', 'timestamp', 'alt', 'tra', 'mach', 't2', 't24', 't30', 't50',
             'p2', 'p15', 'p30', 'nf', 'nc', 'epr', 'ps30', 'phi', 'nrf', 'nrc', 'bpr', 
            'farb', 'htbleed', 'nf_dmd', 'pcnfr_dmd', 'w31', 'w32'] (26 total).
- Description:
  - engine_id: Engine/unit identifier (integer).
  - timestamp: Time step (integer, incrementing per engine).
  - alt, tra, mach: operational conditions
  - the rest: sensor readings

Example row:
[1, 1, 0.5, 0.6, 0.7, 0.05, 0.1, ..., 1.0]