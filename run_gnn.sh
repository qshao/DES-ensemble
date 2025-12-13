#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1  

echo "=== GNN Pipeline Started ==="

# 1. Process Data
#echo "[1/2] Processing Data..."
#python 1_process_data.py

# 2. Train GNN
#echo "[2/2] Training GNN..."
#python train_gnn.py

python advanced_pipeline.py

echo "=== Training Complete ==="
