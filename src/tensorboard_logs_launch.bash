#!/bin/bash

# Define the path to the Python script
PYTHON_SCRIPT="gbm/utils/log_utils/logs_merger.py"

# Run the Python script to copy logs
echo "Running Python script to copy logs..."
python "$PYTHON_SCRIPT"

# Define the destination directory for TensorBoard
DEST_DIR="experiments/performed_experiments/all_logs"

# Start TensorBoard on port 6006
echo "Starting TensorBoard..."
tensorboard --logdir="$DEST_DIR" --port=6006
