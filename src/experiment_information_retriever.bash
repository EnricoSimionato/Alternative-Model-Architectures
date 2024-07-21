#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <experiment_name>"
    exit 1
fi

EXPERIMENT_NAME=$1

# Define the path to the Python script
PYTHON_SCRIPT="gbm/utils/experiment_pipeline/config_printer.py"

# Run the Python script with the provided experiment name
python3 "$PYTHON_SCRIPT" "$EXPERIMENT_NAME"
