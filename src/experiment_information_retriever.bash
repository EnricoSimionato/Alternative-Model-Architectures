#!/bin/bash

# THE CODE HAS TO BE RUN FROM THE ROOT DIRECTORY OF THE PROJECT (i.e. the directory containing the src/ folder)
# >> bash src/experiment_information_retriever.bash <experiment_name>

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <experiment_name>"
    exit 1
fi

EXPERIMENT_NAME=$1

# Define the path to the Python script
PYTHON_SCRIPT="src/gbm/utils/experiment_pipeline/config_printer.py"

# Run the Python script with the provided experiment name
python3 "$PYTHON_SCRIPT" "$EXPERIMENT_NAME"
