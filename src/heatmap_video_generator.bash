#!/bin/bash

# THE CODE HAS TO BE RUN FROM THE ROOT DIRECTORY OF THE PROJECT (i.e. the directory containing the src/ folder)
# >> bash src/heatmap_video_generator.bash <experiment_name> <filter_string>

# Check if at least one argument is provided
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <experiment_name> [filter_string]"
    exit 1
fi

EXPERIMENT_NAME=$1
FILTER_STRING=${2:-""}

# Define the path to the Python script
PYTHON_SCRIPT="src/gbm/utils/image_utils/video_utils.py"

# Set the PYTHONPATH environment variable
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Debug: Print current PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"

# Run the Python script with the provided experiment name and optional filter string
if [ -z "$FILTER_STRING" ]; then
    python "$PYTHON_SCRIPT" "$EXPERIMENT_NAME"
else
    python "$PYTHON_SCRIPT" "$EXPERIMENT_NAME" "$FILTER_STRING"
fi
