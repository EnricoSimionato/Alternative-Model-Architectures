#!/bin/bash

# THE CODE HAS TO BE RUN FROM THE ROOT DIRECTORY OF THE PROJECT (i.e., the directory containing the src/ folder)
# Usage:
# bash src/tensorboard_logs_launch.bash --models model_filter_1 model_filter_2 ... --experiments experiment_filter_1 experiment_filter_2 ...

# Defining the path to the Python script
PYTHON_SCRIPT="src/neuroflex/utils/log_utils/logs_merger.py"

# Defining the destination directory for TensorBoard
DEST_DIR="src/experiments/results/all_logs"

# Initializing arrays for models and experiments
models=()
experiments=()

# Parsing command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --models)
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do
        models+=("$1")
        shift
      done
      ;;
    --experiments)
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do
        experiments+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Constructing the arguments string for the Python script
model_args=""
if [[ ${#models[@]} -gt 0 ]]; then
  model_args="--models ${models[*]}"
fi

experiment_args=""
if [[ ${#experiments[@]} -gt 0 ]]; then
  experiment_args="--experiments ${experiments[*]}"
fi

# Running the Python script to copy logs
echo "Running Python script to copy logs..."
python "$PYTHON_SCRIPT" $model_args $experiment_args

# Startomg TensorBoard on port 6006
echo "Starting TensorBoard..."
tensorboard --logdir="$DEST_DIR" --port=6006
