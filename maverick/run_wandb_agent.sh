#!/bin/bash

# This script simply executes the wandb agent command.
# The Sweep ID is passed as the first argument ($1) from ScriptRunConfig.
echo "Wrapper script starting wandb agent..."
echo "Sweep ID: $1"

# Execute the wandb agent command using the argument passed to this script
wandb agent "$1" 

# Optional: Add error checking or logging if needed
exit_code=$?
echo "Wandb agent finished with exit code: $exit_code"
exit $exit_code