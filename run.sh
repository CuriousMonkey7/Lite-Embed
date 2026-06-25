#!/bin/bash
set -e

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

# create venv if missing
[ -d ".venv" ] || uv venv

# install dependencies
uv sync

# activate venv
source .venv/bin/activate

ZIP_FILE="indian-food-images-dataset.zip"
DATASET_DIR="Indian Food Images Dataset"

# Download only if zip is missing
if [ ! -f "$ZIP_FILE" ]; then
    echo "Downloading dataset..."
    curl -L -o "$ZIP_FILE" \
        "https://www.kaggle.com/api/v1/datasets/download/iamsouravbanerjee/indian-food-images-dataset"
else
    echo "Dataset zip already exists, skipping download."
fi

# Unzip only if dataset directory is missing
if [ ! -d "$DATASET_DIR" ]; then
    echo "Extracting dataset..."
    unzip -q "$ZIP_FILE" -d "./Indian Food Images Dataset/"
else
    echo "Dataset already extracted, skipping unzip."
fi

echo "Starting main.py ..."
echo "Output will be shown in the terminal and saved to out.log"
echo "Started at: $(date)"
echo

WANDB_MODE=disabled python main.py 2>&1 | tee out.log

echo
echo "Finished at: $(date)"