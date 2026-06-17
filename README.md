# [Unofficial] Implementation of LiteEmbed: Adapting CLIP to Rare Classes

This repository contains an unofficial implementation of the paper [*LiteEmbed*](https://arxiv.org/pdf/2601.09661). It attempts to reproduce their Top-1 Accuracy on the [Indian Food Images Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset?select=List+of+Indian+Foods.txt).

## Quick Start

The easiest way to run everything is:

```bash
chmod +x run.sh
./run.sh

```

If `run.sh` is not executable, you can also run:

```bash
bash run.sh

```

> `run.sh` installs `uv` (if not already installed), sets up the virtual environment, installs project dependencies, downloads and extracts the dataset (if needed), and runs `main.py`.

## Main Files

* [main.py](https://www.google.com/search?q=main.py) — the main training/evaluation pipeline
* [precompute_neighborhood.py](https://www.google.com/search?q=precompute_neighborhood.py) — generates the neighborhood JSON file
* [run.sh](run.sh) — setup and run script
* [llm_neighborhoods.json](https://www.google.com/search?q=llm_neighborhoods.json) — cached coarse/fine class neighborhoods
