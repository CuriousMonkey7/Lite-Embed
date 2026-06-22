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

* `main.py` — the main training/evaluation pipeline
* `precompute_neighborhood.py`— generates the neighborhood JSON file
* `run.sh` — setup and run script
* `llm_neighborhoods.json` — cached coarse/fine class neighborhoods

## Assumptions and Limitations

This implementation is based on the information available in the LiteEmbed paper and several implementation details had to be inferred. The following assumptions were made:

### Assumptions

1. **Loss Weights**
   - λ₁ (Coarse Loss) = 0.5
   - λ₂ (Fine Loss) = 0.5

2. **Fine-Grained Pruning**
   - Candidate fine classes are selected using a hard cosine similarity threshold of **0.25** against image embeddings.
   - If no candidates remain after thresholding, a **Top-5 fallback** strategy is used.

3. **PCA Subspace Split**
   - `k = 4` (following fig5 of the paper)

### Current Limitations
- The quality of the precomputed class neighborhoods has a significant impact on final performance. This implementation currently uses `Qwen/Qwen2.5-7B-Instruct` to generate coarse and fine-grained neighborhoods for each class. Performance may improve by using a stronger LLM or by more closely matching the neighborhood generation procedure used in the original paper.
- The implementation currently reproduces strong results only on a **6-class subset** of the Indian Food Images dataset.
- While the reported behavior of LiteEmbed can be observed on small-scale experiments, the current implementation **does not scale effectively to the full dataset**.
