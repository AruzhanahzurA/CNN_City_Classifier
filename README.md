# CNN City Classifier — Astana vs. Almaty

A PyTorch project that trains a convolutional neural network to classify cityscape photos as **Astana** (label `0`) or **Almaty** (label `1`). The workflow lives in a Jupyter notebook and was originally built for Google Colab.

## Overview

| Item | Details |
|------|---------|
| Task | Binary image classification |
| Classes | Astana (`0`), Almaty (`1`) |
| Dataset | ~55 images per city in `Astana/` and `Almata/` |
| Train / test split | 50 + 5 images per city (100 train, 10 test) |
| Input size | 128×128 RGB, ImageNet normalization |
| Framework | PyTorch |

## Project structure

```
CNN_city_classifier/
├── Ast_or_Almaty.ipynb.ipynb   # Main notebook (training + evaluation)
├── Astana/                     # Astana cityscape images
├── Almata/                     # Almaty cityscape images
└── README.md
```

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [Pillow](https://python-pillow.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)

Install dependencies:

```bash
pip install torch torchvision pillow numpy scikit-learn
```

For Google Colab, PyTorch and most libraries are preinstalled; you only need to mount Google Drive if loading data from there.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/AruzhanahzurA/CNN_City_Classifier.git
cd CNN_City_Classifier
```

### 2. Run the notebook

Open `Ast_or_Almaty.ipynb.ipynb` in **Google Colab** or a local Jupyter environment.

**Option A — Use the bundled dataset (local or Colab)**

Point the notebook at the folders included in this repo:

```python
Astana_path = "Astana"
Almata_path = "Almata"
```

**Option B — Google Drive (original Colab workflow)**

1. Upload the `Astana/` and `Almata/` folders to your Google Drive.
2. Mount Drive in Colab.
3. Update the paths in the notebook, for example:

```python
Astana_path = "/content/drive/MyDrive/Astana"
Almata_path = "/content/drive/MyDrive/Almata"
```

> **Note:** In the data-loading cell, make sure Astana images are loaded with `Astana_path` (not `left_bank_path`, which will raise a `NameError`).

Run all cells in order: load data → define model → train → evaluate.

## Model architecture

The `Almaty_or_Astana` CNN includes:

- **Conv block 1:** Conv2d (3→32) → BatchNorm → ReLU → MaxPool
- **Conv block 2:** Conv2d (32→64) → BatchNorm → ReLU → MaxPool
- **Dropout:** 0.25
- **Classifier:** Linear (65536→512) → BatchNorm → ReLU → Dropout → Linear (512→2)

Training settings:

- Optimizer: Adam (`lr=0.01`)
- Loss: CrossEntropyLoss
- Epochs: 50

## Results

Reported metrics from the notebook:

| Split | F1 score | Accuracy |
|-------|----------|----------|
| Train | 0.943 | 94% |
| Test  | 0.727 | 70% |

Evaluation uses scikit-learn `f1_score` and `classification_report`. With only 10 test images, test performance can vary between runs.

## License

MIT License.
