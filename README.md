# BERT Fake News Detection

A machine learning project that fine-tunes BERT (Bidirectional Encoder Representations from Transformers) to detect fake news articles using the Kaggle Fake News Competition dataset.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Citation](#citation)

## Overview

Fake news has become a prevalent issue in today's society, with the ability to spread rapidly and cause serious harm through misinformation. This project implements a fake news detection system using a fine-tuned BERT model to classify news articles as real or fake.

**Key Problem**: Incidents, like a fake tweet claiming insulin is free causing Eli Lilly's stock to plummet 4.4%, demonstrate the real-world impact of misinformation.

## Features

- Fine-tuned BERT model for binary classification (real vs fake news)
- Uses the Kaggle Fake News Competition dataset (20K+ articles)
- Google Colab compatible with GPU acceleration
- Model checkpointing and persistence
- Comprehensive evaluation metrics
- Automated data preprocessing pipeline

## Dataset

The project uses the [Kaggle Fake News Competition dataset](https://www.kaggle.com/competitions/fake-news):

| Split | Articles | Description |
|-------|----------|-------------|
| Train | 20,761 | Labeled news articles (0=reliable, 1=unreliable) |
| Test | 5,193 | Unlabeled articles for evaluation |

**Features**: `id`, `title`, `author`, `text`, `label`

## Installation

### Prerequisites

```bash
pip install transformers datasets torch pandas numpy
```

### Additional Requirements (for Kaggle integration)
- Kaggle API credentials (`kaggle.json`)
- Google Drive access (for checkpoint storage)

### Setup in Google Colab

1. **Upload Kaggle credentials**:
   ```python
   from google.colab import files
   uploaded = files.upload()  # Upload your kaggle.json
   ```

2. **Setup Kaggle API**:
   ```bash
   !mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Mount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Quick Start

### 1. Download Dataset
```bash
!kaggle competitions download -c fake-news
!unzip fake-news.zip
```

### 2. Load and Preprocess Data
```python
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Load data
df = pd.read_csv("train.csv")
df = df[df.text.notna()]  # Remove null texts
df = df[df.text.apply(lambda text: text != '')]  # Remove empty texts

# Setup tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = Dataset.from_pandas(df)
```

### 3. Train Model
```python
from transformers import BertForSequenceClassification
import torch

# Load model
model = BertForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-yelp-polarity"
).cuda()

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# Training loop implementation in notebook...
```

## Usage

### Data Processing Pipeline

```python
def tokenization(example):
    """Tokenize text with BERT tokenizer"""
    return tokenizer(example["text"], padding='max_length', truncation=True)

# Apply tokenization
dataset = dataset.map(tokenization, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])

# Create train/val/test splits
dataset = dataset.train_test_split(test_size=0.2, shuffle=True)
val_test_dataset = dataset["test"].train_test_split(test_size=0.5)
```

### Model Training

```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(num_epochs):
    # Training phase
    total_train_loss = 0
    for batch in train_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        predictions = model(**batch)
        predictions.loss.backward()
        total_train_loss += predictions.loss.item()
        optimizer.step()
    
    # Validation phase
    total_val_loss = 0
    for batch in val_loader:
        with torch.no_grad():
            predictions = model(**batch)
        total_val_loss += predictions.loss.item()
    
    # Save checkpoint every 100 epochs
    if (epoch + 1) % 100 == 0:
        model.save_pretrained(f'{CHECKPOINT_FOLDER}/{epoch + 1}')
```

### Model Evaluation

```python
# Load checkpoint
loaded_model = BertForSequenceClassification.from_pretrained(
    f"{CHECKPOINT_FOLDER}/{epoch_number}"
).cuda()

# Evaluate
total_correct = 0
for batch in test_loader:
    with torch.no_grad():
        predictions = loaded_model(**batch)
        preds = (predictions.logits[:,0] < predictions.logits[:,1]).long()
        total_correct += (preds == batch["labels"]).sum().item()

accuracy = total_correct / len(test_loader.dataset)
print(f"Test Accuracy: {accuracy}")
```

## Model Architecture

| Component | Details |
|-----------|---------|
| **Base Model** | BERT-base-uncased |
| **Pre-training** | textattack/bert-base-uncased-yelp-polarity |
| **Task** | Binary classification (fake news detection) |
| **Input** | Tokenized text (max 512 tokens) |
| **Output** | Binary classification (0=real, 1=fake) |
| **Optimizer** | Adam (lr=1e-5) |

## Results

| Metric | Value | Notes |
|--------|-------|-------|
| **Test Accuracy** | 0.574 | Slightly better than random (0.5) |
| **Dataset** | Kaggle Fake News | 20K+ training articles |

> **Note**: The modest performance suggests room for improvement through hyperparameter tuning, data augmentation, or ensemble methods.

## References

- [BERT Paper](http://arxiv.org/abs/1810.04805) - Devlin et al., 2018
- [Kaggle Competition](https://kaggle.com/competitions/fake-news) - Fake News Dataset
- [Transformers Library](https://huggingface.co/transformers/) - HuggingFace
- [Related Work](https://doi.org/10.48550/arXiv.1905.12616) - Zellers et al., 2019
