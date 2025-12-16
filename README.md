# NLP Sentiment Analysis - XPI Product Feedback

A two-stage, 5-point sentiment classifier for product feedback comments. The model is first domain-adapted on public datasets (Amazon, Google Play, TAWOS), then fine-tuned on labeled XPI feedback data.

## Table of Contents
- [Setup](#setup)
- [Workflow Overview](#workflow-overview)
- [Stage 1: Domain Adaptation](#stage-1-domain-adaptation)
- [Stage 2: Fine-tuning](#stage-2-fine-tuning)
- [Evaluation](#evaluation)
- [Optional: Adding Custom Datasets](#optional-adding-custom-datasets)
- [Labeling New Data](#labeling-new-data)
- [Labels](#labels)

## Setup

### Prerequisites
- Python 3.11+
- NVIDIA GPU with CUDA support (optional but recommended for training)
- Kaggle API credentials (for Amazon Reviews dataset)

### Installation

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### GPU Setup (Optional)

For faster training on NVIDIA GPUs:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU availability:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Workflow Overview

This project uses a **two-stage training approach**:

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Domain Adaptation (Optional but Recommended)        │
│ • Downloads public sentiment datasets (Amazon, Google Play)  │
│ • Trains DistilBERT on composite data for domain knowledge  │
│ Output: ./models/stage1_domain_adapted/                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ Stage 2: Fine-tuning (On Your Labeled Data)                 │
│ • Loads Stage 1 pre-trained model                           │
│ • Fine-tunes on your XPI labeled data                       │
│ • Applies class weighting for imbalanced data               │
│ Output: ./xpi_sentiment_model/                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│ Evaluation (On Full XPI Dataset)                             │
│ • Evaluates fine-tuned model on all XPI feedback            │
│ • Generates classification metrics and confusion matrix     │
│ Output: xpi_confusion_matrix.png                            │
└─────────────────────────────────────────────────────────────┘
```

## Stage 1: Domain Adaptation

This stage trains the model on public sentiment datasets to give it general sentiment knowledge before fine-tuning on your specific domain.

### Datasets Included by Default

The `data_prep.py` script automatically downloads:

1. **Amazon Reviews** (Kaggle) - 50,000 reviews with 1-5 star ratings
2. **Google Play Reviews** (Optional, local CSV) - Finance app reviews
3. **TAWOS/Jira Data** (Optional, local CSV) - Structured feedback with priority levels

### Running Stage 1

```bash
python data_prep.py
python train_stage1_composite.py
```

This will:
- Prepare the composite dataset: `data/composite_dataset`
- Train the Stage 1 model: `models/stage1_domain_adapted`
- Generate training loss graph: `stage1_training_loss.png`

**Output Model**: `models/stage1_domain_adapted`

---

## Stage 2: Fine-tuning

This stage fine-tunes the Stage 1 model on your labeled XPI feedback data.

### Required Data

Place your labeled data in:
```
xpi_labeled_data_augmented.csv
```

Required columns:
- `text` - The feedback text
- `label` - Sentiment label (0-4)

### Running Stage 2

```bash
python train_finetuned.py
```

This will:
- Load the Stage 1 pre-trained model (or fall back to base model if Stage 1 doesn't exist)
- Split data: 80% train, 20% test
- Train for 10 epochs with class weighting
- Save the fine-tuned model: `xpi_sentiment_model`
- Generate confusion matrix: `confusion_matrix_finetuned.png`

**Output Model**: `xpi_sentiment_model`

---

## Evaluation

Evaluate the fine-tuned model on your complete XPI dataset:

```bash
python evaluate_xpi.py
```

This will:
- Load the fine-tuned model (or Stage 1 model as fallback)
- Run predictions on the full `xpi_labeled_data_augmented.csv`
- Print classification metrics (precision, recall, F1-score)
- Generate confusion matrix: `xpi_confusion_matrix.png`

---

## Optional: Adding Custom Datasets

You can add custom datasets to the Stage 1 training by providing local CSV files.

### Google Play Reviews Dataset

1. Download the Google Play Reviews dataset (e.g., from Kaggle)
2. Save it as: `./data/google_play_finance.csv`
3. Required columns: `content` (or `review`) and `score` (or `rating`)

**Example format:**
```csv
content,score
"This app is amazing!",5
"Terrible experience",1
```

4. Run: `python data_prep.py` - it will automatically include Google Play data

### TAWOS / Jira Dataset

1. Export your Jira/TAWOS data as CSV: `./data/tawos_dump.csv`
2. Required columns: `description` (or `body`) and `priority`

**Example format:**
```csv
description,priority
"Critical bug in payment system",Blocker
"Minor UI improvement needed",Minor
```

3. Run: `python data_prep.py` - it will automatically include TAWOS data

### Mapping Priority Levels

The TAWOS importer maps priority to labels:
- `Blocker` → 0 (Very Negative)
- `Critical` → 1 (Negative)
- `Major` → 2 (Neutral)
- `Minor` → 3 (Positive)
- `Trivial` → 4 (Very Positive)

Modify the `priority_map` in `data_prep.py` if your dataset uses different values.

---

## Labeling New Data

Use the interactive labeling GUI to label unlabeled feedback:

```bash
python label_gui.py
```

This opens a GUI where you can:
- View feedback text
- Select sentiment label (0-4)
- Save labeled data to CSV

---

## Labels

The model uses 5 sentiment classes:

| Label | Class | Example |
|-------|-------|---------|
| 0 | Very Negative | "This is broken and unusable. I'm leaving." |
| 1 | Negative | "Essential feature missing. This blocks my work." |
| 2 | Neutral | "It would be nice to have this feature." |
| 3 | Positive | "Good app overall, works well." |
| 4 | Very Positive | "Amazing! Works perfectly, love it." |

---

## Project Structure

```
├── data_prep.py                           # Stage 1: Prepare composite dataset
├── train_stage1_composite.py              # Stage 1: Domain adaptation training
├── train_finetuned.py                     # Stage 2: Fine-tuning on XPI data
├── evaluate_xpi.py                        # Evaluation step
├── label_gui.py                           # Interactive labeling tool
├── xpi_labeled_data_augmented.csv         # Your labeled XPI data
├── xpi_sentiment_model/                   # Final fine-tuned model (deployment)
├── models/
│   └── stage1_domain_adapted/             # Stage 1 pre-trained model
├── data/
│   ├── composite_dataset/                 # Stage 1 training data
│   ├── google_play_finance.csv            # (Optional) Custom dataset
│   └── tawos_dump.csv                     # (Optional) Custom dataset
├── results/                               # Training checkpoints
└── requirements.txt
```

---

## Troubleshooting

### Stage 1 Model Not Found
If you see "Stage 1 model not found", run:
```bash
python data_prep.py
python train_stage1_composite.py
```

### No GPU Detected
Ensure you have:
- NVIDIA drivers installed
- CUDA Toolkit matching your PyTorch version
- Updated PyTorch installation (see GPU Setup section)

Check with:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory Error
If training fails due to GPU memory:
1. Reduce batch size in training script (default: 16)
2. Use smaller max_length (default: 256)
3. Train on CPU instead (slower but works)

---

## Performance Tips

- **For Best Results**: Run all three steps (Stage 1 → Stage 2 → Evaluation)
- **For Quick Testing**: Skip Stage 1 and go straight to Stage 2 (uses base model)
- **For Production**: Use the fine-tuned model saved in `xpi_sentiment_model`