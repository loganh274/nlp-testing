# NLP Sentiment Analysis

A 5-point sentiment classifier for product feedback comments, fine-tuned on domain-specific data.

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Label new data
```bash
python label_web.py
```

### Train model
```bash
python train_finetuned.py
```

### Run predictions
```bash
python predict.py
```

## Labels
- 0: Very Negative
- 1: Negative
- 2: Neutral
- 3: Positive
- 4: Very Positive