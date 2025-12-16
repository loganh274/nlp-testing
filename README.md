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

pip install -r [requirements.txt](http://_vscodecontentref_/1)