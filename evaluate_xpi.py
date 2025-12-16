from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from datasets import Dataset, ClassLabel
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# Configuration
MODEL_PATH = "./xpi_sentiment_model"
FALLBACK_MODEL_PATH = "./models/stage1_domain_adapted"
TEST_DATA_FILE = 'xpi_labeled_data_augmented.csv'
CM_FILE = "xpi_confusion_matrix.png"

# Labels
NUM_LABELS = 5
LABEL_NAMES = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]

def main():
    # 1. Load the Fine-tuned Model
    print(f"Loading fine-tuned model from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=NUM_LABELS)
        print("✓ Successfully loaded fine-tuned model!")
    except OSError:
        print(f"Fine-tuned model not found at {MODEL_PATH}.")
        print(f"Falling back to Stage 1 model at {FALLBACK_MODEL_PATH}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL_PATH, num_labels=NUM_LABELS)
            print("✓ Successfully loaded Stage 1 model!")
        except OSError:
            print(f"CRITICAL ERROR: Could not find model at {FALLBACK_MODEL_PATH}.")
            print("Please run 'train_stage1_composite.py' first, then 'train_finetuned.py'.")
            return

    # 2. Load XPI Data
    print(f"Loading test data from {TEST_DATA_FILE}...")
    if not os.path.exists(TEST_DATA_FILE):
        print(f"Error: File {TEST_DATA_FILE} not found.")
        return
        
    df = pd.read_csv(TEST_DATA_FILE)
    
    # Create Dataset (No split, use full file for testing)
    dataset = Dataset.from_pandas(df[['text', 'label']])
    dataset = dataset.cast_column('label', ClassLabel(names=LABEL_NAMES))

    # Tokenize
    def tokenize(batch):
        return tokenizer(batch['text'], padding=True, truncation=True, max_length=256)

    print("Tokenizing test data...")
    dataset = dataset.map(tokenize, batched=True)

    # 3. Predict (Inference Only)
    print("Running predictions on XPI dataset...")
    trainer = Trainer(model=model)
    preds = trainer.predict(dataset)
    
    pred_labels = np.argmax(preds.predictions, axis=-1)
    true_labels = preds.label_ids

    # 4. Evaluation Metrics
    print("\n" + "="*60)
    print("FINAL EVALUATION: XPI Dataset (Fine-tuned Model)")
    print("="*60)
    print(classification_report(true_labels, pred_labels, target_names=LABEL_NAMES, zero_division=0))

    # 5. Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title('Confusion Matrix - XPI Test Set (Fine-tuned Model)')
    plt.tight_layout()
    plt.savefig(CM_FILE)
    print(f"Confusion matrix saved to {CM_FILE}")

if __name__ == "__main__":
    main()