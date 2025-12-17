import pandas as pd
import torch
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import argparse

def evaluate(data_path, model_path, batch_size=32):
    # 1. Device Management (Crucial for performance)
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    # 2. Data Loading & Split Replication
    # CRITICAL: We must use the exact same random_state as training to separate the test set
    df = pd.read_csv(data_path)
    
    # Check if we have enough data to split
    if len(df) < 5:
        raise ValueError("Dataset too small to split.")

    _, X_test, _, y_test = train_test_split(
        df['text'].tolist(), 
        df['label'].tolist(), 
        test_size=0.2, 
        random_state=42 # Must match train_finetuned.py
    )
    
    print(f"Evaluating on {len(X_test)} unseen test samples.")

    # 3. Pipeline Initialization
    # We load the tokenizer from the same path to ensure vocab consistency
    classifier = pipeline(
        "text-classification", 
        model=model_path, 
        tokenizer=model_path,
        device=device
    )

    # 4. Batched Inference
    # The pipeline handles tokenization and batching internally here
    predictions = []
    print("Running inference...")
    for output in tqdm(classifier(X_test, batch_size=batch_size, truncation=True), total=len(X_test)):
        predictions.append(output['label'])

    # 5. Label Alignment
    # The pipeline returns strings (e.g., "positive"), but y_test might be ints (0, 1, 2)
    # We need to standardize. Assuming model config has id2label.
    
    # Get mapping from model config
    id2label = classifier.model.config.id2label
    label2id = classifier.model.config.label2id
    
    # Convert predictions to IDs for metric calculation
    pred_ids = [label2id[p] for p in predictions]
    
    # 6. Metrics
    print("\nClassification Report:")
    print(classification_report(y_test, pred_ids, target_names=list(label2id.keys())))

    # 7. Visualization
    cm = confusion_matrix(y_test, pred_ids)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(label2id.keys()), 
                yticklabels=list(label2id.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test Set Only)')
    plt.savefig('confusion_matrix_test_set.png')
    print("Confusion matrix saved to confusion_matrix_test_set.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="xpi_labeled_data_augmented.csv")
    parser.add_argument("--model_path", type=str, default="./xpi_sentiment_model")
    args = parser.parse_args()
    
    evaluate(args.data_path, args.model_path)