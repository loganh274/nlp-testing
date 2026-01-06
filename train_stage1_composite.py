from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import os

# Configuration
INPUT_DIR = "./data/composite_dataset"
OUTPUT_DIR = "./models/stage1_domain_adapted"
LOSS_GRAPH_FILE = "stage1_training_loss.png"
MODEL_NAME = "microsoft/deberta-v3-base"
NUM_LABELS = 5

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def main():
    # 1. Load Composite Data (Kaggle/TAWOS/etc)
    if not os.path.exists(INPUT_DIR):
        print(f"Error: {INPUT_DIR} not found. Run data_prep.py first.")
        return

    print(f"Loading composite dataset from {INPUT_DIR}...")
    dataset = load_from_disk(INPUT_DIR)

    # Split for validation (Internal validation during training)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # 2. Setup Model & Tokenizer
    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    def tokenize(batch):
        return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=256)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize, batched=True)

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=False,
        learning_rate=1e-5,
        weight_decay=0.1,
        logging_steps=50, # Log frequently for the graph
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        compute_metrics=compute_metrics,
    )

    # 4. Train
    print("Starting Stage 1 Training...")
    trainer.train()

    # 5. Save Model
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 6. Generate Training Loss Graph
    print("Generating Loss Graph...")
    history = trainer.state.log_history
    train_loss = [x['loss'] for x in history if 'loss' in x]
    train_epochs = [x['epoch'] for x in history if 'loss' in x]
    eval_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
    eval_epochs = [x['epoch'] for x in history if 'eval_loss' in x]

    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_loss, label='Training Loss', alpha=0.6)
    if eval_loss:
        plt.plot(eval_epochs, eval_loss, label='Validation Loss', marker='o', linestyle='--')
    
    plt.title('Stage 1 Training: Kaggle/Composite Dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_GRAPH_FILE)
    print(f"Loss graph saved to {LOSS_GRAPH_FILE}")

if __name__ == "__main__":
    main()