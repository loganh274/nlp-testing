from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, ClassLabel
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import matplotlib.pyplot as plt
import torch

# Configuration
USE_3_CLASSES = False
MODEL_NAME = "distilbert-base-uncased"

# Load augmented data
df = pd.read_csv('xpi_labeled_data_augmented.csv')

if USE_3_CLASSES:
    label_mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
    df['label'] = df['label'].map(label_mapping)
    NUM_LABELS = 3
    LABEL_NAMES = ["Negative", "Neutral", "Positive"]
else:
    NUM_LABELS = 5
    LABEL_NAMES = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]

print("Class distribution:")
print(df['label'].value_counts().sort_index())

# Calculate class weights for imbalance
label_counts = Counter(df['label'])
total = len(df)
class_weights = {i: total / (NUM_LABELS * label_counts.get(i, 1)) for i in range(NUM_LABELS)}
print(f"\nClass weights: {class_weights}")

# Create HuggingFace dataset
dataset = Dataset.from_pandas(df[['text', 'label']])

# Cast label column to ClassLabel for stratified splitting
dataset = dataset.cast_column('label', ClassLabel(names=LABEL_NAMES))

dataset = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column='label')

# Load tokenizer and model
print(f"\nLoading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=256)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Custom Trainer with class weights
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(
            [class_weights[i] for i in range(len(class_weights))],
            dtype=torch.float32
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        weights = self.class_weights.to(logits.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    report = classification_report(labels, preds, target_names=LABEL_NAMES, output_dict=True, zero_division=0)
    
    return {
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score']
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='macro_f1',
    greater_is_better=True,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics,
)

print("\nStarting training...")
trainer.train()

# Final evaluation
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)
preds = trainer.predict(dataset['test'])
pred_labels = np.argmax(preds.predictions, axis=-1)
true_labels = preds.label_ids

print(classification_report(true_labels, pred_labels, target_names=LABEL_NAMES, zero_division=0))

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix - Fine-tuned Model')
plt.tight_layout()
plt.savefig('confusion_matrix_finetuned.png')
print("\nConfusion matrix saved to confusion_matrix_finetuned.png")

# Save model
trainer.save_model('./xpi_sentiment_model')
tokenizer.save_pretrained('./xpi_sentiment_model')
print("\nModel saved to ./xpi_sentiment_model")