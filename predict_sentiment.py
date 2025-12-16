"""
Single Comment Sentiment Prediction
Evaluate sentiment for a single unknown comment using the fine-tuned model.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import sys

# Configuration
MODEL_PATH = "./xpi_sentiment_model"
FALLBACK_MODEL_PATH = "./models/stage1_domain_adapted"

# Labels
LABEL_NAMES = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
LABEL_SCORES = {
    0: "⭐ Very Negative (1/5)",
    1: "⭐⭐ Negative (2/5)",
    2: "⭐⭐⭐ Neutral (3/5)",
    3: "⭐⭐⭐⭐ Positive (4/5)",
    4: "⭐⭐⭐⭐⭐ Very Positive (5/5)"
}

def load_model():
    """Load the fine-tuned model and tokenizer."""
    print("Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        print(f"✓ Successfully loaded fine-tuned model from {MODEL_PATH}\n")
    except OSError:
        print(f"⚠ Fine-tuned model not found at {MODEL_PATH}.")
        print(f"Falling back to Stage 1 model at {FALLBACK_MODEL_PATH}...\n")
        try:
            tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL_PATH)
            print(f"✓ Successfully loaded Stage 1 model from {FALLBACK_MODEL_PATH}\n")
        except OSError:
            print(f"ERROR: Could not find model at {FALLBACK_MODEL_PATH}.")
            print("Please run 'train_finetuned.py' or 'train_stage1_composite.py' first.")
            sys.exit(1)
    
    return model, tokenizer

def predict_sentiment(text, model, tokenizer):
    """Predict sentiment for a single comment."""
    # Tokenize
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get probabilities
    probabilities = torch.softmax(logits, dim=-1)
    pred_label = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][pred_label].item()
    
    return pred_label, confidence, probabilities[0].cpu().numpy()

def display_results(text, pred_label, confidence, probabilities):
    """Display prediction results in a readable format."""
    print("="*70)
    print("SENTIMENT ANALYSIS RESULT")
    print("="*70)
    print(f"\nComment: {text}\n")
    print(f"Predicted Sentiment: {LABEL_SCORES[pred_label]}")
    print(f"Confidence: {confidence*100:.1f}%")
    
    print("\nDetailed Breakdown:")
    print("-"*70)
    for i, (label_name, prob) in enumerate(zip(LABEL_NAMES, probabilities)):
        bar_length = int(prob * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"{label_name:15} | {bar} | {prob*100:5.1f}%")
    print("="*70 + "\n")

def main():
    """Main function."""
    model, tokenizer = load_model()
    
    print("Interactive Sentiment Analysis")
    print("Type 'quit' to exit, or enter a comment to analyze:\n")
    
    while True:
        user_input = input("Enter comment: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not user_input:
            print("Please enter a comment.\n")
            continue
        
        pred_label, confidence, probabilities = predict_sentiment(user_input, model, tokenizer)
        display_results(user_input, pred_label, confidence, probabilities)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line argument provided
        model, tokenizer = load_model()
        comment = " ".join(sys.argv[1:])
        pred_label, confidence, probabilities = predict_sentiment(comment, model, tokenizer)
        display_results(comment, pred_label, confidence, probabilities)
    else:
        # Interactive mode
        main()
