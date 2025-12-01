# NLP Sentiment Analysis Project

A sentiment analysis project using GloVe word embeddings and Logistic Regression to classify text as positive or negative sentiment.

## Setup

### 1. Install Dependencies

First, install the required libraries:

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Ensure you have two CSV files in the project directory:
- `negative.csv` - Contains negative sentiment comments (column 5 should have the text)
- `positive.csv` - Contains positive sentiment comments (column 5 should have the text)

## Project Structure

- `train.py` - Trains the sentiment model on your corpus and saves it
- `test.py` - Tests the trained model on new sentences
- `sentiment_model.pkl` - The saved trained model (generated after running train.py)
- `requirements.txt` - Project dependencies

## Usage

### Step 1: Train the Model

Run the training script to create the sentiment model:

```bash
python train.py
```

This will:
- Load your positive and negative CSV files
- Generate GloVe embeddings for each sentence
- Train a Logistic Regression model
- Save the model as `sentiment_model.pkl`

**Output:** `Model saved!`

### Step 2: Test on New Sentences

Run the test script to make predictions:

```bash
python test.py
```

This will:
- Load the pre-trained model
- Load GloVe word vectors
- Classify each test sentence as positive or negative
- Display the sentiment and confidence score

**Example Output:**
```
Text: 'I love this product!'
Sentiment: Positive (Confidence: 95.23%)

Text: 'This is terrible'
Sentiment: Negative (Confidence: 87.14%)
```

### Step 3: Test Your Own Sentences

Edit `test.py` and modify the `test_sentences` list with your own text:

```python
test_sentences = [
    "Your sentence here",
    "Another sentence to test"
]
```

Then run `python test.py` again.

## How It Works

1. **Data Loading** - Reads positive/negative sentiment data from CSV files
2. **Embeddings** - Converts each sentence into a 100-dimensional GloVe vector by averaging word embeddings
3. **Training** - Trains a Logistic Regression classifier on the embeddings
4. **Prediction** - Classifies new sentences using the trained model with confidence scores

## Requirements

- Python 3.7+
- See `requirements.txt` for package versions

## Notes

- First run will download GloVe vectors (~100MB) - subsequent runs use cached version
- Model accuracy depends on the quality and quantity of training data
- Confidence scores range from 0 to 1 (0-100%)