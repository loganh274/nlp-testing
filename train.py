import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

import gensim.downloader as api
import joblib
# Load data from CSV files (table containing comments, sentiment labels, etc.)
negative_table = pd.read_csv('negative.csv')
positive_table = pd.read_csv('positive.csv')

# Extract the relevant text data from the tables
negative = negative_table.iloc[:, 4].tolist()
positive = positive_table.iloc[:, 4].tolist()

# Combine the data into a single corpus and create corresponding labels
corpus = negative + positive
labels = [0] * len(negative) + [1] * len(positive)

# Load pre-trained word vectors
word_vectors = api.load("glove-wiki-gigaword-100")

# Function to compute document embeddings by averaging word vectors
def get_document_embedding(text, vectors, dim=100):
    words = text.lower().split()
    embeddings = [vectors[word] for word in words if word in vectors]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(dim)

# Compute document embeddings for the entire corpus
X = np.array([get_document_embedding(doc, word_vectors) for doc in corpus])
y = np.array(labels)

# Train a logistic regression model on the document embeddings
model = LogisticRegression()
model.fit(X, y)

# Save the trained model to a file
joblib.dump(model, 'sentiment_model.pkl')
print("Model saved!")