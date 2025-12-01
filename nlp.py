import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

import gensim.downloader as api




negative_table = pd.read_csv('negative.csv')
positive_table = pd.read_csv('positive.csv')

negative = negative_table.iloc[:, 4].tolist()
positive = positive_table.iloc[:, 4].tolist()

corpus = negative + positive
labels = [0] * len(negative) + [1] * len(positive)

print(labels)