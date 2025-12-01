import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


negative_table = pd.read_csv('negative.csv')
positive_table = pd.read_csv('positive.csv')

negative = negative_table.iloc[:, 4].tolist()
positive = positive_table.iloc[:, 4].tolist()

print(len(negative))
print(len(positive))