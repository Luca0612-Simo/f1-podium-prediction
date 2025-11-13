import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

try:
    results = pd.read_csv('results.csv')
    races = pd.read_csv('races.csv')
except FileNotFoundError:
    print("No se encuentran los archivos")
    exit()