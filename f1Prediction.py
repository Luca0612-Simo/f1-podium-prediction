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

races = races[races['year'] >= 2000]

df = pd.merge(results, races, on='raceId')

cols = ['grid', 'circuitId', 'year', 'positionOrder']
data = df[cols].copy()

data['grid'] = data['grid'].replace(0, 24)