import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

try:
    results = pd.read_csv('archive/results.csv')
    races = pd.read_csv('archive/races.csv')
except FileNotFoundError:
    print("No se encuentran los archivos")
    exit()

races = races[races['year'] >= 2000]

df = pd.merge(results, races, on='raceId')

cols = ['circuitId', 'year', 'constructorId', 'positionOrder']
data = df[cols].copy()

data['podium'] = data['positionOrder'].apply(lambda x: 1 if 1 <= x <= 3 else 0)

data = data.drop('positionOrder', axis=1)

print("Dataset listo para el modelo:")
print(data.head())
print(f"\nTotal de carreras a analizar: {len(data)}")

X = data.drop('podium', axis=1)
y = data['podium']

param_grid = {
    'min_samples_split': [2, 10, 20, 50],
    'max_depth': [5,7,10,15, None]
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

print("Mejores parámetros encontrados:", grid.best_params_)
print("Mejor score", grid.best_score_)