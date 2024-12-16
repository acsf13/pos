import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd

# Carregar o dataset
# Substitua pelo caminho correto do arquivo se necessário
df = pd.read_csv('C:/Users/Antonio/Desktop/projeto-pos/pos/Machine_Learn_II/Trabalho/gym.csv')

# Visualizar as primeiras linhas do dataset
print(df.head())

# 1. Pré-processamento dos dados
# Verificar valores nulos e tratar se necessário
print("Valores nulos por coluna:")
print(df.isnull().sum())

# Remover ou imputar valores nulos (se existirem)
df = df.dropna()

# Converter variáveis categóricas em variáveis dummy (One-Hot Encoding)
df = pd.get_dummies(df, columns=['Gender', 'Workout_Type'], drop_first=True)

# Separar as features (variáveis independentes) e a variável alvo (Calories_Burned)
X = df.drop(columns=['Calories_Burned'])
y = df['Calories_Burned']

# Selecionar atributos contínuos e variável alvo
cont_features = [
    "Weight (kg)", "Height (m)", "Session_Duration (hours)",
    "Fat_Percentage", "Water_Intake (liters)", "BMI"
]
target = "Calories_Burned"

X = df[cont_features]
y = df[target]

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar o modelo e o GridSearchCV para RandomForestRegressor
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf_model = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,  # Cross-validation com 5 divisões
    scoring="r2",
    n_jobs=-1
)

# Executar o GridSearchCV no conjunto de treino
grid_search.fit(X_train, y_train)

# Melhor combinação de hiperparâmetros e desempenho correspondente
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Melhores parâmetros:", best_params)
print("Melhor desempenho (R²):", best_score)