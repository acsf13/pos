import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

# Dividir os dados em conjuntos de treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Treinamento do modelo de Regressão Linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 3. Avaliação do modelo
# Fazer previsões no conjunto de teste
y_pred = modelo.predict(X_test)

# Calcular métricas de desempenho
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erro quadrático médio (MSE): {mse}")
print(f"Coeficiente de determinação (R²): {r2}")

# 4. Determinação das variáveis mais importantes
# Os coeficientes do modelo indicam a importância das variáveis
coeficientes = pd.DataFrame({'Variável': X.columns, 'Coeficiente': modelo.coef_})
coeficientes = coeficientes.sort_values(by='Coeficiente', ascending=False)

print("/nImportância das variáveis:")
print(coeficientes)

# 5. Visualizar as variáveis mais importantes
plt.figure(figsize=(10, 6))
plt.barh(coeficientes['Variável'], coeficientes['Coeficiente'])
plt.xlabel('Coeficiente')
plt.ylabel('Variável')
plt.title('Importância das Variáveis')
plt.show()

# Gráfico 1: Previsões vs Valores reais
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.title('Previsões vs Valores Reais')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.show()

# Gráfico 2: Resíduos
residuos = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuos, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Previsões')
plt.ylabel('Resíduos')
plt.title('Resíduos vs Previsões')
plt.show()

# Gráfico 3: Distribuição dos resíduos
plt.figure(figsize=(10, 6))
plt.hist(residuos, bins=30, alpha=0.7, edgecolor='k')
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.title('Distribuição dos Resíduos')
plt.show()
