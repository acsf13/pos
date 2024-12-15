import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Carregar o dataset
df_gym = pd.read_csv('/mnt/data/gym.csv')  # Substituir pelo caminho correto caso necessário

# Remover linhas com valores ausentes na variável alvo
df_gym = df_gym.dropna(subset=['Experience_Level'])

# Codificar variáveis categóricas
label_encoders = {}
categorical_columns = ['Gender', 'Workout_Type']
for col in categorical_columns:
    le = LabelEncoder()
    df_gym[col] = le.fit_transform(df_gym[col])
    label_encoders[col] = le

# Definir features (X) e target (y)
X = df_gym.drop(columns=['Experience_Level'])
y = df_gym['Experience_Level']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir os dados para treino e teste com 25% dos dados para teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Configuração do modelo Random Forest com o melhor n_estimators do exercício 1
random_forest = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=1000)  # Melhor n_estimators ajustado

# Treinar o modelo e fazer previsões
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

# Calcular métricas no conjunto de teste
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted')

# Exibir resultados das métricas
print("Métricas de Avaliação no Conjunto de Teste:")
print(f"Acurácia: {accuracy:.2%}")
print(f"Precisão: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-Score: {f1:.2%}")

# Gerar gráfico das métricas
metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 6))
plt.bar(metrics, values, color='skyblue')
plt.ylim(0, 1)
plt.title('Métricas de Avaliação do Modelo Random Forest')
plt.ylabel('Valores')
plt.xlabel('Métricas')
plt.show()
