import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Carregar o dataset
df_gym = pd.read_csv('C:/Users/Antonio/Desktop/projeto-pos/pos/Machine_Learn_II/Trabalho/gym.csv')

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

# Configuração do modelo Random Forest
random_forest = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=1000)

# Configuração do Cross-Validation com 10 k-folds
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Executando o Cross-Validation
scores = cross_val_score(random_forest, X_scaled, y, cv=kfold, scoring='accuracy')

# Resultados do Cross-Validation
mean_accuracy = scores.mean()
std_accuracy = scores.std()

print(f"Acurácia Média no Cross-Validation: {mean_accuracy:.2%}")
print(f"Desvio Padrão da Acurácia: {std_accuracy:.2%}")
print(f"Acurácias por Fold: {scores}")

# Dividir os dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Treinar o modelo e fazer previsões
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)

# Calcular métricas no conjunto de teste
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted')

print("/nMétricas de Avaliação no Conjunto de Teste:")
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
