import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

<<<<<<< HEAD
#Load dataset
df_gym = pd.read_csv('C:/Users/acsfarias/Desktop/Pós-Ciência de Dados/pos/Machine_Learn_II/Trabalho/gym.csv')
df_gym.head()  
=======
# Carga e processamento dos dados
df_gym = pd.read_csv('C:/Users/Antonio/Desktop/projeto-pos/pos/Machine_Learn_II/Trabalho/gym.csv')
>>>>>>> fc919c1e947e7f0ad04953e6812ffb1375e6118c

# Checar valores ausentes
missing_values = df_gym.isnull().sum()

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

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Treinamento do modelo KNN
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

# Previsões
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Acurácias
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Matrizes de confusão
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)

# Exibir acurácias como tabela
print("\nResultados de Acurácia:")
accuracy_results = pd.DataFrame({
    "Dataset": ["Treinamento", "Teste"],
    "Acurácia": [train_accuracy, test_accuracy]
})
print(accuracy_results)

# Plotar as acurácias
plt.figure(figsize=(8, 5))
plt.bar(accuracy_results["Dataset"], accuracy_results["Acurácia"], color=['blue', 'orange'])
plt.title('Acurácia do Modelo')
plt.ylabel('Acurácia')
plt.ylim(0, 1)
plt.show()

# Plotar matrizes de confusão
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Matriz de confusão do treinamento
sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Matriz de Confusão - Treinamento')
axes[0].set_xlabel('Previsões')
axes[0].set_ylabel('Valores Reais')

# Matriz de confusão do teste
sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Matriz de Confusão - Teste')
axes[1].set_xlabel('Previsões')
axes[1].set_ylabel('Valores Reais')

plt.tight_layout()
plt.show()
