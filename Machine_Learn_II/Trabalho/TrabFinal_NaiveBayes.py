import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB  # Substituindo o MultinomialNB por GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Carregar o dataset
df_gym = pd.read_csv('/home/zeka/Projetos/Pos/pos/data/gym.csv')

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

# Normalizar os dados para o intervalo [0, 1] com MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Configuração da validação cruzada com 10 folds
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Criando o modelo Naive Bayes Gaussiano
nb = GaussianNB()

# Avaliação com validação cruzada (sem GridSearch, pois GaussianNB não tem parâmetros a serem ajustados dessa maneira)
accuracy = cross_val_score(nb, X_scaled, y, cv=cv, scoring='accuracy').mean()
precision = cross_val_score(nb, X_scaled, y, cv=cv, scoring='precision_weighted').mean()
recall = cross_val_score(nb, X_scaled, y, cv=cv, scoring='recall_weighted').mean()

print("Resultados com validação cruzada:")
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Divisão Holdout (70% treino, 30% teste)
X_train_holdout, X_test_holdout, y_train_holdout, y_test_holdout = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Treinar o modelo no conjunto de treino
nb.fit(X_train_holdout, y_train_holdout)

# Previsões no conjunto de teste
y_pred_holdout = nb.predict(X_test_holdout)

# Avaliação no conjunto Holdout
accuracy_holdout = accuracy_score(y_test_holdout, y_pred_holdout)
precision_holdout = precision_score(y_test_holdout, y_pred_holdout, average='weighted')
recall_holdout = recall_score(y_test_holdout, y_pred_holdout, average='weighted')
f1_holdout = f1_score(y_test_holdout, y_pred_holdout, average='weighted')

print("\nResultados no conjunto Holdout:")
print(f"Acurácia: {accuracy_holdout:.4f}")
print(f"Precisão: {precision_holdout:.4f}")
print(f"Recall: {recall_holdout:.4f}")
print(f"F1-score: {f1_holdout:.4f}")

# Matriz de confusão
conf_matrix = confusion_matrix(y_test_holdout, y_pred_holdout)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()

# Relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test_holdout, y_pred_holdout))

# Gráfico comparativo das métricas
metrics = {
    'Acurácia': [accuracy, accuracy_holdout],
    'Precisão': [precision, precision_holdout],
    'Recall': [recall, recall_holdout],
    'F1-score': [None, f1_holdout]  # F1-score não calculado na validação cruzada
}
metrics_df = pd.DataFrame(metrics, index=['Validação Cruzada', 'Holdout'])
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title('Comparação de Métricas entre Validação Cruzada e Holdout')
plt.ylabel('Valor')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.show()


# Treinar o modelo com os melhores parâmetros encontrados
best_model.fit(X_train_holdout, y_train_holdout)

# Previsões no conjunto de teste
y_pred_holdout = best_model.predict(X_test_holdout)

# Avaliação no conjunto Holdout
accuracy_holdout = accuracy_score(y_test_holdout, y_pred_holdout)
precision_holdout = precision_score(y_test_holdout, y_pred_holdout, average='weighted')
recall_holdout = recall_score(y_test_holdout, y_pred_holdout, average='weighted')
f1_holdout = f1_score(y_test_holdout, y_pred_holdout, average='weighted')

print("\nResultados no conjunto Holdout:")
print(f"Acurácia: {accuracy_holdout:.4f}")
print(f"Precisão: {precision_holdout:.4f}")
print(f"Recall: {recall_holdout:.4f}")
print(f"F1-score: {f1_holdout:.4f}")

# Matriz de confusão
conf_matrix = confusion_matrix(y_test_holdout, y_pred_holdout)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()

# Relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test_holdout, y_pred_holdout))

# Gráfico comparativo das métricas
metrics = {
    'Acurácia': [accuracy, accuracy_holdout],
    'Precisão': [precision, precision_holdout],
    'Recall': [recall, recall_holdout],
    'F1-score': [None, f1_holdout]  # F1-score não calculado na validação cruzada
}
metrics_df = pd.DataFrame(metrics, index=['Validação Cruzada', 'Holdout'])
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title('Comparação de Métricas entre Validação Cruzada e Holdout')
plt.ylabel('Valor')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.show()
