import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Carga e processamento dos dados
df_gym = pd.read_csv('C:/Users/Antonio/Desktop/projeto-pos/pos/Machine_Learn_II/Trabalho/gym.csv')

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


####################################################################################################################
# 1.1 Aplique o método Cross-Validation para 10 k-folds (cv =10).
####################################################################################################################

# Configuração dos K-Folds
k_folds = 10
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Instanciar o modelo Random Forest
random_forest = RandomForestClassifier(random_state=42, n_jobs=-1)

# Avaliar o modelo com validação cruzada
scores = cross_val_score(random_forest, X_scaled, y, cv=kf, scoring='accuracy')

# Resultados
mean_score = scores.mean()
std_score = scores.std()

mean_score, std_score, scores

# Imprimir resultados
print("Resultados da Validação Cruzada (Random Forest):")
print(f"Acurácia Média: {mean_score:.4f}")
print(f"Desvio Padrão: {std_score:.4f}")
print(f"Acurácias por Fold: {scores}")

# Gerar gráficos
# Gráfico de barras das acurácias por fold
plt.figure(figsize=(10, 6))
plt.bar(range(1, k_folds + 1), scores, color='blue', alpha=0.7)
plt.axhline(mean_score, color='red', linestyle='--', label=f'Média ({mean_score:.4f})')
plt.title('Acurácia por Fold (Random Forest - Cross Validation)')
plt.xlabel('Fold')
plt.ylabel('Acurácia')
plt.xticks(range(1, k_folds + 1))
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt.close()

# Gráfico de distribuição das acurácias
plt.figure(figsize=(10, 6))
plt.hist(scores, bins=10, color='green', alpha=0.7, edgecolor='black')
plt.axvline(mean_score, color='red', linestyle='--', label=f'Média ({mean_score:.4f})')
plt.title('Distribuição das Acurácias (Random Forest - Cross Validation)')
plt.xlabel('Acurácia')
plt.ylabel('Frequência')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt.close()


# Os resultados da validação cruzada foram apresentados, e os gráficos gerados incluem:

# Gráfico de barras: Mostra a acurácia em cada fold, com uma linha indicando a média.
# Histograma: Mostra a distribuição das acurácias obtidas nos folds, destacando a média.


####################################################################################################################
# 1.2 Altere o parâmetro quantidade de árvores geradas na floresta (n_estimator), podendo ser de 100 a 1000.
####################################################################################################################

# Testar diferentes valores de n_estimators
n_estimators_range = range(100, 1100, 100)
mean_scores_n_estimators = []

for n in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf, X_scaled, y, cv=kf, scoring='accuracy')
    mean_scores_n_estimators.append(scores.mean())

# Gerar gráfico de acurácia média vs. n_estimators
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, mean_scores_n_estimators, marker='o', linestyle='-', color='blue', alpha=0.7)
plt.title('Impacto de n_estimators na Acurácia Média (Random Forest)')
plt.xlabel('n_estimators')
plt.ylabel('Acurácia Média')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(n_estimators_range)
plt.show()
plt.close()

# Exibir os resultados em formato tabular
results_df = pd.DataFrame({
    'n_estimators': n_estimators_range,
    'Mean Accuracy': mean_scores_n_estimators
})

# Salvar os resultados em um arquivo CSV
results_df.to_csv('random_forest_n_estimators_results.csv', index=False)
print("Resultados salvos em 'random_forest_n_estimators_results.csv'")

# Exibir resultados no console
print(results_df)


# Encontre a melhor configuração utilizando as métricas Acuray, Precision e Recall.