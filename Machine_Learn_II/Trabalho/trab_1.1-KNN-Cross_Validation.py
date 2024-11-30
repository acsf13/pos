import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Carga e processamento dos dados
df_gym = pd.read_csv('C:/Users/Antonio/Desktop/projeto-pos/pos/Machine_Learn_II/Trabalho/gym.csv')

# Checar valores ausentes
missing_values = df_gym.isnull().sum()
print(f"Valores ausentes:\n{missing_values}\n")

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

# Configuração dos K-Folds
k_folds = 10
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Testar diferentes valores de k no KNN
k_values = range(1, 31)  # Valores de k entre 1 e 30
mean_scores = []
std_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Realizar validação cruzada
    scores = cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy')
    
    # Registrar as médias e desvios padrão das acurácias
    mean_scores.append(scores.mean())
    std_scores.append(scores.std())

# Encontrar o melhor k com base na acurácia média
optimal_k = k_values[np.argmax(mean_scores)]
best_score = max(mean_scores)

print(f"Melhor valor de k: {optimal_k}")
print(f"Acurácia média com validação cruzada para k={optimal_k}: {best_score:.4f}")

# Visualizar os resultados
plt.figure(figsize=(12, 6))
plt.errorbar(k_values, mean_scores, yerr=std_scores, fmt='o-', label='Acurácia Média com Validação Cruzada')
plt.axvline(optimal_k, color='red', linestyle='--', label=f'Melhor k = {optimal_k}')
plt.title('Acurácia Média vs. Número de Vizinhos (k) com Validação Cruzada')
plt.xlabel('Número de Vizinhos (k)')
plt.ylabel('Acurácia Média')
plt.legend()
plt.grid(True)
plt.show()

# Explicação do Código:
# K-Folds com KFold:

# O conjunto de dados é dividido em 10 folds aleatórios.
# Cada fold é usado como conjunto de teste enquanto os outros 9 são usados para treino.
# Cross-Validation com cross_val_score:

# Avalia o modelo K-NN para cada valor de 𝑘
# k.
# Retorna as acurácias para cada uma das 10 iterações do K-Fold.
# Cálculo da Média e Desvio Padrão:

# Calcula a média e o desvio padrão das acurácias obtidas nos 10 folds.
# Identificação do Melhor 
# 𝑘
# k:

# Seleciona o 
# 𝑘
# k que maximiza a acurácia média.
# Visualização:

# Um gráfico mostra a acurácia média e os intervalos de erro (desvio padrão) para cada valor de 
# 𝑘
# k.
# Benefícios:
# O modelo é avaliado de forma robusta, minimizando o impacto de divisões específicas dos dados.
# O desvio padrão ajuda a entender a variabilidade do desempenho para diferentes divisões.