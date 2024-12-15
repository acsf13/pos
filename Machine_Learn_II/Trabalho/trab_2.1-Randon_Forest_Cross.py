import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Carregar o dataset
# Substitua o caminho pelo local correto do arquivo
caminho_arquivo = 'C:/Users/Antonio/Desktop/projeto-pos/pos/Machine_Learn_II/Trabalho/gym.csv'
df_gym = pd.read_csv(caminho_arquivo)

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

# Exibindo os resultados do Cross-Validation
mean_accuracy = scores.mean()
std_accuracy = scores.std()

print(f"Acurácia Média no Cross-Validation: {mean_accuracy:.2%}")
print(f"Desvio Padrão da Acurácia: {std_accuracy:.2%}")
print(f"Acurácias por Fold: {scores}")

# Gerar gráfico das acurácias por fold
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', label='Acurácia por Fold')
plt.axhline(mean_accuracy, color='r', linestyle='--', label='Acurácia Média')
plt.title('Desempenho do Modelo com 10 K-Folds com n_estimators=1000')
plt.xlabel('Fold')
plt.ylabel('Acurácia')
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()
