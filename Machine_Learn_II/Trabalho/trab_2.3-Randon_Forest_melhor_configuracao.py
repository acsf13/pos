import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
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
random_forest = RandomForestClassifier(random_state=42, n_jobs=-1)

# Configuração de GridSearchCV para encontrar a melhor configuração
param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_scaled, y)

# Melhor configuração e modelo treinado
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("/nMelhor Configuração Encontrada:")
print(best_params)

# Avaliar o modelo com validação cruzada
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='accuracy')
accuracy = np.mean(cv_scores)
print("/nMédia da Acurácia com Validação Cruzada:")
print(f"{accuracy:.2%}")

# Gerar gráfico das métricas
plt.figure(figsize=(8, 6))
plt.bar(['Acurácia'], [accuracy], color='skyblue')
plt.ylim(0, 1)
plt.title('Média da Acurácia com Validação Cruzada (Melhor Configuração)')
plt.ylabel('Valores')
plt.xlabel('Métricas')
plt.show()
