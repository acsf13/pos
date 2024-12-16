import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Carregar o dataset
df_gym = pd.read_csv('/home/zeka/Projetos/Pos/pos/data/gym.csv')

# Separar variáveis preditoras (X) e alvo (y)
X = df_gym.drop(columns=['Experience_Level'])  # Excluindo a variável alvo
y = df_gym['Experience_Level']

# Codificar variáveis categóricas
categorical_columns = ['Gender', 'Workout_Type']
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Normalizar os dados numéricos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão Holdout (será usada posteriormente)
X_train_holdout, X_test_holdout, y_train_holdout, y_test_holdout = train_test_split(
    +
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Configuração do modelo SVM para Cross-Validation (10-fold)
svm = SVC()
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Grid Search com Cross-Validation
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=0)
grid_search.fit(X_scaled, y)

# Melhor modelo encontrado
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Acurácia média com Cross-Validation
cv_accuracy = grid_search.best_score_

# Exibir resultados
print("Melhores Parâmetros:", best_params)
print("Acurácia com Cross-Validation:", cv_accuracy)

# Aplicar o melhor modelo ao conjunto Holdout
best_model.fit(X_train_holdout, y_train_holdout)
y_pred_holdout = best_model.predict(X_test_holdout)

# Matriz de Confusão e Métricas de Desempenho
conf_matrix = confusion_matrix(y_test_holdout, y_pred_holdout)
classification_rep = classification_report(y_test_holdout, y_pred_holdout, digits=4)

# Exibir resultados
print("Matriz de Confusão:\n", conf_matrix)
print("\nRelatório de Classificação:\n", classification_rep)

# Aplicar o melhor modelo ao conjunto Holdout
best_model.fit(X_train_holdout, y_train_holdout)
y_pred_holdout = best_model.predict(X_test_holdout)

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test_holdout, y_pred_holdout)

# Gráfico da Matriz de Confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Matriz de Confusão - Método Holdout")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# Relatório de Classificação (extraído para visualização)
classification_rep = classification_report(y_test_holdout, y_pred_holdout, digits=4, output_dict=True)

# Gráfico das Métricas (Precision, Recall, F1-Score)
metrics = ['precision', 'recall', 'f1-score']
classes = list(classification_rep.keys())[:-3]  # Remove "accuracy", "macro avg", "weighted avg"

# Criar dataframe para gráficos
import pandas as pd
metrics_df = pd.DataFrame({metric: [classification_rep[cls][metric] for cls in classes] for metric in metrics}, index=classes)

# Gráficos
metrics_df.plot(kind='bar', figsize=(10, 6))
plt.title("Métricas por Classe - Método Holdout")
plt.xlabel("Classes")
plt.ylabel("Pontuação")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend(title="Métricas")
plt.grid(axis='y')
plt.show()

