import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# 1. Carregar os dados
file_path = 'C:/Users/Antonio/Desktop/projeto-pos/pos/Machine_Learn_II/Trabalho/gym.csv'
data = pd.read_csv(file_path)

# 2. Pré-processamento (ajustar dependendo da estrutura dos dados)
# Exemplo: separação entre características e rótulos
# (Substituir "target_column" pelo nome real da coluna alvo)
X = data.drop(columns=['target_column'])  # Ajustar a coluna alvo
y = data['target_column']

# Converter rótulos para valores numéricos (se necessário)
if y.dtype == 'object':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

# 3. Configurar o modelo Naive Bayes
model = GaussianNB()

# 4. Aplicar validação cruzada com 10 folds
cv_scores = cross_val_score(model, X, y, cv=10)

# 5. Exibir os resultados
print("Acurácias por fold:", cv_scores)
print("Acurácia média:", cv_scores.mean())
