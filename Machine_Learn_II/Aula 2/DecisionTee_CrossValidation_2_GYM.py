import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import tree

#Load dataset
df_gym = pd.read_csv('gym.csv')
df_gym.head()

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

# Atualizar a variável feature_cols com as colunas do dataframe processado
feature_cols = X.columns.tolist()
print("Features selecionadas:", feature_cols)

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Verificar a distribuição das classes no alvo
unique_elements, counts_elements = np.unique(y, return_counts=True)
print("Classes e contagens:", np.asarray((unique_elements, counts_elements)))

#CART ALGORITHM
#max_depth: poda da árvore. O nodo raiz não conta pois possui todos os dados e não é uma ramificação. If None, 
#then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
#min_samples_leaf: quantidade mínima de sample nos nodos finais (folhas)

tree = DecisionTreeClassifier(criterion = 'gini', random_state=100,max_depth=3,min_samples_leaf=5)
tree.fit(X_scaled,y)
#Cross Validation
predictions = cross_val_predict(tree,X_scaled,y,cv=10)

#Compute accuracy
accuracy_score(y,predictions)*100
#print("The prediction accuracy is: ",tree.score(X,y)*100,"%")

cf = confusion_matrix(y,predictions)
lbl1=['Beginner', 'Intermediate', 'Advanced']
lbl2 = ['Beginner', 'Intermediate', 'Advanced']
sns.heatmap(cf,annot=True,cmap="Greens", fmt="d",xticklabels=lbl1,yticklabels=lbl2)

from sklearn.metrics import classification_report
#Gera a matriz de confusão do test
print(classification_report(y,predictions))

#Generate the tree in a text format
from sklearn.tree import export_text
r = export_text(tree, feature_names=feature_cols)
print(r)
