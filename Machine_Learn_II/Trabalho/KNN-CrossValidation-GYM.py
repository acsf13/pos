import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

#Load dataset
df_gym = pd.read_csv('C:/Users/Antonio/Desktop/projeto-pos/pos/Machine_Learn_II/Trabalho/gym.csv')
df_gym.head()

# Initialize the LabelEncoder
le = LabelEncoder()
df_gym['Gender_Encoded'] = le.fit_transform(df_gym['Gender'])
df_gym['Workout_Type_Encoded'] = le.fit_transform(df_gym['Workout_Type'])
df_gym[['Gender', 'Gender_Encoded', 'Workout_Type', 'Workout_Type_Encoded']].head()
df_gym_new= df_gym.drop(columns=['Gender', 'Workout_Type'])
df_gym_new.head()

df_gym_new.shape

import matplotlib.pyplot as plt

# Compute the correlation matrix
corr = df_gym_new.corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

#Separe training dataset from the target attribute
X = df_gym_new.drop(columns=['Experience_Level'])  # Variáveis independentes
print(X)

#Take the last attribute as a target
y = df_gym_new['Experience_Level']  # Variável dependente (alvo)
print(y)

unique_elements, counts_elements = np.unique(X, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

unique_elements, counts_elements = np.unique(y, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

# 4. Padronizar os atributos preditores
# O K-NN é sensível à escala dos dados, então precisamos normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#######################################################################
#Cross Validation
neigh = KNeighborsClassifier(n_neighbors=3,weights='distance')
predictions_train = cross_val_predict(neigh,X_scaled,y,cv=10)

#Compute accuracy
accuracy_score(y,predictions_train)*100

#Gera a matriz de confusão do treino
confusion_matrix(y,predictions_train)

#Gera a matriz de confusão do treino na visualização de HeatMMap
import seaborn as sns
cf = confusion_matrix(y,predictions_train)
lbl1=['beginner', 'intermediate', 'expert']
lbl2 = ['beginner', 'intermediate', 'expert']
sns.heatmap(cf,annot=True,cmap="Greens", fmt="d",xticklabels=lbl1,yticklabels=lbl2)

#Separe training dataset from the target attribute
X = df_gym_new.drop(columns=['Experience_Level','Age','Max_BPM', 'Avg_BPM','Resting_BPM','Fat_Percentage', 'Workout_Type_Encoded' ])  # Variáveis independentes
print(X)

X.shape

#Take the last attribute as a target
y = df_gym_new['Experience_Level']  # Variável dependente (alvo)
print(y)

unique_elements, counts_elements = np.unique(y, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))

# 4. Padronizar os atributos preditores
# O K-NN é sensível à escala dos dados, então precisamos normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

neigh = KNeighborsClassifier(n_neighbors=3,weights='distance')
predictions_train = cross_val_predict(neigh,X,y,cv=10)

#Compute accuracy
accuracy_score(y,predictions_train)*100

#Gera a matriz de confusão do treino
confusion_matrix(y,predictions_train)
#Gera a matriz de confusão do treino na visualização de HeatMMap
import seaborn as sns
cf = confusion_matrix(y,predictions_train)
lbl1=['beginner', 'intermediate', 'expert']
lbl2 = ['beginner', 'intermediate', 'expert']
sns.heatmap(cf,annot=True,cmap="Greens", fmt="d",xticklabels=lbl1,yticklabels=lbl2)
