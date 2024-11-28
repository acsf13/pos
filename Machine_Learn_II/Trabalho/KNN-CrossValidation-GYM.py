#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[ ]:


# Configurar pandas para exibir mais colunas e linhas
pd.set_option('display.max_columns', None)  # Exibe todas as colunas
# pd.set_option('display.max_rows', None)


# In[ ]:


#Load dataset
df_gym = pd.read_csv('gym_members_exercise_tracking.csv')
df_gym.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
le = LabelEncoder()

# Apply Label Encoding to the Gender column
df_gym['Gender_Encoded'] = le.fit_transform(df_gym['Gender'])

# Apply Label Encoding to the Workout_Type column
df_gym['Workout_Type_Encoded'] = le.fit_transform(df_gym['Workout_Type'])

# Show the updated dataframe
df_gym[['Gender', 'Gender_Encoded', 'Workout_Type', 'Workout_Type_Encoded']].head()
# uuu


# In[ ]:


df_gym_new= df_gym.drop(columns=['Gender', 'Workout_Type'])
df_gym_new.head()


# In[ ]:


df_gym_new.shape


# In[ ]:


import matplotlib.pyplot as plt

# Compute the correlation matrix
corr = df_gym_new.corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:


#Separe training dataset from the target attribute
X = df_gym_new.drop(columns=['Experience_Level'])  # Variáveis independentes
print(X)


# In[ ]:


#Take the last attribute as a target
y = df_gym_new['Experience_Level']  # Variável dependente (alvo)
print(y)


# In[ ]:


unique_elements, counts_elements = np.unique(X, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))


# In[ ]:


unique_elements, counts_elements = np.unique(y, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))


# In[ ]:


# 4. Padronizar os atributos preditores
# O K-NN é sensível à escala dos dados, então precisamos normalizar
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)


# In[ ]:


#######################################################################
#Cross Validation
neigh = KNeighborsClassifier(n_neighbors=3,weights='distance')
predictions_train = cross_val_predict(neigh,X,y,cv=10)

#Compute accuracy
accuracy_score(y,predictions_train)*100


#Gera a matriz de confusão do treino
confusion_matrix(y,predictions_train)


# In[ ]:


#Gera a matriz de confusão do treino na visualização de HeatMMap
import seaborn as sns
cf = confusion_matrix(y,predictions_train)
lbl1=['beginner', 'intermediate', 'expert']
lbl2 = ['beginner', 'intermediate', 'expert']
sns.heatmap(cf,annot=True,cmap="Greens", fmt="d",xticklabels=lbl1,yticklabels=lbl2)


# In[ ]:


#Separe training dataset from the target attribute
X = df_gym_new.drop(columns=['Experience_Level','Age','Max_BPM', 'Avg_BPM','Resting_BPM','Fat_Percentage', 'Workout_Type_Encoded' ])  # Variáveis independentes
print(X)


# In[ ]:


X.shape


# In[ ]:


#Take the last attribute as a target
y = df_gym_new['Experience_Level']  # Variável dependente (alvo)
print(y)


# In[ ]:


unique_elements, counts_elements = np.unique(y, return_counts=True)
print(np.asarray((unique_elements, counts_elements)))


# In[ ]:


# 4. Padronizar os atributos preditores
# O K-NN é sensível à escala dos dados, então precisamos normalizar
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)


# In[ ]:


neigh = KNeighborsClassifier(n_neighbors=3,weights='distance')
predictions_train = cross_val_predict(neigh,X,y,cv=10)


# In[ ]:


#Compute accuracy
accuracy_score(y,predictions_train)*100


# In[ ]:


#Gera a matriz de confusão do treino
confusion_matrix(y,predictions_train)


# In[ ]:


#Gera a matriz de confusão do treino na visualização de HeatMMap
import seaborn as sns
cf = confusion_matrix(y,predictions_train)
lbl1=['beginner', 'intermediate', 'expert']
lbl2 = ['beginner', 'intermediate', 'expert']
sns.heatmap(cf,annot=True,cmap="Greens", fmt="d",xticklabels=lbl1,yticklabels=lbl2)

