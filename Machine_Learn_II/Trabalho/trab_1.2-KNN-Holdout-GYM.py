from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Load dataset
df_gym = pd.read_csv('C:/Users/Antonio/Desktop/projeto-pos/pos/Machine_Learn_II/Trabalho/gym.csv')
df_gym.head()

# Check for missing values
missing_values = df_gym.isnull().sum()
print(missing_values)

# Drop rows with missing target values, if any
data = df_gym.dropna(subset=['Experience_Level'])

# Encode categorical variables (Gender, Workout_Type)
label_encoders = {}
categorical_columns = ['Gender', 'Workout_Type']
for col in categorical_columns:
    le = LabelEncoder()
    df_gym[col] = le.fit_transform(df_gym[col])
    label_encoders[col] = le
df_gym.head

# Define features (X) and target (y)
X = df_gym.drop(columns=['Experience_Level'])
y = df_gym['Experience_Level']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#######################################################################
#create training and testing datasets. 30% for tests selected by random
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.3,random_state=42,stratify=y)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

# Train the KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

# Predictions on training and testing sets
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Calculate accuracies
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy  = accuracy_score(y_test, y_test_pred)

# Generate confusion matrices
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)

# Display results
results = {
    "Train Accuracy": train_accuracy,
    "Test Accuracy": test_accuracy,
    "Train Confusion Matrix": train_conf_matrix,
    "Test Confusion Matrix": test_conf_matrix
}
results

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Training confusion matrix
sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap='Greens', ax=axes[0])
axes[0].set_title('Training Confusion Matrix')
axes[0].set_xlabel('Predicted Labels')
axes[0].set_ylabel('True Labels')

# Testing confusion matrix
sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Testing Confusion Matrix')
axes[1].set_xlabel('Predicted Labels')
axes[1].set_ylabel('True Labels')

plt.tight_layout()
plt.show()

