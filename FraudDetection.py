import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


# Carregando o dataset
df = pd.read_csv("CreditCard_FraudDetection_dataset.csv")

# Preparação dos dados
X = df.drop('Class', axis=1)  # Remove a coluna de classe para formar o conjunto de características
y = df['Class']  # A coluna 'Class' é o nosso target

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização dos dados (exceto para a coluna 'Time')
scaler = StandardScaler()
X_train[['Amount']] = scaler.fit_transform(X_train[['Amount']])
X_test[['Amount']] = scaler.transform(X_test[['Amount']])

# Aplicando SMOTE para tratar o desbalanceamento do dataset
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Treinamento do modelo RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_smote, y_train_smote)

# Predições no conjunto de teste
y_pred = clf.predict(X_test)

# Avaliação do modelo
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
