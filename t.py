from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Charger vos données encodées (remplacez par vos propres données)
data = pd.read_csv('hospital_readmissions.csv')

# Encodage des colonnes non numériques
data_encoded = pd.get_dummies(data, drop_first=True)
X = data_encoded.drop('readmitted', axis=1)
y = data_encoded['readmitted']

# Séparer les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Entraîner un nouveau modèle
clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(X_train, y_train)

# Sauvegarder le modèle correctement
joblib.dump(clf, 'hospital_readmissions_model.pkl')
print("Modèle réentraîné et sauvegardé avec succès !")
