from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Charger le modèle avec diagnostic
model_path = 'hospital_readmissions_model.pkl'
if not os.path.exists(model_path):
    print(f"Erreur : Le fichier modèle '{model_path}' est introuvable dans {os.getcwd()}")
    raise FileNotFoundError(f"Le fichier '{model_path}' est introuvable dans le répertoire : {os.getcwd()}")

try:
    model = joblib.load(model_path)
    print("Modèle chargé avec succès !")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    raise

# Créer un DataFrame gabarit avec les colonnes attendues
columns = model.feature_names_in_
template = pd.DataFrame(columns=columns)

# Route d'accueil pour tester si le serveur fonctionne
@app.route('/', methods=['GET'])
def home():
    return "Bienvenue dans l'API Hospital Readmissions !"

# Route pour les prédictions
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Le modèle n'a pas pu être chargé."}), 500

    try:
        # Récupérer les données envoyées via JSON
        data = request.get_json()

        # Convertir les données en DataFrame
        df = pd.DataFrame([data])

        # Adapter les données au gabarit attendu
        df = pd.concat([template, df], ignore_index=True).fillna(0)
        df = df[model.feature_names_in_]  # Assurer l'ordre correct des colonnes

        # Effectuer une prédiction
        prediction = model.predict(df)

        # Retourner la prédiction
        return jsonify({"readmitted": int(prediction[0]),
                        "message": "Patient sera réadmis" if prediction[0] == 1 else "Patient ne sera pas réadmis"})

    except Exception as e:
        # Gestion des erreurs inattendues
        return jsonify({"error": str(e)}), 500

# Démarrer l'application Flask
if __name__ == '__main__':
    app.run(debug=True) 



