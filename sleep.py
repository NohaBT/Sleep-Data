import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Prédiction de la Durée du Sommeil",
    page_icon="🩺",
    layout="centered"
)

# Chargement du modèle et du scaler sauvegardés avec tes chemins d'accès
@st.cache_resource
def load_assets():
    model = joblib.load('C:\\Users\\Noha\\OneDrive\\Desktop\\AI101\\Machine Learning\\ML\\Projet_Regression_Collect\\best_sleep_model.pkl')
    scaler = joblib.load('C:\\Users\\Noha\\OneDrive\\Desktop\\AI101\\Machine Learning\\ML\\Projet_Regression_Collect\\sleep_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Erreur lors du chargement des fichiers : {e}")
    st.stop()

# Titre principal de l'application
st.title("🩺 Prédiction de la Durée du Sommeil & Mode de Vie")
st.write("Cette application utilise le Machine Learning (Random Forest) pour estimer votre durée de sommeil idéale en fonction de votre profil.")

# Formulaire de saisie des données utilisateur
st.header("👤 Profil de l'utilisateur")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Âge", min_value=10, max_value=100, value=30, step=1)
    gender = st.selectbox("Genre", ["Male", "Female"])
    stress_level = st.slider("Niveau de Stress (1 à 10)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

with col2:
    physical_activity = st.slider("Niveau d'Activité Physique (minutes/jour)", min_value=0, max_value=120, value=60, step=5)
    quality_of_sleep = st.slider("Qualité du Sommeil perçue (1 à 10)", min_value=1.0, max_value=10.0, value=7.0, step=0.5)

st.subheader("📋 Informations Professionnelles et Médicales")

# Liste complète des professions pour correspondre au Scaler
occupation = st.selectbox(
    "Profession (Occupation)", 
    ["Nurse", "Doctor", "Lawyer", "Engineer", "Accountant", "Teacher", "Salesperson", "Software Engineer", "Manager", "Scientist", "Other"]
)

sleep_disorder = st.selectbox(
    "Trouble du sommeil détecté", 
    ["None", "Insomnia", "Sleep Apnea"]
)

# Bouton déclenchant la prédiction
if st.button("🚀 Calculer la Durée du Sommeil"):
    
    # 1. Création du dictionnaire initial avec les valeurs saisies par l'utilisateur
    input_data = {
        'Age': age,
        'Quality of Sleep': quality_of_sleep,
        'Physical Activity Level': physical_activity,
        'Stress Level': stress_level,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Occupation_Doctor': 1 if occupation == "Doctor" else 0,
        'Occupation_Engineer': 1 if occupation == "Engineer" else 0,
        'Occupation_Lawyer': 1 if occupation == "Lawyer" else 0,
        'Occupation_Manager': 1 if occupation == "Manager" else 0,
        'Occupation_Nurse': 1 if occupation == "Nurse" else 0,
        'Occupation_Salesperson': 1 if occupation == "Salesperson" else 0,
        'Occupation_Scientist': 1 if occupation == "Scientist" else 0,
        'Occupation_Software Engineer': 1 if occupation == "Software Engineer" else 0,
        'Occupation_Teacher': 1 if occupation == "Teacher" else 0,
        'Sleep Disorder_none': 1 if sleep_disorder == "None" else 0,
        'Sleep Disorder_Sleep Apnea': 1 if sleep_disorder == "Sleep Apnea" else 0
    }
    
    # 2. Conversion en DataFrame provisoire
    df_provisoire = pd.DataFrame([input_data])
    
    # 3. Réalignement automatique et strict selon l'ordre des caractéristiques du Scaler
    try:
        ordre_strict_du_scaler = scaler.feature_names_in_
        
        # Vérification et création des colonnes manquantes si nécessaire
        for col in ordre_strict_du_scaler:
            if col not in df_provisoire.columns:
                df_provisoire[col] = 0
                
        # Application de l'ordre exact
        df_input = df_provisoire[ordre_strict_du_scaler]
        
    except AttributeError:
        # Ordre de secours manuel si la propriété feature_names_in_ n'est pas détectée
        ordre_secours = [
            'Age', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level',
            'Gender_Male', 'Occupation_Doctor', 'Occupation_Engineer', 
            'Occupation_Lawyer', 'Occupation_Manager', 'Occupation_Nurse', 
            'Occupation_Salesperson', 'Occupation_Scientist', 'Occupation_Software Engineer', 
            'Occupation_Teacher', 'Sleep Disorder_none', 'Sleep Disorder_Sleep Apnea'
        ]
        df_input = df_provisoire[ordre_secours]
    
    # 4. Normalisation des données d'entrée
    df_input_scaled = scaler.transform(df_input)
    
    # 5. Exécution de la prédiction finale avec le modèle
    prediction = model.predict(df_input_scaled)[0]
    
    # 6. Affichage des résultats à l'écran
    st.success("✨ Analyse terminée avec succès !")
    
    # Conversion de la valeur décimale en heures et minutes pour un affichage lisible
    heures = int(prediction)
    minutes = int((prediction - heures) * 60)
    
    st.metric(
        label="⏱️ Durée de sommeil estimée", 
        value=f"{heures}h {minutes}min", 
        delta=f"{prediction:.2f} heures au total"
    )
    
    # Conseils et messages de prévention basés sur le niveau de stress inséré
    if stress_level >= 7:
        st.warning("⚠️ Votre niveau de stress semble élevé, ce qui peut réduire considérablement votre temps de repos nocturne.")
    else:
        st.info("💡 Un bon équilibre d'activité physique favorise la stabilisation et la régularité de cette durée.")