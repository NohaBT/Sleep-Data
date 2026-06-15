import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page Streamlit pour un aspect premium
st.set_page_config(
    page_title="Santé du Sommeil - Diagnostic Clinique",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Chargement du modèle et du scaler sauvegardés
@st.cache_resource
def load_assets():
    model = joblib.load(r'c:\Users\Noha\OneDrive\Desktop\AI101\Machine Learning\ML\Projet_Regression_Collect\best_sleep_model.pkl')
    scaler = joblib.load(r'c:\Users\Noha\OneDrive\Desktop\AI101\Machine Learning\ML\Projet_Regression_Collect\sleep_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Erreur lors du chargement des fichiers de modélisation : {e}")
    st.stop()

# Injection de styles CSS personnalisés adaptatifs (Supportant les thèmes Clair et Sombre)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    h1 {
        font-weight: 800;
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
        text-align: center;
    }
    
    .subtitle {
        font-size: 1.25rem;
        color: var(--text-color);
        opacity: 0.8;
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    /* Style des cartes adaptatif */
    .card {
        background: var(--secondary-background-color, #ffffff);
        color: var(--text-color, #2d3436);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1.5rem;
    }
    
    .card-title {
        color: var(--text-color, #2d3436);
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #6c5ce7;
        padding-bottom: 0.5rem;
    }
    
    /* Bouton principal personnalisé */
    div.stButton > button {
        background: linear-gradient(135deg, #6c5ce7, #a29bfe) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        border-radius: 50px !important;
        box-shadow: 0 4px 15px rgba(108, 92, 231, 0.4) !important;
        transition: all 0.3s ease !important;
        display: block;
        margin: 1.5rem auto !important;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(108, 92, 231, 0.6) !important;
    }
    
    /* Métriques de prédiction adaptées */
    .prediction-container {
        background: linear-gradient(135deg, #2d3436, #0f1112);
        color: white;
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        margin: 2rem 0;
    }
    
    .prediction-value {
        font-size: 4rem;
        font-weight: 800;
        color: #a29bfe;
        margin: 0.5rem 0;
        letter-spacing: -1px;
    }
    
    .prediction-label {
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #b2bec3;
    }
    
    /* Métriques du modèle */
    .metric-box {
        background-color: var(--secondary-background-color, #f1f2f6);
        border-left: 5px solid #6c5ce7;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        color: var(--text-color, #2d3436);
        border-top: 1px solid rgba(128, 128, 128, 0.1);
        border-right: 1px solid rgba(128, 128, 128, 0.1);
        border-bottom: 1px solid rgba(128, 128, 128, 0.1);
    }
    
    .metric-num {
        font-size: 1.8rem;
        font-weight: bold;
        color: var(--text-color, #2d3436);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-color, #747d8c);
        opacity: 0.8;
    }

    /* Styles pour alertes cliniques adaptatives sans couleurs agressives à fort contraste */
    .clinical-alert-stress {
        background-color: rgba(230, 126, 34, 0.12);
        color: #e67e22;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 6px solid #e67e22;
        margin-bottom: 1rem;
    }
    
    .clinical-alert-activity {
        background-color: rgba(99, 110, 114, 0.12);
        color: var(--text-color, #636e72);
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 6px solid #636e72;
        margin-bottom: 1rem;
    }
    
    .clinical-alert-healthy {
        background-color: rgba(0, 184, 148, 0.12);
        color: #00b894;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 6px solid #00b894;
        margin-bottom: 1rem;
    }
    
    .clinical-alert-disorder {
        background-color: rgba(231, 76, 60, 0.12);
        color: #e74c3c;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 6px solid #e74c3c;
        margin-bottom: 1rem;
    }
    
    </style>
""", unsafe_allow_html=True)

# En-tête principal
st.markdown("<h1>Santé du Sommeil & Mode de Vie</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Explorez, analysez et préduisez votre durée idéale de sommeil grâce à l'Intelligence Artificielle.</p>", unsafe_allow_html=True)

# Définition des onglets de navigation
tab1, tab2, tab3 = st.tabs([
    "Simulateur & Prédiction", 
    "Tableau de Bord & Métriques", 
    "Conseils Personnalisés"
])

with tab1:
    col_input, col_result = st.columns([1.1, 0.9])
    
    with col_input:
        st.markdown("""
            <div class='card'>
                <div class='card-title'>Profil Démographique & Habitudes</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Grid interne pour les entrées utilisateur
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            age = st.slider("Âge", min_value=15, max_value=80, value=38, step=1)
            gender = st.selectbox("Genre", ["Male", "Female"])
            stress_level = st.slider("Niveau de Stress (Échelle de 1 à 10)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
            
        with sub_col2:
            physical_activity = st.slider("Activité Physique (minutes par jour)", min_value=0, max_value=120, value=60, step=5)
            quality_of_sleep = st.slider("Qualité Perçue du Sommeil (1 à 10)", min_value=1.0, max_value=10.0, value=7.0, step=0.5)
            
        st.markdown("""
            <div class='card' style='margin-top: 1rem;'>
                <div class='card-title'>Situation Médicale & Professionnelle</div>
            </div>
        """, unsafe_allow_html=True)
        
        sub_col3, sub_col4 = st.columns(2)
        with sub_col3:
            occupation = st.selectbox(
                "Profession", 
                ["Nurse", "Doctor", "Lawyer", "Engineer", "Accountant", "Teacher", "Salesperson", "Software Engineer", "Manager", "Scientist", "Other"]
            )
        with sub_col4:
            sleep_disorder = st.selectbox(
                "Trouble du Sommeil Détecté", 
                ["None", "Insomnia", "Sleep Apnea"]
            )
            
        predict_button = st.button("Calculer ma Durée du Sommeil")
        
    with col_result:
        if predict_button:
            # Structuration des caractéristiques d'entrée
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
            
            df_provisoire = pd.DataFrame([input_data])
            ordre_strict_du_scaler = scaler.feature_names_in_
            
            for col in ordre_strict_du_scaler:
                if col not in df_provisoire.columns:
                    df_provisoire[col] = 0
                    
            df_input = df_provisoire[ordre_strict_du_scaler]
            df_input_scaled = scaler.transform(df_input)
            
            # Prédiction du modèle
            prediction = model.predict(df_input_scaled)[0]
            
            heures = int(prediction)
            minutes = int((prediction - heures) * 60)
            
            # Affichage de la prédiction dans la carte premium
            st.markdown(f"""
                <div class='prediction-container'>
                    <div class='prediction-label'>Durée de Sommeil Recommandée</div>
                    <div class='prediction-value'>{heures}h {minutes:02d}min</div>
                    <div style='color: #b2bec3; font-size: 0.95rem; margin-top: 1rem;'>
                        Modèle de régression : Random Forest Regressor (Forêt Aléatoire)
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Évaluation synthétique
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Interprétation Clinique Rapide</div>", unsafe_allow_html=True)
            
            if stress_level >= 7.0:
                st.markdown("""
                    <div class='clinical-alert-stress'>
                        <strong>Niveau de Stress Très Élevé :</strong> Le modèle indique que votre niveau de stress est un facteur limitant important de votre repos. Pour augmenter votre temps de sommeil profond, des exercices de relaxation en fin de journée sont fortement préconisés.
                    </div>
                """, unsafe_allow_html=True)
            elif physical_activity < 30:
                st.markdown("""
                    <div class='clinical-alert-activity'>
                        <strong>Activité Physique Insuffisante :</strong> Pratiquer au moins 30 minutes d'activité physique modérée par jour aide à réguler les cycles circadiens et améliore la qualité globale de votre sommeil.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='clinical-alert-healthy'>
                        <strong>Équilibre Sain :</strong> Vos habitudes d'activité physique et de gestion de stress soutiennent une durée et une régularité de sommeil saines. Continuez ainsi.
                    </div>
                """, unsafe_allow_html=True)
                
            if sleep_disorder != "None":
                st.markdown(f"""
                    <div class='clinical-alert-disorder'>
                        <strong>Attention :</strong> La présence déclarée de <strong>{sleep_disorder}</strong> perturbe la structure naturelle du sommeil. Un suivi avec un professionnel de santé est recommandé.
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            # Affichage par défaut avant clic
            st.markdown("""
                <div class='prediction-container' style='background: linear-gradient(135deg, #6c5ce7, #341f97);'>
                    <div class='prediction-label'>Simulation Prête</div>
                    <div class='prediction-value' style='color: white; font-size: 2.2rem; margin: 1.5rem 0;'>
                        Remplissez le formulaire et cliquez sur le bouton pour afficher l'estimation.
                    </div>
                </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Métriques Générales d'Évaluation des Modèles</div>", unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown("""
            <div class='metric-box'>
                <div class='metric-num'>0.4487</div>
                <div class='metric-label'>R² Score (Variance Expliquée)</div>
            </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown("""
            <div class='metric-box'>
                <div class='metric-num'>0.4293 h</div>
                <div class='metric-label'>RMSE (Marge Quadratique Moyen)</div>
            </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown("""
            <div class='metric-box'>
                <div class='metric-num'>0.3257 h</div>
                <div class='metric-label'>MAE (Écart Absolu Moyen)</div>
            </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown("""
            <div class='metric-box'>
                <div class='metric-num'>224 / 75</div>
                <div class='metric-label'>Échantillons Train / Test</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
    
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Importance Relatives des Variables (Random Forest)</div>", unsafe_allow_html=True)
        
        # Génération du graphique de Feature Importance adaptatif au thème sombre/clair
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        fig1.patch.set_alpha(0.0)  # Fond transparent pour le graphique
        ax1.patch.set_alpha(0.0)   # Fond transparent pour l'axe
        
        try:
            importances = model.feature_importances_
            feature_names = scaler.feature_names_in_
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True)
            
            # Sélectionner les principales variables pour la clarté
            feat_imp_top = feat_imp.tail(10)
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feat_imp_top)))
            feat_imp_top.plot(kind='barh', color=colors, ax=ax1)
            
            # Stylisation des labels et axes pour la lisibilité sur fond clair et sombre
            ax1.set_xlabel("Score d'Importance", color='#888888', fontsize=11)
            ax1.tick_params(colors='#888888', labelsize=10)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_color('#555555')
            ax1.spines['bottom'].set_color('#555555')
            
            plt.tight_layout()
            st.pyplot(fig1)
        except Exception as e:
            st.write(f"Impossible de générer le graphique d'importance : {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_plot2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Explication Clinique des Résultats</div>", unsafe_allow_html=True)
        st.markdown("""
            L'analyse statistique montre que la durée du sommeil est principalement dictée par :
            
            1. **Niveau de Stress (33.5%)** : Le stress induit une sécrétion prolongée de cortisol, ce qui perturbe l'endormissement et diminue le sommeil lent profond.
            2. **Qualité perçue du Sommeil (20.2%)** : Il existe une corrélation directe et bidirectionnelle importante. Dormir une quantité de temps suffisante augmente la qualité perçue, et inversement.
            3. **Âge de l'individu (15.7%)** : La structure des cycles du sommeil évolue physiologiquement avec l'âge (diminution du besoin brut de sommeil).
            4. **Niveau d'activité physique (14.2%)** : Favorise la fatigue physique saine et augmente le taux de sommeil profond restaurateur.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Plan d'Action Thérapeutique Recommandé</div>", unsafe_allow_html=True)
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("""
            ### Gestion du Stress et Relaxation
            Le stress étant la variable à plus fort impact sur la durée de votre repos, voici les étapes conseillées :
            - **Routine d'extinction des écrans** : Éteignez tout écran 1 heure avant le coucher pour limiter l'impact de la lumière bleue sur la mélatonine.
            - **Respiration guidée (Cohérence cardiaque)** : Pratiquez la respiration ventrale (méthode 365) pendant 5 minutes avant de dormir pour abaisser le rythme cardiaque.
            - **Lecture ou écriture** : Journaliser ses pensées du jour permet de décharger l'anxiété accumulée.
        """)
        
    with col_c2:
        st.markdown("""
            ### Activité Physique & Environnement
            - **Régularité de l'exercice** : Pratiquez une activité physique régulière (marche rapide, vélo, course) de préférence en matinée ou en après-midi. Évitez le sport intensif moins de 3 heures avant le coucher.
            - **Hygiène de la chambre** : Maintenez la température de la chambre entre 16 et 18 degrés Celsius, dans le noir complet et le silence.
            - **Alimentation** : Évitez les repas lourds et la caféine après 14h00.
        """)
    st.markdown("</div>", unsafe_allow_html=True)