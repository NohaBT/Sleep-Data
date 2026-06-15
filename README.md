# Santé du Sommeil et Mode de Vie - Analyse de Régression

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-darkgreen.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20+-blue.svg)](https://numpy.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-blue.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-blue.svg)](https://seaborn.pydata.org/)
[![Joblib](https://img.shields.io/badge/Joblib-1.2+-blue.svg)](https://joblib.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

Ce projet propose une analyse approfondie des facteurs biologiques et comportementaux qui influencent la durée du sommeil. À l'aide de modèles de Machine Learning de régression, nous étudions l'impact du niveau de stress, de l'âge, de l'activité physique et de la profession sur le temps de repos quotidien.

## Objectifs du Projet

- Analyser les relations complexes entre les habitudes de vie et la durée du sommeil.
- Nettoyer et préparer un jeu de données réel comportant des anomalies (valeurs négatives, valeurs aberrantes ou manquantes).
- Entraîner et comparer plusieurs algorithmes de régression (Régression Linéaire, KNN Regressor, Decision Tree, Random Forest et SVR).
- Évaluer les modèles à l'aide de métriques standards (MAE, MSE, RMSE, R² Score) pour identifier le plus performant pour le déploiement.

## Structure du Dépôt

- sleep_data.ipynb : Le notebook Jupyter principal contenant l'analyse exploratoire des données (EDA), le nettoyage, la modélisation et l'évaluation des performances.
- sleep.py : Application interactive Web Streamlit pour la prédiction et l'analyse de la durée du sommeil.
- Sleep_health_and_lifestyle_dataset-dirty.csv : Le jeu de données brut contenant les anomalies à traiter.
- best_sleep_model.pkl : Le modèle de régression Random Forest final sauvegardé.
- sleep_scaler.pkl : Le standardiseur de caractéristiques (StandardScaler) associé au modèle.
- sleep_data.pdf & sleep_data.html : Exports statiques du notebook pour une consultation rapide sans exécution de code.

## Métriques de Performance des Modèles

Après division des données (60% entraînement, 20% validation, 20% test) et normalisation des caractéristiques, voici les résultats obtenus sur l'ensemble de test :

- Random Forest Regressor : R² = 0.4487 | RMSE = 0.4293 | MAE = 0.3257
- Régression Linéaire : R² = 0.3495 | RMSE = 0.4663 | MAE = 0.3718
- SVR (Kernel RBF) : R² = 0.3231 | RMSE = 0.4757 | MAE = 0.3674
- KNN Regressor (k=5) : R² = 0.2753 | RMSE = 0.4922 | MAE = 0.3645
- Decision Tree Regressor : R² = 0.0838 | RMSE = 0.5534 | MAE = 0.3855

Le modèle Random Forest obtient les meilleures performances en capturant efficacement les relations non linéaires. Il prédit la durée de sommeil avec une erreur moyenne absolue d'environ 19.5 minutes.

## Principaux Résultats de l'Analyse

L'analyse de l'importance des variables montre que :
1. Le niveau de stress (Stress Level) est le facteur le plus déterminant, expliquant environ 33.5% des décisions de division du modèle.
2. La qualité ressentie du sommeil (Quality of Sleep) contribue à hauteur de 20.2%.
3. L'âge et le niveau d'activité physique représentent respectivement 15.7% et 14.2% de l'importance du modèle.
4. Les troubles du sommeil déclarés (Sleep Disorder) et le genre ont un impact minime dans ce jeu de données épuré.
## Limites et Perspectives

- Le fichier de données est de taille limitée (374 observations). Un échantillon plus large améliorerait la généralisation des modèles.
- Certaines colonnes physiologiques importantes du jeu de données complet d'origine (comme la pression artérielle, la fréquence cardiaque et le nombre de pas quotidiens) ne sont pas incluses dans ce dataset épuré. Leur ajout permettrait d'augmenter considérablement la précision (R²).

## Application Interactive Streamlit

Une application web interactive est incluse dans ce dépôt (`sleep.py`) pour permettre aux utilisateurs de simuler et prédire leur durée idéale de sommeil à partir de leurs propres informations de santé.

L'application est structurée en trois onglets :
- **Simulateur & Prédiction** : Un formulaire convivial pour saisir vos données démographiques (âge, genre, stress, activité physique) et estimer instantanément votre durée recommandée de sommeil en heures et en minutes.
- **Tableau de Bord & Métriques** : Une visualisation de l'importance des variables sous forme de graphique et le rappel des métriques de performance globales du modèle Random Forest.
- **Conseils Personnalisés** : Un plan d'action et des recommandations d'hygiène de vie adaptés dynamiquement à votre profil.

## Installation et Utilisation

Pour exécuter ce projet localement :

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/NohaBT/Sleep-Data.git
   ```

2. Installez les dépendances requises :
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit
   ```

3. Lancez l'environnement Jupyter pour explorer le notebook :
   ```bash
   jupyter notebook sleep_data.ipynb
   ```

4. Lancez l'application interactive Streamlit :
   ```bash
   streamlit run sleep.py
   ```

