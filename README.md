# OC/DS Projet 4 : Anticipez les besoins en consommation de bâtiments
Formation OpenClassrooms - Parcours data scientist - Projet Professionnalisant (Décembre 2022-Février 2023)

## Secteur : 
Énergie

## Technologies utilisées : 
 * Jupyter Notebook,
 * Python
 * librairies spécifiques : HyperOpt, SHAP

## Mots-clés :
régression supervisée, modèles ensemblistes

## Le contexte : 
Le client, la ville de Seattle, s’est fixé un objectif de neutralité carbone en 2050. 
Pour atteindre cet objectif il a défini plusieurs actions dont réduire la consommation totale d’énergie et les émissions de CO2 des bâtiments non destinés à l’habitation.

Il a donc besoin de monitorer ces deux cibles pour adapter sa stratégie. 

## La mission : 
Proposer deux modèles de prédiction : l’un permettant de prédire la consommation totale d’énergie des bâtiments non destinés à l’habitation, l’autre leurs émissions de CO2. 

## Algorithme retenu : 
GBoost

## Livrables :
* notebook_analyse.ipynb : notebook du nettoyage et de l'analyse exploratoire
* notebook_co2.ipynb : notebook des différents tests de modèles pour prédire les émissions de CO2
* notebook_conso.ipynb : notebook des différents tests de modèles pour prédire la consommation totale d’énergie
* presentation.pdf : Un support de présentation pour la soutenance.

## Méthodologie suivie : 
1. Nettoyage des données :
* sélection des variables pertinentes
* traitement des doublons
* traitement des valeurs aberrantes
* traitement des valeurs manquantes

2. Traitement des données :
* feature engineering
* choix des transformations appliquées aux variables (RobustScaler, OrdinalEncoder, TargetEncoder, Passage au logarithme, Passage au carré)

3. Modélisation :
* définition de la méthode de modélisation : pipeline + validation croisée
* entraînement d’un modèle naïf 
* test de plusieurs modèles :
	- linéaires : régression linéaire, lasso, regression ridge, elastic net, svm
	- à noyau :
	- ensemblistes : Gboost, XGBoost, CatBoost, Bagging, Forêt aléatoire, Adaboost
* analyse de l’impact de chaque variable sur le modèle (librairie SHAP)

On recommence les étapes 2 et 3 (traitement des données et modélisation) pour améliorer les résultats.
Une fois satisfait du choix des variables d’entrées, on passe à l’étape 4.

4. Optimisation :
* choix des meilleurs modèles : temps d’entraînement, MAE,  R2
* optimisation des hyperparamètres des meilleurs modèles (librairie HyperOpt)
* choix du meilleur modèle après optimisation

## Compétences acquises :  
* Mettre en place le modèle d’apprentissage supervisé adapté au problème métier
* Adapter les hyperparamètres d’un algorithme d’apprentissage supervisé afin de l’améliorer
* Transformer les variables pertinentes d’un modèle d’apprentissage supervisé
* Évaluer les performances d’un modèle d’apprentissage supervisé

## Data source : 
https://data.seattle.gov/dataset/2016-Building-Energy-Benchmarking/2bpz-gwpy
