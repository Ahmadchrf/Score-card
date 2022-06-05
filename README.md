# Construction d'une grille de score

L'objectif de ce sujet est de mettre en avant une méthodologie de construction d'une grille de score sur une problématique de risque de crédit à la consommation, et de proposer des méthodes ensemblistes challengeant notre grille de score classique et en déduire une interprétabilité avec la méthode de la valeur de Shapley. 

# Etapes

* Application de plusieus étapes de nettoyage et mise en forme des données (Label encoding, problématique de traitement des données manquantes et outliers).
* Création de variable (variable macroéconomique rendant de l'environnement économique lors de l'octroi) .
* Discrétisatoion des variables quantitatives pour la construction de la grille et regroupement de modalités des variables qualitatives.
* Prétraitement pour la modélisation : Classement des variables explicatives avec notre variable cible en fonction de la corrélation, avavnt de réaliser une regréssion logistique en step by step, visant à minimiser le critère d'information BIC et à omettre tout correlation entre variable explicatives selon la stat du V de Cramer. 
* Modélisation avec un **modèle classique** : régression logistique, **des modèles challenger** : Random Forest et Gradient Boosting, et interprétabilité de ces méthodes ensembemblistes avec la Shapley Value.


# Le code

Le script **programme_scorecard.ipynb** permet de rendre compte de l'entière réalisation du projet

Le script **function_nexialog.py** contient toutes les fonctions utile à la bonne execution du scirpt **programme_scorecard.ipynb**

Le script **librairies_nexialog.py** permet l'import des librairies utile à la bonne exécution des fonctions présentent dans le script **function_nexialog.py** et donc indirectement au script **programme_scorecard.ipynb**
