# Construction d'une grille de score - Domaine du crédit à la consommation

L'objectif de ce sujet est de mettre en avant une méthodologie de construction d'une grille de score sur une problématique de risque de crédit à la consommation, et de proposer des méthodes ensemblistes challengeant notre grille de score classique et en déduire une interprétabilité avec la méthode de la valeur de Shapley.

# Données 

La base de données traitées concerne des données issues d'une banque octroyant des crédits à la consommation. En effet ces crédits concernent des prêts liés à un besoin de financement dans l'éducation, pour des fins médicales, professionnel, personnel en tout genre, rachat de crédit ou encore pour des travaux mineurs personnels. En effet les montants de ces prêts n'excèdent que rarement les quelques dizaines de milliers d'euros. La base de données est constituée de 32 581 observations et de 12 variables. La base contient des informations propres aux contrats type montant du prêt, taux d'intérêt associé, objet du prêt, ou encore la durée écoulée depuis le début du prêt. Elle contient aussi des informations propres aux clients comme l'âge du client, son revenu, sa situation immobilière, sa durée expérience ou encore une note comportementale associée. La variable à modéliser est le défaut du prêt ou non, la variable vaut 1 si le client a fait au moins une fois défaut et 0 sinon.

La base est issue de Kaggle **Lien du challenge : https://www.kaggle.com/datasets/laotse/credit-risk-dataset?resource=download" 

# Etapes

* Application de plusieurs étapes de nettoyage et mise en forme des données (Label encoding, problématique de traitement des données manquantes et outliers).
* Création de variable (variable macroéconomique rendant de l'environnement économique lors de l'octroi).
* Discrétisation des variables quantitatives pour la construction de la grille et regroupement de modalités des variables qualitatives.
* Prétraitement pour la modélisation : Classement des variables explicatives avec notre variable cible en fonction de la corrélation, avant de réaliser une régression logistique en step by step, visant à minimiser le critère d'information BIC et à omettre toute corrélation entre variable explicative selon la stat du V de Cramer.
* Modélisation avec un modèle classique : régression logistique, des modèles challenger : Random Forest et Gradient Boosting, et interprétabilités de ces méthodes ensemblistes avec la Shapley Value.


# Le code

Le script **programme_scorecard.ipynb** permet de rendre compte de l'entière réalisation du projet

Le script **function_score.py** contient toutes les fonctions utiles à la bonne exécution du script **programme_scorecard.ipynb**

Le script **librairies_score.py** permet l'import des librairies utile à la bonne exécution des fonctions présentes dans le script **function_score.py** et donc indirectement au script **programme_scorecard.ipynb**
