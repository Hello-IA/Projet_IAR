# Communicating Artificial Neural Networks Develop Efficient Color-Naming Systems

Ce dépôt contient le code et les expériences réalisées dans le cadre du **projet IAR** à **Sorbonne Université**.  
Le travail s’appuie sur l’article :

*Chaabouni et al. (2021), “Communicating artificial neural networks develop efficient color-naming systems”, PNAS.*

L’objectif est d’étudier l’émergence de systèmes de dénomination des couleurs chez des agents artificiels et de les comparer aux langues humaines à l’aide du cadre de l’**Information Bottleneck (IB)**.

---

## Objectif du projet

Les langues humaines utilisent des symboles discrets pour décrire un monde perceptif continu.  
Ce projet cherche à déterminer si des propriétés similaires peuvent émerger dans des réseaux de neurones artificiels entraînés à communiquer sur des couleurs.

Nous analysons :
- des langues humaines (World Color Survey),
- des langages émergents produits par des agents artificiels,
- leur efficacité informationnelle via le compromis précision / complexité.

---

## Cadre théorique : Information Bottleneck

La communication est analysée à l’aide du principe du goulot d’étranglement de l’information.

Variables principales :
- `C` : couleur cible (puces du World Color Survey),
- `W` : mot discret produit par le Speaker,
- `U` : représentation perceptive continue reconstruite par le Listener.

Mesures :
- `I(C; W)` : complexité lexicale,
- `I(U; W)` : précision communicationnelle.

La frontière Information Bottleneck sert de référence pour évaluer l’efficacité des langages humains et artificiels.

---

## Organisation du dépôt

### Librairie
Le projet utilise **EGG (Emergence of lanGuage in Games)**, une librairie de recherche permettant d’implémenter des jeux de communication Speaker / Listener avec des canaux discrets.

### Scripts principaux
- `arch.py` : architectures des agents Speaker et Listener  
- `train.py` : entraînement des agents dans le jeu de communication  
- `features.py` : gestion des représentations perceptives et du besoin discriminatif  
- `best_parameters.py` : optimisation des hyperparamètres (Optuna)  
- `langueg_maker.py` : extraction des langages émergents  
- `accuracy_complexity.py` : calcul des mesures Information Bottleneck et génération des figures

### Données
- Données humaines issues du **World Color Survey**
- Représentations perceptives en espace CIELAB
- Langages émergents extraits sous forme de distributions `P(w | c)`

---

## Pipeline expérimental

1. Préparation des données perceptives (WCS)
2. Entraînement des agents artificiels à communiquer
3. Extraction des langages émergents
4. Calcul de la complexité lexicale et de la précision perceptive
5. Comparaison avec les langues humaines et la frontière IB

---

## Résultats principaux

Les résultats reproduisent les observations de l’article :
- les langages émergents sont proches de la frontière Information Bottleneck,
- le besoin discriminatif influence la complexité des langages,
- les langues humaines présentent une plus grande variabilité, suggérant des contraintes supplémentaires.

---

## Auteurs

Simon GROC  
Shirel AMOZIEG  

Sorbonne Université – Master Intelligence Artificielle  
Projet IAR – Janvier 2026
