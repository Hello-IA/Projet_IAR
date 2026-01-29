=== EGG : Emergence of lanGuage in Games ===
	Librairie de recherche développée par certains auteurs de l’article dans des travaux antérieurs.
	Permet d’implémenter des jeux de communication entre agents (Speaker / Listener)
	et de faire émerger des langages discrets.

EGG fournit :
- une architecture Speaker / Listener
- des jeux de communication (signal game, referential game, etc.)
– des méthodes d’apprentissage adaptées aux canaux discrets
  (REINFORCE, Gumbel-Softmax),
- la gestion du canal discret (le mot)

Dans le projet, EGG est utilisée pour entraîner les agents
et générer les langages ensuite analysés avec l’Information Bottleneck.


=== arch.py === 
définit l’architecture des réseaux (Speaker / Listener)
(dessin tableau)

=== features.py ===
représentation perceptive des couleurs
distance en fct des percentiles entre la cible et le distracteur
-> fig 4

=== langueg_maker.py ===
transforme la sortie du modèle en langage exploitable sous forme de distributions
p(w∣c), qui sont ensuite utilisées pour l’analyse Information Bottleneck
Sauvegarde matrice complète 330x1024
330 lignes : couleurs 
1024 colonnes : un mot donné
-> pour chaque couleur, proba d'un mot donné produit par le speaker

=== accuracy_complexity.py ===
Permet de générer la courbe IB
Une couleur correspond à une distribution perceptive floue
Chaque langue -> un point sur le plan IB
-> fig 3

– Calcule les mesures Information Bottleneck :
  • Complexité I(M;W)
  • Précision perceptive I(U;W)
– Applique ces mesures aux langues WCS et aux langages NN
– Calcule la frontière théorique IB
– Produit la figure de comparaison (langues humaines vs NN)

=== best_parameters.py ===
Utilise Optuna pour optimiser les hyperparamètres du jeu de communication
Maximise l’accuracy de validation (réussite au jeu)
l’objectif est d’obtenir des agents capables de communiquer correctement

=== term.txt ===
données brutes WCS (langues humaines)
Chaque ligne ressemble à : lang  speaker  chip  term
 Exemple : Dans la langue X, le locuteur Y a nommé la couleur chip Z avec le mot "blue".

=== ours_images_single_sm0.h5 ===
contient les représentations perceptives des 330 couleurs WCS dans l’espace CIELAB
les 330 couleurs WCS représentées comme vecteurs numériques (CIELAB)

=== nn_language_seed_0.txt ===
des mots produits par le Speaker pour chaque couleur après entraînement
sous forme :
speaker chip word

similaire à term.txt mais pour les réseaux

=== nn_language_seed_X.npz ===
La matrice complète P(w∣c) du réseau.
C’est exactement l’entrée de accuracy_complexite.py pour les NN.
Chaque fichier nn_language_seed_X.npz contient la matrice complète 
P(w∣c) d’un langage émergent obtenu avec une initialisation différente.

=== wcs_accuracy_complexity.csv ===
Un fichier CSV contenant :
Language | I(M;W) | I(U;W)
Un point par langue humaine.

=== train.py ===
lance l’entraînement du jeu de communication
implémente le discrimination game
rf et gs -> fig 5


==========================
=== Pipeline du projet ===
==========================
Objectif : 
Analyser des systèmes de dénomination des couleurs (langues humaines et langages 
de réseaux de neurones) à l’aide du cadre Information Bottleneck, afin de mesurer 
le compromis entre complexité lexicale et précision communicationnelle.

M = la bulle que le Speaker a en tête
U = la bulle que l’auditeur reconstruit
Et le langage sert à :
M⟶W⟶U

Si tout marche bien :
U≈M
la communication est précise

M représente l’intention perceptive du Speaker, tandis que U représente la perception 
reconstruite par le Listener. Le langage sert à transmettre M via W pour que U s’en rapproche 
le plus possible.

1. Données et représentations
term.txt : données brutes du World Color Survey (langues humaines).
ours_images_single_sm0.h5 : représentations perceptives (CIELAB) des 330 couleurs WCS.

2. Génération des langages artificiels
Librairie : EGG (Emergence of lanGuage in Games).
archs.py : architectures Speaker / Listener.
train.py : implémentation du jeu de communication (signal game) et entraînement des agents 
	   via un canal discret (REINFORCE ou Gumbel-Softmax).
best_parameters.py : optimisation des hyperparamètres pour assurer un entraînement stable.

3. Extraction du langage appris
langueg_maker.py :
	clonage du Speaker entraîné,
	extraction de la distribution complète P(w∣c),
	sauvegarde sous forme de matrices 330×1024 (nn_language_seed_X.npz),
	représentant un langage émergent par seed.

4. Modèle perceptif et analyse IB
accuracy_complexite.py :
	modélisation perceptive des couleurs par des gaussiennes en CIELAB,
	calcul de la complexité I(M;W) et de la précisionI(U;W),
	analyse des langues humaines et artificielles,
	calcul de la frontière théorique Information Bottleneck.

5. Comparaison finale
Chaque langage est représenté par un point dans le plan (I(M;W),I(U;W)).
Comparaison entre langues humaines, langages NN et courbe IB (Figures 3, 4 et 5 de l’article).
Résultats sauvegardés dans wcs_accuracy_complexity.csv et figures associées.