1. Qu'est-ce qu'un réseau de neurones à propagation avant ?
Un réseau de neurones à propagation avant (feedforward neural network) est une architecture où l'information circule exclusivement dans une seule direction : des neurones d'entrée vers les neurones de sortie, sans aucune boucle ni retour en arrière.
Comment l'information se déplace :

L'information entre par la couche d'entrée (input layer)
Elle traverse une ou plusieurs couches cachées (hidden layers) où chaque neurone applique une transformation : sortie = f(Σ(poids × entrée) + biais)
Elle atteint finalement la couche de sortie (output layer)
Aucune connexion ne revient en arrière pendant la phase de prédiction

Le perceptron multicouche (MLP) est l'exemple classique de ce type d'architecture. Les CNN (réseaux convolutifs) sont également des réseaux à propagation avant.

2. Fonctionnement des réseaux de neurones récurrents (RNN)
Les RNN (Recurrent Neural Networks) sont des réseaux qui possèdent des connexions récurrentes permettant de maintenir une mémoire des informations précédentes.
Comment l'information se déplace :

À chaque pas de temps t, le RNN reçoit une entrée x(t)
Il possède un état caché h(t) qui mémorise l'information des pas précédents
La formule : h(t) = f(W_hh × h(t-1) + W_xh × x(t) + b)
Cet état caché est réutilisé au pas suivant, créant une boucle récurrente

Différence avec les réseaux à propagation avant :

Feedforward : pas de mémoire, chaque prédiction est indépendante
RNN : possède une mémoire via les connexions récurrentes, peut traiter des séquences
Feedforward : architecture statique
RNN : architecture dynamique qui se "déploie" dans le temps

Note importante : La backpropagation n'est PAS spécifique aux RNN ! C'est l'algorithme d'apprentissage utilisé par TOUS les réseaux de neurones (feedforward et récurrents). Pour les RNN, on parle de BPTT (Backpropagation Through Time).

3. Cas d'utilisation adaptés
MLP (Multilayer Perceptron) :

Données tabulaires et classification simple
Problèmes de régression
Reconnaissance de patterns simples
Exemple : prédiction de prix, classification binaire

CNN (Convolutional Neural Network) :

Traitement d'images et vision par ordinateur
Détection d'objets, segmentation d'images
Reconnaissance faciale
Également efficace pour certains signaux 1D (audio, séries temporelles)

RNN (Recurrent Neural Network) :

Traitement du langage naturel (NLP)
Séries temporelles et prédictions temporelles
Génération de texte, traduction automatique
Reconnaissance vocale
Analyse de vidéos (séquences d'images)


4. Types de RNN
Architecture de base :
a) Simple RNN (Vanilla RNN)

Structure basique avec une seule couche récurrente
Illustration : Input → [État caché ⟲] → Output

b) RNN Bidirectionnel (Bi-RNN)

Deux RNN : un traite la séquence de gauche à droite, l'autre de droite à gauche
Combine les deux sorties pour capturer le contexte passé ET futur
Exemple : analyse de sentiment où le contexte complet de la phrase est important

c) RNN Profond (Deep RNN)

Multiple couches de RNN empilées verticalement
Chaque couche extrait des représentations de plus haut niveau
Illustration :

Entrée → RNN₁ → RNN₂ → RNN₃ → Sortie
         ⟲      ⟲      ⟲
d) Architectures par type de séquence :

One-to-Many : Une entrée → séquence de sorties (ex: génération d'image vers description)
Many-to-One : Séquence → une sortie (ex: analyse de sentiment d'un texte)
Many-to-Many : Séquence → séquence (ex: traduction automatique)


5. Exploding et Vanishing Gradients
Vanishing Gradients (gradients évanescents) :

Définition : Les gradients deviennent extrêmement petits lors de la rétropropagation à travers de nombreuses couches/pas de temps
Conséquence : Le réseau n'apprend plus les dépendances à long terme
Cause : Multiplication répétée de valeurs < 1

Exploding Gradients (gradients explosifs) :

Définition : Les gradients deviennent extrêmement grands
Conséquence : Instabilité de l'apprentissage, poids qui divergent
Cause : Multiplication répétée de valeurs > 1

Solutions :
Pour Vanishing Gradients :

Utiliser LSTM ou GRU (architectures avec portes)
Fonctions d'activation appropriées (ReLU au lieu de sigmoid/tanh)
Initialisation des poids adaptée (Xavier, He)
Connexions résiduelles (skip connections)

Pour Exploding Gradients :

Gradient Clipping : limiter la norme des gradients
Réduction du taux d'apprentissage
Normalisation par batch (Batch Normalization)


6. Limites et désavantages des RNN
Limites principales :

Dépendances à long terme : difficulté à capturer des relations distantes dans les séquences
Problèmes de gradients : vanishing/exploding gradients limitent l'apprentissage
Traitement séquentiel : impossible de paralléliser, donc lent à entraîner
Mémoire limitée : l'état caché a une capacité limitée
Coût computationnel : BPTT est gourmand en mémoire et calcul
Sensibilité à l'ordre : petites variations dans la séquence peuvent affecter fortement la sortie


7. Architectures avancées de RNN
a) LSTM (Long Short-Term Memory)
Architecture :

Possède une cellule mémoire en plus de l'état caché
Utilise 3 portes pour contrôler le flux d'information :

Forget gate (porte d'oubli) : décide quelles informations oublier
Input gate (porte d'entrée) : décide quelles nouvelles informations stocker
Output gate (porte de sortie) : décide quelles informations transmettre



Fonctionnement :

Les portes utilisent des fonctions sigmoïdes (valeurs entre 0 et 1) pour "filtrer" l'information
La cellule mémoire maintient l'information à long terme
Résout largement le problème de vanishing gradients

b) GRU (Gated Recurrent Unit)
Architecture :

Version simplifiée du LSTM avec seulement 2 portes :

Reset gate : contrôle combien d'information passée ignorer
Update gate : contrôle combien d'information passée conserver



Fonctionnement :

Plus simple et rapide que LSTM (moins de paramètres)
Performances souvent similaires au LSTM
Pas de cellule mémoire séparée

c) Autres architectures :

Attention Mechanisms : permet au réseau de "se concentrer" sur certaines parties de la séquence
Transformers : ont largement remplacé les RNN pour le NLP (mais techniquement ne sont plus des RNN)