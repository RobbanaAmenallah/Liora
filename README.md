# Neural Network Lab — Masterclass Deep Learning Interactive

Application web interactive de simulation, de visualisation et d'enseignement du Deep Learning, conçue pour animer une **Masterclass de 20 minutes en direct (Zoom / Présentiel)** à destination de Data Scientists et d'étudiants maîtrisant déjà `scikit-learn`.

Tous les entraînements, calculs de gradients, prédictions probabilistes et courbes d'apprentissage sont **100 % réels** et exécutés en temps réel dans le navigateur via **TensorFlow.js**.

---

## Sommaire

1. [Installation et Lancement Rapide](#1-installation-et-lancement-rapide)
2. [Vue d'Ensemble de l'Interface Globale (`App.tsx`)](#2-vue-densemble-de-linterface-globale-apptsx)
3. [Partie 1 — Onglet 1 : `1. Explore the Neuron`](#3-partie-1--onglet-1--1-explore-the-neuron)
4. [Partie 2 — Onglet 2 : `2. Train a Neural Network`](#4-partie-2--onglet-2--2-train-a-neural-network)
5. [Partie 3 — Onglet 3 : `3. Neural Network Challenge`](#5-partie-3--onglet-3--3-neural-network-challenge)
6. [Partie 4 — Composants Éducatifs Transversaux (Fiches & Glossaire)](#6-partie-4--composants-éducatifs-transversaux-fiches--glossaire)
7. [Partie 5 — Moteur Machine Learning & Rigueur Scientifique (`src/lib/`)](#7-partie-5--moteur-machine-learning--rigueur-scientifique-srclib)
8. [Arborescence Complète du Projet](#8-arborescence-complète-du-projet)
9. [Stack Technologique](#9-stack-technologique)

---

## 1. Installation et Lancement Rapide

### Prérequis
- **Node.js** (v18+ ou v20+ recommandé)
- **npm**

### Commandes
```bash
# 1. Installer les dépendances
npm install

# 2. Lancer le serveur de développement local
npm run dev

# 3. Compiler pour la production (vérification TypeScript + bundle Vite)
npm run build

# 4. Prévisualiser le build de production
npm run preview
```

Ouvrez ensuite l'URL affichée dans le terminal (par défaut `http://localhost:5173`) en plein écran (1920×1080) pour votre partage d'écran.

---

## 2. Vue d'Ensemble de l'Interface Globale (`App.tsx`)

L'application s'articule autour d'une enveloppe principale (`src/App.tsx`) qui coordonne la navigation, les outils de présentation pour l'animateur et les ressources pédagogiques :

### A. En-tête de Navigation & Contrôles Formateur (Header)
- **Identité visuelle :** Logo **Neural Network Lab**, badge *« Masterclass Live 20 min »* et sous-titre résumant le fil conducteur (*Du Machine Learning `scikit-learn` aux Réseaux de Neurones Profonds*).
- **Sélecteur des 3 Onglets Principaux :**
  1. `1. Explore the Neuron` — Anatomie mathématique et géométrique d'un neurone unique.
  2. `2. Train a Neural Network` — Laboratoire complet d'entraînement TensorFlow.js en temps réel.
  3. `3. Neural Network Challenge` — Jeu interactif en 3 missions notées sur 300 points.
- **Bouton « Démarrer la démonstration » :** Bascule automatiquement sur l'Onglet 2 et charge l'**Expérience A** du parcours pédagogique guidé.
- **Bouton « Presentation Mode » (Mode Zoom 1080p) :**
  - Masque les panneaux de configuration latéraux et les fiches théoriques du bas de page afin de dédier **100 % de la largeur de l'écran** aux visualisations graphiques.
  - Affiche une **barre d'accès rapide formateur** contenant :
    - L'objectif pédagogique dynamique de l'écran ou de l'expérience en cours.
    - Les boutons de sélection directe **`Exp. A`**, **`Exp. B`**, **`Exp. C`**, **`Exp. D`**.
    - Le bouton **`Reset Demo Defaults`** pour réinitialiser instantanément la démonstration.

### B. Pied de page (Footer)
- Affiche les crédits de l'application (**Robbana Amenallah — Liora**) ainsi que les boutons d'accès direct : **Email**, **LinkedIn**, **Portfolio** et **GitHub**.

---

## 3. Partie 1 — Onglet 1 : `1. Explore the Neuron`

**Fichier :** `src/components/tabs/ExploreNeuronTab.tsx`  
**Objectif pédagogique (Minutes 0 à 4) :** Faire le pont entre `LogisticRegression` / `LinearRegression` de `scikit-learn` (`coef_`, `intercept_`) et le neurone biologique/formel en décomposant ses deux opérations : la **combinaison linéaire pondérée** $z = w_1 x_1 + w_2 x_2 + b$ et l'**activation non-linéaire** $a = \varphi(z)$.

### A. Panneau Latéral de Contrôle (3 Cartes de Paramètres)
*(En mode Présentation, ce panneau est remplacé par une barre horizontale compacte au-dessus du diagramme).*
1. **Carte 1 — Variables d'entrée (`Features` $x_1, x_2$) :**
   - Deux curseurs continus dans $[-3.0, +3.0]$ (pas de $0.1$) représentant les coordonnées d'une observation (une ligne de la matrice `X` dans `scikit-learn`).
2. **Carte 2 — Paramètres appris (`Poids` $w_1, w_2$ et `Biais` $b$) :**
   - Curseurs pour $w_1 \in [-3.0, +3.0]$ et $w_2 \in [-3.0, +3.0]$ avec code couleur dynamique : **turquoise** si le poids est excitateur ($w \ge 0$), **orange** s'il est inhibiteur ($w < 0$).
   - Curseur pour le biais $b \in [-3.0, +3.0]$ (ordonnée à l'origine qui translate le seuil d'activation).
3. **Carte 3 — Sélecteur de Fonction d'Activation $\varphi(z)$ :**
   - 4 boutons interactifs avec leur formule mathématique :
     - **Linear :** $\varphi(z) = z$
     - **ReLU :** $\varphi(z) = \max(0, z)$
     - **Sigmoid :** $\varphi(z) = \frac{1}{1 + e^{-z}}$
     - **Tanh :** $\varphi(z) = \tanh(z) \in [-1, 1]$

### B. Diagramme SVG Animé du Flux de Propagation Avant
- **Nœuds d'entrée ($x_1, x_2$) :** Affichent les valeurs numériques en temps réel.
- **Connexions synaptiques animées :**
  - Des impulsions lumineuses (`framer-motion`) circulent en continu des entrées vers le sommateur $\Sigma$, puis vers l'activation et la sortie.
  - L'épaisseur de chaque trait est proportionnelle à $|w_i|$ et sa couleur reflète son signe.
  - Des badges sur chaque connexion affichent en direct la valeur du poids $w_i$ et le produit partiel $w_i \cdot x_i$.
- **Nœud Sommateur $\Sigma (z)$ :** Affiche la somme pondérée exacte $z = w_1 x_1 + w_2 x_2 + b$ avec l'apport du biais $b$ en pointillés violets.
- **Boîte de Fonction d'Activation avec Mini-Graphique Temps Réel :**
  - Trace dynamiquement la courbe de la fonction $\varphi(z)$ choisie sur l'intervalle $[-4, +4]$ et positionne un **point turquoise** exactement aux coordonnées actuelles $(z, a)$.
- **Nœud de Sortie Finale ($a$) :** Affiche la valeur finale $a = \varphi(z)$ après activation.

### C. Boîte d'Équations Substituées en Direct
- Sous le schéma SVG, deux lignes de formules substituent en temps réel les valeurs numériques choisies par l'utilisateur :
  1. **Pré-activation :** `z = w₁·x₁ + w₂·x₂ + b = (1.50 × 1.20) + (-1.00 × -0.80) + (-0.30) = 2.300`
  2. **Activation non-linéaire :** `a = RELU(z) = RELU(2.300) = 2.3000`

### D. Journal d'Analyse en Direct & Défi « Prédisez avant de manipuler »
- **Journal d'analyse en direct :** Explique en français l'impact exact du dernier curseur manipulé et rappelle l'interprétation géométrique (dans le plan $(x_1, x_2)$, $w_1 x_1 + w_2 x_2 + b = 0$ est une droite dont le vecteur normal est $(w_1, w_2)$).
- **Défi « Intuition ReLU » :** Pose une question prédictive (*« Que se passe-t-il sur la sortie $a$ si vous diminuez le biais $b$ jusqu'à $z < 0$ avec ReLU ? »*) avec bouton de vérification expliquant l'effet d'interrupteur non-linéaire ($a = 0$).

---

## 4. Partie 2 — Onglet 2 : `2. Train a Neural Network`

**Fichier :** `src/components/tabs/TrainNetworkTab.tsx`  
**Objectif pédagogique (Minutes 4 à 15) :** Entraîner un vrai réseau de neurones TensorFlow.js en direct et observer comment l'architecture (nombre de couches cachées, neurones, fonction d'activation) sculpte la **Decision Boundary** et influence les **Learning Curves** (Underfitting vs Overfitting).

### A. Bannière du Mode Démonstration Guidée (4 Expériences Masterclass)
Permet de charger en 1 clic l'une des **4 expériences scénarisées** (`GUIDED_EXPERIMENTS` dans `src/lib/educationalContent.ts`) :
- **Expérience A — Modèle linéaire sur Two Moons (`0 Hidden Layer`) :**
  - Montre qu'un réseau sans couche cachée est strictement équivalent à une `LogisticRegression` : la frontière $P(y=1)=0.5$ reste une ligne droite (~85–87 % d'accuracy max, cas d'**Underfitting**).
- **Expérience B — 1 couche cachée (`8 neurones + ReLU`) :**
  - Montre comment chaque neurone caché crée une « charnière » linéaire et comment la couche de sortie les combine pour épouser le couloir courbe entre les deux croissants (>96 % d'accuracy).
- **Expérience C — 2 couches cachées (`2 × 8 neurones + ReLU`) :**
  - Illustre la composition hiérarchique des représentations et permet d'introduire le compromis entre finesse de la frontière et nombre de paramètres entraînables.
- **Expérience D — Fort bruit (`0.35`) & Généralisation (`3 × 14 neurones`) :**
  - Sur un petit jeu de données très bruité avec un réseau sur-dimensionné, met en évidence l'**Overfitting** : la *Training Loss* continue de baisser alors que la *Validation Loss* stagne ou remonte.

Chaque expérience déploie un **panneau pédagogique interactif** comprenant :
1. Le contexte et le message clé à retenir.
2. Une **question QCM à poser à la classe avant de lancer l'entraînement** (avec validation immédiate de la prédiction).
3. Un bouton direct **« Lancer l'entraînement de l'Expérience »** et un lien vers l'expérience suivante.

### B. Barre d'Actions & Bandeau de Métriques en Temps Réel
- **5 Boutons de Contrôle du Cycle de Vie du Modèle :**
  - `Initialize Model` : Reconstruit le réseau TensorFlow.js et réinitialise les poids selon la graine aléatoire.
  - `Train` / `Continuer Train` : Lance ou reprend la boucle d'entraînement asynchrone époque par époque.
  - `Pause` / `Resume` : Fige l'entraînement à l'époque courante pour permettre à l'enseignant de commenter la frontière ou les poids, puis reprend sans perte d'état.
  - `Reset` : Arrête l'entraînement et réinitialise le modèle à l'époque 0.
  - `Compare Models (N)` : Capture un instantané (*snapshot*) complet des performances du modèle actuel et l'ajoute au tableau comparatif.
- **5 Compteurs Temps Réel :**
  - `Epoch` (ex. `45 / 80`), `Train Acc` (%), `Val Acc` (%), `Current Loss` (Binary Cross-Entropy), et `Paramètres` (nombre exact de poids et biais entraînables).

### C. Panneau Latéral de Configuration (Masqué en Presentation Mode)
1. **Carte « Jeu de Données (Dataset) » :**
   - **4 topologies synthétiques 2D :** `Two Moons`, `Circles` (cercles concentriques), `XOR` (damier 4 quadrants), et `Linéaire` (nuages gaussiens séparables).
   - **Échantillons (`Samples`) :** Curseur de $100$ à $800$ points.
   - **Bruit gaussien (`Noise`) :** Curseur de $0.00$ à $0.45$.
   - **Split Train / Validation :** Boutons `70/30`, `80/20`, `90/10` (découpage stratifié).
2. **Carte « Architecture & Hyperparamètres » :**
   - **Couches cachées (`Hidden Layers`) :** Curseur de `0` (modèle linéaire) à `4` couches cachées.
   - **Neurones par couche :** Curseur de `1` à `16` neurones par couche cachée.
   - **Activation cachée :** Boutons `RELU`, `TANH`, `SIGMOID` (la couche de sortie utilise toujours `Sigmoid` pour prédire $P(y=1) \in [0, 1]$).
   - **Learning Rate (Adam) :** Sélecteur rapide `0.003`, `0.01`, `0.03`, `0.1`.
   - **Epochs :** Liste déroulante (`40`, `60`, `80`, `120`, `200` époques).
   - **Batch Size :** Liste déroulante (`16`, `32`, `64`, `128` échantillons par lot).

### D. Les 3 Panneaux de Visualisation Synchronisés (`src/components/visualizations/`)

#### 1. Panneau A — Frontière de Décision 2D (`DecisionBoundaryCanvas.tsx`)
- **Carte de chaleur probabiliste (Heatmap) :** Évalue le modèle TensorFlow.js sur une grille $34 \times 34$ et interpole les probabilités $P(y=1)$ du bleu ciel ($0.0$, Classe 0) à l'orange ($1.0$, Classe 1) via un `offscreenCanvas` haute performance.
- **Contour exact $P(y=1) = 0.50$ (Marching Squares) :** Trace une courbe vectorielle turquoise continue représentant la frontière de décision exacte où le modèle bascule entre la Classe 0 et la Classe 1.
- **Nuage de points Train / Validation :**
  - Les points d'entraînement sont cerclés de bleu nuit ; les points de validation sont **cerclés de blanc** pour être immédiatement distinguables.
  - Deux boutons filtres (`Train` et `Validation`) permettent d'afficher ou de masquer chaque sous-ensemble en 1 clic (idéal pour montrer comment les points de validation se comportent vis-à-vis de la frontière apprise sur le Train).
- **Inspecteur de probabilité au survol (Zero-Lag Crosshair) :** Déplacer la souris sur le canevas affiche instantanément les coordonnées $(x_1, x_2)$ et la probabilité prédite $P(y=1)$ en pourcentage sans déclencher de re-rendu React.

#### 2. Panneau B — Architecture & Poids Synaptiques Appris (`NetworkArchitectureSvg.tsx`)
- **Rendu dynamique de la topologie :** Dessine les nœuds d'entrée ($x_1, x_2$), les couches cachées ($1$ à $4$ couches de $1$ à $16$ neurones) et le neurone de sortie ($\hat{y}$).
- **Codage visuel des poids appris en temps réel :**
  - À chaque rafraîchissement d'époque, les tenseurs de poids et de biais sont extraits du modèle TensorFlow.js.
  - **Couleur de la synapse :** Turquoise si $w \ge 0$, orange si $w < 0$.
  - **Épaisseur et opacité :** Proportionnelles à l'amplitude $|w|$.
  - **Biais ($b$) :** Affichés directement à l'intérieur des nœuds cachés lorsque le nombre de neurones par couche est $\le 10$.
- **Inspecteur de connexion au survol :** Survoler n'importe quelle synapse la met en surbrillance blanche et affiche une infobulle technique précisant la couche source/destination, le poids exact $w$ (à $10^{-4}$ près) et le biais du neurone cible $b_{\text{dest}}$.

#### 3. Panneau C — Courbes d'Apprentissage (`LearningCurvesChart.tsx`)
Deux graphiques interactifs (`Recharts`) mis à jour en direct pendant l'entraînement :
- **Graphique gauche — Courbe de Coût (`Binary Cross-Entropy Loss`) :**
  - Compare la **Training Loss** (trait plein turquoise) et la **Validation Loss** (pointillés orange).
  - Permet de repérer immédiatement le point de retournement caractéristique du sur-apprentissage (*Overfitting*).
- **Graphique droit — Précision de Classification (`Accuracy %`) :**
  - Compare la **Training Accuracy** (trait plein bleu ciel) et la **Validation Accuracy** (pointillés violets) de $0\%$ à $100\%$.

### E. Tableau Comparatif des Modèles Enregistrés (`Compare Models`)
- Affiche tous les modèles sauvegardés au fil de la session avec :
  - Nom et résumé de l'architecture (ex. `Linéaire (0 couche cachée)`, `1×8 (RELU)`, `2×8 (RELU)`).
  - Type de dataset et niveau de bruit.
  - Nombre de paramètres entraînables et époque finale.
  - `Train Acc`, `Val Acc`, `Train Loss`, `Val Loss`.
  - **Colonne Diagnostic Automatique :** Analyse l'écart Train/Validation et la précision pour qualifier le modèle :
    - `Underfitting (Linéaire)` (si `valAcc < 82%` avec `0` couche cachée),
    - `Risque Overfitting` (si `trainAcc - valAcc > 8%` ou `valLoss > 1.35 × trainLoss`),
    - `Équilibré` (dans les cas optimaux).

---

## 5. Partie 3 — Onglet 3 : `3. Neural Network Challenge`

**Fichier :** `src/components/tabs/ChallengeTab.tsx`  
**Objectif pédagogique (Minutes 15 à 20) :** Valider les acquis de la Masterclass par la pratique grâce à un jeu interactif structuré en **3 missions progressives** notées sur un total de **300 points** (100 points par mission).

### A. Système de Score (100 points par mission)
Chaque mission est évaluée selon 3 critères objectifs :
- **50 points :** Atteindre ou dépasser la cible de **Validation Accuracy** fixée par la mission (25 points si proche à moins de 4 points de la cible).
- **20 points :** Respecter le **budget maximal de paramètres entraînables** (frugalité / rasoir d'Occam, accordé si la cible de précision est atteinte).
- **30 points :** Répondre correctement à la **question théorique de compréhension**.

### B. Les 3 Missions du Challenge
1. **Mission 1 — Vaincre la non-linéarité (`Two Moons`) :**
   - *Problème initial :* Le réseau démarre avec `0` couche cachée (3 paramètres) et plafonne autour de ~86 % en traçant une droite.
   - *Objectif :* Choisir une architecture non-linéaire simple atteignant **$\ge 92\%$ de Validation Accuracy** avec **$\le 80$ paramètres** (ex. 1 couche cachée de 6 ou 8 neurones ReLU/Tanh).
2. **Mission 2 — Entourer le cercle central (`Concentric Circles`) :**
   - *Problème initial :* Une droite coupe les cercles concentriques en deux (~50 % de réussite, équivalent d'un pile ou face).
   - *Objectif :* Construire un réseau capable de fermer un polygone autour de la classe centrale pour atteindre **$\ge 92\%$ de Validation Accuracy** avec **$\le 90$ paramètres**.
   - *Concept clé vérifié :* Il faut au minimum 3 neurones cachés en 2D (3 demi-plans formant un triangle fermé) pour entourer une région bornée.
3. **Mission 3 — Éviter le Sur-apprentissage / Overfitting (`Two Moons très bruité`) :**
   - *Problème initial :* Sur un petit dataset très bruité (`noise = 0.33`, `200` échantillons), le réseau par défaut est sur-dimensionné (`4` couches de `16` neurones = **881 paramètres**) et mémorise le bruit.
   - *Objectif :* Réduire drastiquement l'architecture à **$\le 65$ paramètres** tout en maintenant **$\ge 84\%$ de Validation Accuracy** (ex. 1 couche de 6 neurones = 25 paramètres).

### C. Interface en 2 Colonnes de chaque Mission
- **Colonne gauche :**
  - *Étape 1 :* Sélecteurs rapides par boutons pour le nombre de couches cachées (`0`, `1`, `2`, `4`), le nombre de neurones par couche (`4`, `6`, `8`, `16`) et l'activation (`RELU`, `TANH`), avec calcul instantané du nombre de paramètres et indicateur vert/orange selon le budget maximal autorisé.
  - *Étape 2 :* Question théorique à choix multiples (3 options).
  - *Bouton d'action unique :* **`Tester & Valider ma Mission`** qui entraîne réellement le modèle TensorFlow.js sur 45 époques (en 3 salves rapides de 15 époques avec barre de progression) et évalue le score.
- **Colonne droite :**
  - Mini-canevas 2D affichant le dataset et la frontière de décision apprise.
  - Affichage côte à côte de la `Train Accuracy` et de la `Validation Accuracy`.
  - Encadré de **feedback pédagogique personnalisé** expliquant pourquoi la mission est réussie ou comment corriger l'architecture (trop rigide, trop complexe, ou erreur au quiz), avec bouton **`Mission suivante`**.

---

## 6. Partie 4 — Composants Éducatifs Transversaux (Fiches & Glossaire)

### A. Section des 7 Fiches Pédagogiques & Passerelles `scikit-learn`
**Fichiers :** `src/components/educational/ConceptCardsSection.tsx` et `src/lib/educationalContent.ts`  
Affichée au bas de l'application (repliable et automatiquement masquée en *Presentation Mode*), cette section propose **7 fiches de référence** reliant systématiquement les concepts du Deep Learning aux acquis `scikit-learn` :

| # | Concept | Passerelle `scikit-learn` | Défi « Predict before you train » inclus |
| :--- | :--- | :--- | :--- |
| **1** | **Weights & Bias ($w, b$)** | `model.coef_` et `model.intercept_` dans `LogisticRegression` | Effet de l'inversion du signe de $w_1$ et $w_2$ sur la frontière |
| **2** | **Activation Functions** | Paramètre `activation="relu"` dans `MLPClassifier` | Pourquoi ReLU évite le *vanishing gradient* face à Sigmoid |
| **3** | **Hidden Layers** | Remplace `PolynomialFeatures` ou le noyau RBF manuel par un apprentissage automatique de représentation | Nombre minimal de neurones ReLU pour fermer une zone 2D (3 neurones) |
| **4** | **Forward Propagation** | Appel `model.predict_proba(X)` | Pourquoi la passe avant ne modifie jamais les poids |
| **5** | **Binary Cross-Entropy Loss** | `sklearn.metrics.log_loss` | Valeur théorique de la Loss initiale quand $\hat{p} = 0.5$ ($-\ln(0.5) \approx 0.693$) |
| **6** | **Backpropagation & Learning Rate** | `learning_rate_init` et `solver="adam"` dans `MLPClassifier` | Comportement de la Loss lorsque le Learning Rate est trop élevé |
| **7** | **Underfitting vs Overfitting** | Diagnostic par `learning_curve` / `validation_curve` | Signature visuelle de l'Overfitting entre Training Loss et Validation Loss |

Chaque fiche comporte un bouton interactif **« → Révéler la réponse & le message clé »** permettant d'interroger les participants avant d'afficher la solution.

### B. Infobulles Contextuelles (`FrenchTooltip.tsx`)
**Fichier :** `src/components/ui/FrenchTooltip.tsx`  
Dispersées à côté de chaque réglage et graphique (icône `?`), ces infobulles affichent au survol ou au clic :
- La définition claire du terme technique en français.
- Un encadré **« Repère scikit-learn »** donnant l'équivalent exact en Python (`StandardScaler().fit_transform(X_train)`, `train_test_split(..., stratify=y)`, `accuracy_score`, `log_loss`, etc.).

---

## 7. Partie 5 — Moteur Machine Learning & Rigueur Scientifique (`src/lib/`)

L'application respecte les bonnes pratiques d'ingénierie Machine Learning :

### A. Générateur de Données Déterministe & Sans Fuite (`src/lib/datasets.ts`)
- **Reproductibilité à 100 % (`SeededRandom`) :** Implémente le générateur pseudo-aléatoire déterministe **Mulberry32** couplé à la transformation de **Box-Muller** pour le bruit gaussien. Une même graine produit toujours exactement le même nuage de points.
- **Split Train / Validation Stratifié :** Les classes $0$ et $1$ sont mélangées et découpées séparément selon le ratio choisi ($70\%$, $80\%$ ou $90\%$) afin de garantir un équilibre parfait des classes dans les deux ensembles.
- **Zéro Data Leakage (Aucune fuite de données) :** La moyenne $(\mu_{x_1}, \mu_{x_2})$ et l'écart-type $(\sigma_{x_1}, \sigma_{x_2})$ de standardisation sont calculés **exclusivement sur `trainPoints`**, puis appliqués aux points de validation (`valPoints`) et aux points de la grille de décision 2D (exactement comme `scaler.fit(X_train)` suivi de `scaler.transform(X_val)`).

### B. Moteur TensorFlow.js Temps Réel (`src/lib/tfModel.ts`)
- **Backend CPU Optimisé pour les Réseaux 2D :** Pour de petits réseaux denses (2 entrées, 1 à 4 couches de 1 à 16 neurones), le backend CPU de TensorFlow.js est configuré par défaut car il évite la latence de transfert mémoire GPU $\leftrightarrow$ CPU à chaque époque, offrant une fluidité 10× à 30× supérieure.
- **Initialisation Scientifique des Poids :**
  - Couches cachées avec activation **ReLU** : initialisation **He Normal** (`tf.initializers.heNormal`).
  - Couches cachées **Tanh / Sigmoid** et couche de sortie : initialisation **Glorot / Xavier Normal** (`tf.initializers.glorotNormal`).
  - Biais initialisés à zéro (`tf.initializers.zeros`).
- **Fonction de Coût & Optimiseur :** `binaryCrossentropy` optimisée par `tf.train.adam(learningRate)`.
- **Évaluation Rapide & Gestion Mémoire (`tf.tidy` / `dispose`) :**
  - La fonction `evaluateSplit` calcule la Binary Cross-Entropy et l'Accuracy dans une boucle `Float32Array` encapsulée dans `tf.tidy()` pour éviter toute fuite de tenseurs.
  - Le tenseur de la grille 2D (`buildNormalizedGridTensor`) est pré-alloué une seule fois au début de l'entraînement et nettoyé dans le bloc `finally`, garantissant zéro fuite mémoire même après des dizaines d'entraînements consécutifs.

---

## 8. Arborescence Complète du Projet

```text
Liora/
├── index.html                                         # Point d'entrée HTML
├── package.json                                       # Dépendances et scripts npm
├── tailwind.config.js                                 # Palette de couleurs (navy, turquoise, violet, orange)
├── tsconfig.json                                      # Configuration TypeScript
├── vite.config.ts                                     # Configuration du bundler Vite
├── README.md                                          # Documentation complète du projet
└── src/
    ├── main.tsx                                       # Montage React dans le DOM
    ├── App.tsx                                        # En-tête, navigation 3 onglets, Presentation Mode, Footer
    ├── index.css                                      # Styles globaux et directives Tailwind CSS
    ├── types/
    │   └── ml.ts                                      # Interfaces TypeScript (Dataset, ModelConfig, EpochMetric...)
    ├── lib/
    │   ├── datasets.ts                                # PRNG Mulberry32, génération 2D, split stratifié, normalisation
    │   ├── tfModel.ts                                 # Création du modèle TF.js, entraînement, extraction des poids
    │   ├── educationalContent.ts                      # Données des 4 Expériences guidées (A-D) et des 7 Fiches concepts
    │   └── utils.ts                                   # Utilitaires de classes CSS (clsx + tailwind-merge)
    └── components/
        ├── tabs/
        │   ├── ExploreNeuronTab.tsx                   # Onglet 1 : Simulateur de neurone formel & SVG animé
        │   ├── TrainNetworkTab.tsx                    # Onglet 2 : Laboratoire d'entraînement TF.js & comparaison
        │   └── ChallengeTab.tsx                       # Onglet 3 : Jeu interactif en 3 missions (score / 300 pts)
        ├── visualizations/
        │   ├── DecisionBoundaryCanvas.tsx             # Heatmap 2D + contour Marching Squares + inspecteur souris
        │   ├── NetworkArchitectureSvg.tsx             # Graphe SVG dynamique des poids et biais appris
        │   └── LearningCurvesChart.tsx                # Courbes Recharts (Train/Val Loss & Train/Val Accuracy)
        ├── educational/
        │   └── ConceptCardsSection.tsx                # Grille repliable des 7 fiches pédagogiques & quiz rapides
        └── ui/
            └── FrenchTooltip.tsx                      # Infobulle du glossaire avec passerelles scikit-learn
```

---

## 9. Stack Technologique

- **Framework UI :** React 18 + TypeScript
- **Moteur Deep Learning :** TensorFlow.js (`@tensorflow/tfjs` v4.22)
- **Visualisations :**
  - **HTML5 Canvas 2D + Algorithme Marching Squares** (Frontière de décision et Heatmap probabiliste)
  - **SVG Dynamique + Framer Motion** (Anatomie du neurone et graphe d'architecture du réseau)
  - **Recharts** (Courbes d'apprentissage Loss et Accuracy en temps réel)
- **Design & Styles :** Tailwind CSS + Lucide React (icônes)
- **Build & Serveur de développement :** Vite 6

---

## Auteur

**Robbana Amenallah — Liora**  
- **Email :** [amenallah.robbana@dauphine.eu](mailto:amenallah.robbana@dauphine.eu)  
- **LinkedIn :** [linkedin.com/in/robbana-amenallah](https://www.linkedin.com/in/robbana-amenallah/)  
- **Portfolio :** [robbanaamenallah.github.io](https://robbanaamenallah.github.io/)  
- **GitHub :** [github.com/RobbanaAmenallah](https://github.com/RobbanaAmenallah)
