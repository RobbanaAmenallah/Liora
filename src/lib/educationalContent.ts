import { GuidedExperiment } from '@/types/ml';

export const GUIDED_EXPERIMENTS: GuidedExperiment[] = [
  {
    id: 'A',
    title: 'Expérience A — Modèle linéaire sur Two Moons',
    subtitle: 'Comprendre la limite fondamentale d’un classifieur linéaire (équivalent LogisticRegression de scikit-learn)',
    dataset: 'moons',
    noise: 0.15,
    samples: 400,
    trainSplit: 0.8,
    config: {
      hiddenLayers: 0,
      neuronsPerLayer: 4,
      activation: 'relu',
      learningRate: 0.05,
      epochs: 60,
      batchSize: 32,
    },
    explanationFr:
      'Sans couche cachée (0 Hidden Layer), notre réseau se réduit à z = w₁x₁ + w₂x₂ + b suivi d’une Sigmoid. C’est exactement une régression logistique ! Essayons de séparer les deux croissants entrelacés (Two Moons).',
    questionFr:
      'Avant de lancer l’entraînement : quelle forme géométrique aura la Decision Boundary (frontière de décision) ?',
    optionsFr: [
      'Une ligne strictement droite incapable d’épouser les croissants (~85-88% accuracy max)',
      'Une courbe sinusoïdale parfaite grâce à la fonction Sigmoid de sortie',
      'Deux cercles concentriques autour de chaque classe',
    ],
    correctOptionIndex: 0,
    takeawayFr:
      'À retenir : La Sigmoid finale transforme le score en probabilité entre 0 et 1, mais la frontière P(y=1)=0.5 correspond à w₁x₁ + w₂x₂ + b = 0, qui reste toujours un hyperplan (une droite en 2D). On est en situation d’Underfitting.',
  },
  {
    id: 'B',
    title: 'Expérience B — 1 couche cachée (8 neurones + ReLU)',
    subtitle: 'Découvrir comment une couche cachée non-linéaire plie l’espace de représentation',
    dataset: 'moons',
    noise: 0.15,
    samples: 400,
    trainSplit: 0.8,
    config: {
      hiddenLayers: 1,
      neuronsPerLayer: 8,
      activation: 'relu',
      learningRate: 0.03,
      epochs: 80,
      batchSize: 32,
    },
    explanationFr:
      'Ajoutons 1 couche cachée de 8 neurones avec l’activation non-linéaire ReLU(z) = max(0, z). Chaque neurone caché crée une cassure linéaire (« charnière »), et le neurone de sortie les recombine.',
    questionFr:
      'Que va-t-il se passer sur la Decision Boundary avec 8 neurones ReLU ?',
    optionsFr: [
      'Elle reste une simple droite mais change d’inclinaison',
      'Elle devient une frontière non-linéaire par morceaux qui épouse le couloir entre les deux lunes',
      'Le modèle diverge systématiquement car ReLU annule les valeurs négatives',
    ],
    correctOptionIndex: 1,
    takeawayFr:
      'À retenir : Une seule couche cachée dotée d’une fonction d’activation non-linéaire suffit déjà à approximer des frontières courbes complexes en combinant plusieurs demi-plans !',
  },
  {
    id: 'C',
    title: 'Expérience C — 2 couches cachées et comparaison',
    subtitle: 'Comparer la hiérarchie de représentation et le nombre de paramètres',
    dataset: 'moons',
    noise: 0.15,
    samples: 400,
    trainSplit: 0.8,
    config: {
      hiddenLayers: 2,
      neuronsPerLayer: 8,
      activation: 'relu',
      learningRate: 0.03,
      epochs: 80,
      batchSize: 32,
    },
    explanationFr:
      'Passons à 2 couches cachées (8 neurones chacune). La deuxième couche recombine les caractéristiques déjà abstraites par la première couche, permettant une frontière plus fluide en moins d’époques — mais avec davantage de paramètres à estimer.',
    questionFr:
      'Un réseau plus profond garantit-il toujours une meilleure Validation Accuracy ?',
    optionsFr: [
      'Oui, plus il y a de couches et de neurones, meilleure est la généralisation sur tout jeu de données',
      'Non : il offre plus de capacité expressive, mais sur un problème simple il peut sur-apprendre (Overfitting) ou être superflu',
      'Non, car TensorFlow ne peut pas rétropropager le gradient sur 2 couches',
    ],
    correctOptionIndex: 1,
    takeawayFr:
      'À retenir : La profondeur apporte de l’expressivité hiérarchique, mais « plus gros » ne veut jamais dire automatiquement « meilleur ». Comparez toujours la Validation Loss et le nombre de paramètres !',
  },
  {
    id: 'D',
    title: 'Expérience D — Fort bruit et Généralisation (Overfitting)',
    subtitle: 'Observer le compromis biais-variance face à des données très bruitées',
    dataset: 'moons',
    noise: 0.35,
    samples: 220,
    trainSplit: 0.7,
    config: {
      hiddenLayers: 3,
      neuronsPerLayer: 14,
      activation: 'relu',
      learningRate: 0.03,
      epochs: 120,
      batchSize: 16,
    },
    explanationFr:
      'Augmentons fortement le bruit (0.35) avec un échantillon plus restreint et un réseau très capacitaire (3 couches de 14 neurones). Les deux classes se chevauchent désormais largement.',
    questionFr:
      'Quel phénomène allez-vous observer entre les courbes de Training et de Validation ?',
    optionsFr: [
      'La Validation Accuracy dépassera la Training Accuracy de 15 points',
      'Un écart croissant : la Training Accuracy continue de monter en mémorisant le bruit tandis que la Validation Loss stagne ou remonte',
      'Les poids deviennent tous exactement égaux à zéro',
    ],
    correctOptionIndex: 1,
    takeawayFr:
      'À retenir : Quand la capacité du réseau dépasse la structure réelle du signal, il mémorise les anomalies du jeu d’entraînement (Overfitting). Surveillez toujours le croisement entre Training Loss et Validation Loss !',
  },
];

export interface ConceptCardData {
  id: string;
  term: string;
  titleFr: string;
  beginnerExplanationFr: string;
  sklearnBridgeFr: string;
  interactiveTipFr: string;
  predictQuestionFr: string;
  predictAnswerFr: string;
  takeawayFr: string;
}

export const EDUCATIONAL_CONCEPTS: ConceptCardData[] = [
  {
    id: 'weights-bias',
    term: 'Weights & Bias (w, b)',
    titleFr: 'Poids synaptiques et Biais',
    beginnerExplanationFr:
      'Les poids (w) mesurent l’importance et le signe de chaque variable d’entrée, tandis que le biais (b) décale le seuil de déclenchement du neurone.',
    sklearnBridgeFr:
      'Exactement comme model.coef_ et model.intercept_ dans LogisticRegression ou LinearRegression.',
    interactiveTipFr:
      'Dans l’onglet 1, mettez w₁ = 0 : vous verrez que l’entrée x₁ n’a plus aucun impact sur la sortie.',
    predictQuestionFr: 'Que se passe-t-il si on inverse le signe de w₁ et w₂ ?',
    predictAnswerFr: 'L’orientation de la frontière de décision s’inverse (les zones positives et négatives s’échangent).',
    takeawayFr: 'Les poids orientent la frontière, le biais la translate dans l’espace.',
  },
  {
    id: 'activation',
    term: 'Activation Functions',
    titleFr: 'Fonctions d’activation (ReLU, Tanh, Sigmoid)',
    beginnerExplanationFr:
      'Sans fonction d’activation non-linéaire, empiler 10 couches revient mathématiquement à une seule multiplication matricielle linéaire.',
    sklearnBridgeFr:
      'Dans MLPClassifier(activation="relu"), c’est ce qui distingue un réseau profond d’une simple régression linéaire.',
    interactiveTipFr:
      'Comparez ReLU (frontières anguleuses par morceaux) et Tanh (frontières courbes et lisses) sur Two Moons.',
    predictQuestionFr: 'Pourquoi ReLU est-elle préférée à Sigmoid dans les couches cachées ?',
    predictAnswerFr: 'ReLU ne sature pas pour les valeurs positives (gradient = 1), évitant la disparition du gradient (vanishing gradient).',
    takeawayFr: 'La non-linéarité est la clé qui permet de plier l’espace de représentation.',
  },
  {
    id: 'hidden-layers',
    term: 'Hidden Layers',
    titleFr: 'Couches cachées & Feature Engineering automatique',
    beginnerExplanationFr:
      'Les couches cachées transforment automatiquement les coordonnées brutes (x₁, x₂) en nouvelles caractéristiques où les classes deviennent linéairement séparables.',
    sklearnBridgeFr:
      'Au lieu de créer manuellement des PolynomialFeatures ou un noyau RBF, les couches cachées apprennent la transformation par descente de gradient.',
    interactiveTipFr:
      'Passez de 0 à 1 couche cachée sur Concentric Circles (au moins 4 neurones) pour fermer une frontière autour du cercle central.',
    predictQuestionFr: 'Combien de neurones ReLU minimum faut-il dans 1 couche cachée pour entourer complètement un cercle en 2D ?',
    predictAnswerFr: 'Au moins 3 neurones (qui forment un triangle fermé autour du centre) ! Avec 2 neurones, la région reste ouverte.',
    takeawayFr: 'Chaque couche cachée recombine les détecteurs de la couche précédente.',
  },
  {
    id: 'forward-prop',
    term: 'Forward Propagation',
    titleFr: 'Propagation avant (Inférence)',
    beginnerExplanationFr:
      'Les données traversent le réseau de gauche à droite : chaque couche calcule z = W·x + b puis applique son activation jusqu’à la probabilité finale P(y=1).',
    sklearnBridgeFr:
      'Correspond à l’appel model.predict_proba(X) dans scikit-learn.',
    interactiveTipFr:
      'Survolez la carte de chaleur 2D : chaque pixel correspond à une Forward Propagation d’un point (x₁, x₂).',
    predictQuestionFr: 'La Forward Propagation modifie-t-elle les poids du réseau ?',
    predictAnswerFr: 'Non, jamais ! Elle évalue simplement la fonction mathématique actuelle avec les poids figés.',
    takeawayFr: 'La passe avant prédit ; seule la passe arrière (Backpropagation) apprend.',
  },
  {
    id: 'loss-function',
    term: 'Binary Cross-Entropy Loss',
    titleFr: 'Fonction de coût (Log-Loss)',
    beginnerExplanationFr:
      'La Binary Cross-Entropy mesure l’erreur entre la probabilité prédite p̂ et la vraie étiquette y ∈ {0, 1}. Elle pénalise très fortement les prédictions confiantes mais fausses.',
    sklearnBridgeFr:
      'Identique à sklearn.metrics.log_loss.',
    interactiveTipFr:
      'Observez comment la Loss peut continuer à baisser même quand l’Accuracy plafonne : le modèle devient plus confiant loin de la frontière.',
    predictQuestionFr: 'Que vaut approximativement la Loss initiale quand le réseau prédit p ≈ 0.5 partout ?',
    predictAnswerFr: '-ln(0.5) ≈ 0.693. C’est le point de départ typique à l’époque 0.',
    takeawayFr: 'On optimise la Loss (continue et dérivable), pas directement l’Accuracy (en escalier).',
  },
  {
    id: 'backprop-gd',
    term: 'Backpropagation & Learning Rate',
    titleFr: 'Rétropropagation du gradient & Taux d’apprentissage',
    beginnerExplanationFr:
      'Par la règle de dérivation en chaîne (chain rule), le réseau calcule la responsabilité de chaque poids dans l’erreur finale, puis ajuste les poids dans la direction opposée au gradient.',
    sklearnBridgeFr:
      'Contrôlé par learning_rate_init et solver="adam" dans MLPClassifier.',
    interactiveTipFr:
      'Testez un Learning Rate de 0.3 vs 0.001 : trop élevé, la frontière saute de façon chaotique ; trop bas, l’apprentissage est très lent.',
    predictQuestionFr: 'Que se passe-t-il si le Learning Rate est beaucoup trop grand ?',
    predictAnswerFr: 'Les mises à jour dépassent le minimum de la vallée de Loss : la Loss oscille fortement ou diverge.',
    takeawayFr: 'Backpropagation calcule la pente ; Gradient Descent fait le pas pondéré par le Learning Rate.',
  },
  {
    id: 'under-overfitting',
    term: 'Underfitting vs Overfitting',
    titleFr: 'Sous-apprentissage vs Sur-apprentissage',
    beginnerExplanationFr:
      'Underfitting : modèle trop rigide (ex. droite sur Two Moons). Overfitting : modèle trop flexible qui mémorise le bruit du jeu d’entraînement au détriment de la validation.',
    sklearnBridgeFr:
      'Même diagnostic qu’avec learning_curve ou validation_curve dans scikit-learn.',
    interactiveTipFr:
      'Lancez l’Expérience D ou la Mission 3 du Challenge pour diagnostiquer un écart Train/Validation en conditions réelles.',
    predictQuestionFr: 'Quel indicateur trahit l’Overfitting sur les Learning Curves ?',
    predictAnswerFr: 'La Training Loss continue de diminuer alors que la Validation Loss stagne ou remonte (écart grandissant).',
    takeawayFr: 'La vraie performance d’un réseau se juge toujours sur les données de validation non vues à l’entraînement.',
  },
];
