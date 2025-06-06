import numpy as np
from sklearn.linear_model import RidgeClassifier
from   sklearn.exceptions import NotFittedError
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop
import numpy as np
from sklearn.linear_model import RidgeClassifier
from tensorflow.keras.optimizers import SGD

class NTKRR:
    """
    Implémentation du Neural Tangent Kernel Ridge Regression (NTK-RR).

    Cette classe permet d'entraîner un modèle de Neural Tangent Kernel Ridge Regression à une couche cachée avec activation ReLU.
    
    Une meilleure implémentation est possible en utilisant le module neural-tangents 
    de la bibliothèque JAX. Cependant, des problèmes avec CUDA et cuDNN ont empêché cette approche.
    """

    def __init__(self, hidden_dim, output_dim, image_size, reg_lambda=1e-3):
        """
        Initialise les paramètres du modèle NTKRR.

        Args:
            hidden_dim (int): Nombre de neurones dans la couche cachée.
            output_dim (int): Nombre de classes de sortie.
            image_size (tuple): Dimensions des images en entrée (hauteur, largeur).
            reg_lambda (float, optionnel): Facteur de régularisation pour la régression de crête. Défaut à `1e-3`.
        """
        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.reg_lambda = reg_lambda

        # Initialisation des poids
        self.W1 = np.random.randn(image_size[0], image_size[1], hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)

    def forward(self, X):
        """
        Effectue un passage avant (forward pass) dans le réseau.

        Args:
            X (numpy.ndarray): Données d'entrée de taille `(n_samples, hauteur, largeur)`.

        Returns:
            numpy.ndarray: Prédictions du modèle de taille `(n_samples, output_dim)`.
        """
        Z1 = np.tensordot(X, self.W1, axes=([1, 2], [0, 1])) + self.b1
        A1 = np.maximum(0, Z1)  # Activation ReLU
        Z2 = np.dot(A1, self.W2) + self.b2
        return Z2

    def compute_ntk(self, X1, X2):
        """
        Calcule le noyau tangent du réseau pour deux ensembles d'entrées.

        Args:
            X1 (numpy.ndarray): Première matrice de données `(n_samples1, hauteur, largeur)`.
            X2 (numpy.ndarray): Deuxième matrice de données `(n_samples2, hauteur, largeur)`.

        Returns:
            numpy.ndarray: Matrice du noyau tangent de taille `(n_samples1, n_samples2)`.
        """
        # Activation cachée avec ReLU
        A1_X1 = np.maximum(0, np.tensordot(X1, self.W1, axes=([1, 2], [0, 1])))
        A1_X2 = np.maximum(0, np.tensordot(X2, self.W1, axes=([1, 2], [0, 1])))

        # Calcul du noyau tangent
        NTK = np.dot(A1_X1, A1_X2.T)  # Contribution des activations
        NTK += np.tensordot(X1, X2, axes=([1, 2], [1, 2]))  # Contribution brute
        return NTK
    
    def fit(self, X_train, y_train):
        """
        Entraîne le modèle NTKRR en ajustant les coefficients de la régression de crête.

        Args:
            X_train (numpy.ndarray): Données d'entraînement `(n_samples, hauteur, largeur)`.
            y_train (numpy.ndarray): Labels associés aux échantillons `(n_samples, output_dim)`.
        """
        K = self.compute_ntk(X_train, X_train)
        n = K.shape[0]
        # Solution analytique de la regression KRR
        self.alpha = np.linalg.inv(K + self.reg_lambda * np.eye(n)).dot(y_train)

    def predict(self, X_test, X_train):
        """
        Effectue une prédiction sur de nouvelles données en utilisant le modèle entraîné.

        Args:
            X_test (numpy.ndarray): Données de test `(n_samples_test, hauteur, largeur)`.
            X_train (numpy.ndarray): Données d'entraînement utilisées pour `fit()`.

        Returns:
            numpy.ndarray: Prédictions des classes (`-1` ou `1` pour classification binaire).

        Raises:
            AttributeError: Si le modèle n'a pas été entraîné (`fit` non appelé).
        """
        if not hasattr(self, "alpha"):
            raise AttributeError("Le modèle doit être entraîné avec `fit` avant d'effectuer des prédictions.")
        
        K_test = self.compute_ntk(X_test, X_train)
        return np.sign(K_test.dot(self.alpha))  # Classification binaire


class RFKRR:
    """
    Kernel Ridge Regression utilisant des caractéristiques de Fourier aléatoires (Random Fourier Features).

    Ce code est inspiré du projet disponible sur GitHub : https://github.com/gwgundersen/random-fourier-features/tree/master

    Cette implémentation permet d'approximer une régression à noyau de manière efficace
    en projetant les données dans un espace de caractéristiques de Fourier aléatoire, 
    puis en appliquant un RidgeClassifier.
    """

    def __init__(self, rff_dim=4096, alpha=1.0, sigma=1.0):
        """
        Initialise le modèle RFKRR.

        Args:
            rff_dim (int, optionnel) : Dimension de la couche cachée du RFKRR. 
                                       Défaut à 4096.
            alpha (float, optionnel) : Intensité de la régularisation du `RidgeClassifier`. 
                                       Défaut à 1.0.
            sigma (float, optionnel) : Paramètre de lissage pour les caractéristiques de Fourier. 
                                       Défaut à 1.0.
        """
        self.fitted = False
        self.rff_dim = rff_dim
        self.sigma = sigma
        self.lm = RidgeClassifier(alpha=alpha)
        self.b_ = None
        self.W_ = None

    def fit(self, X, y):
        """
        Ajuste le modèle sur les données d'entraînement.

        Args:
            X (numpy.ndarray) : Données d'entraînement de taille (n_samples, n_features).
            y (numpy.ndarray) : Cibles associées aux échantillons d'entraînement.

        Returns:
            RFKRR : L'instance du modèle entraîné.
        """
        Z, W, b = self._get_rffs(X, return_vars=True)
        self.lm.fit(Z.T, y)
        self.b_ = b
        self.W_ = W
        self.fitted = True
        return self

    def predict(self, X):
        """
        Prédit les étiquettes des nouvelles données à l'aide du modèle ajusté.

        Args:
            X (numpy.ndarray) : Données de test de taille (n_samples, n_features).

        Returns:
            numpy.ndarray : Prédictions du modèle.

        Raises:
            NotFittedError : Si le modèle n'a pas encore été entraîné (`fit` non appelé).
        """
        if not self.fitted:
            msg = "Appelez la méthode 'fit' avec des données appropriées avant de prédire."
            raise NotFittedError(msg)
        Z = self._get_rffs(X, return_vars=False)
        return self.lm.predict(Z.T)

    def _get_rffs(self, X, return_vars):
        """
        Génère les caractéristiques de Fourier aléatoires pour les données données.

        Args:
            X (numpy.ndarray) : Données d'entrée de taille (n_samples, n_features).
            return_vars (bool) : Si True, retourne aussi les variables aléatoires W et b.

        Returns:
            numpy.ndarray : Matrice de caractéristiques de Fourier de taille (rff_dim, n_samples).
            (optionnel) numpy.ndarray, numpy.ndarray : Matrices W et b utilisées pour la transformation.
        """
        N, D = X.shape
        if self.W_ is not None:
            W, b = self.W_, self.b_
        else:
            W = np.random.normal(loc=0, scale=1, size=(self.rff_dim, D))
            b = np.random.uniform(0, 2 * np.pi, size=self.rff_dim)

        B = np.repeat(b[:, np.newaxis], N, axis=1)
        norm = 1. / np.sqrt(self.rff_dim)
        Z = norm * np.sqrt(2) * np.cos(self.sigma * W @ X.T + B)

        if return_vars:
            return Z, W, b
        return Z


class NN(Model):
    """
    Implémente un réseau de neurones en TensorFlow avec deux architectures possibles :
      - Réseau dense (MLP) si conv=False : contient 3 couches cachées (512, 256, 128) avec Dropout.
      - Réseau convolutionnel (CNN) si conv=True : contient 2 couches de convolution suivies de max pooling.
    """

    def __init__(self, num_classes=10, learning_rate=0.001, conv=False):
        """
        Initialise l'architecture du réseau de neurones.

        Args:
            num_classes (int, optionnel) : Nombre de classes en sortie. Défaut à 10.
            learning_rate (float, optionnel) : Taux d'apprentissage de l'optimiseur. Défaut à 0.001.
            conv (bool, optionnel) : Si `True`, utilise un CNN. Sinon, utilise un réseau dense (MLP). Défaut à False.
        """
        super(NN, self).__init__()
        self.conv = conv  
        self.num_classes = num_classes

        if conv:
            # Architecture CNN
            self.conv1 = Conv2D(24, kernel_size=5, padding="same", activation="relu")
            self.pool1 = MaxPooling2D(pool_size=2)

            self.conv2 = Conv2D(48, kernel_size=5, padding="same", activation="relu")
            self.pool2 = MaxPooling2D(pool_size=2)

            self.flatten = Flatten()  
            self.fc1 = Dense(256, activation="relu")
        else:
            # Architecture MLP (Dense)
            self.flatten = Flatten()
            self.fc1 = Dense(512, activation="relu")
            self.dropout1 = Dropout(0.2)
            self.fc2 = Dense(512, activation="relu")
            self.dropout2 = Dropout(0.2)
            self.fc3 = Dense(256, activation="relu")
            self.dropout3 = Dropout(0.2)

        self.fc_out = Dense(num_classes, activation="softmax")
    
        # Dans le papier l'optimiseur utilisé est un SGD avec momentum 0.9 (pour le NN) avec un cosine decay pour le learning rate.
        # Néanmoins ces hyperparamètres font parfois diverger notre modèle lorsqu'on l'entraine. 
        # On choisit donc l'optimiseur classique Adam.
        optimizer = Adam(learning_rate=learning_rate)

        self.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def call(self, inputs, training=False):
        """
        Effectue un passage avant (forward pass) du modèle.

        Args:
            inputs (tensor) : Données d'entrée, typiquement des images de taille (batch_size, height, width, channels).
            training (bool, optionnel) : Indique si le modèle est en mode entraînement (True) ou inférence (False). Défaut à False.

        Returns:
            tensor : Sortie du modèle, un tenseur de dimension `(batch_size, num_classes)` contenant les probabilités de classification.
        """
        x = inputs

        if self.conv:
            # Forward pass du CNN
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.flatten(x)  
            x = self.fc1(x)
        else:
            # Forward pass du MLP
            x = self.flatten(x) 
            x = self.fc1(x)
            x = self.dropout1(x, training=training)
            x = self.fc2(x)
            x = self.dropout2(x, training=training)
            x = self.fc3(x)
            x = self.dropout3(x, training=training)

        x = self.fc_out(x)
        return x


