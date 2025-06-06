import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.fftpack import dct, idct 
import tensorflow as tf


class SpikedDataset:
    """
    Desc:
        Classe qui permet de générer un dataset suivant le modèle de covariable spiked décrit dans le papier suivant :
        https://arxiv.org/abs/2006.13409
        Du bruit gaussien est ajouté aux hautes fréquences des images afin d'étudier l'impact d'un tel bruit sur les modèles de classification.
    """
    def __init__(self, dataset='MNIST', subset_size=(10000, 2000)):
        """
        Args:
            dataset (str, optional): 'MNIST' dataset par défaut
            subset_size (tuple, optional): Choisi la taille du dataset train/test utilisé. (10000, 2000) par défaut
        """
      
        train_size, test_size = subset_size

        if dataset == 'MNIST':
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        elif dataset == 'FMNIST':
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        else:
            raise ValueError("Dataset non reconnu. Utilisez 'MNIST' ou 'FMNIST'.")

        train_indices = np.random.choice(len(X_train), train_size, replace=False)
        test_indices = np.random.choice(len(X_test), test_size, replace=False)

        self.X_train, self.y_train = X_train[train_indices], y_train[train_indices]
        self.X_test, self.y_test = X_test[test_indices], y_test[test_indices]

        
    def generate_circular_high_frequency_filter(self, img_sz):
        """
        Desc:
            Génère un filtre circulaire pour sélectionner les hautes fréquences dans une matrice k x k.


        Args:
            img_sz: Taille de l'image utilisée 

        Returns:
            F: Filtre circulaire pour les hautes fréquences de la DCT.
        """
        (k,v) = img_sz
        F = np.zeros((k,v))
        center = k  
        for i in range(k):
            for j in range(k):
                if np.sqrt((k - i) ** 2 + (k - j) ** 2) <= k :
                    F[i, j] = 1  
        F[k-1, 0] = 1
        F[0, k-1] = 1

        return F


    def add_high_frequency_noise_with_circle(self, x, filter, tau=0.1):
        
        """
        Desc:
            Ajoute du bruit haute fréquence avec un filtre circulaire à une image en utilisant la DCT.
        
        Args:
            x: Image d'entrée (k x k).
            tau: Contrôle l'amplitude du bruit.
            noise_level: Niveau de bruit.
 
        Returns:
            x_noisy_normalized: Image normalisée et bruité selon le modèle des covariables spiked.
        """
        k = x.shape[0]

        # Étape (a) : Convertir l'image en domaine fréquentiel avec DCT-II
        x_tilde = dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')
        Z = np.random.normal(0, 1, (k, k))
        
        Z_tilde = Z * filter
        # Z_tilde = np.flipud(Z_tilde)  # Inversion verticale du filtre

        x_noisy_tilde = x_tilde + tau * (x_tilde / (np.abs(x_tilde) + 1e-10)) * Z_tilde
        
        x_noisy = idct(idct(x_noisy_tilde, axis=0, norm='ortho'), axis=1, norm='ortho')
        
        d = k * k
        x_noisy_normalized = x_noisy / np.linalg.norm(x_noisy) * np.sqrt(d)
        
        return x_noisy_normalized

    def filter_classes(self, X, y, classes_to_keep):
        """
        Desc:
            Filtre le dataset pour ne conserver que les échantillons des classes spécifiées.
            Les labels sont également remappés pour être cohérents (0, 1, ..., n_classes-1).

        Args:
            X: images
            y: label
            classes_to_keep: choix aribitraire des classes à garder
        """
        mask = np.isin(y, classes_to_keep)
        X_filtered = X[mask]
        y_filtered = y[mask]
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(classes_to_keep)}
        y_remapped = np.vectorize(label_mapping.get)(y_filtered)
        return X_filtered, y_remapped


    def generate_spiked_dataset(self, classes_to_keep, tau):
        """
        Desc:
            Prétraite les données en filtrant les classes spécifiées, en ajoutant du bruit,
            en normalisant et en redimensionnant les images.
   
        Args:
            classes_to_keep: choix aribitraire des classes à garder
            tau (_type_): intensité du bruit rajouté

        Returns:
            X_train, X_test, y_train, y_test : Dataset complet
        """
       
        X_train, y_train = self.filter_classes(self.X_train, self.y_train, classes_to_keep)
        X_test, y_test = self.filter_classes(self.X_test, self.y_test, classes_to_keep)

        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        image_size = self.X_train[0].shape
        F = self.generate_circular_high_frequency_filter(image_size)
        
        if tau == 0:
            return X_train, X_test, y_train, y_test

        print("train set noising")
        for k in tqdm(range(len(self.X_train)), desc='Generating dataset'):
            X_train[k] = self.add_high_frequency_noise_with_circle(X_train[k],F, tau=tau)
        print("test set noising")
        for k in tqdm(range(len(self.X_test))):
            X_test[k] = self.add_high_frequency_noise_with_circle(X_test[k],F, tau=tau)
        
        return X_train, X_test, y_train, y_test
    
    def plot_eigenvalues(self, tau_values=[0, 1, 2, 3]):
        """
        Desc:

            Affiche les valeurs propres de la matrice de covariance empirique des images 
            pour différents niveaux de bruit tau.
        Args:
            tau_values (list, optionnel): Liste des niveaux de bruit (tau) à considérer. 
                                          Par défaut [0, 1, 2, 3].
        """
        classes_to_keep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        plt.figure(figsize=(8, 6))
        
        for tau in tau_values:
  
            X_train_processed, _, _, _ = self.generate_spiked_dataset(classes_to_keep=classes_to_keep, tau=tau)
            X_flatten = X_train_processed.reshape(X_train_processed.shape[0], -1) 
            covariance_matrix = np.cov(X_flatten, rowvar=False)
            eigenvalues = np.linalg.eigvalsh(covariance_matrix)
            plt.plot(eigenvalues, label=f"$\\tau$ = {tau}")
     
        plt.xlabel("Index des valeurs propres", fontsize=12)
        plt.ylabel("Magnitude des valeurs propres", fontsize=12)
        plt.title("Valeurs propres de la covariance empirique des images", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_frequency_repartition(self):
        """
        Desc:
            Calcule la répartition fréquentielle moyenne des images du dataset en appliquant la DCT,
            et affiche le filtre basse fréquence utilisé.
        """
        train_images = self.X_train.astype('float32')
        d = train_images.shape[1] * train_images.shape[2]
        train_images = (train_images - np.mean(train_images)) / np.linalg.norm(train_images) * np.sqrt(d)

        num_images = train_images.shape[0]
        img_size = (train_images.shape[1], train_images.shape[2])

        freq_sum = np.zeros(img_size, dtype=np.float64) 
        filter_low_freq = self.generate_circular_high_frequency_filter(img_sz=img_size)

        for img in tqdm(train_images, desc='Frequency decomposition'):
            img_dct = dct(dct(img, axis=1, norm='ortho'), axis=0, norm='ortho')
            freq_sum += np.abs(img_dct)

        freq_avg = freq_sum / num_images
        freq_avg /= np.max(freq_avg)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(filter_low_freq, cmap="viridis", vmin=0, vmax=1)
        plt.title("Noise Filter F")
        plt.xlabel("X")
        plt.ylabel("Y")

        plt.subplot(1, 2, 2)
        im = plt.imshow(freq_avg, cmap="jet", vmin=0, vmax=1)
        plt.title("Average Absolute Frequency Component")
        plt.xlabel("Frequency X")
        plt.ylabel("Frequency Y")

        plt.colorbar(im, label="Amplitude")

        plt.show()


    

