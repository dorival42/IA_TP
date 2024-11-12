# Chargement des bibliothèques nécessaires
from ndslib import load_data
import numpy as np
import matplotlib.pyplot as plt

# Charger les données

# brain = load_data("bold_volume")
# brain = np.load('brain.npy')
brain = np.load(r'C:\Users\DELL\Desktop\Cours SMSD\IA\images (2)\images\brain.npy')

# Prendre une coupe horizontale au milieu du cerveau
slice10 = brain[:, :, 10]

# Visualiser l'image
fig, ax = plt.subplots()
im = ax.imshow(slice10, cmap="bone")
ax.set_title("Coupe du cerveau")
plt.axis('off')
plt.show()

# Calculer l'histogramme
fig, ax = plt.subplots()
ax.hist(slice10.flat, bins=50)
ax.set_title("Histogramme des valeurs d'intensité")
plt.xlabel("Intensité")
plt.ylabel("Fréquence")
plt.show()

# Implémentation de la méthode d'Otsu pour trouver le seuil
min_intraclass_variance = np.inf
threshold = 0

for candidate in np.unique(slice10):
    foreground = slice10[slice10 >= candidate]
    background = slice10[slice10 < candidate]
    if len(foreground) > 0 and len(background) > 0:
        foreground_variance = np.var(foreground) * len(foreground)
        background_variance = np.var(background) * len(background)
        intraclass_variance = foreground_variance + background_variance
        if intraclass_variance < min_intraclass_variance:
            min_intraclass_variance = intraclass_variance
            threshold = candidate

# Afficher la moyenne et le seuil d'Otsu sur l'histogramme
mean = np.mean(slice10)
fig, ax = plt.subplots()
ax.hist(slice10.flat, bins=50)
ax.axvline(mean, linestyle='dashed', color='red', label='Moyenne')
ax.axvline(threshold, linestyle='dotted', color='blue', label='Seuil Otsu')
ax.set_title("Histogramme avec seuils")
plt.xlabel("Intensité")
plt.ylabel("Fréquence")
plt.legend()
plt.show()

# Segmentation de l'image
segmentation = np.zeros_like(slice10)
segmentation[slice10 >= threshold] = 1

# Visualiser la segmentation
fig, ax = plt.subplots()
ax.imshow(slice10, cmap="bone")
ax.imshow(segmentation, alpha=0.5, cmap="gray")
ax.set_title("Segmentation avec seuil Otsu")
plt.axis('off')
plt.show()
