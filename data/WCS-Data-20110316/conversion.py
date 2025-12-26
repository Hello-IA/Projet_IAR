import numpy as np
import h5py

# Charger la table (on ignore les colonnes non utiles)
# Colonnes attendues : cnum, L*, a*, b*
data = np.genfromtxt(
    "cnum-vhcm-lab-new.txt",
    delimiter="\t",
    skip_header=1,
    usecols=(0, 6, 7, 8)  # cnum, L*, a*, b*
)

# Trier par cnum pour avoir un ordre stable (optionnel mais recommandé)
data = data[data[:, 0].argsort()]

# Extraire uniquement les features LAB
features = data[:, 1:].astype(np.float32)  # shape = (330, 3)

print(features.shape)  # doit afficher (330, 3)
print(features)
# Normalisation simple
features[:, 0] /= 100.0   # L* ∈ [0,100]
features[:, 1:] /= 128.0  # a*, b* ≈ [-128,128]
print(data)
# Sauvegarde au format attendu par EGG
with h5py.File("ours_images_single_sm0.h5", "w") as f:
    f.create_dataset("features", data=features)
