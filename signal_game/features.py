# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data as data
import h5py

class _BatchIterator:
    def __init__(self, loader, n_batches, seed=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.n_batches = n_batches
        self.batch_size = loader.batch_size
        self.game_size = loader.opt.game_size
        self.percentile = loader.opt.percentile
        self.batches_generated = 0
        self.rng = np.random.RandomState(seed)

        # seuil de distance minimale entre target et distracteur
        self.dist_min = self.dataset.thresholds[self.percentile]

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_generated >= self.n_batches:
            raise StopIteration()
        self.batches_generated += 1
        return self.get_batch()

    def get_batch(self):
        X_sender = torch.zeros(self.game_size, self.batch_size, 3)
        X_receiver = torch.zeros_like(X_sender)
        y = torch.zeros(self.batch_size, dtype=torch.long)

        for b in range(self.batch_size):
            # 1️ Choisir la cible
            c_t = self.rng.randint(0, len(self.dataset.colors))

            # 2️ Choisir un distracteur valide
            valid = torch.where(self.dataset.dist_matrix[c_t] >= self.dist_min)[0].numpy()
            valid = valid[valid != c_t]  # sécurité
            c_d = self.rng.choice(valid)

            # 3️ Mettre la cible en position 0
            X_sender[0, b] = self.dataset.colors[c_t]
            X_sender[1, b] = self.dataset.colors[c_d]

            # 4️ Mélanger pour le Receiver
            perm = torch.randperm(self.game_size)
            X_receiver[:, b] = X_sender[perm, b]

            # 5️ Label = position de la cible après mélange
            y[b] = (perm == 0).nonzero(as_tuple=True)[0]

        return X_sender, y, X_receiver

class ImagenetLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        self.opt = kwargs.pop("opt")
        self.seed = kwargs.pop("seed")
        self.batches_per_epoch = kwargs.pop("batches_per_epoch")

        super(ImagenetLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        if self.seed is None:
            seed = np.random.randint(0, 2 ** 31-1)
        else:
            seed = self.seed
        return _BatchIterator(self, n_batches=self.batches_per_epoch, seed=seed)





class WCSFeat(data.Dataset):
    def __init__(self, h5_file, percentiles=(10, 25, 50, 75)):
        # Recuperer les couleur CIELAB dans le ficher .h5 taille de colors (330, 3) avec les dimantion (L, A, B) pour chaque chips
        with h5py.File(h5_file, "r") as f:
            self.colors = torch.tensor(f["features"][:]).float()

        # Calcul de la matrice des distances perceptives (CIELAB) shape = (330, 330)
        self.dist_matrix = torch.cdist(self.colors, self.colors, p=2) 

        # 3. Extraire les distances uniques (k > l) toute les indise au decus de la diagonale pour ne pas avoir les distence deux fois.
        tri_i, tri_j = torch.triu_indices(
            self.dist_matrix.size(0),
            self.dist_matrix.size(1),
            offset=1,
        )
        all_dists = self.dist_matrix[tri_i, tri_j] # ne garde que les distence des indice strictement audecus de la diagonalle.

        # 4. Calcul des seuils dist_min (percentiles) dans un dictioner de seulle caculer avec les distence de tous les chips WCS deux a deux
        self.thresholds = {
            p: torch.quantile(all_dists, p / 100.0).item()
            for p in percentiles
        }


    def __len__(self):
        return self.colors.size(0)

    def __getitem__(self, idx):
        return self.colors[idx], idx
