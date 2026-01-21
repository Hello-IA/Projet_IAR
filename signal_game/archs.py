# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class ColorSenderRF(nn.Module):
    def __init__(self, game_size=2, embedding_size=1000, vocab_size=1024, temp=1.0):
        super().__init__()
        self.game_size = game_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.temp = temp
        # réseau simple MLP sur couleurs
        self.lin1 = nn.Linear(3, embedding_size)
        self.lin2 = nn.Linear(embedding_size, embedding_size)
        self.lin3 = nn.Linear(embedding_size, embedding_size)
        self.lin4 = nn.Linear(embedding_size, vocab_size)
        self.leaky_reLU = nn.LeakyReLU(0.01)

    def forward(self, x, _aux_input=None):
        """
        x: tensor shape (game_size, batch_size, 3)
        """
        batch_size = x.size(1)

        # On ne prend que la couleur cible en position 0
        h = self.lin1(x[0])  # (batch_size, embedding_size)
        hr = self.leaky_reLU(h)
        h1 = self.lin2(hr)
        hr1 = self.leaky_reLU(h1)
        h2 = self.lin3(hr1)
        hr2 = self.leaky_reLU(h2)
        logits = torch.clamp(self.lin4(hr2) / self.temp, min=-50, max=50)  # (batch_size, vocab_size)
        log_probs = F.log_softmax(logits, dim=1)
        return log_probs

class ColorSenderGS(nn.Module):
    def __init__(self, game_size=2, embedding_size=1000, vocab_size=1024, temp=1.0):
        super().__init__()
        self.game_size = game_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.temp = temp
        # réseau simple MLP sur couleurs
        self.lin1 = nn.Linear(3, embedding_size)
        self.lin2 = nn.Linear(embedding_size, embedding_size)
        self.lin3 = nn.Linear(embedding_size, embedding_size)
        self.lin4 = nn.Linear(embedding_size, vocab_size)
        self.leaky_reLU = nn.LeakyReLU(0.01)

    def forward(self, x, _aux_input=None):
        """
        x: tensor shape (game_size, batch_size, 3)
        """
        batch_size = x.size(1)

        # On ne prend que la couleur cible en position 0
        h = self.lin1(x[0])  # (batch_size, embedding_size)
        hr = self.leaky_reLU(h)
        h1 = self.lin2(hr)
        hr1 = self.leaky_reLU(h1)
        h2 = self.lin3(hr1)
        hr2 = self.leaky_reLU(h2)
        logits = torch.clamp(self.lin4(hr2) / self.temp, min=-50, max=50)  # (batch_size, vocab_size)
        return logits

class ColorReceiverMLP(nn.Module):
    def __init__(self, game_size: int, embedding_size: int = 5,vocab_size: int = 1024,reinforce: bool = True,):
        super().__init__()

        self.game_size = game_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.reinforce = reinforce

        # --- Embedding des couleurs (linéaire, capacité contrôlée) ---
        self.lin_colors = nn.Linear(3, embedding_size, bias=False)

        # --- Embedding du signal ---
        if reinforce:
            # signal = indice discret
            self.lin_signal = nn.Embedding(vocab_size, embedding_size)
        else:
            # signal = vecteur continu (GS)
            self.lin_signal = nn.Linear(vocab_size, embedding_size, bias=False)

    def forward(self, signal, x, _aux_input=None):
        """
        signal :
            - REINFORCE : LongTensor (batch_size,)
            - GS        : FloatTensor (batch_size, vocab_size)

        x :
            tensor shape (game_size, batch_size, 3)
        """

        batch_size = x.size(1)

        # --- Embedding des couleurs ---
        # (batch_size, game_size, embedding_size)
        color_embs = []
        for i in range(self.game_size):
            h_i = self.lin_colors(x[i])
            color_embs.append(h_i.unsqueeze(1))
        color_embs = torch.cat(color_embs, dim=1)

        
        h_s = self.lin_signal(signal)  # (batch_size, embedding_size)

        # (batch_size, embedding_size, 1)
        h_s = h_s.unsqueeze(2)

        # --- Score ---
        # (batch_size, game_size, 1)
        out = torch.bmm(color_embs, h_s)

        # (batch_size, game_size)
        out = out.squeeze(-1)

        # --- Probabilités ---
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
