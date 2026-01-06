# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os

import torch.nn.functional as F
import numpy as np
import egg.core as core
from archs import ColorSenderMLP, ColorReceiverMLP
from features import WCSFeat, ImagenetLoader
import random
from train import *

import torch
import os
def clone_analysis_sender(game, tau_s, device="cpu"):
    """
    On enlève ReinforceWrapper et on récupère la distribution complète
    """
    game_size=2
    vocab_size = 1024
    embedding_size = 1000
    analysis_sender = ColorSenderMLP(
        game_size=game_size,
        embedding_size=embedding_size,  
        vocab_size=vocab_size,
        temp=tau_s,
    ).to(device)

    # Copier les poids du sender entraîné
    analysis_sender.load_state_dict(
        game.sender.agent.state_dict()
    )

    analysis_sender.eval()
    return analysis_sender

def extract_nn_language(
    game,
    tau_s,
    dataset,
    output_path,
    n_samples_per_chip=25,
    device="cpu",
):
    """
    game        : jeu EGG entraîné
    dataset     : WCSFeat (330 couleurs)
    output_path : fichier texte de sortie
    """

    sender = game.sender
    sender.eval()

    game_size = 2
    speaker_id = 1  # un seul speaker NN

    copy_sender = clone_analysis_sender(game, tau_s)

    with open(output_path, "w") as f:
        for chip_id in range(len(dataset)):
            color = dataset.colors[chip_id].to(device)

            # --- construire input sender ---
            x = torch.zeros(game_size, 1, 3, device=device)
            x[0, 0] = color          # target
            x[1, 0] = color          # distracteur dummy (inutile)

            with torch.no_grad():
                log_probs = copy_sender(x)          # (1, vocab)
                probs = torch.exp(log_probs)   # P(w|c)

            # --- distribution catégorielle ---
            dist = torch.distributions.Categorical(probs=probs.squeeze(0))

            samples = dist.sample((n_samples_per_chip,))  # (25,)

            for w in samples.tolist():
                f.write(f"{speaker_id} {chip_id+1} w{w}\n")


def langueg(n_epochs, game_size, mode, gs_tau, tau_s, seed):
    opts = parse_arguments()
    torch.manual_seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)
    data_folder = os.path.join("data", "train\\")
    dataset = WCSFeat(data_folder+"ours_images_single_sm0.h5")
    batch_size = 128
    train_loader = ImagenetLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        opt=opts,
        batches_per_epoch=opts.batches_per_epoch,
        seed=seed,
    )
    validation_loader = ImagenetLoader(
        dataset,
        opt=opts,
        batch_size=batch_size,
        batches_per_epoch=opts.batches_per_epoch,
        seed=seed,
    )
    game = get_game(game_size, mode, gs_tau, tau_s, 0.09511187187723279, 0.029903872692200614)
    optimizer = core.build_optimizer(game.parameters())
    for g in optimizer.param_groups:
        g['lr'] = 0.0009488916108443118
    callback = None
    if mode == "gs":
        callbacks = [core.TemperatureUpdater(agent=game.sender, decay=0.9, minimum=0.1)]
    else:
        callbacks = []

    callbacks.append(core.ConsoleLogger(as_json=True, print_train_loss=True))
    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=callbacks,
    )

    trainer.train(n_epochs=n_epochs)
    extract_nn_language(
	    game=game,
	    tau_s=tau_s,
	    dataset=dataset,
	    output_path=f"nn_language_seed_{seed}.txt",
	    n_samples_per_chip=25,
	    device="cpu")


    core.close()


langueg(50, 2, "rf", 1, 2.3167829883799427, 0)