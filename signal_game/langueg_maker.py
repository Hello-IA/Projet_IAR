# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
import egg.core as core
from archs import ColorSenderMLP, ColorReceiverMLP
from features import WCSFeat, ImagenetLoader
import random
from train import *



import torch
import os
def clone_analysis_sender(game, tau_s, device="cuda"):
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

def extract_nn_language_full_probs(
    game,
    tau_s,
    dataset,
    output_path,
    device="cuda",
):
    """
    Récupère la matrice complète 330x1024 P(w|c) et la sauvegarde dans un fichier .npz
    """
    copy_sender = clone_analysis_sender(game, tau_s, device=device)

    U = len(dataset)       # 330 chips
    V = 1024               # vocabulaire

    P_w_c_matrix = np.zeros((U, V), dtype=np.float32)

    copy_sender.eval()
    with torch.no_grad():
        for chip_id in range(U):
            color = dataset.colors[chip_id].to(device)
            x = torch.zeros(2, 1, 3, device=device)
            x[0, 0] = color       # target
            x[1, 0] = color       # distracteur dummy
            log_probs = copy_sender(x)  # (1, vocab)
            probs = torch.exp(log_probs).squeeze(0).cpu().numpy()
            P_w_c_matrix[chip_id] = probs

    # Sauvegarde compressée
    np.savez_compressed(output_path, P_w_c=P_w_c_matrix)
    print(f"Saved full P(w|c) matrix to {output_path}")


def langueg(n_epochs, game_size, mode, gs_tau, tau_s, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    opts = parse_arguments()
    torch.manual_seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)
    data_folder = os.path.join("data", "train\\")
    dataset = WCSFeat(data_folder+"ours_images_single_sm0.h5")
    batch_size = 128

    train_loader = ImagenetLoader(dataset, batch_size=batch_size, shuffle=True, opt=opts,
                                  batches_per_epoch=opts.batches_per_epoch, seed=seed)
    validation_loader = ImagenetLoader(dataset, opt=opts, batch_size=batch_size,
                                       batches_per_epoch=opts.batches_per_epoch, seed=seed)

    game = get_game(game_size, mode, gs_tau, tau_s, 0.09511187187723279, 0.029903872692200614)
    game.sender.agent.to(device)
    game.receiver.agent.to(device)

    optimizer = core.build_optimizer(game.parameters())
    for g in optimizer.param_groups:
        g['lr'] = 0.0009488916108443118

    callbacks = [core.ConsoleLogger(as_json=True, print_train_loss=True)]
    if mode == "gs":
        callbacks.append(core.TemperatureUpdater(agent=game.sender, decay=0.9, minimum=0.1))

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=validation_loader,
        callbacks=callbacks,
    )

    trainer.train(n_epochs=n_epochs)

    # --- Sauvegarde matrice complète 330x1024 ---
    extract_nn_language_full_probs(
        game=game,
        tau_s=tau_s,
        dataset=dataset,
        output_path=f"fig5/temp10/nn_language_seed_{seed}.npz",
        device=device
    )

    core.close()


for seed in range(0, 60):
    langueg(200, 2, "rf", 1, 2.3167829883799427, seed)
