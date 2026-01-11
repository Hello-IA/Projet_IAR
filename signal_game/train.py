# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch.nn.functional as F

import egg.core as core
from archs import ColorSenderMLP, ColorReceiverMLP
from features import WCSFeat, ImagenetLoader


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="", help="data root folder")
    # 2-agents specific parameters
    parser.add_argument(
        "--tau_s", type=float, default=2.3167829883799427, help="Sender Gibbs temperature"
    )
    parser.add_argument(
        "--game_size", type=int, default=2, help="Number of images seen by an agent"
    )
    parser.add_argument("--same", type=int, default=0, help="Use same concepts")
    parser.add_argument("--embedding_size", type=int, default=50, help="embedding size")
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=20,
        help="hidden size (number of filters informed sender)",
    )
    parser.add_argument(
        "--batches_per_epoch",
        type=int,
        default=100,
        help="Batches in a single training/validation epoch",
    )
    parser.add_argument("--inf_rec", type=int, default=0, help="Use informed receiver")
    parser.add_argument(
        "--mode",
        type=str,
        default="gs",
        help="Training mode: Gumbel-Softmax (gs) or Reinforce (rf). Default: rf.",
    )
    parser.add_argument("--gs_tau", type=float, default=10.0, help="GS temperature")
    parser.add_argument(
        "--percentile",
        type=int,
        default=60,
        help="Percentile for dist_min (discriminative need)",
    )

    opt = core.init(parser)
    assert opt.game_size >= 1

    return opt


def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
    """
    Accuracy loss - non-differetiable hence cannot be used with GS
    """
    #print("receiver_output", receiver_output, "labels", labels)
    acc = (receiver_output  == labels).float()
    return -acc, {"acc": acc} 


def loss_nll(
    _sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
):
    """
    NLL loss - differentiable and can be used with both GS and Reinforce
    """
    nll = F.nll_loss(receiver_output, labels, reduction="none")
    acc = (labels == receiver_output.argmax(dim=1)).float()
    return nll, {"acc": acc}


def get_game(game_size, mode, gs_tau, tau_s, sender_entropy, receiver_entropy):
    feat_size = 3
    vocab_size = 1024
    embedding_size = 1000
    sender = ColorSenderMLP(
        game_size=game_size,
        embedding_size=embedding_size,  # par ex. 50
        vocab_size=vocab_size,
        temp=tau_s,
    )
    receiver = ColorReceiverMLP(
        game_size=game_size,
        embedding_size=5,  # doit matcher sender
        vocab_size=vocab_size,
        reinforce = True if mode == "rf" else False
    )
    if mode == "rf":
        sender = core.ReinforceWrapper(sender)
        receiver = core.ReinforceWrapper(receiver)
        game = core.SymbolGameReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=sender_entropy,
            receiver_entropy_coeff=receiver_entropy,
        )
    elif mode == "gs":
        sender = core.GumbelSoftmaxWrapper(sender, temperature=gs_tau)
        game = core.SymbolGameGS(sender, receiver, loss_nll)
    else:
        raise RuntimeError(f"Unknown training mode: {mode}")

    return game


if __name__ == "__main__":
    opts = parse_arguments()

    data_folder = os.path.join(opts.root, "train\\")
    dataset = WCSFeat(data_folder+"ours_images_single_sm0.h5")
    batch_size = 128
    train_loader = ImagenetLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        opt=opts,
        batches_per_epoch=opts.batches_per_epoch,
        seed=None,
    )
    validation_loader = ImagenetLoader(
        dataset,
        opt=opts,
        batch_size=batch_size,
        batches_per_epoch=opts.batches_per_epoch,
        seed=7,
    )
    game = get_game(opts.game_size, opts.mode, opts.gs_tau, opts.tau_s, 0.09511187187723279, 0.029903872692200614)
    optimizer = core.build_optimizer(game.parameters())
    for g in optimizer.param_groups:
        g['lr'] = 0.0009488916108443118
    callback = None
    if opts.mode == "gs":
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

    trainer.train(n_epochs=opts.n_epochs)

    core.close()
