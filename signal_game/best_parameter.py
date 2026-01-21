# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import optuna
import os
import egg.core as core
from features import(WCSFeat,
    ImagenetLoader)
from train import (
    parse_arguments,
    get_game)

# =========================
# OBJECTIVE OPTUNA
# =========================


class ValidationAccCallback(core.Callback):
    def __init__(self):
        self.last_val_acc = None

    def on_validation_end(self, validation_loss, validation_interaction, epoch):
        # validation_interaction est un Interaction
        aux = validation_interaction.aux

        if aux is not None and "acc" in aux:
            self.last_val_acc = aux["acc"].mean().item()
        else:
            self.last_val_acc = None


def objective(trial):
    game_size = 2
    # ------------------ hyperparams ------------------
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    tau_s = trial.suggest_float("tau_s", 1.0, 5.0)
    sender_entropy = trial.suggest_float("sender_entropy", 0.01, 0.3)
    receiver_entropy = trial.suggest_float("receiver_entropy", 0.01, 0.15)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # ------------------ opts ------------------
    opts = parse_arguments()
    opts.tau_s = tau_s
    opts.batch_size = batch_size
    opts.mode = "gs"
    opts.gs_tau = 1

    # ------------------ dataset ------------------
    dataset = WCSFeat("ours_images_single_sm0.h5")

    train_loader = ImagenetLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        opt=opts,
        batches_per_epoch=opts.batches_per_epoch,
        seed=None,
    )

    val_loader = ImagenetLoader(
        dataset,
        batch_size=batch_size,
        opt=opts,
        batches_per_epoch=opts.batches_per_epoch,
        seed=7,
    )

    # ------------------ game ------------------
    game = get_game(game_size, opts.mode, opts.gs_tau, opts.tau_s, sender_entropy, receiver_entropy)

    optimizer = core.build_optimizer(game.parameters())
    for g in optimizer.param_groups:
        g["lr"] = lr

    # ------------------ callback ------------------
    val_cb = ValidationAccCallback()

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=val_loader,
        callbacks=[val_cb],
    )

    trainer.train(n_epochs=10)

    if val_cb.last_val_acc is None:
        return 0.0

    return val_cb.last_val_acc




# =========================
# LANCEMENT OPTUNA
# =========================
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)

    print("\n Meilleurs hyperparamètres :")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")

    print(f"\n Accuracy max: {study.best_value:.4f}")
