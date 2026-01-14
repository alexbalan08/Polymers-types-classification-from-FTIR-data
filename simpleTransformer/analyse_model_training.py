import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

PATH = r"checkpoints"


def run():
    # analyse_dir(os.path.join(PATH, r"basic_model"), True)
    analyse_dir(os.path.join(PATH, r"pretraining_sample"), False)

    compare()


def compare():
    pars = ["D_MODEL", "NUM_HEADS", "NUM_LAYERS"]

    basic_models = load_model_losses(os.path.join(os.path.join(PATH, r"basic_model")))
    pretrain = load_model_losses(os.path.join(os.path.join(PATH, r"pretraining_sample")))

    basic_models = basic_models.loc[basic_models.set_index(pars).index.isin(pretrain.set_index(pars).index)]
    # missing_cols = [c for c in pretrain.columns if c not in basic_models.columns]
    # for mc in missing_cols:
    #     basic_models = basic_models.assign(from_df1=False)

    basic_models = basic_models.assign(pretrain=False)
    pretrain = pretrain.assign(pretrain=True)

    models = pd.concat([basic_models, pretrain], ignore_index=True)
    with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.max_colwidth", None,
            "display.width", None,
            "display.expand_frame_repr", False,
    ):
        print(basic_models)
        print(pretrain)
        print(models)

    pars.append('pretrain')
    analyse_individual(models, pars)


def analyse_dir(path, search):
    losses = load_model_losses(os.path.join(path))
    with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.max_colwidth", None,
            "display.width", None,
            "display.expand_frame_repr", False,
    ):
        print(losses)

    analyse_losses(losses, search=search)


def load_model_losses(path):
    configs = []
    for d in Path(path).iterdir():
        if d.is_dir():
            config = pd.read_csv(d / "config_log.csv")
            loss = pd.read_csv(d / "training_history.csv")
            config["TRAIN_TIME"] = (pd.to_datetime(config["END_TIME"], format="%Y-%m-%d_%H-%M-%S") -
                                    pd.to_datetime(config["START_TIME"], format="%Y-%m-%d_%H-%M-%S"))

            for fold in loss["fold"].unique():
                fold_loss = loss[loss["fold"] == fold].sort_values("epoch")
                config[f"train_loss_f{fold}"] = [fold_loss["loss"].to_numpy()]
                config[f"val_loss_f{fold}"] = [fold_loss["val_loss"].to_numpy()]

            configs.append(config)
    configs = pd.concat(configs, ignore_index=True)
    return configs


def analyse_losses(losses, search=True):
    if search:
        analyse_hyper_parameter(losses, "D_MODEL", ["NUM_HEADS", "NUM_LAYERS"])
        analyse_hyper_parameter(losses, "NUM_HEADS", ["D_MODEL", "NUM_LAYERS"])
        analyse_hyper_parameter(losses, "NUM_LAYERS", ["D_MODEL", "NUM_HEADS"])
    else:
        analyse_individual(losses, ["D_MODEL", "NUM_HEADS", "NUM_LAYERS"])


def analyse_hyper_parameter(losses, par, other_pars):
    n = losses.groupby(other_pars).ngroups
    cols = min(3, n)
    rows = np.ceil(n / cols).astype(int)

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(n)]

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows), sharex=True)
    axes = axes.flatten()

    for i, ((o_par1, o_par2), loss) in enumerate(losses.groupby(other_pars)):
        for j, (par_val, lo) in enumerate(loss.groupby(par)):
            for fold in range(1, 4):
                train_loss, val_loss = lo[f"train_loss_f{fold}"].iloc[0], lo[f"val_loss_f{fold}"].iloc[0]
                epochs = np.arange(len(train_loss))

                ax = axes[i]
                if fold == 1:
                    ax.plot(epochs, train_loss, label=f"train {par}={par_val}", color=colors[j], linestyle='--', alpha=0.5)
                    ax.plot(epochs, val_loss, label=f"val {par}={par_val}", color=colors[j])
                else:
                    ax.plot(epochs, train_loss, color=colors[j], linestyle='--', alpha=0.5)
                    ax.plot(epochs, val_loss, color=colors[j])
                ax.set_yscale("log")
                ax.set_ylim(0.05, 0.25)

                ax.set_title(f"{other_pars[0]}={o_par1} {other_pars[1]}={o_par2}")
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend()

    # Remove any empty axes if N < rows*cols
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def analyse_individual(losses, pars):
    n = losses.groupby(pars).ngroups
    cols = 1  # min(3, n)
    rows = 1  # np.ceil(n / cols).astype(int)

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(n)]

    scale_fig = 2
    fig, axes = plt.subplots(rows, cols, figsize=(5 * scale_fig*cols, 3 * scale_fig*rows), sharex=True)
    if rows != 1:
        axes.flatten()

    for i, (par_vals, loss) in enumerate(losses.groupby(pars)):
        model_name = " ".join([f"{p}={v}" for v, p in zip(par_vals, pars)])
        if i == 0:
            ax = axes[0] if cols != 1 else axes
            ax.plot([1, 2], [-1, -1], color='black', label=f"train loss", linestyle='--', alpha=0.5)
            ax.plot([1, 2], [-1, -1], color='black', label=f"val loss")
            ax.plot([1, 2], [-1, -1], color='black', label=f"pretrain loss", linestyle='-.', alpha=0.3)
            ax.plot([1, 2], [-1, -1], color='black', label=f"preval loss", linestyle=':')

        for fold in range(1, 4):
            has_pretrain = "pretrain" in losses.columns
            print(has_pretrain)
            if has_pretrain:
                has_pretrain = loss["pretrain"].iloc[0]
                print(has_pretrain)
            else:
                has_pretrain = f"train_loss_fpre{fold}" in losses.columns

            if has_pretrain:
                pre_train_loss, pre_val_loss = loss[f"train_loss_fpre{fold}"].iloc[0], loss[f"val_loss_fpre{fold}"].iloc[0]
                pre_epochs = np.arange(len(pre_train_loss))
            train_loss, val_loss = loss[f"train_loss_f{fold}"].iloc[0], loss[f"val_loss_f{fold}"].iloc[0]
            epochs = np.arange(len(train_loss))

            ax.plot(epochs, train_loss, color=colors[i], linestyle='--', alpha=0.5)
            if fold == 1:
                ax.plot(epochs, val_loss, color=colors[i], label=model_name)
            else:
                ax.plot(epochs, val_loss, color=colors[i])
            if has_pretrain:
                ax.plot(pre_epochs, pre_train_loss, color=colors[i], linestyle='-.', alpha=0.3)
                ax.plot(pre_epochs, pre_val_loss, color=colors[i], linestyle=':')
            ax.set_yscale("log")
            ax.set_ylim(0.05, 0.5)

            # ax.set_title()
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()

    # Remove any empty axes if N < rows*cols
    if rows != 1:
        for j in range(n, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
