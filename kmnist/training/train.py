import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pathlib import Path

from CONFIG import TRAINING
from kmnist.data import build_dataloaders
from kmnist.models import Autoencoder
from kmnist.utils.paths import checkpoint_dir, outputs_dir


def build_classifier_checkpoint_callback(dirpath: Path | None = None) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=dirpath or checkpoint_dir(),
        filename=TRAINING.classifier_checkpoint_filename,
        monitor=TRAINING.classifier_checkpoint_monitor,
        mode=TRAINING.classifier_checkpoint_mode,
        save_top_k=TRAINING.save_top_k,
        save_last=TRAINING.save_last,
    )


def build_embedding_checkpoint_callback(dirpath: Path | None = None) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=dirpath or checkpoint_dir(),
        filename=TRAINING.embedding_checkpoint_filename,
        monitor=TRAINING.embedding_checkpoint_monitor,
        mode=TRAINING.embedding_checkpoint_mode,
        save_top_k=TRAINING.save_top_k,
        save_last=TRAINING.save_last,
    )


def build_trainer(
    max_epochs: int = TRAINING.max_epochs,
    checkpoint_dirpath: Path | None = None,
    early_stopping: EarlyStopping | None = None,
    limit_train_batches: float = TRAINING.limit_train_batches,
) -> L.Trainer:
    precision = TRAINING.cuda_precision if torch.cuda.is_available() else TRAINING.cpu_precision
    callbacks = [
        build_classifier_checkpoint_callback(checkpoint_dirpath),
        build_embedding_checkpoint_callback(checkpoint_dirpath),
    ]
    if TRAINING.use_swa:
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=TRAINING.swa_lrs,
                swa_epoch_start=TRAINING.swa_epoch_start,
                annealing_epochs=TRAINING.swa_annealing_epochs,
            )
        )
    if early_stopping is not None:
        callbacks.append(early_stopping)
    return L.Trainer(
        accelerator=TRAINING.accelerator,
        precision=precision,
        devices=TRAINING.devices,
        log_every_n_steps=TRAINING.log_every_n_steps,
        enable_progress_bar=True,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        default_root_dir=outputs_dir(),
        callbacks=callbacks,
    )


def train(
    pseudo_labels_csv=None,
    pseudo_loss_weight: float = 0.25,
    max_epochs: int = TRAINING.max_epochs,
    checkpoint_dirpath: Path | None = None,
    early_stopping: EarlyStopping | None = None,
) -> None:
    torch.set_float32_matmul_precision(TRAINING.float32_matmul_precision)
    model = Autoencoder(pseudo_loss_weight=pseudo_loss_weight, max_epochs=max_epochs)
    train_loader, validation_loader = build_dataloaders(pseudo_labels_csv=pseudo_labels_csv)
    trainer = build_trainer(max_epochs=max_epochs, checkpoint_dirpath=checkpoint_dirpath, early_stopping=early_stopping)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader)


def main() -> None:
    train()


if __name__ == "__main__":
    main()
