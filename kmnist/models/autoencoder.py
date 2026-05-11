import lightning as L
import torch
from torch import nn
from torch.nn import functional as F

from CONFIG import LOSSES, MODEL, PSEUDO_LABELS, TRAINING
from kmnist.losses.embedding_losses import supervised_contrastive_loss
from kmnist.metrics.reference import reference_labeling_metrics
from kmnist.models.architecture import build_decoder, build_encoder


VALIDATION_METRICS_TO_LOG = {
    "labeled_knn_1_acc",
    "labeled_prototype_acc",
    "labeled_cluster_score",
}


class Autoencoder(L.LightningModule):
    def __init__(
        self,
        pseudo_loss_weight: float = PSEUDO_LABELS.loss_weight,
        consistency_loss_weight: float = LOSSES.consistency_loss_weight,
        consistency_confidence_threshold: float = LOSSES.consistency_confidence_threshold,
        max_epochs: int = TRAINING.max_epochs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = build_encoder()
        self.decoder = build_decoder()
        self.classifier = self._build_classifier()
        self.validation_embeddings: list[torch.Tensor] = []
        self.validation_labels: list[torch.Tensor] = []
        self.pseudo_loss_weight = pseudo_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.consistency_confidence_threshold = consistency_confidence_threshold
        self.max_epochs = max_epochs

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.encode(x)
        reconstructions = self.decoder(embeddings)
        logits = self.classifier(embeddings)
        return reconstructions, logits, embeddings

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @staticmethod
    def _build_classifier() -> nn.Sequential:
        layers: list[nn.Module] = []
        in_features = MODEL.embedding_size
        for hidden_size in MODEL.classifier_hidden_sizes:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(MODEL.classifier_dropout),
                ]
            )
            in_features = hidden_size
        layers.append(nn.Linear(in_features, MODEL.num_classes))
        return nn.Sequential(*layers)

    def training_step(self, batch, batch_idx):
        batch = self._combined_batch(batch)
        return self._training_step(batch)

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        images, labels = batch
        _, logits, embeddings = self(images)
        classification_loss = F.cross_entropy(logits, labels)

        self.log(
            "val_classification_loss",
            classification_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=labels.size(0),
        )
        self.validation_embeddings.append(embeddings.detach())
        self.validation_labels.append(labels.detach())
        return classification_loss

    def _training_step(self, batch) -> torch.Tensor:
        unlabeled_views, _ = batch["unlabeled"]
        unlabeled_weak_images, unlabeled_strong_images = self._unlabeled_views(unlabeled_views)
        labeled_images, labels = batch["labeled"]
        pseudo_batch = batch.get("pseudo")

        all_images = torch.cat([unlabeled_strong_images, labeled_images], dim=0)
        unlabeled_batch_size = unlabeled_strong_images.size(0)
        reconstruction_batch_size = all_images.size(0)
        labeled_batch_size = labeled_images.size(0)

        reconstructions, logits, embeddings = self(all_images)
        labeled_logits = logits[unlabeled_batch_size:]
        labeled_embeddings = embeddings[unlabeled_batch_size:]

        reconstruction_loss = F.mse_loss(reconstructions, all_images)
        classification_loss = F.cross_entropy(
            labeled_logits,
            labels,
            label_smoothing=LOSSES.classification_label_smoothing,
        )
        contrastive_loss = supervised_contrastive_loss(
            labeled_embeddings,
            labels,
            temperature=LOSSES.supervised_contrastive_temperature,
        )
        total_loss = (
            LOSSES.reconstruction_loss_weight * reconstruction_loss
            + LOSSES.classification_loss_weight * classification_loss
            + LOSSES.supervised_contrastive_loss_weight * contrastive_loss
        )
        consistency_loss, consistency_acceptance_rate = self._consistency_loss(
            unlabeled_weak_images,
            unlabeled_strong_images,
        )
        consistency_weight = self._consistency_weight()
        total_loss = total_loss + consistency_weight * consistency_loss

        if pseudo_batch is not None:
            pseudo_images, pseudo_labels, pseudo_weights = self._pseudo_batch_items(pseudo_batch)
            _, pseudo_logits, _ = self(pseudo_images)
            pseudo_losses = F.cross_entropy(
                pseudo_logits,
                pseudo_labels,
                label_smoothing=LOSSES.classification_label_smoothing,
                reduction="none",
            )
            pseudo_weights = pseudo_weights.to(device=pseudo_losses.device, dtype=pseudo_losses.dtype)
            pseudo_classification_loss = (pseudo_losses * pseudo_weights).sum() / pseudo_weights.sum().clamp_min(1e-12)
            total_loss = total_loss + self.pseudo_loss_weight * LOSSES.classification_loss_weight * pseudo_classification_loss

        self._log_training_metrics(
            {
                "train_total_loss": (total_loss, reconstruction_batch_size, True),
                "train_has_pseudo_labels": (
                    torch.tensor(float(pseudo_batch is not None), device=self.device),
                    labeled_batch_size,
                    False,
                ),
                "train_consistency_loss": (consistency_loss, unlabeled_batch_size, False),
                "train_consistency_acceptance_rate": (consistency_acceptance_rate, unlabeled_batch_size, False),
                "train_consistency_weight": (
                    torch.tensor(consistency_weight, device=self.device),
                    unlabeled_batch_size,
                    False,
                ),
                "train_pseudo_loss_weight": (
                    torch.tensor(self.pseudo_loss_weight, device=self.device),
                    labeled_batch_size,
                    False,
                ),
            }
        )
        return total_loss

    def _consistency_loss(
        self,
        weak_images: torch.Tensor,
        strong_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            _, weak_logits, _ = self(weak_images)
            weak_probabilities = F.softmax(weak_logits, dim=1)
            confidences, pseudo_labels = weak_probabilities.max(dim=1)
            accepted_mask = confidences >= self.consistency_confidence_threshold

        acceptance_rate = accepted_mask.float().mean()
        if not accepted_mask.any():
            return strong_images.new_tensor(0.0), acceptance_rate

        _, strong_logits, _ = self(strong_images)
        loss = F.cross_entropy(strong_logits[accepted_mask], pseudo_labels[accepted_mask])
        return loss, acceptance_rate

    def _consistency_weight(self) -> float:
        if self.consistency_loss_weight <= 0:
            return 0.0
        ramp_epochs = max(1.0, self.max_epochs * LOSSES.consistency_ramp_fraction)
        return float(self.consistency_loss_weight * min(1.0, (self.current_epoch + 1) / ramp_epochs))

    @staticmethod
    def _unlabeled_views(unlabeled_views) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(unlabeled_views, (tuple, list)) and len(unlabeled_views) == 2:
            return unlabeled_views[0], unlabeled_views[1]
        return unlabeled_views, unlabeled_views

    def _pseudo_batch_items(self, pseudo_batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(pseudo_batch) == 3:
            return pseudo_batch
        pseudo_images, pseudo_labels = pseudo_batch
        return pseudo_images, pseudo_labels, torch.ones_like(pseudo_labels, dtype=torch.float32, device=self.device)

    def on_validation_epoch_start(self):
        self.validation_embeddings = []
        self.validation_labels = []

    def on_validation_epoch_end(self):
        if not self.validation_embeddings:
            return

        embeddings = torch.cat(self.validation_embeddings, dim=0)
        labels = torch.cat(self.validation_labels, dim=0)
        metrics = reference_labeling_metrics(embeddings, labels)
        batch_size = labels.size(0)
        for metric_name, metric_value in metrics.items():
            if metric_name not in VALIDATION_METRICS_TO_LOG:
                continue
            validation_metric_name = self._validation_metric_name(metric_name)
            self.log(
                validation_metric_name,
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=validation_metric_name in {
                    "val_knn_1_acc",
                    "val_prototype_acc",
                    "val_cluster_score",
                },
                logger=True,
                batch_size=batch_size,
            )

        self.validation_embeddings = []
        self.validation_labels = []

    @staticmethod
    def _validation_metric_name(metric_name: str) -> str:
        if metric_name.startswith("labeled_"):
            return f"val_{metric_name.removeprefix('labeled_')}"
        return f"val_{metric_name}"

    @staticmethod
    def _combined_batch(batch):
        if isinstance(batch, tuple) and batch and isinstance(batch[0], dict):
            return batch[0]
        return batch

    def _log_training_metrics(
        self,
        metrics: dict[str, tuple[torch.Tensor, int, bool]],
    ) -> None:
        for metric_name, (metric_value, batch_size, prog_bar) in metrics.items():
            self.log(
                metric_name,
                metric_value,
                on_step=True,
                on_epoch=True,
                prog_bar=prog_bar,
                logger=True,
                batch_size=batch_size,
            )

    def configure_optimizers(self):
        trainable_parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=TRAINING.model_learning_rate,
            weight_decay=TRAINING.model_weight_decay,
        )
        warmup_epochs = max(1, int(round(self.max_epochs * TRAINING.warmup_fraction)))
        cosine_epochs = max(1, self.max_epochs - warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=TRAINING.model_min_learning_rate,
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / warmup_epochs,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
