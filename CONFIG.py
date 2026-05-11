from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PathsConfig:
    # Absolute path to the repository root.
    project_root: Path = Path(__file__).resolve().parent
    # Directory containing input CSVs and image folders.
    data_dir_name: str = "Data"
    # Directory used for new generated artifacts.
    outputs_dir_name: str = "outputs"
    @property
    def data_dir(self) -> Path:
        return self.project_root / self.data_dir_name
    @property
    def outputs_dir(self) -> Path:
        return self.project_root / self.outputs_dir_name


@dataclass(frozen=True)
class DataConfig:
    # Folder of labeled KMNIST PNG images.
    labeled_dir_name: str = "Labeled"
    # Folder of unlabeled KMNIST PNG images.
    unlabeled_dir_name: str = "Unlabeled"
    # CSV file mapping labeled image paths to class labels.
    labels_csv_name: str = "Labeled-labels.csv"
    # CSV template defining submission image order.
    sample_submission_name: str = "sample-submission.csv"
    # Expected image file extension for dataset loading.
    image_glob: str = "*.png"
    # CSV column containing image paths.
    image_path_column: str = "ImagePath"
    # CSV column containing class labels.
    label_column: str = "Label"


@dataclass(frozen=True)
class ModelConfig:
    # Flattened 28x28 grayscale KMNIST image size.
    input_size: int = 28 * 28
    # Input image shape used by the convolutional autoencoder.
    image_channels: int = 1
    image_size: int = 28
    # Channel widths for residual encoder stages.
    encoder_channels: tuple[int, ...] = (96, 160, 256, 384)
    # Residual block count for each encoder stage.
    encoder_blocks_per_stage: tuple[int, ...] = (1, 2, 2, 2)
    # Spatial feature-map size after the strided encoder blocks.
    decoder_seed_size: int = 7
    # Channel width used by the lightweight decoder seed feature map.
    decoder_seed_channels: int = 32
    # Dropout probability after hidden activations in the encoder.
    encoder_dropout: float = 0.20
    # Embedding vector width produced by avg-pool and max-pool concatenation.
    embedding_size: int = 768
    # Dropout probability after hidden activations in the decoder.
    decoder_dropout: float = 0.15
    # Hidden widths for the classifier head that consumes embeddings.
    classifier_hidden_sizes: tuple[int, ...] = (1536, 1024, 768)
    # Dropout probability in the classifier head.
    classifier_dropout: float = 0.35
    # Number of KMNIST target classes.
    num_classes: int = 10


@dataclass(frozen=True)
class TrainingConfig:
    # Batch size used by the joint labeled/unlabeled training loaders.
    batch_size: int = 512
    # Batch size used for reference embedding metrics.
    reference_batch_size: int = 512
    # Maximum DataLoader worker count when not supplied explicitly.
    max_num_workers: int = 4
    # Number of epochs for the Lightning trainer.
    max_epochs: int = 100
    # Number of devices requested from Lightning.
    devices: int = 1
    # Trainer accelerator selection.
    accelerator: str = "auto"
    # GPU mixed precision mode.
    cuda_precision: str = "16-mixed"
    # CPU precision mode.
    cpu_precision: int = 32
    # Lightning logging cadence.
    log_every_n_steps: int = 1
    # Fraction of training batches to run each epoch.
    limit_train_batches: float = 1.0
    # Directory under outputs/ for future checkpoints.
    checkpoint_dir_name: str = "checkpoints"
    # Fraction of labeled samples held out for validation.
    validation_fraction: float = 0.1
    # Random seed for deterministic labeled train/validation splitting.
    validation_seed: int = 42
    # Filename format for the best classifier-loss checkpoint.
    classifier_checkpoint_filename: str = "best-classifier-{epoch:03d}-{val_classification_loss:.4f}"
    # Metric used to choose the best classifier checkpoint.
    classifier_checkpoint_monitor: str = "val_classification_loss"
    # Direction used for classifier checkpoint metric comparison.
    classifier_checkpoint_mode: str = "min"
    # Filename format for the best embedding-geometry checkpoint.
    embedding_checkpoint_filename: str = "best-embedding-{epoch:03d}-{val_prototype_acc:.4f}"
    # Metric used to choose the best embedding checkpoint.
    embedding_checkpoint_monitor: str = "val_prototype_acc"
    # Direction used for embedding checkpoint metric comparison.
    embedding_checkpoint_mode: str = "max"
    # Number of best checkpoints to keep.
    save_top_k: int = 1
    # Whether to keep the latest checkpoint.
    save_last: bool = True
    # Matmul precision requested from PyTorch.
    float32_matmul_precision: str = "medium"
    # Learning rate for model parameters.
    model_learning_rate: float = 5e-4
    # Minimum learning rate reached by cosine annealing at the end of training.
    model_min_learning_rate: float = 1e-6
    # AdamW weight decay for regularization.
    model_weight_decay: float = 1e-3
    # Fraction of epochs used for linear learning-rate warmup before cosine decay.
    warmup_fraction: float = 0.10
    # Whether to use stochastic weight averaging near the end of training.
    use_swa: bool = True
    # Learning rate used by stochastic weight averaging.
    swa_lrs: float = 5e-5
    # Fraction of training after which stochastic weight averaging starts.
    swa_epoch_start: float = 0.75
    # Number of epochs used to anneal into the SWA learning rate.
    swa_annealing_epochs: int = 10


@dataclass(frozen=True)
class LossConfig:
    # Low reconstruction weight so unlabeled images shape embeddings without dominating class geometry.
    reconstruction_loss_weight: float = 0.05
    # Weight applied to labeled classifier cross-entropy.
    classification_loss_weight: float = 1.0
    # Label smoothing for training cross-entropy only; validation loss stays unsmoothed.
    classification_label_smoothing: float = 0.1
    # Weight applied to supervised contrastive embedding separation.
    supervised_contrastive_loss_weight: float = 1.0
    # Temperature for supervised contrastive similarities.
    supervised_contrastive_temperature: float = 0.1
    # Maximum weight for FixMatch-style unlabeled consistency loss.
    consistency_loss_weight: float = 0.5
    # Fraction of each stage used to ramp consistency loss from 0 to max weight.
    consistency_ramp_fraction: float = 0.2
    # Minimum weak-view confidence required before applying consistency loss.
    consistency_confidence_threshold: float = 0.95


@dataclass(frozen=True)
class AnalysisConfig:
    # Directory under outputs/ for embeddings and visualizations.
    output_dir_name: str = "embedding_analysis"
    # Default batch size for embedding extraction.
    batch_size: int = 512
    # Default worker count for analysis loaders.
    num_workers: int = 0
    # Random seed used by TSNE and UMAP.
    random_state: int = 42
    # TSNE perplexity.
    tsne_perplexity: int = 30
    # UMAP neighbor count.
    umap_neighbors: int = 100
    # UMAP minimum distance.
    umap_min_dist: float = 0.0
    # UMAP distance metric.
    umap_metric: str = "cosine"
    # Matplotlib config directory for headless runs.
    matplotlib_config_dir: str = "/tmp/matplotlib"


@dataclass(frozen=True)
class SubmissionConfig:
    # Directory under outputs/ for generated submissions.
    output_dir_name: str = "submissions"
    # Default output filename for a submission CSV.
    output_filename: str = "1.csv"
    # Default batch size for submission embedding extraction.
    batch_size: int = 512
    # Default worker count for submission loaders.
    num_workers: int = 0
    # Whether to average classifier logits over deterministic test-time augmentations.
    use_tta: bool = True
    # Deterministic TTA rotation angles in degrees. The first entry should be the identity view.
    tta_rotation_degrees: tuple[float, ...] = (0.0, -5.0, 5.0, 0.0, 0.0)
    # Deterministic TTA translations in pixels, paired with tta_rotation_degrees.
    tta_translate_pixels: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0), (-1, 0), (1, 0))


@dataclass(frozen=True)
class EnsembleConfig:
    # Prototype margin required to trust prototype prediction on classifier disagreement.
    prototype_margin_gate: float = 0.20
    # Classifier softmax confidence required to trust classifier prediction on prototype disagreement.
    classifier_confidence_gate: float = 0.90
    # Deterministic fallback for low-confidence disagreements.
    low_confidence_fallback: str = "prototype"


@dataclass(frozen=True)
class PseudoLabelConfig:
    # Directory under outputs/ for pseudo-label generation runs.
    output_dir_name: str = "pseudo_labels"
    # If set, training includes this pseudo-label CSV as weakly supervised data.
    labels_csv: Optional[Path] = None
    # Relative loss weight for pseudo-labeled supervised losses.
    loss_weight: float = 0.25
    # Nearest-prototype distance threshold for selected pseudo labels.
    max_distance: float = 0.10
    # Prototype margin threshold for selected pseudo labels.
    min_margin: float = 0.20
    # Maximum selected pseudo-label count per predicted class.
    class_cap: int = 50
    # Optional total selected pseudo-label cap. Class-balanced selection is used when set.
    total_cap: Optional[int] = None
    # Pseudo-label threshold mode: "labeled_relative" or "static".
    threshold_mode: str = "labeled_relative"
    # Labeled own-prototype distance quantile used for class-specific calibration.
    labeled_distance_quantile: float = 0.95
    # Multiplier applied to labeled distance quantiles for unlabeled candidate limits.
    labeled_distance_scale: float = 40.0
    # Labeled true-class margin quantile used for class-specific calibration.
    labeled_margin_quantile: float = 0.05
    # Multiplier applied to labeled margin quantiles for unlabeled candidate limits.
    labeled_margin_scale: float = 0.10
    # CSV column containing per-sample pseudo-label training weights.
    weight_column: str = "Weight"
    # Minimum pseudo-label weight after confidence/margin quality scaling.
    min_sample_weight: float = 0.25


@dataclass(frozen=True)
class StagedTrainingConfig:
    # Directory under outputs/ for staged self-training experiments.
    output_dir_name: str = "staged_training"
    # Total pseudo-label budgets used by successive stages. Stage 0 must remain 0.
    pseudo_label_budgets: tuple[int, ...] = (0,)
    # Seeds used by the multi-seed staged experiment runner.
    seeds: tuple[int, ...] = (42, 43, 44)
    # Maximum epochs per stage before early stopping.
    max_epochs: int = 160
    # Metric used for early stopping and stage comparison.
    monitor: str = "val_prototype_acc"
    # Direction used for monitored metric comparison.
    mode: str = "max"
    # Early-stopping patience within a stage.
    patience: int = 25
    # Minimum monitored-metric improvement treated as meaningful.
    min_delta: float = 0.001
    # First pseudo-label stage supervised pseudo-loss weight.
    pseudo_loss_start_weight: float = 0.03
    # Final pseudo-label stage supervised pseudo-loss weight.
    pseudo_loss_end_weight: float = 0.05
    # Whether pseudo-label stages initialize from the previous stage's best checkpoint.
    warm_start_pseudo_stages: bool = True
    # Warm-start strategy for pseudo-label stages. Currently "weights_only".
    warm_start_mode: str = "weights_only"
    # Default staged-run submission mode.
    staged_submission_mode: str = "ensemble"
    # Number of deterministic validation splits used for post-run model selection.
    repeated_validation_splits: int = 10
    # Number of top checkpoints used in default staged-run submission ensembles.
    ensemble_size: int = 3
    # Whether staged training should generate richer summaries/submission artifacts after training.
    postprocess: bool = True
    # Whether staged training postprocessing should run embedding analysis for the best checkpoint.
    postprocess_embedding_analysis: bool = True
    # Whether staged training postprocessing should generate default submission artifacts.
    postprocess_submission: bool = True


@dataclass(frozen=True)
class TensorBoardConfig:
    # Directory under outputs/ where Lightning writes logs.
    log_dir_name: str = "lightning_logs"
    # Host TensorBoard binds to.
    host: str = "0.0.0.0"
    # Port TensorBoard listens on.
    port: int = 6006
    # Whether TensorBoard should disable fast loading.
    load_fast: str = "false"


PATHS = PathsConfig()
DATA = DataConfig()
MODEL = ModelConfig()
TRAINING = TrainingConfig()
LOSSES = LossConfig()
ANALYSIS = AnalysisConfig()
SUBMISSION = SubmissionConfig()
ENSEMBLE = EnsembleConfig()
PSEUDO_LABELS = PseudoLabelConfig()
STAGED_TRAINING = StagedTrainingConfig()
TENSORBOARD = TensorBoardConfig()
