from dataclasses import dataclass

import numpy as np

from CONFIG import ENSEMBLE
from kmnist.submission.embeddings import normalize_rows


@dataclass(frozen=True)
class EnsemblePredictionResult:
    labels: np.ndarray
    prototype_labels: np.ndarray
    prototype_distances: np.ndarray
    prototype_margins: np.ndarray
    classifier_labels: np.ndarray
    classifier_confidences: np.ndarray
    decision_reasons: np.ndarray


def predict_labels(embeddings: np.ndarray, classes: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    result = prototype_predictions(embeddings, classes, prototypes)
    return result[0]


def prototype_predictions(
    embeddings: np.ndarray,
    classes: np.ndarray,
    prototypes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized_embeddings = normalize_rows(embeddings)
    distances = 1 - normalized_embeddings @ prototypes.T
    sorted_indices = np.argsort(distances, axis=1)
    nearest_indices = sorted_indices[:, 0]
    second_indices = sorted_indices[:, 1]
    nearest_distances = distances[np.arange(len(distances)), nearest_indices]
    second_distances = distances[np.arange(len(distances)), second_indices]
    return classes[nearest_indices], nearest_distances, second_distances - nearest_distances


def classifier_predictions(logits: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted_logits)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    indices = probabilities.argmax(axis=1)
    return classes[indices], probabilities[np.arange(len(probabilities)), indices]


def ensemble_predict_labels(
    embeddings: np.ndarray,
    logits: np.ndarray,
    classes: np.ndarray,
    prototypes: np.ndarray,
    prototype_margin_gate: float = ENSEMBLE.prototype_margin_gate,
    classifier_confidence_gate: float = ENSEMBLE.classifier_confidence_gate,
) -> EnsemblePredictionResult:
    prototype_labels, prototype_distances, prototype_margins = prototype_predictions(
        embeddings,
        classes,
        prototypes,
    )
    classifier_labels, classifier_confidences = classifier_predictions(logits, classes)

    labels = prototype_labels.copy()
    decision_reasons = np.full(len(labels), "agreement", dtype=object)
    agreement_mask = prototype_labels == classifier_labels
    disagreement_mask = ~agreement_mask
    high_prototype_margin_mask = prototype_margins >= prototype_margin_gate
    high_classifier_confidence_mask = classifier_confidences >= classifier_confidence_gate

    prototype_margin_mask = disagreement_mask & high_prototype_margin_mask
    decision_reasons[prototype_margin_mask] = "prototype_margin"

    classifier_confidence_mask = disagreement_mask & ~high_prototype_margin_mask & high_classifier_confidence_mask
    labels[classifier_confidence_mask] = classifier_labels[classifier_confidence_mask]
    decision_reasons[classifier_confidence_mask] = "classifier_confidence"

    fallback_mask = disagreement_mask & ~high_prototype_margin_mask & ~high_classifier_confidence_mask
    decision_reasons[fallback_mask] = "prototype_fallback"

    return EnsemblePredictionResult(
        labels=labels.astype(np.int64),
        prototype_labels=prototype_labels.astype(np.int64),
        prototype_distances=prototype_distances.astype(np.float32),
        prototype_margins=prototype_margins.astype(np.float32),
        classifier_labels=classifier_labels.astype(np.int64),
        classifier_confidences=classifier_confidences.astype(np.float32),
        decision_reasons=decision_reasons,
    )


def diagnostic_mask(result: EnsemblePredictionResult) -> np.ndarray:
    disagreement_mask = result.prototype_labels != result.classifier_labels
    low_confidence_mask = result.decision_reasons == "prototype_fallback"
    return disagreement_mask | low_confidence_mask
