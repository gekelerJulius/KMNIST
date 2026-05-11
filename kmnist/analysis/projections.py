import numpy as np
from sklearn.manifold import TSNE

from CONFIG import ANALYSIS


def run_tsne(embeddings: np.ndarray) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=ANALYSIS.random_state,
        perplexity=ANALYSIS.tsne_perplexity,
        verbose=1,
    )
    return tsne.fit_transform(embeddings)


def run_umap(embeddings: np.ndarray) -> np.ndarray:
    try:
        import umap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "UMAP is not installed. Install it with `pip install umap-learn` and rerun the script."
        ) from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=ANALYSIS.umap_neighbors,
        min_dist=ANALYSIS.umap_min_dist,
        metric=ANALYSIS.umap_metric,
        random_state=ANALYSIS.random_state,
    )
    return reducer.fit_transform(embeddings)
