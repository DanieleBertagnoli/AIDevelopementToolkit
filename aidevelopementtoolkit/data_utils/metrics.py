from typing import Dict, Optional, Tuple

from aidevelopementtoolkit.logging_utils.logger import get_formatted_logger
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.utils.parallel import Parallel, delayed
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    completeness_score,
    homogeneity_score,
    v_measure_score,
    fowlkes_mallows_score,
)

logger = get_formatted_logger(name=__name__, level="ERROR")

def compute_classification_metrics(confusion_matrix: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute classification metrics from a confusion matrix.

    This function computes classification metrics directly from an accumulated
    confusion matrix. It is useful when predictions and labels are aggregated
    over multiple batches or distributed processes, avoiding the need to store
    all predictions and labels.

    Notes
    -----
    The following metrics are computed:
    - Accuracy
    - Precision
    - Recall
    - F1 score

    The function automatically detects the classification problem type:
    - Binary classification if the confusion matrix has shape `(2, 2)`.
    - Multiclass classification otherwise.

    For multiclass problems, Precision, Recall, and F1 Score are computed
    using macro averaging.

    The confusion matrix is expected to follow the convention:

        rows -> ground-truth classes
        columns -> predicted classes

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Confusion matrix with shape `(C, C)` where:
        - `C`: Number of classes

        Element `(i, j)` represents the number of samples belonging to
        ground-truth class `i` that were predicted as class `j`.

    Returns
    -------
    Tuple[Dict[str, float], np.ndarray]
        Respectively:
            - Dictionary containing the computed classification metrics.
            - Row-normalized confusion matrix.

    Examples
    --------
    >>> cm = np.array(
    ...     [
    ...         [8, 2],
    ...         [1, 9],
    ...     ]
    ... )
    >>> metrics, normalized_cm = compute_classification_metrics_from_cm(cm)
    """

    confusion_matrix = np.asarray(confusion_matrix, dtype=np.float64,)

    if confusion_matrix.ndim != 2:
        logger.error(
            "The given confusion matrix must be a 2-dimensional square matrix. "
            f"Received shape {confusion_matrix.shape}."
        )
        raise ValueError()

    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        logger.error(
            "The given confusion matrix must be square. "
            f"Received shape {confusion_matrix.shape}."
        )
        raise ValueError()


    num_classes = confusion_matrix.shape[0]

    # Compute correctly classified samples
    true_positives = np.diag(confusion_matrix)

    # Compute accuracy
    total_samples = confusion_matrix.sum()
    accuracy = true_positives.sum() / total_samples if total_samples > 0 else 0.0

    if num_classes == 2:

        fp = confusion_matrix[0, 1]
        fn = confusion_matrix[1, 0]
        tp = confusion_matrix[1, 1]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    else:

        # Multiclass classification:
        precisions = np.zeros(num_classes, dtype=np.float64)
        recalls = np.zeros(num_classes, dtype=np.float64)

        for class_idx in range(num_classes):

            tp = true_positives[class_idx]

            predicted_as_class = confusion_matrix[:, class_idx].sum()
            actual_class = confusion_matrix[class_idx, :].sum()

            if predicted_as_class > 0:
                precisions[class_idx] = tp / predicted_as_class

            if actual_class > 0:
                recalls[class_idx] = tp / actual_class


        # Macro averaging: every class has equal importance
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


    metrics = {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1 Score": float(f1_score),
    }


    # Normalize the confusion matrix by row so each true-class sum is 1
    normalized_confusion_matrix = confusion_matrix.copy()

    row_sums = normalized_confusion_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normalized_confusion_matrix /= row_sums

    return metrics, normalized_confusion_matrix


def compute_confusion_matrix(
        predictions: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
        padding_mask: np.ndarray | None = None,
    ) -> np.ndarray:
    """Compute the confusion matrix from predictions and labels.

    This function computes a confusion matrix that can be accumulated across
    multiple batches or distributed processes before computing classification
    metrics.

    Notes
    -----
    The confusion matrix follows the convention:

        rows -> ground-truth classes
        columns -> predicted classes

    Element `(i, j)` represents the number of samples belonging to
    ground-truth class `i` that were predicted as class `j`.

    Padding positions are ignored when a padding mask is provided.

    Parameters
    ----------
    predictions : np.ndarray
        Array containing the predicted classes. Accepted shapes:
        `(N,)` or any shape that can be flattened consistently with labels.

    labels : np.ndarray
        Array containing the ground-truth classes.
        Same shape as `predictions`.

    num_classes : int
        Number of classes in the classification problem.

    padding_mask : np.ndarray, default=None
        Boolean array where `True` indicates padding positions that must be
        ignored. Same shape as `predictions`.

    Returns
    -------
    np.ndarray
        Confusion matrix with shape `(num_classes, num_classes)`.

    Examples
    --------
    >>> predictions = np.array([0, 1, 1, 0])
    >>> labels = np.array([0, 1, 0, 0])
    >>> cm = compute_confusion_matrix(
    ...     predictions=predictions,
    ...     labels=labels,
    ...     num_classes=2,
    ... )
    """

    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    if predictions.shape != labels.shape:
        logger.error(
            "The given `predictions` and `labels` have different shapes: "
            f"{predictions.shape} vs {labels.shape}"
        )
        raise ValueError()


    if padding_mask is not None:

        padding_mask = np.asarray(padding_mask, dtype=bool)

        if predictions.shape != padding_mask.shape:
            logger.error(
                "The given `predictions` and `padding_mask` have different "
                "shapes: "
                f"{predictions.shape} vs {padding_mask.shape}"
            )
            raise ValueError()

        valid = ~padding_mask

        predictions = predictions[valid]
        labels = labels[valid]


    # Flatten all dimensions
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    if np.any(predictions < 0) or np.any(predictions >= num_classes):
        logger.error(
            "Predictions contain class indices outside the valid range "
            f"[0, {num_classes - 1}]."
        )
        raise ValueError()

    if np.any(labels < 0) or np.any(labels >= num_classes):
        logger.error(
            "Labels contain class indices outside the valid range "
            f"[0, {num_classes - 1}]."
        )
        raise ValueError()

    return confusion_matrix(
        labels,
        predictions,
        labels=np.arange(num_classes),
    )


def cluster_distance_stats(
        embeddings: np.ndarray,
        cluster_ids: np.ndarray,
        metric: str = "euclidean",
    ) -> Tuple[float, float]:
    """Compute intra- and inter-cluster distance statistics.

    Notes
    -----
    Intra-cluster distance is the mean pairwise distance between every pair of
    points that belong to the **same** cluster, averaged across all clusters.

    Inter-cluster distance is the mean pairwise distance between every pair of
    points that belong to **different** clusters.

    Any distance metric accepted by `scipy.spatial.distance.cdist` can be
    used (e.g. `"euclidean"`, `"cosine"`, `"cityblock"`).

    Parameters
    ----------
    embeddings : np.ndarray
        2-D array of point coordinates with shape `(N, D)` where:
        - `N`: Number of points
        - `D`: Embedding dimensionality

    cluster_ids : np.ndarray
        1-D integer array of cluster assignments with shape `(N,)`.
        Each element is the cluster id of the corresponding embedding.

    metric : str, default="euclidean"
        Distance metric forwarded to `scipy.spatial.distance.cdist`.

    Returns
    -------
    Tuple[float, float]
        Respectively:
            - Mean intra-cluster distance (`nan` when every cluster has
              fewer than two points).
            - Mean inter-cluster distance (`nan` when all points belong to
              the same cluster).

    Examples
    --------
    >>> embeddings = np.array(
    ...     [
    ...         [0.0, 0.0],
    ...         [1.0, 0.0],
    ...         [5.0, 0.0],
    ...         [6.0, 0.0],
    ...     ]
    ... )
    >>> cluster_ids = np.array([0, 0, 1, 1])
    >>> intra_d, inter_d = cluster_distance_stats(
    ...     embeddings=embeddings,
    ...     cluster_ids=cluster_ids,
    ...     metric="euclidean",
    ... )
    """

    embeddings = np.asarray(embeddings, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)

    if embeddings.ndim != 2:
        logger.error(
            "The given `embeddings` must be a 2-D array with shape `(N, D)`. "
            f"Received shape {embeddings.shape}."
        )
        raise ValueError()

    if cluster_ids.ndim != 1 or cluster_ids.shape[0] != embeddings.shape[0]:
        logger.error(
            "The given `cluster_ids` must be a 1-D array with length N matching "
            f"`embeddings`. Received shape {cluster_ids.shape} vs {embeddings.shape}."
        )
        raise ValueError()

    unique_ids = np.unique(cluster_ids)

    # Intra cluster distances
    intra_means = []
    for cid in unique_ids:
        mask = cluster_ids == cid
        pts = embeddings[mask]
        if pts.shape[0] < 2:
            continue
        dists = cdist(pts, pts, metric=metric)
        # upper triangle only (exclude diagonal zeros)
        upper = dists[np.triu_indices(pts.shape[0], k=1)]
        intra_means.append(upper.mean())

    intra_d = float(np.mean(intra_means)) if intra_means else float(np.nan)

    # Inter cluster distances
    inter_distances = []
    for i, cid_a in enumerate(unique_ids):
        for cid_b in unique_ids[i + 1:]:
            pts_a = embeddings[cluster_ids == cid_a]
            pts_b = embeddings[cluster_ids == cid_b]
            dists = cdist(pts_a, pts_b, metric=metric)
            inter_distances.append(dists.mean())

    inter_d = float(np.mean(inter_distances)) if inter_distances else float(np.nan)

    return intra_d, inter_d


def compute_clustering_metrics(
        predictions: np.ndarray,
        labels: np.ndarray,
        padding_mask: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        metric: str = "euclidean",
        n_jobs: int = -1,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Compute clustering metrics for one or more sequences.

    Notes
    -----
    The following metrics are computed for every sequence:
    - Completeness
    - Homogeneity
    - V-Measure Score
    - Fowlkes-Mallows Score
    - Elements Like Me (ELM) Score
    - Intra-Cluster Distance (only when `embeddings` is provided)
    - Inter-Cluster Distance (only when `embeddings` is provided)

    The first four metrics are computed through `sklearn.metrics` (https://scikit-learn.org/stable/api/sklearn.metrics.html).
    The ELM score is computed by following the paper: https://link.springer.com/article/10.1007/s10791-024-09436-7.
    Intra/Inter-Cluster distances are computed via :func:`cluster_distance_stats`.

    Degenerate clusterings (only one predicted or one ground-truth cluster) are excluded from the averaged 
    Homogeneity, Completeness and V-Measure statistics for this evaluation protocol.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted cluster labels. Accepted shapes `(T,)` or `(B, T)`: 
        - `B`: Batch dimension
        - `T`: Sequence length

    labels : np.ndarray
        Ground-truth cluster labels. Same shape as `predictions`.

    padding_mask : np.ndarray
        Boolean array where `True` indicates padding. Same shape as `predictions`.

    embeddings : np.ndarray, default=None
        Point embeddings used to compute distance-based statistics via
        :func:`cluster_distance_stats`. Accepted shapes `(T, D)` or
        `(B, T, D)` where `D` is the embedding dimensionality. When
        `None` the distance metrics are omitted from the output.

    metric : str, default="euclidean"
        Distance metric forwarded to :func:`cluster_distance_stats`.
        Ignored when `embeddings` is `None`.

    n_jobs : int, default=-1
        Number of parallel workers for batch processing. `-1` uses all
        available CPU cores. Forwarded to :class:`joblib.Parallel`.

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, float]]

        - non_averaged:
            Dictionary containing one metric per sequence.

        - averaged:
            Dictionary containing the average metric across sequences.

    Examples
    --------
    >>> predictions = np.array([0, 0, 1, 1])
    >>> labels = np.array([0, 0, 1, 2])
    >>> padding_mask = np.zeros_like(labels, dtype=bool)
    >>> non_averaged, averaged = compute_clustering_metrics(
    ...     predictions=predictions,
    ...     labels=labels,
    ...     padding_mask=padding_mask,
    ... )
    """

    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    padding_mask = np.asarray(padding_mask, dtype=bool)

    use_embeddings = embeddings is not None
    if use_embeddings:
        embeddings = np.asarray(embeddings, dtype=np.float64)

    if predictions.shape != labels.shape:
        logger.error(
            "The given `predictions` and `labels` have different shapes: "
            f"{predictions.shape} vs {labels.shape}"
        )
        raise ValueError()

    if predictions.shape != padding_mask.shape:
        logger.error(
            "The given `predictions` and `padding_mask` have different shapes: "
            f"{predictions.shape} vs {padding_mask.shape}"
        )
        raise ValueError()

    if predictions.ndim == 1:
        predictions = predictions[None, :]
        labels = labels[None, :]
        padding_mask = padding_mask[None, :]
        if use_embeddings:
            embeddings = embeddings[None, :]

    if predictions.ndim != 2:
        logger.error(
            "The given arrays must have shape `(T,)` or `(B, T)`. "
            f"Received {predictions.shape}."
        )
        raise ValueError()

    if use_embeddings and embeddings.ndim != 3:
        logger.error(
            "The given `embeddings` must have shape `(T, D)` or `(B, T, D)`. "
            f"Received {embeddings.shape}."
        )
        raise ValueError()

    B, T = predictions.shape

    def _process_batch(batch_idx: int) -> Dict[str, float]:
        result: Dict[str, float] = {}

        valid = ~padding_mask[batch_idx]
        if not np.any(valid):
            return result

        pred = predictions[batch_idx][valid]
        gt = labels[batch_idx][valid]

        if pred.size < 2:
            return result

        result["Fowlkes-Mallows Score"] = fowlkes_mallows_score(gt, pred)
        result["Elements Like Me Score"] = _elm_score(pred, gt)

        if use_embeddings:
            emb = embeddings[batch_idx][valid]
            intra_d, inter_d = cluster_distance_stats(
                embeddings=emb,
                cluster_ids=pred,
                metric=metric,
            )
            result["Intra-Cluster Distance"] = intra_d
            result["Inter-Cluster Distance"] = inter_d

        n_pred_clusters = np.unique(pred).size
        n_gt_clusters = np.unique(gt).size

        if n_pred_clusters > 1 and n_gt_clusters > 1:
            result["Completeness"] = completeness_score(gt, pred)
            result["Homogeneity"] = homogeneity_score(gt, pred)
            result["V-Measure Score"] = v_measure_score(gt, pred)

        return result

    batch_results = Parallel(n_jobs=n_jobs)(
        delayed(_process_batch)(i)
        for i in tqdm(
            range(B),
            leave=False,
            colour="cyan",
            desc="Computing clustering metrics 📊",
        )
    )

    metric_keys = [
        "Completeness",
        "Homogeneity",
        "V-Measure Score",
        "Fowlkes-Mallows Score",
        "Elements Like Me Score",
    ]
    if use_embeddings:
        metric_keys += ["Intra-Cluster Distance", "Inter-Cluster Distance"]

    non_averaged = {k: np.full(B, np.nan) for k in metric_keys}
    for batch_idx, result in enumerate(batch_results):
        for k, v in result.items():
            non_averaged[k][batch_idx] = v

    averaged = {
        metric: float(np.nanmean(values))
        for metric, values in non_averaged.items()
    }

    return non_averaged, averaged


def _elm_score(
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> float:
    """
    Compute the Elements Like Me (ELM) F1 score. It's modified version of 
    the BCubed score. 
    
    Notes
    -----
    Official paper: https://link.springer.com/article/10.1007/s10791-024-09436-7.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted cluster labels. Shape (N,).

    labels : np.ndarray
        Ground-truth cluster labels. Shape (N,).

    Returns
    -------
    float
        Mean ELM F1 score over all elements.
    """

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    N = len(predictions)

    if N == 0:
        return np.nan

    f1_scores = np.empty(N, dtype=float)

    for i in range(N):

        pred_cluster = np.flatnonzero(predictions == predictions[i])
        true_cluster = np.flatnonzero(labels == labels[i])

        # Remove the element itself
        pred_others = pred_cluster[pred_cluster != i]
        true_others = true_cluster[true_cluster != i]

        tp = len(np.intersect1d(pred_others, true_others))
        fp = len(np.setdiff1d(pred_others, true_others))
        fn = len(np.setdiff1d(true_others, pred_others))

        # F1 (paper Eq. 3 with modified TP)
        if len(pred_others) == 0 and len(true_others) == 0:
            f1 = 1.0
        elif tp == 0:
            f1 = 0.0
        else:
            f1 = tp / (tp + 0.5 * (fp + fn))

        f1_scores[i] = f1

    return float(np.mean(f1_scores))