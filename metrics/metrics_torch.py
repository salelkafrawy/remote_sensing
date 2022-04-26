import torch
import numbers
import warnings


def top_k_error_rate_top_30_set(y_probas, y_true):
    r"""Predicts the top-30 sets error rate from scores.

    Parameters
    ----------
    y_probas: 2d array, [n_samples, n_classes]
        Scores for each sample and label.
    y_true: 1d array, [n_samples]
            True labels.

    Returns
    -------
    2d array, [n_samples, 30]:
        Predicted top-30 sets for each sample.


    Notes
    -----
    Complexity: :math:`O( n_\text{samples} \times n_\text{classes} )`.
    """
    k = 30
    n_classes = y_probas.shape[1]
    _, pred_ids = torch.topk(y_probas, k)

    pointwise_accuracy = torch.sum(pred_ids == y_true[:, None], axis=1)
    return 1 - torch.Tensor.float(pointwise_accuracy).mean()


def predict_top_30_set(y_probas):
    r"""Predicts the top-30 sets from scores/probabilities

    Parameters
    ----------
    y_probas: 2d array, [n_samples, n_classes]
        Scores for each sample and label.

    Returns
    -------
    2d array, [n_samples, 30]:
        Predicted top-30 sets for each sample.

    Notes
    -----
    Complexity: :math:`O( n_\text{samples} \times n_\text{classes} )`.
    """

    k = 30
    n_classes = y_probas.shape[1]
    vals, pred_ids = torch.topk(y_probas, k)
    return pred_ids


def top_k_error_rate_from_sets(y_true, s_pred):
    r"""Computes the top-k error rate from predicted sets.

    Parameters
    ----------
    y_true: 1d array, [n_samples]
        True labels.
    s_pred: 2d array, [n_samples, k]
        Previously computed top-k sets for each sample.

    Returns
    -------
    float:
        Error rate value.
    """
    #     y_true = validate_labels(y_true)
    #     s_pred = validate_top_k_sets(s_pred)

    # validate_labels(y_true)
    # validate_top_k_sets(s_pred)

    pointwise_accuracy = torch.sum(s_pred == y_true[:, None], axis=1)
    return 1 - torch.Tensor.float(pointwise_accuracy).mean()


def top_k_error_rate(y_true, y_score, k, disable_warning=False):
    r"""Computes the top-k error rate for a given k.

    Parameters
    ----------
    y_true: 1d array, [n_samples]
        True labels.
    y_score: 2d array, [n_samples, n_classes]
        Scores for each label.
    k: int
        Value of k to use, should range from 1 to n_classes.
    disable_warning: bool
        Disables the warning trigger if y_score looks like it contains top-k sets rather than scores

    Returns
    -------
    float:
        Error rate value.

    Notes
    -----
    Complexity: :math:`O( n_\text{samples} \times n_\text{classes} )`.
    """
    s_pred = predict_top_k_set(y_score, k, disable_warning=disable_warning)
    return top_k_error_rate_from_sets(y_true, s_pred)


def top_30_error_rate(y_true, y_score):
    r"""Computes the top-30 error rate.

    Parameters
    ----------
    y_true: 1d array, [n_samples]
        True labels.
    y_score: 2d array, [n_samples, n_classes]
        Scores for each label.

    Returns
    -------
    float:
        Top-30 error rate value.

    Notes
    -----
    Complexity: :math:`O( n_\text{samples} \times n_\text{classes} )`.
    """
    return top_k_error_rate(y_true, y_score, k=30)
