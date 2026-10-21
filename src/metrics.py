from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Sestavi 2x2 kontingecni tabulku ``[[TN, FP], [FN, TP]]``."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    assert len(y_true) == len(y_pred), "Vektory y_true a y_pred musí mít stejnou délku."
    assert np.isin(y_true, [0, 1]).all(), "Vektor y_true musí obsahovat pouze binární hodnoty {0, 1}."
    assert np.isin(y_pred, [0, 1]).all(), "Vektor y_pred musí obsahovat pouze binární hodnoty {0, 1}."

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([[tn, fp], [fn, tp]], dtype=int)


def accuracy(cm: np.ndarray) -> float:
    """Presnost (accuracy) — podil spravne zaradenych vzorku."""
    assert cm.shape == (2, 2), "Kontingenční tabulka cm musí mít tvar (2, 2)."
    total = np.sum(cm)
    if total == 0:
        return 0.0
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    return float((tp + tn) / total)


def precision(cm: np.ndarray) -> float:
    """Preciznost (precision, pozitivni prediktivni hodnota)."""
    assert cm.shape == (2, 2), "Kontingenční tabulka cm musí mít tvar (2, 2)."
    fp = cm[0, 1]
    tp = cm[1, 1]
    denominator = tp + fp
    if denominator == 0:
        return 0.0
    return float(tp / denominator)


def recall_sensitivity(cm: np.ndarray) -> float:
    """Recall = sensitivita = Se = TPR (true positive rate)."""
    assert cm.shape == (2, 2), "Kontingenční tabulka cm musí mít tvar (2, 2)."
    fn = cm[1, 0]
    tp = cm[1, 1]
    denominator = tp + fn
    if denominator == 0:
        return 0.0
    return float(tp / denominator)


def specificity(cm: np.ndarray) -> float:
    """Specificita = Sp = TNR (true negative rate)."""
    assert cm.shape == (2, 2), "Kontingenční tabulka cm musí mít tvar (2, 2)."
    tn = cm[0, 0]
    fp = cm[0, 1]
    denominator = tn + fp
    if denominator == 0:
        return 0.0
    return float(tn / denominator)


def f1_score(cm: np.ndarray) -> float:
    """F1 skore — harmonicky prumer preciznosti a recallu."""
    assert cm.shape == (2, 2), "Kontingenční tabulka cm musí mít tvar (2, 2)."
    prec = precision(cm)
    rec = recall_sensitivity(cm)
    denominator = prec + rec
    if denominator == 0:
        return 0.0
    return float(2 * prec * rec / denominator)