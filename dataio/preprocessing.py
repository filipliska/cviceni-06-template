from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def standardize_fit(x_train: np.ndarray) -> dict:
    """Spocita parametry z-skorove standardizace **jen z trenovacich dat**."""
    x_train = np.asarray(x_train, dtype=np.float64)
    assert x_train.ndim == 2, "Vstupní matice x_train musí mít 2 rozměry."

    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0, ddof=0)  # Populační směrodatná odchylka (ddof=0)

    return {"mean": mean, "std": std}


def standardize_apply(x: np.ndarray, params: dict) -> np.ndarray:
    """Aplikuje ulozene parametry standardizace (z-skore) na matici ``X``."""
    assert "mean" in params and "std" in params, "Slovník params musí obsahovat klíče 'mean' a 'std'."
    
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(params["mean"], dtype=np.float64)
    std = np.asarray(params["std"], dtype=np.float64)

    assert x.shape[1] == mean.shape[0], "Počet příznaků v x se musí shodovat s délkou mean v params."

    # Ochrana proti dělení nulou: nulové směrodatné odchylky nahradíme jedničkou
    std_safe = np.where(std == 0.0, 1.0, std)

    return (x - mean) / std_safe


def subsample_imbalance(
        x: np.ndarray,
        y: np.ndarray,
        minority_ratio: float,
        random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Podvzorkovanim vytvori silne nevyvazeny dataset (~90/10)."""
    assert 0.0 < minority_ratio < 0.5, "Minority ratio musí být v intervalu (0, 0.5)."
    y = np.asarray(y).ravel()
    classes = np.unique(y)
    assert len(classes) == 2, "Cílová proměnná y musí obsahovat přesně dvě třídy (0 i 1)."

    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(random_state)

    # 1. Zjištění minoritní a majoritní třídy (minlength=2 zajišťuje správné počty pro obě třídy 0 a 1)
    counts = np.bincount(y, minlength=2)
    minority_class = int(np.argmin(counts))
    majority_class = int(np.argmax(counts))

    min_indices = np.flatnonzero(y == minority_class)
    maj_indices = np.flatnonzero(y == majority_class)

    n_min = len(min_indices)
    n_maj = len(maj_indices)
    r = minority_ratio

    # 2. Výpočet počtu ponechaných řádků podle detailu výpočtu
    current_ratio = n_min / (n_min + n_maj)

    if current_ratio > r:
        # Ponecháme všechny majoritní a ořízneme minoritní
        n_min_keep = round(r * n_maj / (1.0 - r))
        n_min_keep = max(1, min(n_min_keep, n_min))

        selected_min = rng.choice(min_indices, size=n_min_keep, replace=False)
        keep = np.concatenate([maj_indices, selected_min])
    else:
        # Ponecháme všechny minoritní a ořízneme majoritní
        n_maj_keep = round(n_min * (1.0 - r) / r)
        n_maj_keep = max(1, min(n_maj_keep, n_maj))

        selected_maj = rng.choice(maj_indices, size=n_maj_keep, replace=False)
        keep = np.concatenate([min_indices, selected_maj])

    # 3. Zamíchání výsledného pořadí řádků
    rng.shuffle(keep)

    return x[keep], y[keep]


def demonstrate_leakage(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    random_state: int,
) -> dict:
    """Demonstruje unik dat: standardizace PRED vs. PO rozdeleni na train/test."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    test_size = 0.2

    # --- SPATNE: standardizace cele matice pred rozdelenim -----------------
    params_all = standardize_fit(x)
    x_all_std = standardize_apply(x, params_all)
    xtr_w, xte_w, ytr_w, yte_w = train_test_split(
        x_all_std, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf_w = KNeighborsClassifier(n_neighbors=k)
    clf_w.fit(xtr_w, ytr_w)
    acc_wrong = float(np.mean(clf_w.predict(xte_w) == yte_w))

    # --- SPRAVNE: nejdriv split, scaler jen z trenovacich dat -------------
    xtr_r, xte_r, ytr_r, yte_r = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    params_tr = standardize_fit(xtr_r)
    xtr_r_std = standardize_apply(xtr_r, params_tr)
    xte_r_std = standardize_apply(xte_r, params_tr)
    clf_r = KNeighborsClassifier(n_neighbors=k)
    clf_r.fit(xtr_r_std, ytr_r)
    acc_right = float(np.mean(clf_r.predict(xte_r_std) == yte_r))

    return {
        "acc_wrong": acc_wrong,
        "acc_right": acc_right,
        "test_size": test_size,
        "k": k,
    }