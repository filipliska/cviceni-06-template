from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from src.metrics import confusion_matrix


class Validator(ABC):
    """Spolecne rozhrani vsech prevzorkovacich strategii (navrhovy vzor Strategy)."""

    @abstractmethod
    def split(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yielduje dvojice ``(train_idx, test_idx)`` celociselnych indexu."""


class HoldOut(Validator):
    """Jedno nahodne rozdeleni na train/test (nejjednodussi prevzorkovani)."""

    def __init__(
        self,
        test_size: float = 0.2,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, x: np.ndarray, y: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = x.shape[0]
        indices = np.arange(n_samples)

        rng = np.random.default_rng(self.random_state)
        if self.shuffle:
            rng.shuffle(indices)

        n_test = int(round(self.test_size * n_samples))
        n_test = max(1, min(n_test, n_samples - 1))
        n_train = n_samples - n_test

        train_idx = np.sort(indices[:n_train])
        test_idx = np.sort(indices[n_train:])
        yield train_idx, test_idx


class KFold(Validator):
    """k-nasobna krizova validace — ``n_folds`` rovnocennych rozdeleni."""

    def __init__(
        self,
        n_folds: int,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, x: np.ndarray, y: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yielduje ``n_folds`` dvojic ``(train_idx, test_idx)``[cite: 5]."""
        n_samples = x.shape[0]
        
        assert self.n_folds >= 2, "Počet foldů n_folds musí být alespoň 2."
        assert self.n_folds <= n_samples, "Počet foldů n_folds nemůže být větší než počet vzorků[cite: 5]."

        indices = np.arange(n_samples)
        rng = np.random.default_rng(self.random_state)
        if self.shuffle:
            rng.shuffle(indices)

        # Rozdělení indexů na n_folds přihrádek pomocí array_split (ošetří nerovnoměrné délky)
        folds = np.array_split(indices, self.n_folds)

        for i in range(self.n_folds):
            test_idx = folds[i]
            # Train indexy tvoří sjednocení všech ostatních foldů
            train_idx = np.concatenate([folds[j] for j in range(self.n_folds) if j != i])
            
            # Seřazení indexů vzestupně pro zachování původního pořádku
            yield np.sort(train_idx), np.sort(test_idx)


class Bootstrap(Validator):
    """Bootstrap — vyber trenovaci mnoziny losovanim s opakovanim."""

    def __init__(
        self,
        n_bootstrap: int,
        random_state: int | None = None,
    ) -> None:
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def split(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yielduje (nejvyse) ``n_bootstrap`` dvojic ``(train_idx, test_idx)``[cite: 5]."""
        assert self.n_bootstrap >= 1, "Počet bootstrapových iterací musí být alespoň 1."
        n_samples = x.shape[0]
        assert n_samples > 0, "Matice příznaků x musí obsahovat alespoň jeden vzorek."

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_bootstrap):
            # a. Výběr n trénovacích indexů s opakováním
            train_idx = rng.integers(0, n_samples, size=n_samples)
            
            # b. Out-of-bag (OOB) testovací indexy = komplement unikátních hodnot train_idx
            unique_train = np.unique(train_idx)
            test_idx = np.setdiff1d(np.arange(n_samples), unique_train)

            # c. Pokud je testovací množina prázdná, iteraci přeskočíme
            if test_idx.size == 0:
                continue

            # d. Jinak yieldujeme dvojici (train_idx v původním pořadí/délce n, test_idx seřazené)
            yield train_idx, np.sort(test_idx)


def cross_validate(
    model: Any,
    validator: Validator,
    x: np.ndarray,
    y: np.ndarray,
    metric_fns: dict[str, Callable[[np.ndarray], float]],
) -> dict:
    """Spusti model na vsech splitech dane strategie a agreguje metriky."""
    per_split_scores: dict[str, list[float]] = {name: [] for name in metric_fns}
    n_splits = 0

    for train_idx, test_idx in validator.split(x, y):
        n_splits += 1
        model.fit(x[train_idx], y[train_idx])
        y_pred = model.predict(x[test_idx])
        cm = confusion_matrix(y[test_idx], y_pred)
        for name, fn in metric_fns.items():
            per_split_scores[name].append(float(fn(cm)))

    result: dict = {}
    for name, scores in per_split_scores.items():
        arr = np.asarray(scores, dtype=float)
        result[f"{name}_mean"] = float(arr.mean()) if arr.size else float("nan")
        result[f"{name}_std"] = float(arr.std()) if arr.size else float("nan")
    result["per_split_scores"] = per_split_scores
    result["n_splits"] = n_splits
    return result