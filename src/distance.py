from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Distance(ABC):
    """
    Abstraktní základní třída pro výpočet vzdálenosti (nepodobnosti)
    mezi dvěma vektory příznaků.

    Konkrétní podtřídy implementují vlastnost ``is_metric``, která říká,
    zda daná míra splňuje axiomy metriky (zejména trojúhelníkovou
    nerovnost). Např. eukleidovská a manhattanská vzdálenost jsou skutečné
    metriky, zatímco kosinová vzdálenost (1 - kosinová podobnost) trojúhelníkovou
    nerovnost obecně nesplňuje, a tedy metrikou není.
    """

    @property
    @abstractmethod
    def is_metric(self) -> bool:
        """Vrátí True, pokud vzdálenost splňuje axiomy metriky."""

    @abstractmethod
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Vypočítá vzdálenost (nepodobnost) mezi dvěma 1D vektory příznaků
        x a y.

        Parametry
        ---------
        x : np.ndarray
            První vektor příznaků (1D pole).
        y : np.ndarray
            Druhý vektor příznaků (1D pole), stejné délky jako x.

        Návratová hodnota
        ------------------
        float
            Nezáporné číslo vyjadřující vzdálenost/nepodobnost mezi x a y.
        """
        raise NotImplementedError(
            "Úkol: Implementujte výpočet vzdálenosti mezi dvěma vektory x a y."
        )

    def create_distance_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        Sestaví čtvercovou matici vzdáleností pro všechny dvojice vzorků
        v X.
        """
        n_samples = x.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples), dtype=float)

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distance = self.calculate(x[i], x[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        return distance_matrix


class EuclideanDistance(Distance):
    """Eukleidovská (L2) vzdálenost mezi dvěma vektory."""

    @property
    def is_metric(self) -> bool:
        """Eukleidovská vzdálenost je pravá metrika."""
        return True

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Vypočítá eukleidovskou vzdálenost mezi dvěma 1D vektory příznaků
        x a y.
        """
        assert isinstance(x, np.ndarray), "Vektor x musí být typu np.ndarray"
        assert isinstance(y, np.ndarray), "Vektor y musí být typu np.ndarray"
        assert x.shape == y.shape, "Vektory x a y musí mít stejný tvar."

        return float(np.linalg.norm(x - y))


class ManhattanDistance(Distance):
    """Manhattanská (L1, taxicab) vzdálenost mezi dvěma vektory."""

    @property
    def is_metric(self) -> bool:
        """Manhattanská vzdálenost je pravá metrika."""
        return True

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Vypočítá manhattanskou vzdálenost mezi dvěma 1D vektory příznaků
        x a y.
        """
        assert isinstance(x, np.ndarray), "Vektor x musí být typu np.ndarray"
        assert isinstance(y, np.ndarray), "Vektor y musí být typu np.ndarray"
        assert x.shape == y.shape, "Vektory x a y musí mít stejný tvar."

        return float(np.sum(np.abs(x - y)))


class CosineCoeficient(Distance):
    """
    Kosinová vzdálenost (1 - kosinová podobnost) mezi dvěma vektory.
    """

    @property
    def is_metric(self) -> bool:
        """Kosinová vzdálenost není pravá metrika (porušuje trojúhelníkovou nerovnost)."""
        return False

    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Vypočítá kosinovou vzdálenost (1 - kosinový koeficient podobnosti)
        mezi dvěma 1D vektory příznaků x a y.
        """
        assert isinstance(x, np.ndarray), "Vektor x musí být typu np.ndarray"
        assert isinstance(y, np.ndarray), "Vektor y musí být typu np.ndarray"
        assert x.shape == y.shape, "Vektory x a y musí mít stejný tvar."

        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        assert norm_x > 0 and norm_y > 0, "Vektory x a y nesmí být nulové (dělení nulou)."

        similarity = np.dot(x, y) / (norm_x * norm_y)
        # Ošetření případných numerických nepřesností mimo interval [-1, 1]
        similarity = np.clip(similarity, -1.0, 1.0)

        return float(1.0 - similarity)