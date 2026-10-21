from __future__ import annotations

from typing import TYPE_CHECKING
import os
import numpy as np

if TYPE_CHECKING:
    from src.distance import Distance


class KNNClassifier:
    """Induktivni klasifikator k nejblizsich sousedu (nededi z niceho).

    Metrika vzdalenosti se predava zavislosti (dependency injection) —
    klasifikator sam nevi, jak se vzdalenost pocita, jen vola
    ``self.distance.calculate(x, y)``. Diky tomu lze stejny kNN pouzit s
    eukleidovskou, manhattanskou i kosinovou vzdalenosti bez zmeny kodu.

    Atributy (naplneny az v ``fit``, do te doby ``None``):
        X_train_: np.ndarray tvaru (n_samples, n_features) — zapamatovana
            trenovaci data (float64).
        y_train_: np.ndarray tvaru (n_samples,) — zapamatovane popisky
            trenovacich vzorku (cela cisla).
    """

    def __init__(self, k: int, distance: "Distance") -> None:
        self.k = k
        self.distance = distance
        # Atributy naucene ve fit(); pred zavolanim fit() jsou None.
        self.x_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Zapamatuje si trenovaci data; u kNN je "trenink" pouhe ulozeni."""
        self.x_train_ = np.asarray(x, dtype=np.float64)
        self.y_train_ = np.asarray(y).ravel().astype(int)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Zaradi kazdy radek ``X`` vetsinovym hlasem jeho ``k`` nejblizsich sousedu."""
        assert self.x_train_ is not None and self.y_train_ is not None, "Model nebyl natrénován (fit nebyl zavolán)."
        
        # Převedení vstupních dat na numpy pole, pokud už nejsou
        x = np.asarray(x, dtype=np.float64)
        assert x.ndim == 2, "Vstupní data x musí mít 2 dimenze (n_queries, n_features)."
        assert x.shape[1] == self.x_train_.shape[1], "Počet příznaků v dotazu se musí shodovat s trénovacími daty[cite: 3]."

        n_queries = x.shape[0]
        predictions = np.empty(n_queries, dtype=int)

        for i in range(n_queries):
            query = x[i]
            # 1. Spočítejte vzdálenost ke všem trénovacím bodům přes self.distance.calculate[cite: 2, 3]
            distances = np.array([self.distance.calculate(query, x_train_i) for x_train_i in self.x_train_])
            
            # 2. Vyberte k nejbližších sousedů (stabilní řazení zajistí argpartition nebo argsort)
            # argsort zajistí deterministické řešení shod podle indexů v poli x_train_
            nearest_indices = np.argsort(distances)[:self.k]
            
            # 3. Z popisků těchto k sousedů vezměte většinový hlas[cite: 3]
            nearest_labels = self.y_train_[nearest_indices]
            
            # Hlasování s deterministickým řešením shod (při shodě hlasů vyhrává nižší třída)
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            max_count = np.max(counts)
            
            # Najdeme všechny třídy, které mají maximální počet hlasů
            best_labels = unique_labels[counts == max_count]
            # Při shodě vrátíme třídu s nejnižším číselným označením
            predictions[i] = np.min(best_labels)

        return predictions

    def save(self, path: str) -> None:
        """Ulozi nauceny model do souboru ``.npz`` (format NumPy)[cite: 2, 3]."""
        assert self.x_train_ is not None and self.y_train_ is not None, "Nelze uložit netrénovaný model."
        
        np.savez(
            path,
            x_train=self.x_train_,
            y_train=self.y_train_,
            k=np.asarray(self.k)
        )

    def load(self, path: str) -> KNNClassifier:
        """Nacte nauceny model z ``.npz`` souboru (format NumPy)[cite: 2, 3]."""
        assert os.path.exists(path), f"Soubor {path} neexistuje."
        
        # Načtení dat ze souboru .npz
        with np.load(path) as data:
            assert "x_train" in data and "y_train" in data and "k" in data, "Soubor .npz neobsahuje požadovaná data modelu."
            self.x_train_ = data["x_train"]
            self.y_train_ = data["y_train"]
            self.k = int(data["k"])
            
        return self