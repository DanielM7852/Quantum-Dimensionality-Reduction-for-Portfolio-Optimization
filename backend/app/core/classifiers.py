import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class LogisticTrader:
    def __init__(self):
        self.name = "Logistic"
        self.model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0).astype(int)
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict_proba(x)[:, 1]


class RandomForestTrader:
    def __init__(self):
        self.name = "Random-Forest"
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=4,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
        )

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0).astype(int)
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict_proba(x)[:, 1]


class GradientBoostingTrader:
    def __init__(self):
        self.name = "Gradient-Boosting"
        self.model = GradientBoostingClassifier(
            n_estimators=30,
            max_depth=2,
            learning_rate=0.05,
            min_samples_split=20,
            random_state=42,
        )

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0).astype(int)
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict_proba(x)[:, 1]
