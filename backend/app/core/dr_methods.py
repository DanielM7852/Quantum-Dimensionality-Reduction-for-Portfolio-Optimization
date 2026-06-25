import numpy as np
from sklearn.decomposition import PCA, KernelPCA


class PCAMethod:
    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.name = "PCA"

    def fit_transform(self, x: np.ndarray, y=None) -> np.ndarray:
        return np.nan_to_num(self.pca.fit_transform(x), nan=0.0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.nan_to_num(self.pca.transform(x), nan=0.0)


class KernelPCAMethod:
    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.kpca = KernelPCA(n_components=n_components, kernel="rbf", gamma=0.01)
        self.name = "Kernel-PCA"

    def fit_transform(self, x: np.ndarray, y=None) -> np.ndarray:
        try:
            return np.nan_to_num(self.kpca.fit_transform(x), nan=0.0)
        except (ValueError, np.linalg.LinAlgError):
            return np.zeros((x.shape[0], self.n_components))

    def transform(self, x: np.ndarray) -> np.ndarray:
        try:
            return np.nan_to_num(self.kpca.transform(x), nan=0.0)
        except (ValueError, np.linalg.LinAlgError):
            return np.zeros((x.shape[0], self.n_components))


class NoDRMethod:
    """Identity passthrough — classifier on scaled raw features."""

    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.name = "No-DR"

    def fit_transform(self, x: np.ndarray, y=None) -> np.ndarray:
        cols = min(self.n_components, x.shape[1])
        self.selected_indices = list(range(cols))
        return np.nan_to_num(x[:, self.selected_indices], nan=0.0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.nan_to_num(x[:, self.selected_indices], nan=0.0)


class EntangleDRMethod:
    """
    Correlation-entanglement dimensionality reduction via sequential Givens rotations.
    See docs/method_entangledr.md for algorithm specification.
    """

    def __init__(
        self,
        n_components: int = 8,
        correlation_threshold: float = 0.7,
        iterations: int = 3,
        rotation_scale: float = np.pi / 36,
    ):
        self.n_components = n_components
        self.correlation_threshold = correlation_threshold
        self.iterations = iterations
        self.rotation_scale = rotation_scale
        self.name = "EntangleDR"

    def _build_transformation(self, x_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        corr = np.nan_to_num(np.corrcoef(x_std.T), nan=0.0)
        entanglement = np.abs(corr)
        transformation = np.eye(x_std.shape[1])
        n_features = x_std.shape[1]

        for _ in range(self.iterations):
            for i in range(n_features - 1):
                for j in range(i + 1, n_features):
                    if entanglement[i, j] > self.correlation_threshold:
                        theta = self.rotation_scale * entanglement[i, j]
                        rotation = np.eye(n_features)
                        rotation[i, i] = np.cos(theta)
                        rotation[i, j] = -np.sin(theta)
                        rotation[j, i] = np.sin(theta)
                        rotation[j, j] = np.cos(theta)
                        transformation = rotation @ transformation

        x_transformed = np.nan_to_num(x_std @ transformation.T, nan=0.0)
        variances = np.var(x_transformed, axis=0)
        top_indices = np.argsort(variances)[-self.n_components :][::-1]
        return transformation, top_indices

    def fit_transform(self, x: np.ndarray, y=None) -> np.ndarray:
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-10
        x_std = np.nan_to_num((x - self.mean) / self.std, nan=0.0)
        self.transformation, self.selected_indices = self._build_transformation(x_std)
        x_transformed = np.nan_to_num(x_std @ self.transformation.T, nan=0.0)
        return x_transformed[:, self.selected_indices]

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_std = np.nan_to_num((x - self.mean) / self.std, nan=0.0)
        x_transformed = np.nan_to_num(x_std @ self.transformation.T, nan=0.0)
        return x_transformed[:, self.selected_indices]


# Archived 2025-06-14: QuantumInspiredMethod renamed to EntangleDRMethod.
# class QuantumInspiredMethod:
#     """Legacy name — fixed hyperparams (τ=0.7, I=3, R=π/36)."""
#     def __init__(self, n_components: int = 8):
#         self._delegate = EntangleDRMethod(n_components=n_components)
#         self.name = "Quantum-Inspired"
#     ...


class QuantumInspiredMethod(EntangleDRMethod):
    """Backward-compatible alias; displays as Quantum-Inspired in legacy UI."""

    def __init__(self, n_components: int = 8):
        super().__init__(
            n_components=n_components,
            correlation_threshold=0.7,
            iterations=3,
            rotation_scale=np.pi / 36,
        )
        self.name = "Quantum-Inspired"
