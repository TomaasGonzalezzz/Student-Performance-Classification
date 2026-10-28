from __future__ import annotations

import numpy as np


class RegresionLogisticaBinariaL2:
    """Regresion logistica binaria con descenso por gradiente por lotes.

    La funcion objetivo sigue la convencion usada en la materia:
        mean(log-loss) + (lambda / 2) * ||w||^2

    La penalizacion L2 actua solo sobre los coeficientes. El intercepto queda
    libre para que el modelo pueda acomodar la probabilidad base.
    """

    def __init__(
        self,
        lambda_: float = 1.0,
        learning_rate: float = 0.05,
        max_iter: int = 4000,
        tolerance: float = 1e-6,
    ):
        self.lambda_ = float(lambda_)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.intercept_ = 0.0
        self.coef_ = None
        self.loss_history_ = []
        self.n_iter_ = 0

    @staticmethod
    def sigmoid(valores: np.ndarray) -> np.ndarray:
        clipped = np.clip(valores, -500, 500)
        return 1.0 / (1.0 + np.exp(-clipped))

    def calcular_perdida(
        self,
        X: np.ndarray,
        y: np.ndarray,
        pesos_muestra: np.ndarray,
        intercept: float,
        coef: np.ndarray,
    ) -> float:
        logits = intercept + X @ coef
        probabilities = self.sigmoid(logits)
        eps = 1e-12
        weighted_log_terms = pesos_muestra * (
            y * np.log(probabilities + eps)
            + (1 - y) * np.log(1 - probabilities + eps)
        )
        log_loss = -np.mean(weighted_log_terms)
        penalty = 0.5 * self.lambda_ * np.sum(coef ** 2)
        return float(log_loss + penalty)

    def fit(self, X, y, pesos_muestra=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if pesos_muestra is None:
            pesos_muestra = np.ones_like(y, dtype=float)
        else:
            pesos_muestra = np.asarray(pesos_muestra, dtype=float)

        cantidad_muestras, n_features = X.shape
        intercept = 0.0
        coef = np.zeros(n_features, dtype=float)
        previous_loss = None
        self.loss_history_ = []

        for iteration in range(self.max_iter):
            logits = intercept + X @ coef
            probabilities = self.sigmoid(logits)

            weighted_errors = pesos_muestra * (probabilities - y)

            intercept_gradient = weighted_errors.mean()
            coef_gradient = (X.T @ weighted_errors) / cantidad_muestras
            coef_gradient += self.lambda_ * coef

            intercept -= self.learning_rate * intercept_gradient
            coef -= self.learning_rate * coef_gradient

            current_loss = self.calcular_perdida(X, y, pesos_muestra, intercept, coef)
            self.loss_history_.append(current_loss)

            if previous_loss is not None:
                if abs(previous_loss - current_loss) < self.tolerance:
                    self.n_iter_ = iteration + 1
                    break

            previous_loss = current_loss
            self.n_iter_ = iteration + 1

        self.intercept_ = float(intercept)
        self.coef_ = coef
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        logits = self.intercept_ + X @ self.coef_
        return self.sigmoid(logits)

    def predict(self, X, umbral: float = 0.5) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= umbral).astype(int)


class RegresionLogisticaMulticlaseL2:
    """Regresion logistica multiclase con salida softmax.

    Usa la misma idea de la version binaria, pero ahora calcula una
    probabilidad para cada clase:
        mean(cross-entropy) + (lambda / 2) * ||W||^2

    El intercepto no se regulariza.
    """

    def __init__(
        self,
        lambda_: float = 1.0,
        learning_rate: float = 0.05,
        max_iter: int = 4000,
        tolerance: float = 1e-6,
    ):
        self.lambda_ = float(lambda_)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.classes_ = None
        self.intercept_ = None
        self.coef_ = None
        self.loss_history_ = []
        self.n_iter_ = 0

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def calcular_perdida(
        self,
        X: np.ndarray,
        y_one_hot: np.ndarray,
        intercept: np.ndarray,
        coef: np.ndarray,
    ) -> float:
        logits = X @ coef + intercept
        probabilities = self.softmax(logits)
        eps = 1e-12
        cross_entropy = -np.mean(np.sum(y_one_hot * np.log(probabilities + eps), axis=1))
        penalty = 0.5 * self.lambda_ * np.sum(coef ** 2)
        return float(cross_entropy + penalty)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.classes_ = np.array(sorted(np.unique(y)))
        cantidad_muestras, n_features = X.shape
        cantidad_clases = len(self.classes_)

        class_to_index = {class_value: idx for idx, class_value in enumerate(self.classes_)}
        y_indices = np.array([class_to_index[valor] for valor in y], dtype=int)
        y_one_hot = np.eye(cantidad_clases, dtype=float)[y_indices]

        intercept = np.zeros(cantidad_clases, dtype=float)
        coef = np.zeros((n_features, cantidad_clases), dtype=float)
        previous_loss = None
        self.loss_history_ = []

        for iteration in range(self.max_iter):
            logits = X @ coef + intercept
            probabilities = self.softmax(logits)

            errors = probabilities - y_one_hot

            intercept_gradient = errors.mean(axis=0)
            coef_gradient = (X.T @ errors) / cantidad_muestras
            coef_gradient += self.lambda_ * coef

            intercept -= self.learning_rate * intercept_gradient
            coef -= self.learning_rate * coef_gradient

            current_loss = self.calcular_perdida(X, y_one_hot, intercept, coef)
            self.loss_history_.append(current_loss)

            if previous_loss is not None:
                if abs(previous_loss - current_loss) < self.tolerance:
                    self.n_iter_ = iteration + 1
                    break

            previous_loss = current_loss
            self.n_iter_ = iteration + 1

        self.intercept_ = intercept
        self.coef_ = coef
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        logits = X @ self.coef_ + self.intercept_
        return self.softmax(logits)

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes_[predicted_indices]


class AnalisisDiscriminanteLinealCovarianzaCompartida:
    """LDA multiclase simple con una covarianza compartida.

    El modelo estima:
    - una media por clase;
    - una covarianza global compartida;
    - priors segun la frecuencia observada de cada clase.

    Se suma un ridge diagonal chico para estabilizar la inversion de la
    covarianza cuando hay colinealidad o escalas parecidas.
    """

    def __init__(self, covariance_ridge: float = 1e-4):
        self.covariance_ridge = float(covariance_ridge)
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariance_ = None
        self.inv_covariance_ = None
        self.linear_coef_ = None
        self.intercept_ = None

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.classes_ = np.array(sorted(np.unique(y)))
        cantidad_muestras, n_features = X.shape
        cantidad_clases = len(self.classes_)

        means = np.zeros((cantidad_clases, n_features), dtype=float)
        priors = np.zeros(cantidad_clases, dtype=float)
        covariance_sum = np.zeros((n_features, n_features), dtype=float)

        for indice_clase, class_value in enumerate(self.classes_):
            class_mask = y == class_value
            X_class = X[class_mask]
            means[indice_clase] = X_class.mean(axis=0)
            priors[indice_clase] = X_class.shape[0] / cantidad_muestras
            centered = X_class - means[indice_clase]
            covariance_sum += centered.T @ centered

        pooled_covariance = covariance_sum / max(cantidad_muestras - cantidad_clases, 1)
        pooled_covariance += self.covariance_ridge * np.eye(n_features)
        inv_covariance = np.linalg.pinv(pooled_covariance)

        linear_coef = means @ inv_covariance
        intercept = (
            -0.5 * np.sum(linear_coef * means, axis=1)
            + np.log(priors + 1e-12)
        )

        self.priors_ = priors
        self.means_ = means
        self.covariance_ = pooled_covariance
        self.inv_covariance_ = inv_covariance
        self.linear_coef_ = linear_coef
        self.intercept_ = intercept
        return self

    def decision_function(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.linear_coef_.T + self.intercept_

    def predict_proba(self, X) -> np.ndarray:
        puntajes = self.decision_function(X)
        return self.softmax(puntajes)

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes_[predicted_indices]

def entropia_desde_conteos(class_counts: np.ndarray) -> float:
    total = int(np.sum(class_counts))
    if total == 0:
        return 0.0

    probabilities = class_counts[class_counts > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


class NodoArbolDecision:

    def __init__(self, predicted_class: int, class_probabilities: np.ndarray):
        self.predicted_class = int(predicted_class)
        self.class_probabilities = class_probabilities
        self.feature_index = None
        self.umbral = None
        self.left = None
        self.right = None


class ClasificadorArbolDecisionEntropia:
    """Arbol de decision multiclase que divide con entropia.
    Se usa dentro del Random Forest y, por eso, asume que 
    las clases ya estan codificadas como enteros.
    """

    def __init__(
        self,
        cantidad_clases: int,
        max_depth: int | None = None,
        min_samples_split: int = 20,
        min_samples_leaf: int = 5,
        max_features: str | int | float = "sqrt",
        semilla: int = 42,
    ):
        self.cantidad_clases = int(cantidad_clases)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.semilla = int(semilla)
        self.root_ = None
        self.n_features_ = 0
        self.feature_importances_ = None
        self._rng = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_, dtype=float)
        self._rng = np.random.default_rng(self.semilla)
        self.root_ = self.crecer_arbol(X, y, depth=0)

        total_importance = float(self.feature_importances_.sum())
        if total_importance > 0:
            self.feature_importances_ = self.feature_importances_ / total_importance

        return self

    def resolver_n_features(self) -> int:
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(self.n_features_)))
        if self.max_features == "all":
            return self.n_features_
        if isinstance(self.max_features, float):
            return max(1, int(np.ceil(self.max_features * self.n_features_)))
        return max(1, min(int(self.max_features), self.n_features_))

    def muestrear_indices_features(self) -> np.ndarray:
        n_selected = self.resolver_n_features()
        return self._rng.choice(self.n_features_, size=n_selected, replace=False)

    def crecer_arbol(self, X: np.ndarray, y: np.ndarray, depth: int):
        class_counts = np.bincount(y, minlength=self.cantidad_clases)
        predicted_class = int(np.argmax(class_counts))
        class_probabilities = class_counts / max(class_counts.sum(), 1)
        node = NodoArbolDecision(predicted_class, class_probabilities)

        pure_node = np.count_nonzero(class_counts) == 1
        depth_limit_reached = (
            self.max_depth is not None and depth >= self.max_depth
        )
        too_small_to_split = len(y) < self.min_samples_split
        not_enough_samples_for_children = len(y) < 2 * self.min_samples_leaf

        if pure_node or depth_limit_reached or too_small_to_split:
            return node
        if not_enough_samples_for_children:
            return node

        feature_indices = self.muestrear_indices_features()
        best_feature, best_threshold, best_gain = self.best_split(
            X,
            y,
            feature_indices,
        )

        if best_feature is None or best_gain <= 0:
            return node

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if left_mask.sum() < self.min_samples_leaf:
            return node
        if right_mask.sum() < self.min_samples_leaf:
            return node

        self.feature_importances_[best_feature] += best_gain * len(y)
        node.feature_index = int(best_feature)
        node.umbral = float(best_threshold)
        node.left = self.crecer_arbol(X[left_mask], y[left_mask], depth + 1)
        node.right = self.crecer_arbol(X[right_mask], y[right_mask], depth + 1)
        return node

    def best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray,
    ):
        total_counts = np.bincount(y, minlength=self.cantidad_clases)
        parent_entropy = entropia_desde_conteos(total_counts)
        cantidad_muestras = len(y)

        best_feature = None
        best_threshold = None
        best_gain = 0.0

        for feature_idx in feature_indices:
            valores = X[:, feature_idx]
            order = np.argsort(valores, kind="mergesort")
            sorted_values = valores[order]
            sorted_y = y[order]

            left_counts = np.zeros(self.cantidad_clases, dtype=int)
            right_counts = total_counts.copy()

            for split_idx in range(cantidad_muestras - 1):
                indice_clase = int(sorted_y[split_idx])
                left_counts[indice_clase] += 1
                right_counts[indice_clase] -= 1

                if sorted_values[split_idx] == sorted_values[split_idx + 1]:
                    continue

                n_left = split_idx + 1
                n_right = cantidad_muestras - n_left
                if n_left < self.min_samples_leaf:
                    continue
                if n_right < self.min_samples_leaf:
                    continue

                left_entropy = entropia_desde_conteos(left_counts)
                right_entropy = entropia_desde_conteos(right_counts)
                weighted_entropy = (
                    (n_left / cantidad_muestras) * left_entropy
                    + (n_right / cantidad_muestras) * right_entropy
                )
                information_gain = parent_entropy - weighted_entropy

                if information_gain > best_gain:
                    umbral = 0.5 * (
                        sorted_values[split_idx] + sorted_values[split_idx + 1]
                    )
                    best_feature = int(feature_idx)
                    best_threshold = float(umbral)
                    best_gain = float(information_gain)

        return best_feature, best_threshold, best_gain

    def predecir_probabilidades_fila(self, fila: np.ndarray) -> np.ndarray:
        node = self.root_
        while node.left is not None and node.right is not None:
            if fila[node.feature_index] <= node.umbral:
                node = node.left
            else:
                node = node.right
        return node.class_probabilities

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        probabilities = np.zeros((X.shape[0], self.cantidad_clases), dtype=float)

        for row_idx in range(X.shape[0]):
            probabilities[row_idx] = self.predecir_probabilidades_fila(X[row_idx])

        return probabilities

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


class ClasificadorRandomForestEntropia:

    def __init__(
        self,
        n_estimators: int = 30,
        max_depth: int | None = 6,
        min_samples_split: int = 20,
        min_samples_leaf: int = 5,
        max_features: str | int | float = "sqrt",
        bootstrap: bool = True,
        semilla: int = 42,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.semilla = int(semilla)
        self.classes_ = None
        self.trees_ = []
        self.feature_importances_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.classes_ = np.array(sorted(np.unique(y)))
        class_to_index = {class_value: idx for idx, class_value in enumerate(self.classes_)}
        y_indices = np.array([class_to_index[valor] for valor in y], dtype=int)

        generador = np.random.default_rng(self.semilla)
        cantidad_muestras, n_features = X.shape
        cantidad_clases = len(self.classes_)
        self.trees_ = []
        self.feature_importances_ = np.zeros(n_features, dtype=float)

        for _ in range(self.n_estimators):
            if self.bootstrap:
                sample_idx = generador.choice(cantidad_muestras, size=cantidad_muestras, replace=True)
            else:
                sample_idx = np.arange(cantidad_muestras)

            tree = ClasificadorArbolDecisionEntropia(
                cantidad_clases=cantidad_clases,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                semilla=int(generador.integers(0, 1_000_000)),
            )
            tree.fit(X[sample_idx], y_indices[sample_idx])
            self.trees_.append(tree)
            self.feature_importances_ += tree.feature_importances_

        total_importance = float(self.feature_importances_.sum())
        if total_importance > 0:
            self.feature_importances_ = self.feature_importances_ / total_importance

        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        tree_probabilities = [tree.predict_proba(X) for tree in self.trees_]
        stacked = np.stack(tree_probabilities, axis=0)
        return stacked.mean(axis=0)

    def predict(self, X) -> np.ndarray:
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes_[predicted_indices]
