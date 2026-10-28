from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def dividir_random_estratificado(
    datos: pd.DataFrame,
    columna_objetivo: str,
    test_size: float = 0.2,
    semilla: int = 42,
):
    generador = np.random.default_rng(semilla)
    validation_indices = []

    for target_value in sorted(datos[columna_objetivo].unique()):
        # Cada clase aporta aproximadamente la misma proporcion al holdout.
        indices_objetivo = datos.index[datos[columna_objetivo] == target_value].to_numpy()
        shuffled = generador.permutation(indices_objetivo)
        n_validation = int(round(len(shuffled) * test_size))
        n_validation = min(max(n_validation, 1), len(shuffled) - 1)
        validation_indices.extend(shuffled[:n_validation].tolist())

    validation_indices = np.array(sorted(validation_indices))
    train_mask = ~datos.index.isin(validation_indices)
    validation_mask = datos.index.isin(validation_indices)

    return datos.loc[train_mask].copy(), datos.loc[validation_mask].copy()


def elegir_group_holdout_schools(
    datos: pd.DataFrame,
    columna_grupo: str = "escuela",
    columna_objetivo: str = "rendimiento_binario_num",
    n_validation_groups: int = 2,
    target_validation_share: float = 0.25,
    approval_gap_weight: float = 2.0,
):
    grupos = sorted(datos[columna_grupo].astype(str).unique())
    overall_rate = float(datos[columna_objetivo].mean())
    overall_n = len(datos)
    filas = []

    for candidate in itertools.combinations(grupos, n_validation_groups):
        # Buscamos un holdout que tenga tamano parecido al deseado y una tasa
        # de aprobacion no demasiado distinta del promedio general.
        mask = datos[columna_grupo].astype(str).isin(candidate)
        datos_validation = datos.loc[mask]
        validation_share = len(datos_validation) / overall_n
        approval_gap = abs(float(datos_validation[columna_objetivo].mean()) - overall_rate)
        size_gap = abs(validation_share - target_validation_share)
        puntaje = size_gap + approval_gap_weight * approval_gap

        filas.append(
            {
                "validation_groups": "-".join(candidate),
                "validation_share": validation_share,
                "validation_approval_rate": float(datos_validation[columna_objetivo].mean()),
                "approval_gap": approval_gap,
                "size_gap": size_gap,
                "score": puntaje,
            }
        )

    candidates_tabla = pd.DataFrame(filas).sort_values("score").reset_index(drop=True)
    best_groups = candidates_tabla.loc[0, "validation_groups"].split("-")
    return best_groups, candidates_tabla


def dividir_group_holdout(
    datos: pd.DataFrame,
    validation_groups,
    columna_grupo: str = "escuela",
):
    validation_groups = {str(valor) for valor in validation_groups}
    group_values = datos[columna_grupo].astype(str)
    validation_mask = group_values.isin(validation_groups)
    train_mask = ~validation_mask
    return datos.loc[train_mask].copy(), datos.loc[validation_mask].copy()


def dividir_holdout_temporal(
    datos: pd.DataFrame,
    time_col: str = "semestre",
):
    ordered_semesters = [
        valor
        for valor in datos[time_col].astype(str).unique().tolist()
        if valor != "nan"
    ]
    # Ordenamos los semestres como strings porque en el dataset vienen con formato estable.
    ordered_semesters = sorted(ordered_semesters)
    validation_semester = ordered_semesters[-1]
    train_mask = datos[time_col].astype(str) != validation_semester
    validation_mask = datos[time_col].astype(str) == validation_semester
    return (
        datos.loc[train_mask].copy(),
        datos.loc[validation_mask].copy(),
        validation_semester,
    )


def construir_indices_random_kfold(
    cantidad_muestras: int,
    n_folds: int = 5,
    semilla: int = 42,
):
    generador = np.random.default_rng(semilla)
    indices = generador.permutation(np.arange(cantidad_muestras))
    fold_indices = np.array_split(indices, n_folds)

    folds = []
    for fold_id in range(n_folds):
        # Cada fold toma una parte como validation y el resto como train.
        indices_validation = np.sort(fold_indices[fold_id])
        indices_entrenamiento = np.sort(
            np.concatenate(
                [fold_indices[idx] for idx in range(n_folds) if idx != fold_id]
            )
        )
        folds.append((indices_entrenamiento, indices_validation))

    return folds


def construir_indices_kfold_estratificado(
    y,
    n_folds: int = 4,
    semilla: int = 42,
):
    y = np.asarray(y)
    generador = np.random.default_rng(semilla)
    fold_buckets = [[] for _ in range(n_folds)]

    for target_value in sorted(np.unique(y)):
        # Repartimos cada clase por separado para que todos los folds queden
        # con una composicion parecida.
        indices_objetivo = np.where(y == target_value)[0]
        shuffled = generador.permutation(indices_objetivo)
        split_indices = np.array_split(shuffled, n_folds)

        for fold_id, fold_part in enumerate(split_indices):
            fold_buckets[fold_id].extend(fold_part.tolist())

    folds = []
    all_indices = np.arange(len(y))
    for fold_id in range(n_folds):
        indices_validation = np.array(sorted(fold_buckets[fold_id]))
        indices_entrenamiento = np.array(sorted(np.setdiff1d(all_indices, indices_validation)))
        folds.append((indices_entrenamiento, indices_validation))

    return folds


def construir_indices_group_kfold(
    grupos,
    n_folds: int | None = None,
):
    serie_grupo = pd.Series(grupos).astype(str).reset_index(drop=True)
    unique_groups = serie_grupo.unique().tolist()

    if n_folds is None:
        n_folds = max(2, len(unique_groups) // 2)

    group_sizes = serie_grupo.value_counts().sort_values(ascending=False)
    fold_groups = [[] for _ in range(n_folds)]
    fold_sizes = np.zeros(n_folds, dtype=int)

    for nombre_grupo, group_size in group_sizes.items():
        # Asignamos cada grupo al fold mas chico hasta el momento para evitar
        # folds extremadamente desbalanceados en cantidad de filas.
        fold_id = int(np.argmin(fold_sizes))
        fold_groups[fold_id].append(nombre_grupo)
        fold_sizes[fold_id] += int(group_size)

    folds = []
    all_indices = np.arange(len(serie_grupo))
    for groups_in_validation in fold_groups:
        validation_mask = serie_grupo.isin(groups_in_validation).to_numpy()
        indices_validation = np.where(validation_mask)[0]
        indices_entrenamiento = np.where(~validation_mask)[0]
        folds.append((indices_entrenamiento, indices_validation))

    return folds
