#!/usr/bin/env python3
"""
test_temporal_cv.py — Prueba unitaria para validación cruzada temporal.

Verifica que TimeSeriesSplit preserve el orden cronológico en cada fold:
  - El último índice de train debe ser menor que el primer índice de val.
  - Los índices de train y val deben estar ordenados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit


def _make_dummy_data(n_samples: int = 200) -> tuple:
    """Crea datos sintéticos simples para la prueba."""
    X = np.random.default_rng(42).normal(size=(n_samples, 5))
    y = np.random.default_rng(42).normal(size=n_samples)
    return X, y


def test_temporal_cv_preserves_chronological_order():
    """Test base: cada fold debe mantener el orden cronológico."""
    X, y = _make_dummy_data(500)
    tscv = TimeSeriesSplit(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        # El último índice de entrenamiento debe ser menor que el primero de validación
        assert max(train_idx) < min(val_idx), (
            f"Fold {fold}: max(train_idx)={max(train_idx)} "
            f">= min(val_idx)={min(val_idx)} — orden cronológico violado"
        )
        # Los índices deben estar ordenados ascendentemente
        assert list(train_idx) == sorted(train_idx), (
            f"Fold {fold}: train_idx no está ordenado"
        )
        assert list(val_idx) == sorted(val_idx), (
            f"Fold {fold}: val_idx no está ordenado"
        )
        # No debe haber solapamiento
        assert len(set(train_idx) & set(val_idx)) == 0, (
            f"Fold {fold}: hay solapamiento entre train y val"
        )


def test_temporal_cv_no_shuffle():
    """Verifica que TimeSeriesSplit no mezcle (shuffle=False por defecto)."""
    X, y = _make_dummy_data(100)
    tscv = TimeSeriesSplit(n_splits=3)
    # Al ser cronológico, los primeros folds tienen menos datos de train
    prev_train_end = -1
    for train_idx, val_idx in tscv.split(X):
        assert max(train_idx) > prev_train_end, (
            "Los folds no son progresivos"
        )
        prev_train_end = max(train_idx)
        # Verificar que train está al inicio y val al final
        assert list(val_idx) == list(range(min(val_idx), max(val_idx) + 1))


def test_temporal_cv_overlap_forbidden():
    """Verifica que ningún fold tenga solapamiento train-val."""
    X, y = _make_dummy_data(300)
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, val_idx in tscv.split(X):
        assert len(set(train_idx) & set(val_idx)) == 0


def test_temporal_cv_with_pandas_index():
    """Verifica que funcione con índices de pandas no contiguos."""
    rng = np.random.default_rng(42)
    # Crear un DataFrame con índices no contiguos (simulando un subset)
    X_df = pd.DataFrame(rng.normal(size=(200, 5)))
    # Eliminar algunas filas aleatorias para romper la contigüidad
    drop_idx = rng.choice(X_df.index, size=20, replace=False)
    X_df = X_df.drop(drop_idx).reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=3)
    for train_idx, val_idx in tscv.split(X_df):
        assert max(train_idx) < min(val_idx)
        assert list(train_idx) == sorted(train_idx)
        assert list(val_idx) == sorted(val_idx)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])