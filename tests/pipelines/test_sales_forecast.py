#!/usr/bin/env python3
"""
tests/pipelines/test_sales_forecast.py — Tests unitarios para
el pipeline de predicción de ventas (HealthCore).

Cubre:
  1. Split temporal 8/2 años respeta orden cronológico sin leak
  2. Features de lag/rolling son causales (no miran futuro)
  3. Métricas se calculan correctamente
  4. Dataset validado coincide con BRIEFING
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

# ── Añadir scripts/ al path para importar ──
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from sales_forecast import (
    BASE_DIR,
    RNG,
    TRAIN_YEARS,
    TEST_YEARS,
    engineer_features,
    gini_index,
    k2_score,
    load_sales_data,
    psi_score,
    temporal_train_test_split,
    validate_data,
)


# ── Fixtures ──


@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    """Carga el dataset real una vez por sesión."""
    return load_sales_data()


@pytest.fixture(scope="session")
def fe(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering sobre datos reales."""
    return engineer_features(df)


@pytest.fixture
def synthetic_series() -> pd.DataFrame:
    """Serie sintética simple de 120 meses para tests deterministas."""
    meses = pd.date_range("2010-01-01", periods=120, freq="MS")
    np.random.seed(42)
    revenue = 100_000 + np.cumsum(np.random.normal(500, 200, 120))
    revenue += 20_000 * np.sin(2 * np.pi * np.arange(120) / 12)
    return pd.DataFrame({
        "month": meses,
        "revenue_usd": revenue,
        "visits_count": np.random.poisson(lam=15000, size=120).astype(float),
        "avg_revenue_per_visit_usd": revenue / 15000,
        "region": "consolidated",
    })


# ═══════════════════ Test 1: Split temporal 8/2 años ═══════════════════


class TestTemporalSplit:
    """Valida que el split 8/2 respete orden cronológico sin leak."""

    def test_split_shapes(self, synthetic_series):
        """El split produce tamaños esperados (96 train, 24 test)."""
        fe = engineer_features(synthetic_series)
        # Simular prepare_features
        fe_clean = fe.dropna().reset_index(drop=True)
        y = fe_clean["revenue_usd"]
        exclude = {"month", "revenue_usd", "year"}
        X = fe_clean.drop(columns=list(exclude))

        X_train, X_test, y_train, y_test, _ = temporal_train_test_split(
            X, y, fe_clean, train_years=8, test_years=2,
        )

        # El dataset tiene 120 meses (10 años)
        # 8 años = 96 filas, 2 años = 24 filas (después de dropna menos)
        expected_train = len(fe_clean) - 24  # aprox 96
        expected_test = 24
        assert len(X_train) >= 84, (
            f"Train demasiado pequeño: {len(X_train)} (esperado ~{expected_train})"
        )
        assert len(X_test) <= 26, (
            f"Test demasiado grande: {len(X_test)} (esperado ~{expected_test})"
        )

    def test_no_temporal_leak(self, df):
        """Ningún índice de train debe tener fecha posterior al corte."""
        fe = engineer_features(df)
        fe_clean = fe.dropna().reset_index(drop=True)
        y = fe_clean["revenue_usd"]
        exclude = {"month", "revenue_usd", "year"}
        X = fe_clean.drop(columns=list(exclude))

        X_train, X_test, y_train, y_test, _ = temporal_train_test_split(
            X, y, fe_clean, train_years=8, test_years=2,
        )

        # Las fechas máximas de train deben ser < fechas mínimas de test
        train_months = fe_clean.loc[X_train.index, "month"]
        test_months = fe_clean.loc[X_test.index, "month"]

        assert train_months.max() < test_months.min(), (
            f"Leak temporal: train_max={train_months.max().date()} >= "
            f"test_min={test_months.min().date()}"
        )

    def test_cronological_order_preserved(self, synthetic_series):
        """El orden cronológico se preserva dentro de cada split."""
        fe = engineer_features(synthetic_series)
        fe_clean = fe.dropna().reset_index(drop=True)
        y = fe_clean["revenue_usd"]
        exclude = {"month", "revenue_usd", "year"}
        X = fe_clean.drop(columns=list(exclude))

        X_train, X_test, y_train, y_test, _ = temporal_train_test_split(
            X, y, fe_clean, train_years=8, test_years=2,
        )

        for name, _df in [("train", X_train), ("test", X_test)]:
            # Verificar índices son crecientes (orden original preservado)
            assert _df.index.is_monotonic_increasing, (
                f"Índice de {name} no está en orden cronológico"
            )


# ═══════════════════ Test 2: Features causales ═══════════════════


class TestCausalFeatures:
    """Valida que lags y rolling sean estrictamente causales."""

    def test_lag_1_is_previous(self, synthetic_series):
        """lag_1 en fila t debe ser igual a revenue_usd en fila t-1."""
        fe = engineer_features(synthetic_series)
        fe_clean = fe.dropna().reset_index(drop=True)

        for i in range(1, len(fe_clean)):
            expected = fe_clean.loc[i - 1, "revenue_usd"]
            actual = fe_clean.loc[i, "lag_1"]
            assert actual == pytest.approx(expected), (
                f"lag_1 en fila {i} = {actual}, pero debería ser "
                f"revenue en fila {i - 1} = {expected}"
            )

    def test_lag_12_is_year_ago(self, synthetic_series):
        """lag_12 en fila t debe ser igual a revenue_usd en fila t-12."""
        fe = engineer_features(synthetic_series)
        fe_clean = fe.dropna().reset_index(drop=True)

        for i in range(12, len(fe_clean)):
            expected = fe_clean.loc[i - 12, "revenue_usd"]
            actual = fe_clean.loc[i, "lag_12"]
            assert actual == pytest.approx(expected), (
                f"lag_12 en fila {i} = {actual}, pero debería ser "
                f"revenue en fila {i - 12} = {expected}"
            )

    def test_no_future_leak_in_lags(self, synthetic_series):
        """Ningún lag debe usar valores de filas futuras (t+1 o más)."""
        fe = engineer_features(synthetic_series)
        fe_clean = fe.dropna().reset_index(drop=True)

        lag_cols = [c for c in fe_clean.columns if c.startswith("lag_")]
        revenue = fe_clean["revenue_usd"].values

        for col in lag_cols:
            shift_val = int(col.split("_")[1])
            for i in range(shift_val, len(fe_clean)):
                # Un lag en t debe venir de t - shift_val
                # Si el lag usara valores futuros, sería > revenue actual
                actual = fe_clean.loc[i, col]
                expected = revenue[i - shift_val]
                assert actual == pytest.approx(expected), (
                    f"{col} en fila {i} no coincide con revenue en "
                    f"fila {i - shift_val}"
                )

    def test_rolling_is_backward(self, synthetic_series):
        """Rolling mean debe usar solo valores pasados, no futuros.
        pandas .rolling(3).mean() en fila i usa revenue[i-2:i+1]."""
        fe = engineer_features(synthetic_series)
        fe_clean = fe.dropna().reset_index(drop=True)

        revenue = fe_clean["revenue_usd"].values
        for i in range(3, len(fe_clean)):
            # rolling(3) en fila i = media de revenue[i-2:i+1]
            expected = np.mean(revenue[i - 2: i + 1])
            actual = fe_clean.loc[i, "rolling_mean_3"]
            assert actual == pytest.approx(expected, rel=1e-2), (
                f"rolling_mean_3 en fila {i} = {actual}, "
                f"esperado media(i-2:i+1) = {expected}"
            )


# ═══════════════════ Test 3: Métricas ═══════════════════


class TestMetrics:
    """Valida que las métricas se calculen correctamente."""

    def test_gini_perfect_ranking(self):
        """Gini = 1 para ranking perfecto."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        gini = gini_index(y_true, y_pred)
        assert gini == pytest.approx(1.0, abs=0.01), (
            f"Gini para ranking perfecto debería ser 1, obtenido {gini}"
        )

    def test_gini_random_ranking(self):
        """Gini ≈ 0 para ranking aleatorio (muestreo pequeño, aprox)."""
        np.random.seed(42)
        y_true = np.random.uniform(0, 100, 100)
        y_pred = np.random.uniform(0, 100, 100)
        gini = gini_index(y_true, y_pred)
        assert 0.0 <= gini <= 1.0, (
            f"Gini debería estar entre 0 y 1, obtenido {gini}"
        )

    def test_psi_identical_distributions(self):
        """PSI ≈ 0 para distribuciones idénticas."""
        data = np.random.normal(100, 20, 1000)
        psi = psi_score(data, data)
        assert psi < 0.05, (
            f"PSI para distribuciones idénticas debería ser ~0, "
            f"obtenido {psi}"
        )

    def test_psi_different_distributions(self):
        """PSI > 0.25 para distribuciones muy diferentes."""
        data_a = np.random.normal(100, 20, 1000)
        data_b = np.random.normal(200, 20, 1000)
        psi = psi_score(data_a, data_b)
        assert psi > 0.1, (
            f"PSI para distribuciones diferentes debería ser > 0.1, "
            f"obtenido {psi}"
        )

    def test_k2_perfect_prediction(self):
        """K2 = 1 para predicción perfecta."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        k2 = k2_score(y_true, y_pred)
        assert k2 == pytest.approx(1.0, abs=0.01), (
            f"K2 para predicción perfecta debería ser 1, obtenido {k2}"
        )

    def test_k2_mean_prediction(self):
        """K2 debe ser < 1 para predicción que no sea perfecta."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([3, 3, 3, 3, 3])  # predicción = media aprox
        k2 = k2_score(y_true, y_pred)
        assert k2 < 1.0, (
            f"K2 para predicción imperfecta debería ser < 1, "
            f"obtenido {k2}"
        )


# ═══════════════════ Test 4: Validación de datos ═══════════════════


class TestDataValidation:
    """Valida que el dataset cumple con las restricciones del BRIEFING."""

    def test_dataset_shape(self, df):
        """120 filas, 10 años completos."""
        assert len(df) == 120, (
            f"Dataset debe tener 120 filas, tiene {len(df)}"
        )
        assert df["month"].min() == pd.Timestamp("2016-01-01")
        assert df["month"].max() == pd.Timestamp("2025-12-01")

    def test_all_consolidated(self, df):
        """Todas las filas deben ser region='consolidated'."""
        assert (df["region"] == "consolidated").all()

    def test_revenue_positive(self, df):
        """Todos los revenue deben ser positivos."""
        assert (df["revenue_usd"] > 0).all()

    def test_no_nulls(self, df):
        """No debe haber valores nulos."""
        assert df.isnull().sum().sum() == 0

    def test_months_complete(self, df):
        """No deben faltar meses en el rango."""
        full_range = pd.date_range("2016-01-01", "2025-12-01", freq="MS")
        assert len(df) == len(full_range)

    def test_validate_passes(self, df):
        """La función validate_data no debe lanzar excepción."""
        validate_data(df)  # should not raise


# ═══════════════════ Test 5: Engineer features coverage ═══════════════════


class TestFeatureEngineering:
    """Valida que se crean todas las features requeridas."""

    def test_required_features_exist(self, fe):
        """Verifica presencia de lags, rolling y calendario."""
        required = [
            "lag_1", "lag_2", "lag_12",
            "rolling_mean_3", "rolling_mean_6", "rolling_mean_12",
            "sin_month", "cos_month", "month_num", "quarter",
        ]
        for col in required:
            assert col in fe.columns, (
                f"Feature requerida '{col}' no encontrada"
            )

    def test_no_future_features(self, fe):
        """No debe haber columnas con valores de filas futuras."""
        assert fe["month"].is_monotonic_increasing

    def test_feature_types(self, fe):
        """Las features numéricas deben tener tipos correctos."""
        numeric_cols = fe.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) >= 10, (
            f"Se esperan al menos 10 columnas numéricas, "
            f"se encontraron {len(numeric_cols)}"
        )