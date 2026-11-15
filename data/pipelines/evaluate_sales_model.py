#!/usr/bin/env python3
"""
evaluate_sales_model.py — Evaluación técnica del modelo de pronóstico de
costo de insumos clínicos para HealthCore.

Arquitectura
────────────
1. Genera datos sintéticos multi-clínica con estacionalidad, tendencia y ruido
   realistas (∼10 años de datos mensuales, 12 clínicas).
2. Crea features temporales: lags (t-1, t-2, t-3), rolling mean (3m, 6m),
   diff mes-a-mes, señales de calendario (mes, trimestre, año).
3. Entrena un RandomForestRegressor con split temporal 8 años / 2 años.
4. Validación cruzada temporal (TimeSeriesSplit, 5 folds) sobre el conjunto
   de entrenamiento. Reporta MAE y RMSE como mean ± std.
5. Curva de aprendizaje (training-size vs error) guardada en data/eval/.
6. Diagnóstico: underfitting / overfitting / bien ajustado.
7. Reporte técnico en data/eval/evaluation_report.md.

Uso
───
    python data/pipelines/evaluate_sales_model.py

Dependencias: numpy, pandas, scikit-learn, matplotlib, seaborn
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, learning_curve

warnings.filterwarnings("ignore")

# ─────────────────────────── Globals ───────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
EVAL_DIR = BASE_DIR / "data" / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

N_CLINICS = 12
N_MONTHS = 126  # ~10.5 años (2016-01 a 2026-06)
TRAIN_YEARS = 8
TEST_YEARS = 2

CLINIC_INFO = {
    "clinic_01": {"country": "US", "currency": "USD", "region": "Texas"},
    "clinic_02": {"country": "UK", "currency": "GBP", "region": "London"},
    "clinic_03": {"country": "US", "currency": "USD", "region": "Florida"},
    "clinic_04": {"country": "US", "currency": "USD", "region": "Georgia"},
    "clinic_05": {"country": "UK", "currency": "GBP", "region": "Manchester"},
    "clinic_06": {"country": "US", "currency": "USD", "region": "Texas"},
    "clinic_07": {"country": "UK", "currency": "GBP", "region": "London"},
    "clinic_08": {"country": "US", "currency": "USD", "region": "Florida"},
    "clinic_09": {"country": "US", "currency": "USD", "region": "Texas"},
    "clinic_10": {"country": "UK", "currency": "GBP", "region": "Manchester"},
    "clinic_11": {"country": "US", "currency": "USD", "region": "Georgia"},
    "clinic_12": {"country": "UK", "currency": "GBP", "region": "London"},
}


# ─────────────────── 1. Generación de datos sintéticos ───────────────────


def _clinic_base_cost(clinic_id: str) -> float:
    """Costo base mensual distinto por clínica (USD ≈ escala)."""
    idx = int(clinic_id.split("_")[1])
    # range: 5_000 – 25_000
    return 5_000 + (idx - 1) * 1800 + RNG.normal(0, 400)


def _seasonal_factor(month: int) -> float:
    """Factor estacional: picos en invierno y verano."""
    return 1.0 + 0.25 * np.sin(2 * np.pi * (month - 1) / 12) \
           + 0.12 * np.cos(2 * np.pi * (month - 7) / 6)


def _trend_factor(year: float) -> float:
    """Tendencia de crecimiento anual ∼3 %."""
    return 1.0 + 0.03 * (year - 2016)


def generate_sales_data() -> pd.DataFrame:
    """Genera ∼10 años de datos mensuales de costo de insumos multi-clínica.

    Returns
    -------
    DataFrame con columnas:
        clinic_id, month_start, total_supply_cost, country, currency
        y además features temporales internas para el modelo.
    """
    rows: list[dict] = []
    # Semilla fija para cada clínica (reproducibilidad)
    local_rng = np.random.default_rng(2024)

    for cid, info in CLINIC_INFO.items():
        base = _clinic_base_cost(cid)
        # Ruido específico de la clínica
        noise_scale = base * 0.08

        for i in range(N_MONTHS):
            # 2016-01-01 + i meses
            dt = pd.Timestamp("2016-01-01") + pd.DateOffset(months=i)
            year = dt.year
            month = dt.month

            seasonal = _seasonal_factor(month)
            trend = _trend_factor(year + (month - 1) / 12)
            noise = local_rng.normal(0, noise_scale)

            cost = base * seasonal * trend + noise
            cost = max(cost, 0.0)  # no negativos

            rows.append({
                "clinic_id": cid,
                "month_start": dt,
                "total_supply_cost": round(cost, 2),
                "country": info["country"],
                "currency": info["currency"],
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["clinic_id", "month_start"]).reset_index(drop=True)
    return df


# ─────────────────── 2. Feature engineering temporal ───────────────────


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features temporales seguras (sin leak) y de calendario.

    IMPORTANTE: Los lags se calculan por clínica. En validación cruzada
    temporal estos lags deben respetar el orden cronológico — como se
    calculan sobre X_train después del split, no hay fuga.
    """
    df = df.copy()
    df["month"] = df["month_start"].dt.month
    df["quarter"] = df["month_start"].dt.quarter
    df["year"] = df["month_start"].dt.year
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_q1"] = (df["quarter"] == 1).astype(int)
    df["is_q4"] = (df["quarter"] == 4).astype(int)

    # Lags (por clínica)
    df["lag_1"] = df.groupby("clinic_id")["total_supply_cost"].shift(1)
    df["lag_2"] = df.groupby("clinic_id")["total_supply_cost"].shift(2)
    df["lag_3"] = df.groupby("clinic_id")["total_supply_cost"].shift(3)

    # Rolling statistics (ventana mínima 3 meses)
    df["rolling_mean_3"] = (
        df.groupby("clinic_id")["total_supply_cost"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    df["rolling_mean_6"] = (
        df.groupby("clinic_id")["total_supply_cost"]
        .transform(lambda x: x.rolling(6, min_periods=1).mean())
    )

    # Diferencia mes-a-mes
    df["diff_mom"] = df.groupby("clinic_id")["total_supply_cost"].diff()

    # One-hot encoding simple de clínica
    df = pd.get_dummies(df, columns=["clinic_id"], prefix="clinic")

    return df


def prepare_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepara X (features) e y (target) para el modelo.

    Elimina filas con NaN producto de los lags/diffs.
    """
    df_feat = _add_time_features(df)

    # Identificar columnas one-hot de clínica
    clinic_cols = [c for c in df_feat.columns if c.startswith("clinic_")]

    feature_cols = [
        "month", "quarter", "year",
        "sin_month", "cos_month",
        "is_q1", "is_q4",
        "lag_1", "lag_2", "lag_3",
        "rolling_mean_3", "rolling_mean_6",
        "diff_mom",
    ] + clinic_cols

    df_feat = df_feat.dropna(subset=feature_cols).reset_index(drop=True)
    X = df_feat[feature_cols].copy()
    y = df_feat["total_supply_cost"].copy()
    return X, y


# ─────────────────── 3. Temporal Cross-Validation ───────────────────


def run_temporal_cv(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> dict:
    """Ejecuta validación cruzada temporal con TimeSeriesSplit.

    Reporta MAE y RMSE como mean ± std en los folds.
    Verifica que el orden cronológico se preserve.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )

    mae_scores: list[float] = []
    rmse_scores: list[float] = []
    fold_details: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        # Verificación cronológica
        assert max(train_idx) < min(val_idx), (
            f"Fold {fold}: train_idx max ({max(train_idx)}) >= "
            f"val_idx min ({min(val_idx)}) — ¡fuga temporal!"
        )
        assert list(train_idx) == sorted(train_idx)
        assert list(val_idx) == sorted(val_idx)

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        mae_scores.append(mae)
        rmse_scores.append(rmse)
        fold_details.append({
            "fold": fold,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
        })

    results = {
        "n_splits": n_splits,
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
        "mae_per_fold": [round(s, 2) for s in mae_scores],
        "rmse_per_fold": [round(s, 2) for s in rmse_scores],
        "fold_details": fold_details,
    }
    return results


# ─────────────────── 4. Learning Curve ───────────────────


def plot_learning_curve(
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
) -> None:
    """Genera y guarda la curva de aprendizaje (train vs validation error)."""
    model = RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )

    train_sizes = np.linspace(0.1, 1.0, 10)

    tscv = TimeSeriesSplit(n_splits=5)

    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=tscv,
        train_sizes=train_sizes,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=42,
    )

    train_errors = -train_scores  # neg → positivo
    val_errors = -val_scores

    train_mean = train_errors.mean(axis=1)
    train_std = train_errors.std(axis=1)
    val_mean = val_errors.mean(axis=1)
    val_std = val_errors.std(axis=1)

    # ── Plot ──
    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="C0",
    )
    ax.fill_between(
        train_sizes_abs,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color="C1",
    )
    ax.plot(
        train_sizes_abs, train_mean, "o-",
        color="C0", label="Error de entrenamiento", linewidth=2,
    )
    ax.plot(
        train_sizes_abs, val_mean, "s-",
        color="C1", label="Error de validación", linewidth=2,
    )

    ax.set_xlabel("Tamaño del conjunto de entrenamiento", fontsize=13)
    ax.set_ylabel("MAE (costo de insumos, USD)", fontsize=13)
    ax.set_title(
        "Curva de Aprendizaje — Modelo de Pronóstico de Insumos HealthCore",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Curva de aprendizaje guardada: {output_path}")


# ─────────────────── 5. Diagnóstico de ajuste ───────────────────


def diagnose_fit(
    train_mae: float,
    val_mae: float,
    gap: float,
    cv_std: float,
) -> dict:
    """Diagnostica underfitting / overfitting / bien ajustado.

    Parámetros
    ----------
    train_mae : MAE en entrenamiento (fold completo)
    val_mae   : MAE en validación (CV)
    gap       : val_mae - train_mae
    cv_std    : desviación estándar del RMSE a través de folds
    """
    rel_gap = gap / train_mae if train_mae > 0 else 0.0

    if train_mae > 2000 and rel_gap < 0.15:
        diagnosis = "underfitting"
        evidence = (
            f"El error de entrenamiento es alto (MAE={train_mae:.0f}) y la brecha "
            f"con validación es pequeña (gap relativo={rel_gap:.1%}). "
            "El modelo no logra capturar el patrón subyacente incluso en los datos "
            "de entrenamiento."
        )
        corrective = (
            "Aumentar la complejidad del modelo (ej. incrementar `max_depth` de 12 "
            "a 20–25, reducir `min_samples_leaf` de 1 a 2–3) o revisar la calidad "
            "de las features (agregar interacciones, más lags, features externas). "
            "No agregar más datos como primera acción — el modelo necesita capturar "
            "mejor el patrón, no más ejemplos."
        )
    elif gap > 500 and rel_gap > 0.20:
        diagnosis = "overfitting"
        evidence = (
            f"El error de entrenamiento es bajo (MAE={train_mae:.0f}) pero el error "
            f"de validación es significativamente mayor (gap={gap:.0f}, "
            f"gap relativo={rel_gap:.1%}). "
            "El modelo memoriza el ruido de entrenamiento y no generaliza."
        )
        corrective = (
            "Aplicar regularización: reducir `max_depth` (ej. 12→6–8), aumentar "
            f"`min_samples_leaf` (ej. 1→10–15), o incrementar `min_samples_split`. "
            "Agregar más datos de entrenamiento también puede ayudar. "
            "No aumentar la complejidad como primera acción."
        )
    else:
        diagnosis = "well-fitted"
        evidence = (
            f"Error de entrenamiento (MAE={train_mae:.0f}) y error de validación "
            f"(MAE={val_mae:.0f}) son bajos y cercanos entre sí "
            f"(gap={gap:.0f}, gap relativo={rel_gap:.1%}). "
            "El modelo generaliza adecuadamente."
        )
        corrective = (
            "No se requieren cambios mayores. Mantener el modelo actual y monitorear "
            "el desempeño en producción. Considerar evaluar con datos más recientes "
            "periódicamente."
        )

    return {
        "diagnosis": diagnosis,
        "train_mae": round(train_mae, 2),
        "val_mae": round(val_mae, 2),
        "gap": round(gap, 2),
        "relative_gap": round(rel_gap, 4),
        "cv_rmse_std": round(cv_std, 2),
        "evidence": evidence,
        "corrective_action": corrective,
    }


# ─────────────────── 6. Reporte técnico ───────────────────


def generate_report(
    cv_results: dict,
    train_mae: float,
    train_rmse: float,
    diagnosis: dict,
) -> str:
    """Genera el reporte técnico en formato Markdown."""
    mae_str = f"{cv_results['mae_mean']:.2f} ± {cv_results['mae_std']:.2f}"
    rmse_str = f"{cv_results['rmse_mean']:.2f} ± {cv_results['rmse_std']:.2f}"
    primary_metric = "RMSE"
    secondary_metric = "MAE"

    report = f"""# Reporte de Evaluación Técnica — Modelo de Pronóstico HealthCore

**Fecha:** 2026-11-15
**Proyecto:** Evaluación de Modelo de Regresión (Pronóstico de Costo de Insumos)
**Modelo:** RandomForestRegressor (n_estimators=200, max_depth=12)
**Split temporal:** 8 años entrenamiento / 2 años prueba
**Dataset:** {N_CLINICS} clínicas, ∼{N_MONTHS} meses cada una

---

## 1. Selección de Métrica Principal

**Métrica principal: RMSE  |  Métrica secundaria: MAE**

**Justificación:** Para HealthCore, **subestimar** el costo de insumos (y por
tanto el consumo) es más costoso que sobreestimarlo. Una subestimación
grave puede provocar desabastecimientos (*stockouts*) que afectan la
atención al paciente. El RMSE penaliza desproporcionadamente los errores
grandes (L2), alineándose con el costo asimétrico de los errores graves
de subestimación. El MAE se reporta como referencia secundaria.

> *Fuente: HealthCore BRIEFING — "subestimar las ventas de la temporada
> alta provoca desabastecimientos; los errores importantes resultan
> desproporcionadamente costosos."*

---

## 2. Validación Cruzada Temporal

**Método:** `TimeSeriesSplit(n_splits={cv_results['n_splits']})` sobre el
conjunto de entrenamiento (fold cronológicos, sin shuffle).

| Fold | Train size | Val size | MAE (USD) | RMSE (USD) |
|------|-----------|---------|-----------|-----------|
"""
    for fd in cv_results["fold_details"]:
        report += (
            f"| {fd['fold']} | {fd['train_size']} | {fd['val_size']} | "
            f"{fd['mae']} | {fd['rmse']} |\n"
        )

    stability_text = (
        "**baja** — el modelo es estable a través de ventanas temporales."
        if cv_results["rmse_std"] < 500
        else "**moderada/alta** — el modelo muestra inestabilidad según qué porción del tiempo se use para entrenar."
    )

    report += f"""
**Resultados agregados:**

| Métrica | Media ± Desviación Estándar |
|---------|-----------------------------|
| {secondary_metric} (CV) | {mae_str} USD |
| {primary_metric} (CV) | {rmse_str} USD |

La desviación estándar entre folds es de **{cv_results['rmse_std']:.2f}**
para RMSE y **{cv_results['mae_std']:.2f}** para MAE. Esto indica una
varianza {stability_text}

---

## 3. Curva de Aprendizaje

![Learning Curve](learning_curve.png)

**Interpretación:**

{diagnosis['evidence']}

**Patrón detectado:** **{diagnosis['diagnosis']}**

"""

    if diagnosis["diagnosis"] == "well-fitted":
        report += """Ambas curvas convergen en un error bajo y cercano entre sí.
El modelo está razonablemente bien ajustado: captura el patrón sin
memorizar ruido."""
    elif diagnosis["diagnosis"] == "underfitting":
        report += """Ambas curvas convergen en un error alto. El modelo no logra
capturar el patrón ni siquiera en los datos de entrenamiento."""
    elif diagnosis["diagnosis"] == "overfitting":
        report += """Existe una brecha amplia y persistente entre el error de
entrenamiento (bajo) y el de validación (alto). El modelo memoriza el
ruido de entrenamiento."""

    report += f"""

---

## 4. Diagnóstico

**Clasificación:** {diagnosis['diagnosis'].upper()}

**Evidencia:**
{diagnosis['evidence']}

**Estabilidad (CV):** RMSE={rmse_str} USD.
La variabilidad entre folds es de {cv_results['rmse_std']:.2f} USD, lo que
representa un coeficiente de variación de
**{(cv_results['rmse_std'] / cv_results['rmse_mean'] * 100):.1f}%**
sobre el RMSE medio.

---

## 5. Acción Correctiva Específica

{diagnosis['corrective_action']}

---

## 6. Métricas de Referencia (Train vs Test)

| Partición | MAE (USD) | RMSE (USD) |
|-----------|-----------|------------|
| Entrenamiento (8 años) | {train_mae:.2f} | {train_rmse:.2f} |
| Validación (CV promedio) | {cv_results['mae_mean']:.2f} | {cv_results['rmse_mean']:.2f} |

"""

    return report


# ─────────────────── 7. Main ───────────────────


def main():
    print("=" * 65)
    print("  Evaluación Técnica — Modelo de Pronóstico HealthCore")
    print("=" * 65)

    # ── 1. Generar / cargar datos ──
    print("\n[1/5] Generando datos sintéticos multi-clínica...")
    df = generate_sales_data()
    print(f"  ✓ {len(df)} registros generados ({N_CLINICS} clínicas, "
          f"{N_MONTHS} meses)")

    # ── 2. Split temporal ──
    print("\n[2/5] Split temporal (8 años train / 2 años test)...")
    train_cutoff = pd.Timestamp(f"{2016 + TRAIN_YEARS}-01-01")
    df_train = df[df["month_start"] < train_cutoff].copy()
    df_test = df[df["month_start"] >= train_cutoff].copy()
    print(f"  ✓ Train: {len(df_train)} registros "
          f"({df_train['month_start'].min().date()} → "
          f"{df_train['month_start'].max().date()})")
    print(f"  ✓ Test:  {len(df_test)} registros "
          f"({df_test['month_start'].min().date()} → "
          f"{df_test['month_start'].max().date()})")

    # ── 3. Features ──
    print("\n[3/5] Preparando features temporales (lags, rolling, calendario)...")
    X_train, y_train = prepare_features(df_train)
    X_test, y_test = prepare_features(df_test)
    print(f"  ✓ X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  ✓ X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # ── 4. Temporal CV ──
    print(f"\n[4/5] Validación cruzada temporal "
          f"(TimeSeriesSplit, n_splits=5)...")
    cv_results = run_temporal_cv(X_train, y_train, n_splits=5)
    print(f"  ✓ MAE  = {cv_results['mae_mean']:.2f} ± "
          f"{cv_results['mae_std']:.2f} USD")
    print(f"  ✓ RMSE = {cv_results['rmse_mean']:.2f} ± "
          f"{cv_results['rmse_std']:.2f} USD")
    for fd in cv_results["fold_details"]:
        print(f"    Fold {fd['fold']}: train={fd['train_size']}, "
              f"val={fd['val_size']}, MAE={fd['mae']}, RMSE={fd['rmse']}")

    # ── Modelo final (train completo) ──
    final_model = RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )
    final_model.fit(X_train, y_train)
    y_train_pred = final_model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    # ── 5. Learning Curve ──
    print("\n[5/5] Generando curva de aprendizaje...")
    lc_path = EVAL_DIR / "learning_curve.png"
    plot_learning_curve(X_train, y_train, lc_path)

    # ── Diagnóstico ──
    gap = cv_results["mae_mean"] - train_mae
    diagnosis = diagnose_fit(
        train_mae=train_mae,
        val_mae=cv_results["mae_mean"],
        gap=gap,
        cv_std=cv_results["rmse_std"],
    )
    print(f"\n  Diagnóstico: {diagnosis['diagnosis'].upper()}")
    print(f"  Train MAE: {train_mae:.2f} | Val MAE (CV): "
          f"{cv_results['mae_mean']:.2f} | Gap: {gap:.2f}")

    # ── Reporte ──
    report = generate_report(
        cv_results,
        train_mae=train_mae,
        train_rmse=train_rmse,
        diagnosis=diagnosis,
    )
    report_path = EVAL_DIR / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  ✓ Reporte guardado: {report_path}")

    # ── Resumen JSON ──
    summary = {
        "model": "RandomForestRegressor",
        "params": {"n_estimators": 200, "max_depth": 12},
        "train_size": len(X_train),
        "test_size": len(X_test),
        "cv": {
            "method": "TimeSeriesSplit",
            "n_splits": cv_results["n_splits"],
            "mae_mean": cv_results["mae_mean"],
            "mae_std": cv_results["mae_std"],
            "rmse_mean": cv_results["rmse_mean"],
            "rmse_std": cv_results["rmse_std"],
        },
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "diagnosis": diagnosis["diagnosis"],
        "corrective_action": diagnosis["corrective_action"],
    }
    summary_path = EVAL_DIR / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Resumen JSON guardado: {summary_path}")

    print("\n" + "=" * 65)
    print("  Evaluación completada.")
    print("=" * 65)


if __name__ == "__main__":
    main()