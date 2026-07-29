"""
Con labels.json (salida del visor HTML) + calibration_sample.csv (que ya trae
las features baratas, y para esa muestra chica tambien las de mascara real
si las calculaste), esto:

1. Ajusta un arbol de decision POCO profundo (max_depth=2 o 3) sobre las
   features -> da reglas legibles del tipo "si X < a y Y < b => no distinguible",
   que es literalmente el threshold que buscas, pero elegido por un criterio
   estadistico (impureza de Gini/entropia) y no por percentiles arbitrarios.
2. Valida con leave-one-scene-out: entrena excluyendo una escena, evalua en
   esa escena. Si el AUC/F1 y las reglas del arbol son estables entre folds,
   el umbral generaliza (esto es justo la robustez "independiente de la
   escena" que buscabas, y es una prueba, no una suposicion).
3. (Opcional) si en la muestra tambien tienes visible-pixels/occupancy-ratio
   reales (de mascara), reporta la correlacion con las features baratas
   (ang_width_deg, completeness_ratio analitico) para confirmar que el proxy
   barato es confiable antes de aplicarlo a los episodios largos sin mascara.
"""

import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, f1_score

FEATURES = ["ang-width-deg", "ang-height-deg", "expected-bbox-area",
            "min-side", "occupancy-ratio"]
# occupancy-ratio ya viene de la mascara real (visible-pixels/bbox-area). Si
# agregas completeness_ratio = visible-pixels/expected-bbox-area calculalo
# antes de esto y sumalo a la lista.


def load_data(manifest_csv, labels_json):
    """manifest_csv = calibration_manifest.csv, generado por build_labeling_set.py"""
    df = pd.read_csv(manifest_csv)
    with open(labels_json) as f:
        labels = json.load(f)
    df["item_id"] = df["item_id"].astype(str)
    lab_df = pd.DataFrame(labels).T.rename(columns={"label": "y"})
    lab_df.index.name = "item_id"
    lab_df = lab_df.reset_index()[["item_id", "y"]]
    lab_df["item_id"] = lab_df["item_id"].astype(str)
    df = df.merge(lab_df, on="item_id", how="inner")
    return df


def fit_tree(df, features=FEATURES, max_depth=3):
    X = df[features].values
    y = df["y"].astype(int).values
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=10,
                                  class_weight="balanced", random_state=42)
    clf.fit(X, y)
    print(export_text(clf, feature_names=features))
    return clf


def validate_leave_one_scene_out(df, features=FEATURES, scene_col="scene_id", max_depth=3):
    """
    Prueba de robustez real: para cada escena, entrena con las otras y
    evalua en la excluida. Si AUC/F1 se mantienen razonablemente parecidos
    entre folds, el umbral no depende de la escena particular.
    """
    if scene_col not in df.columns:
        print(f"Aviso: no encuentro columna '{scene_col}'; agregala al hacer merge de las 3 escenas.")
        return

    X = df[features].values
    y = df["y"].astype(int).values
    groups = df[scene_col].values

    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(X, y, groups):
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=5,
                                      class_weight="balanced", random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        pred = clf.predict(X[test_idx])
        held_out_scene = df[scene_col].values[test_idx][0]
        auc = roc_auc_score(y[test_idx], proba) if len(set(y[test_idx])) > 1 else float("nan")
        f1 = f1_score(y[test_idx], pred)
        print(f"[fold sin escena={held_out_scene}] AUC={auc:.3f} F1={f1:.3f} n_test={len(test_idx)}")


def check_cheap_vs_mask_agreement(df, cheap_col="ang-width-deg", mask_col="visible-pixels"):
    if mask_col not in df.columns:
        print(f"No hay columna '{mask_col}' en la muestra (mascara real) -- solo aplica a la calibracion.")
        return
    corr = df[[cheap_col, mask_col]].corr().iloc[0, 1]
    print(f"Correlacion {cheap_col} vs {mask_col} (real, con mascara): {corr:.3f}")


if __name__ == "__main__":
    import sys
    manifest_csv = sys.argv[1] if len(sys.argv) > 1 else "calibration_manifest.csv"
    labels_json = sys.argv[2] if len(sys.argv) > 2 else "labels.json"

    df = load_data(manifest_csv, labels_json)
    print(f"Muestra etiquetada: {len(df)} items, {df['y'].mean():.1%} marcados distinguibles\n")

    print("=== Arbol de decision (reglas / thresholds) ===")
    fit_tree(df)

    print("\n=== Validacion leave-one-scene-out ===")
    validate_leave_one_scene_out(df)

    print("\n=== Acuerdo feature barata vs mascara real (si aplica) ===")
    check_cheap_vs_mask_agreement(df)