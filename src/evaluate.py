import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# FIX: Import both AttentionLayer AND focal_loss from train.py.
# Both are custom objects decorated with @register_keras_serializable().
# If either is missing at load time, Keras throws "Could not locate function".
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import AttentionLayer, focal_loss  # noqa: F401

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
MODELS_DIR    = os.path.join(BASE_DIR, 'models')


def evaluate_pipeline():

    # ── 1. Load data + model ──────────────────────────────────────────────────
    print("1. Loading test data and model...")

    X_seq_test     = np.load(os.path.join(PROCESSED_DIR, 'X_seq_test.npy'),     allow_pickle=True).astype(np.float32)
    X_static_test  = np.load(os.path.join(PROCESSED_DIR, 'X_static_test.npy'),  allow_pickle=True).astype(np.float32)
    y_test         = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'),          allow_pickle=True).astype(np.int32)
    X_seq_train    = np.load(os.path.join(PROCESSED_DIR, 'X_seq_train.npy'),     allow_pickle=True).astype(np.float32)
    X_static_train = np.load(os.path.join(PROCESSED_DIR, 'X_static_train.npy'),  allow_pickle=True).astype(np.float32)

    home_test  = np.load(os.path.join(PROCESSED_DIR, 'home_team_test.npy')).reshape(-1)
    away_test  = np.load(os.path.join(PROCESSED_DIR, 'away_team_test.npy')).reshape(-1)
    home_train = np.load(os.path.join(PROCESSED_DIR, 'home_team_train.npy')).reshape(-1)
    away_train = np.load(os.path.join(PROCESSED_DIR, 'away_team_train.npy')).reshape(-1)

    keras_path = os.path.join(MODELS_DIR, 'best_model.keras')
    h5_path    = os.path.join(MODELS_DIR, 'best_model.h5')

    if os.path.exists(keras_path):
        model_path = keras_path
    elif os.path.exists(h5_path):
        model_path = h5_path
    else:
        raise FileNotFoundError(
            f"No saved model found. Expected:\n  {keras_path}\n  {h5_path}"
        )

    print(f"   Loading from: {model_path}")
    # FIX: Pass focal_loss in custom_objects so Keras can resolve it on load.
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'AttentionLayer': AttentionLayer,
            'focal_loss'    : focal_loss,
        }
    )

    # ── 2. Predictions ────────────────────────────────────────────────────────
    print("2. Running predictions on test set...")
    y_pred_proba = model.predict(
        {
            'lstm_input': X_seq_test, 
            'static_input': X_static_test,
            'home_team': home_test,
            'away_team': away_test
        },
        verbose=1
    )
    print(f"   Prediction output shape: {y_pred_proba.shape}")
    y_pred = np.argmax(y_pred_proba, axis=1)

    print("Class prediction distribution:")
    print(np.bincount(y_pred))

    # ── 3. Metrics ────────────────────────────────────────────────────────────
    print("3. Computing evaluation metrics...")
    acc = accuracy_score(y_test, y_pred)

    precision_mac, recall_mac, f1_mac, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    precision_wt, recall_wt, f1_wt, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    cm     = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred,
        target_names=['Away Win', 'Draw', 'Home Win'],
        output_dict=True,
        zero_division=0
    )

    class_avg_proba = {
        'away_win': float(np.mean(y_pred_proba[:, 0])),
        'draw'    : float(np.mean(y_pred_proba[:, 1])),
        'home_win': float(np.mean(y_pred_proba[:, 2])),
    }

    metrics = {
        'accuracy'             : float(acc),
        'macro'                : {
            'precision': float(precision_mac),
            'recall'   : float(recall_mac),
            'f1'       : float(f1_mac)
        },
        'weighted'             : {
            'precision': float(precision_wt),
            'recall'   : float(recall_wt),
            'f1'       : float(f1_wt)
        },
        'confusion_matrix'     : cm,
        'classification_report': report,
        'class_avg_proba'      : class_avg_proba,
        'n_test_samples'       : int(len(y_test)),
        'class_distribution'   : {
            'away_win': int(np.sum(y_test == 0)),
            'draw'    : int(np.sum(y_test == 1)),
            'home_win': int(np.sum(y_test == 2)),
        }
    }

    metrics_path = os.path.join(MODELS_DIR, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"   Saved → {metrics_path}")

    # ── 4. Permutation Feature Importance ─────────────────────────────────────
    print("4. Computing permutation feature importance...")
    n_seq_feat  = X_seq_test.shape[2]
    n_stat_feat = X_static_test.shape[1]
    feature_importance = []
    np.random.seed(42)

    print(f"   Shuffling {n_seq_feat} sequence features...")
    for i in range(n_seq_feat):
        X_seq_shuf = X_seq_test.copy()
        idx = np.random.permutation(len(X_seq_shuf))
        X_seq_shuf[:, :, i] = X_seq_shuf[idx, :, i]

        preds_shuf = model.predict(
            {
                'lstm_input': X_seq_shuf, 
                'static_input': X_static_test,
                'home_team': home_test,
                'away_team': away_test
            },
            verbose=0
        )
        acc_shuf = accuracy_score(y_test, np.argmax(preds_shuf, axis=1))
        feature_importance.append({
            'feature_index': int(i),
            'feature_name' : f'seq_feature_{i}',
            'type'         : 'sequence',
            'importance'   : float(acc - acc_shuf)
        })

    print(f"   Shuffling {n_stat_feat} static features...")
    for i in range(n_stat_feat):
        X_stat_shuf = X_static_test.copy()
        idx = np.random.permutation(len(X_stat_shuf))
        X_stat_shuf[:, i] = X_stat_shuf[idx, i]

        preds_shuf = model.predict(
            {
                'lstm_input': X_seq_test, 
                'static_input': X_stat_shuf,
                'home_team': home_test,
                'away_team': away_test
            },
            verbose=0
        )
        acc_shuf = accuracy_score(y_test, np.argmax(preds_shuf, axis=1))
        feature_importance.append({
            'feature_index': int(i + n_seq_feat),
            'feature_name' : f'static_feature_{i}',
            'type'         : 'static',
            'importance'   : float(acc - acc_shuf)
        })

    feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    top20 = feature_importance[:20]

    fi_path = os.path.join(MODELS_DIR, 'feature_importance.json')
    with open(fi_path, 'w') as f:
        json.dump(top20, f, indent=2)
    print(f"   Saved → {fi_path}")

    # ── 5. SHAP values ────────────────────────────────────────────────────────
    print("5. Computing SHAP values (may take a minute)...")
    try:
        import shap

        background  = [X_seq_train[:100],  X_static_train[:100], home_train[:100], away_train[:100]]
        test_sample = [X_seq_test[:10],    X_static_test[:10], home_test[:10], away_test[:10]]

        explainer   = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(test_sample)

        mean_shap = {}
        for c in range(3):
            seq_shap  = shap_values[c][0]
            stat_shap = shap_values[c][1]

            mean_shap[f'class_{c}'] = {
                'label'            : ['Away Win', 'Draw', 'Home Win'][c],
                'sequence_features': np.mean(np.abs(seq_shap), axis=(0, 1)).tolist(),
                'static_features'  : np.mean(np.abs(stat_shap), axis=0).tolist()
            }

        shap_path = os.path.join(MODELS_DIR, 'shap_values.json')
        with open(shap_path, 'w') as f:
            json.dump(mean_shap, f, indent=2)
        print(f"   Saved → {shap_path}")

    except Exception as e:
        print(f"   SHAP failed (common with custom TF layers): {e}")
        print("   Saving empty shap_values.json so Flask API doesn't crash.")
        with open(os.path.join(MODELS_DIR, 'shap_values.json'), 'w') as f:
            json.dump({'error': str(e)}, f, indent=2)

    # ── 6. Training history CSV → JSON ────────────────────────────────────────
    print("6. Converting training history CSV → JSON...")
    try:
        history_df   = pd.read_csv(os.path.join(MODELS_DIR, 'training_history.csv'))
        history_dict = history_df.to_dict(orient='records')

        hist_path = os.path.join(MODELS_DIR, 'training_history.json')
        with open(hist_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        print(f"   Saved → {hist_path}  ({len(history_dict)} epochs)")
    except Exception as e:
        print(f"   Could not load training_history.csv: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  EVALUATION COMPLETE")
    print("=" * 50)
    print(f"  Test accuracy  : {acc * 100:.2f}%")
    print(f"  Macro F1       : {f1_mac:.4f}")
    print(f"  Macro Precision: {precision_mac:.4f}")
    print(f"  Macro Recall   : {recall_mac:.4f}")
    print(f"\n  Confusion matrix (rows=Actual, cols=Predicted):")
    print(f"                Away  Draw  Home")
    labels = ['Away', 'Draw', 'Home']
    for i, row in enumerate(cm):
        print(f"  Actual {labels[i]:4s} :  {str(row)}")
    print("=" * 50)
    print("\nNext step → python src/app.py")


if __name__ == "__main__":
    evaluate_pipeline()