# experiments/exp_018_evaluate_and_submit_threshold.py

"""
Experiment 18: Load trained models, evaluate threshold models, submit.
No training. Reuses everything from exp_017.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score

from src.preprocessing import DataSplitter
from src.submission import Submitter


# ====================== CONFIG ======================

MODELS_DIR = 'models'
THRESHOLD_DIR = 'data/processed/threshold_data'


def full_metrics(y_true, pred):
    fpr, tpr, _ = roc_curve(y_true, pred)
    J = (tpr - fpr).max()
    return {
        'auc_binary': (1 + J) / 2,
        'acc': accuracy_score(y_true, pred),
        'f1': f1_score(y_true, pred),
        'prec': precision_score(y_true, pred),
        'rec': recall_score(y_true, pred),
    }


# ====================== 1. LOAD DATA ======================

print("=" * 85)
print("EXP 018: Evaluate threshold models and submit (no training)")
print("=" * 85)

splitter = DataSplitter(
    data_dir='data/raw',
    processed_dir='data/processed',
    plot_dir='src/plots',
    seed=43
)
splitter.run()

X_test_sc = splitter.X_test_sc
y_test = splitter.y_test
X_kaggle_sc = splitter.scaler.transform(splitter.X_kaggle_test)
test_ids = splitter.test_ids

print(f"\n  Meta-val: {X_test_sc.shape}  y_test: {y_test.shape}")
print(f"  Kaggle:   {X_kaggle_sc.shape}  test_ids: {test_ids.shape}")


# ====================== 2. LOAD FULL ENSEMBLE ======================

print(f"\n{'=' * 85}")
print("  Loading full ensemble")
print(f"{'=' * 85}")

router_bundle = joblib.load(os.path.join(MODELS_DIR, 'lr_router.joblib'))
meta_lr = router_bundle['router']
model_names = router_bundle['model_names']
print(f"  Router loaded: {len(model_names)} models")

# predict on meta-val
val_score_list = []
for name in model_names:
    t0 = time.time()
    safe_name = name.lower().replace(' ', '_')
    bundle = joblib.load(os.path.join(MODELS_DIR, f'{safe_name}.joblib'))
    fe = bundle['fe']
    model = bundle['model']
    scores = model.predict_proba(fe.transform(X_test_sc))
    val_score_list.append(scores)
    print(f"  {name:20s}  meta-val predicted ({time.time()-t0:.1f}s)")

val_matrix = np.column_stack(val_score_list)
p_val = meta_lr.predict_proba(val_matrix)[:, 1]

# predict on kaggle
kaggle_score_list = []
for name in model_names:
    t0 = time.time()
    safe_name = name.lower().replace(' ', '_')
    bundle = joblib.load(os.path.join(MODELS_DIR, f'{safe_name}.joblib'))
    fe = bundle['fe']
    model = bundle['model']
    scores = model.predict_proba(fe.transform(X_kaggle_sc))
    kaggle_score_list.append(scores)
    print(f"  {name:20s}  kaggle predicted ({time.time()-t0:.1f}s)")

kaggle_matrix = np.column_stack(kaggle_score_list)
p_kaggle = meta_lr.predict_proba(kaggle_matrix)[:, 1]

print(f"\n  p_val:    mean={p_val.mean():.4f}  std={p_val.std():.4f}")
print(f"  p_kaggle: mean={p_kaggle.mean():.4f}  std={p_kaggle.std():.4f}")


# ====================== 3. BASELINE (global threshold) ======================

print(f"\n{'=' * 85}")
print("  Baseline: global threshold")
print(f"{'=' * 85}")

fpr, tpr, thr_arr = roc_curve(y_test, p_val)
J = tpr - fpr
global_best_thr = thr_arr[int(np.argmax(J))]
baseline_pred = (p_val >= global_best_thr).astype(int)
baseline_metrics = full_metrics(y_test, baseline_pred)

print(f"  Threshold: {global_best_thr:.4f}")
print(f"  AUC_bin={baseline_metrics['auc_binary']:.4f}  acc={baseline_metrics['acc']:.4f}  "
      f"f1={baseline_metrics['f1']:.4f}  prec={baseline_metrics['prec']:.4f}  rec={baseline_metrics['rec']:.4f}")


# ====================== 4. EVALUATE THRESHOLD MODELS ======================

print(f"\n{'=' * 85}")
print("  Evaluating threshold models on meta-val")
print(f"{'=' * 85}")

results = {'baseline_global_thr': baseline_metrics}

# load threshold models
threshold_models = {}

lr_path = os.path.join(THRESHOLD_DIR, 'threshold_model_lr.joblib')
xgb_path = os.path.join(THRESHOLD_DIR, 'threshold_model_xgb.joblib')

if os.path.exists(lr_path):
    threshold_models['LR_threshold'] = joblib.load(lr_path)
    print(f"  Loaded: LR_threshold")

if os.path.exists(xgb_path):
    threshold_models['XGB_threshold'] = joblib.load(xgb_path)
    print(f"  Loaded: XGB_threshold")

for thresh_name, thresh_model in threshold_models.items():
    t_pred = thresh_model.predict(X_test_sc)
    pred = (p_val >= t_pred).astype(int)
    m = full_metrics(y_test, pred)
    results[thresh_name] = m

    print(f"\n  {thresh_name}:")
    print(f"    AUC_bin={m['auc_binary']:.4f}  acc={m['acc']:.4f}  "
          f"f1={m['f1']:.4f}  prec={m['prec']:.4f}  rec={m['rec']:.4f}")

# comparison
print(f"\n{'=' * 95}")
print("COMPARISON")
print(f"{'=' * 95}")

df_res = pd.DataFrame([
    {'model': k, **v} for k, v in results.items()
]).sort_values('auc_binary', ascending=False)
print(df_res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

best = df_res.iloc[0]
print(f"\n  Best: {best['model']}  AUC_bin={best['auc_binary']:.4f}")

# ====================== 5. SUBMISSION ======================

print(f"\n{'=' * 85}")
print("  Submission")
print(f"{'=' * 85}")

submitter = Submitter(submission_dir='submissions')

# always submit baseline
baseline_kaggle_preds = (p_kaggle >= global_best_thr).astype(int)
submitter.submit_scores('exp018_baseline_global_thr', baseline_kaggle_preds, test_ids, threshold=global_best_thr)

# submit threshold models separately for comparison
for thresh_name, thresh_model in threshold_models.items():
    kaggle_t = thresh_model.predict(X_kaggle_sc)
    kaggle_preds = (p_kaggle >= kaggle_t).astype(int)
    submitter.submit_scores(f'exp018_{thresh_name}', kaggle_preds, test_ids, threshold=None)

submitter.show_history()