# experiments/exp_015_build_threshold_dataset.py

"""
Experiment 15: Build threshold model dataset (both routers).
Loads trained ensembles from models/ (LR + LGBM routers),
runs on training data, computes correct range and ideal threshold per sample.

Outputs saved to data/processed/threshold_data/
  - X_features.npy                    (160k × 200, raw features)
  - y_labels.npy                      (160k, true labels)
  - lr_ensemble_probs.npy             (160k, LR ensemble probability p)
  - lr_t_star_median.npy              (160k, median of correct range)
  - lr_t_star_lower.npy               (160k, lower bound of correct range)
  - lr_t_star_upper.npy               (160k, upper bound of correct range)
  - lgbm_ensemble_probs.npy           (160k, LGBM ensemble probability p)
  - lgbm_t_star_median.npy            (160k, median of correct range)
  - lgbm_t_star_lower.npy             (160k, lower bound of correct range)
  - lgbm_t_star_upper.npy             (160k, upper bound of correct range)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import joblib
import time

from src.preprocessing import DataSplitter


# ====================== CONFIG ======================

MODELS_DIR = 'models'
OUTPUT_DIR = 'data/processed/threshold_data'

ROUTERS = {
    'lr':   'lr_router.joblib',
    'lgbm': 'lgbm_router.joblib',
}


# ====================== 1. LOAD DATA ======================

print("=" * 85)
print("EXP 015: Build threshold model dataset (both routers)")
print("=" * 85)

splitter = DataSplitter(
    data_dir='data/raw',
    processed_dir='data/processed',
    plot_dir='src/plots',
    seed=43
)
splitter.run()

X_train_sc = splitter.X_train_sc
X_test_sc = splitter.X_test_sc
y_train = splitter.y_train
y_test = splitter.y_test

print(f"\n  X_train_sc: {X_train_sc.shape}  y_train: {y_train.shape}")


# ====================== 2. LOAD ROUTERS ======================

print(f"\n{'=' * 85}")
print("  Loading routers")
print(f"{'=' * 85}")

routers = {}
for router_key, router_file in ROUTERS.items():
    path = os.path.join(MODELS_DIR, router_file)
    if os.path.exists(path):
        bundle = joblib.load(path)
        routers[router_key] = bundle
        print(f"  Loaded {router_key}: {len(bundle['model_names'])} models")
    else:
        print(f"  {router_file} not found, skipping")


# ====================== 3. GET MODEL PREDICTIONS ON TRAINING DATA ======================

print(f"\n{'=' * 85}")
print("  Predicting with each model on training data")
print(f"{'=' * 85}")

# use model list from first available router (all routers share same models)
model_names = routers[list(routers.keys())[0]]['model_names']
train_score_list = []

for name in model_names:
    t0 = time.time()
    safe_name = name.lower().replace(' ', '_')
    bundle = joblib.load(os.path.join(MODELS_DIR, f'{safe_name}.joblib'))
    fe = bundle['fe']
    model = bundle['model']

    X_train_fe = fe.transform(X_train_sc)
    scores = model.predict_proba(X_train_fe)
    train_score_list.append(scores)

    elapsed = time.time() - t0
    print(f"  {name:20s}  predicted on {X_train_sc.shape[0]} rows  ({elapsed:.1f}s)")

train_matrix = np.column_stack(train_score_list)
print(f"\n  Model score matrix: {train_matrix.shape}")


# ====================== 4. COMPUTE RANGE AND TARGETS PER ROUTER ======================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# save shared files
np.save(os.path.join(OUTPUT_DIR, 'X_features.npy'), X_train_sc)
np.save(os.path.join(OUTPUT_DIR, 'y_labels.npy'), y_train)

y = y_train.astype(float)

for router_key, bundle in routers.items():
    print(f"\n{'=' * 85}")
    print(f"  Computing targets for {router_key} router")
    print(f"{'=' * 85}")

    router = bundle['router']
    p = router.predict_proba(train_matrix)[:, 1]

    print(f"  p stats: mean={p.mean():.4f}  std={p.std():.4f}  "
          f"min={p.min():.4f}  max={p.max():.4f}")

    # range bounds
    # y=1: t ∈ [0, p]      (correct when p >= t → predict 1)
    # y=0: t ∈ (p, 1]      (correct when p < t  → predict 0)
    t_lower = np.where(y == 1, 0.0, p)
    t_upper = np.where(y == 1, p, 1.0)
    t_star_median = (t_lower + t_upper) / 2.0

    t_lower = np.clip(t_lower, 0.0, 1.0)
    t_upper = np.clip(t_upper, 0.0, 1.0)
    t_star_median = np.clip(t_star_median, 0.0, 1.0)

    # stats
    print(f"\n  Target (median) stats:")
    print(f"    y=1: mean={t_star_median[y==1].mean():.4f}  "
          f"std={t_star_median[y==1].std():.4f}  "
          f"min={t_star_median[y==1].min():.4f}  max={t_star_median[y==1].max():.4f}")
    print(f"    y=0: mean={t_star_median[y==0].mean():.4f}  "
          f"std={t_star_median[y==0].std():.4f}  "
          f"min={t_star_median[y==0].min():.4f}  max={t_star_median[y==0].max():.4f}")

    # separation
    print(f"\n  Separation check:")
    print(f"    y=1 targets should be LOW:  {t_star_median[y==1].mean():.4f}")
    print(f"    y=0 targets should be HIGH: {t_star_median[y==0].mean():.4f}")
    print(f"    Gap: {t_star_median[y==0].mean() - t_star_median[y==1].mean():.4f}")

    # verify correctness
    pred_at_tstar = (p >= t_star_median).astype(int)
    correct_at_tstar = (pred_at_tstar == y).mean()
    print(f"\n  Verification: predictions at t* correct = {correct_at_tstar:.6f} (should be 1.0)")

    # save
    np.save(os.path.join(OUTPUT_DIR, f'{router_key}_ensemble_probs.npy'), p)
    np.save(os.path.join(OUTPUT_DIR, f'{router_key}_t_star_median.npy'), t_star_median)
    np.save(os.path.join(OUTPUT_DIR, f'{router_key}_t_star_lower.npy'), t_lower)
    np.save(os.path.join(OUTPUT_DIR, f'{router_key}_t_star_upper.npy'), t_upper)

    print(f"\n  Saved: {router_key}_ensemble_probs.npy")
    print(f"  Saved: {router_key}_t_star_median.npy")
    print(f"  Saved: {router_key}_t_star_lower.npy")
    print(f"  Saved: {router_key}_t_star_upper.npy")


# ====================== 5. SUMMARY ======================

print(f"\n{'=' * 85}")
print("  All files saved to data/processed/threshold_data/")
print(f"{'=' * 85}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"    {f:40s}  {size / 1024 / 1024:.1f} MB")