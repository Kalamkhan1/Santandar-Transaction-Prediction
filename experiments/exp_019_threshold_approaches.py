# experiments/exp_019_threshold_approaches.py

"""
Experiment 19: 6 threshold approaches on top of full ensemble.
Loads ensemble predictions (cached or computed), tests 6 methods
for per-sample threshold prediction.

Approaches 1-4: predict threshold t from raw features, pred = 1 if p >= t
Approaches 5-6: predict threshold t from PCA-30 features, pred = 1 if p >= t

  1. Linear Regression + median t*     (raw)
  2. XGBoost Regressor + median t*     (raw)
  3. MLP + median t* (MSE loss)        (raw)
  4. MLP + range-aware loss            (raw)
  5. PCA-30 + KNN regression on t*     (PCA)
  6. PCA-30 + MLP regression on t*     (PCA)

Each approach gets its own submission file.
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score

from xgboost import XGBRegressor

from src.preprocessing import DataSplitter
from src.submission import Submitter


# ====================== CONFIG ======================

MODELS_DIR = 'models'
THRESHOLD_DIR = 'data/processed/threshold_data'
CACHE_DIR = 'data/processed/ensemble_cache'

os.makedirs(THRESHOLD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

ROUTER_KEY = 'lr'


# ====================== HELPERS ======================

def threshold_metrics(y_true, p, t_pred):
    """pred = 1 if p >= t. Compute binary AUC + metrics."""
    pred = (p >= t_pred).astype(int)
    fpr, tpr, _ = roc_curve(y_true, pred)
    J = (tpr - fpr).max()
    return {
        'auc_binary': (1 + J) / 2,
        'acc': accuracy_score(y_true, pred),
        'f1': f1_score(y_true, pred),
        'prec': precision_score(y_true, pred),
        'rec': recall_score(y_true, pred),
        'pred': pred,
    }


def score_metrics(y_true, scores):
    """Continuous scores. Find best threshold via Youden's J."""
    fpr, tpr, thr_arr = roc_curve(y_true, scores)
    J = tpr - fpr
    idx = int(np.argmax(J))
    pred = (scores >= thr_arr[idx]).astype(int)
    return {
        'auc_binary': (1 + J[idx]) / 2,
        'best_thr': thr_arr[idx],
        'acc': accuracy_score(y_true, pred),
        'f1': f1_score(y_true, pred),
        'prec': precision_score(y_true, pred),
        'rec': recall_score(y_true, pred),
        'pred': pred,
    }


class RangeAwareLoss(nn.Module):
    def __init__(self, M=10.0, lam=0.1):
        super().__init__()
        self.M = M
        self.lam = lam

    def forward(self, t_pred, p, y):
        center = torch.where(y == 1, p / 2.0, (p + 1.0) / 2.0)
        violation = torch.where(
            y == 1,
            torch.clamp(t_pred - p, min=0),
            torch.clamp(p - t_pred, min=0)
        )
        return (self.M * violation ** 2 + self.lam * (t_pred - center) ** 2).mean()


class MLPRegressor(nn.Module):
    def __init__(self, n_input, hidden=(256, 128)):
        super().__init__()
        layers = []
        prev = n_input
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.Tanh(), nn.Dropout(0.2)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


# ====================== 1. LOAD DATA ======================

print("=" * 85)
print("EXP 019: 6 threshold approaches")
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
X_kaggle_sc = splitter.scaler.transform(splitter.X_kaggle_test)
test_ids = splitter.test_ids


# ====================== 2. ENSEMBLE PREDICTIONS (cached) ======================

print(f"\n{'=' * 85}")
print("  Ensemble predictions (cached or computed)")
print(f"{'=' * 85}")

p_val_path = os.path.join(CACHE_DIR, 'p_val.npy')
p_kaggle_path = os.path.join(CACHE_DIR, 'p_kaggle.npy')

if os.path.exists(p_val_path) and os.path.exists(p_kaggle_path):
    p_val = np.load(p_val_path)
    p_kaggle = np.load(p_kaggle_path)
    print(f"  Loaded from cache: p_val={p_val.shape}  p_kaggle={p_kaggle.shape}")
else:
    router_bundle = joblib.load(os.path.join(MODELS_DIR, f'{ROUTER_KEY}_router.joblib'))
    meta_lr = router_bundle['router']
    model_names = router_bundle['model_names']

    val_score_list = []
    for name in model_names:
        t0 = time.time()
        safe_name = name.lower().replace(' ', '_')
        bundle = joblib.load(os.path.join(MODELS_DIR, f'{safe_name}.joblib'))
        scores = bundle['model'].predict_proba(bundle['fe'].transform(X_test_sc))
        val_score_list.append(scores)
        print(f"  {name:20s}  meta-val ({time.time()-t0:.1f}s)")
    p_val = meta_lr.predict_proba(np.column_stack(val_score_list))[:, 1]

    kaggle_score_list = []
    for name in model_names:
        t0 = time.time()
        safe_name = name.lower().replace(' ', '_')
        bundle = joblib.load(os.path.join(MODELS_DIR, f'{safe_name}.joblib'))
        scores = bundle['model'].predict_proba(bundle['fe'].transform(X_kaggle_sc))
        kaggle_score_list.append(scores)
        print(f"  {name:20s}  kaggle ({time.time()-t0:.1f}s)")
    p_kaggle = meta_lr.predict_proba(np.column_stack(kaggle_score_list))[:, 1]

    np.save(p_val_path, p_val)
    np.save(p_kaggle_path, p_kaggle)
    print(f"  Cached to {CACHE_DIR}/")

print(f"\n  p_val:    mean={p_val.mean():.4f}  std={p_val.std():.4f}  "
      f"min={p_val.min():.4f}  max={p_val.max():.4f}")
print(f"  p_kaggle: mean={p_kaggle.mean():.4f}  std={p_kaggle.std():.4f}  "
      f"min={p_kaggle.min():.4f}  max={p_kaggle.max():.4f}")


# ====================== 3. ENSEMBLE AUC (target to beat) ======================

print(f"\n{'=' * 85}")
print("  ENSEMBLE PERFORMANCE (the target)")
print(f"{'=' * 85}")

fpr, tpr, thr_arr = roc_curve(y_test, p_val)
J = tpr - fpr
auc_prob = auc(fpr, tpr)
idx = int(np.argmax(J))
global_best_thr = thr_arr[idx]
auc_bin_global = (1 + J[idx]) / 2

print(f"  AUC (probability)                 = {auc_prob:.4f}")
print(f"  AUC (binary, global threshold)    = {auc_bin_global:.4f}")
print(f"  Gap to close:                     = {auc_prob - auc_bin_global:.4f}")
print(f"  Global threshold:                 = {global_best_thr:.4f}")


# ====================== 4. LOAD THRESHOLD DATASET ======================

print(f"\n{'=' * 85}")
print("  Loading threshold dataset")
print(f"{'=' * 85}")

oof_path = os.path.join(THRESHOLD_DIR, 'oof_t_star_median.npy')
if os.path.exists(oof_path):
    print(f"  Using OOF dataset")
    X_features = np.load(os.path.join(THRESHOLD_DIR, 'oof_X_features.npy'))
    t_star = np.load(os.path.join(THRESHOLD_DIR, 'oof_t_star_median.npy'))
    t_lower = np.load(os.path.join(THRESHOLD_DIR, 'oof_t_star_lower.npy'))
    t_upper = np.load(os.path.join(THRESHOLD_DIR, 'oof_t_star_upper.npy'))
    y_for_range = np.load(os.path.join(THRESHOLD_DIR, 'oof_y_labels.npy'))
    p_for_range = np.load(os.path.join(THRESHOLD_DIR, 'oof_ensemble_probs.npy'))
else:
    print(f"  Using in-sample dataset (no OOF)")
    X_features = np.load(os.path.join(THRESHOLD_DIR, 'X_features.npy'))
    t_star = np.load(os.path.join(THRESHOLD_DIR, f'{ROUTER_KEY}_t_star_median.npy'))
    t_lower = np.load(os.path.join(THRESHOLD_DIR, f'{ROUTER_KEY}_t_star_lower.npy'))
    t_upper = np.load(os.path.join(THRESHOLD_DIR, f'{ROUTER_KEY}_t_star_upper.npy'))
    y_for_range = np.load(os.path.join(THRESHOLD_DIR, 'y_labels.npy'))
    p_for_range = np.load(os.path.join(THRESHOLD_DIR, f'{ROUTER_KEY}_ensemble_probs.npy'))

print(f"  X_features:  {X_features.shape}")
print(f"  t_star:      {t_star.shape}")
print(f"  y_labels:    {y_for_range.shape}")
print(f"  p_for_range: {p_for_range.shape}")


# ====================== 5. PCA-30 ======================

print(f"\n{'=' * 85}")
print("  Computing PCA-30 features")
print(f"{'=' * 85}")

pca = PCA(n_components=30, random_state=42)
X_train_pca = pca.fit_transform(X_features)
X_test_pca = pca.transform(X_test_sc)
X_kaggle_pca = pca.transform(X_kaggle_sc)

print(f"  X_train_pca:  {X_train_pca.shape}")
print(f"  X_test_pca:   {X_test_pca.shape}")
print(f"  X_kaggle_pca: {X_kaggle_pca.shape}")
print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.4f}")


# ====================== 6. SIX APPROACHES ======================

results = {}
all_preds = {}

# ---- Approach 1: Linear Regression + median t* (raw) ----
print(f"\n{'=' * 70}")
print(f"  1. Linear Regression + median t* (raw features)")
print(f"{'=' * 70}")

t0 = time.time()
lr_t = LinearRegression()
lr_t.fit(X_features, t_star)
t_val = lr_t.predict(X_test_sc)
t_kaggle = lr_t.predict(X_kaggle_sc)
m = threshold_metrics(y_test, p_val, t_val)
m['time'] = time.time() - t0
results['1_LR_median'] = m
all_preds['1_LR_median'] = (p_kaggle >= t_kaggle).astype(int)
print(f"  AUC_bin={m['auc_binary']:.4f}  acc={m['acc']:.4f}  f1={m['f1']:.4f}  time={m['time']:.1f}s")


# ---- Approach 2: XGBoost Regressor + median t* (raw) ----
print(f"\n{'=' * 70}")
print(f"  2. XGBoost Regressor + median t* (raw features)")
print(f"{'=' * 70}")

t0 = time.time()
xgb_t = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)
xgb_t.fit(X_features, t_star)
t_val = xgb_t.predict(X_test_sc)
t_kaggle = xgb_t.predict(X_kaggle_sc)
m = threshold_metrics(y_test, p_val, t_val)
m['time'] = time.time() - t0
results['2_XGB_median'] = m
all_preds['2_XGB_median'] = (p_kaggle >= t_kaggle).astype(int)
print(f"  AUC_bin={m['auc_binary']:.4f}  acc={m['acc']:.4f}  f1={m['f1']:.4f}  time={m['time']:.1f}s")


# ---- Approach 3: MLP + median t* (MSE, raw) ----
print(f"\n{'=' * 70}")
print(f"  3. MLP + median t* (MSE loss, raw features)")
print(f"{'=' * 70}")

t0 = time.time()
torch.manual_seed(42)
mlp_mse = MLPRegressor(n_input=X_features.shape[1])
opt = torch.optim.Adam(mlp_mse.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()

X_t = torch.FloatTensor(X_features)
y_t = torch.FloatTensor(t_star)
dl = DataLoader(TensorDataset(X_t, y_t), batch_size=4096, shuffle=True)

for epoch in range(50):
    mlp_mse.train()
    for Xb, yb in dl:
        opt.zero_grad()
        loss = criterion(mlp_mse(Xb), yb)
        loss.backward()
        opt.step()

mlp_mse.eval()
with torch.no_grad():
    t_val = mlp_mse(torch.FloatTensor(X_test_sc)).numpy()
    t_kaggle = mlp_mse(torch.FloatTensor(X_kaggle_sc)).numpy()

m = threshold_metrics(y_test, p_val, t_val)
m['time'] = time.time() - t0
results['3_MLP_median'] = m
all_preds['3_MLP_median'] = (p_kaggle >= t_kaggle).astype(int)
print(f"  AUC_bin={m['auc_binary']:.4f}  acc={m['acc']:.4f}  f1={m['f1']:.4f}  time={m['time']:.1f}s")


# ---- Approach 4: MLP + range-aware loss (raw) ----
print(f"\n{'=' * 70}")
print(f"  4. MLP + range-aware loss (raw features)")
print(f"{'=' * 70}")

t0 = time.time()
torch.manual_seed(42)
mlp_range = MLPRegressor(n_input=X_features.shape[1])
opt = torch.optim.Adam(mlp_range.parameters(), lr=0.001, weight_decay=1e-4)
range_loss = RangeAwareLoss(M=10.0, lam=0.1)

X_t = torch.FloatTensor(X_features)
p_t = torch.FloatTensor(p_for_range)
y_t = torch.FloatTensor(y_for_range)
dl = DataLoader(TensorDataset(X_t, p_t, y_t), batch_size=4096, shuffle=True)

for epoch in range(50):
    mlp_range.train()
    for Xb, pb, yb in dl:
        opt.zero_grad()
        t_pred = mlp_range(Xb)
        loss = range_loss(t_pred, pb, yb)
        loss.backward()
        opt.step()

mlp_range.eval()
with torch.no_grad():
    t_val = mlp_range(torch.FloatTensor(X_test_sc)).numpy()
    t_kaggle = mlp_range(torch.FloatTensor(X_kaggle_sc)).numpy()

m = threshold_metrics(y_test, p_val, t_val)
m['time'] = time.time() - t0
results['4_MLP_range'] = m
all_preds['4_MLP_range'] = (p_kaggle >= t_kaggle).astype(int)
print(f"  AUC_bin={m['auc_binary']:.4f}  acc={m['acc']:.4f}  f1={m['f1']:.4f}  time={m['time']:.1f}s")


# ---- Approach 5: PCA-30 + KNN regression on t* ----
print(f"\n{'=' * 70}")
print(f"  5. PCA-30 + KNN regression on t* (weighted avg of neighbors)")
print(f"{'=' * 70}")

t0 = time.time()
knn_reg = KNeighborsRegressor(
    n_neighbors=301, weights='distance', n_jobs=-1)
knn_reg.fit(X_train_pca, t_star)
t_val = knn_reg.predict(X_test_pca)
t_kaggle = knn_reg.predict(X_kaggle_pca)
m = threshold_metrics(y_test, p_val, t_val)
m['time'] = time.time() - t0
results['5_PCA_KNN_t'] = m
all_preds['5_PCA_KNN_t'] = (p_kaggle >= t_kaggle).astype(int)
print(f"  AUC_bin={m['auc_binary']:.4f}  acc={m['acc']:.4f}  f1={m['f1']:.4f}  time={m['time']:.1f}s")


# ---- Approach 6: PCA-30 + MLP regression on t* ----
print(f"\n{'=' * 70}")
print(f"  6. PCA-30 + MLP regression on t*")
print(f"{'=' * 70}")

t0 = time.time()
torch.manual_seed(42)
mlp_pca = MLPRegressor(n_input=30)
opt = torch.optim.Adam(mlp_pca.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()

X_t = torch.FloatTensor(X_train_pca)
y_t = torch.FloatTensor(t_star)
dl = DataLoader(TensorDataset(X_t, y_t), batch_size=4096, shuffle=True)

for epoch in range(50):
    mlp_pca.train()
    for Xb, yb in dl:
        opt.zero_grad()
        loss = criterion(mlp_pca(Xb), yb)
        loss.backward()
        opt.step()

mlp_pca.eval()
with torch.no_grad():
    t_val = mlp_pca(torch.FloatTensor(X_test_pca)).numpy()
    t_kaggle = mlp_pca(torch.FloatTensor(X_kaggle_pca)).numpy()

m = threshold_metrics(y_test, p_val, t_val)
m['time'] = time.time() - t0
results['6_PCA_MLP_t'] = m
all_preds['6_PCA_MLP_t'] = (p_kaggle >= t_kaggle).astype(int)
print(f"  AUC_bin={m['auc_binary']:.4f}  acc={m['acc']:.4f}  f1={m['f1']:.4f}  time={m['time']:.1f}s")


# ====================== 7. COMPARISON ======================

print(f"\n{'=' * 105}")
print("ALL 6 APPROACHES + BASELINE")
print(f"{'=' * 105}")
print(f"  Ensemble AUC (prob) = {auc_prob:.4f}  ← the target for binary AUC")
print(f"  Ensemble AUC (binary, global threshold) = {auc_bin_global:.4f}")
print(f"  Gap: {auc_prob - auc_bin_global:.4f}")

baseline_pred = (p_val >= global_best_thr).astype(int)
baseline_m = {
    'auc_binary': auc_bin_global,
    'acc': accuracy_score(y_test, baseline_pred),
    'f1': f1_score(y_test, baseline_pred),
    'prec': precision_score(y_test, baseline_pred),
    'rec': recall_score(y_test, baseline_pred),
    'time': 0,
}
all_results = {'0_baseline_global': baseline_m, **results}

rows = []
for name, m in all_results.items():
    rows.append({
        'approach': name,
        'auc_binary': m['auc_binary'],
        'gap_to_auc_prob': auc_prob - m['auc_binary'],
        'acc': m['acc'],
        'f1': m['f1'],
        'prec': m['prec'],
        'rec': m['rec'],
        'time': m.get('time', 0),
    })

df = pd.DataFrame(rows).sort_values('auc_binary', ascending=False)
print(f"\n{df.to_string(index=False, float_format=lambda v: f'{v:.4f}')}")

best = df.iloc[0]
print(f"\n  Best: {best['approach']}  AUC_bin={best['auc_binary']:.4f}")
print(f"  Gap to ensemble AUC: {best['gap_to_auc_prob']:.4f}")

df.to_csv(os.path.join(THRESHOLD_DIR, 'exp019_comparison.csv'), index=False)
print(f"  Saved: {THRESHOLD_DIR}/exp019_comparison.csv")


# ====================== 8. SUBMISSIONS ======================

print(f"\n{'=' * 85}")
print("  Submissions")
print(f"{'=' * 85}")

submitter = Submitter(submission_dir='submissions')

# baseline
baseline_kaggle_preds = (p_kaggle >= global_best_thr).astype(int)
submitter.submit_scores('exp019_baseline', baseline_kaggle_preds, test_ids,
                        threshold=global_best_thr)

# each approach
for name, preds in all_preds.items():
    submitter.submit_scores(f'exp019_{name}', preds, test_ids, threshold=None)

submitter.show_history()