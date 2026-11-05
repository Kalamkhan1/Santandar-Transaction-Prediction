# experiments/exp_016_train_threshold_models.py

"""
Experiment 16: Train threshold models and evaluate.
Loads threshold dataset from exp_015, trains 6 threshold models,
evaluates on meta-val using ensemble probabilities.

Tests:
  1. Linear Regression + median target
  2. XGBoost Regressor + median target
  3. XGBoost Regressor + two-row augmentation
  4. MLP + median target (MSE loss)
  5. MLP + range-aware loss
  6. MLP + two-row augmentation

Compares all against baseline (global Youden's J threshold).
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
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score

from src.preprocessing import DataSplitter
from src.features import FeatureEngineer


# ====================== CONFIG ======================

MODELS_DIR = 'models'
THRESHOLD_DIR = 'data/processed/threshold_data'
PLOT_DIR = 'src/plots'

os.makedirs(PLOT_DIR, exist_ok=True)

# use LR router by default (swap to lgbm if testing that)
ROUTER_KEY = 'lr'


# ====================== HELPERS ======================

def binary_auc(y_true, pred):
    """Binary AUC from 0/1 predictions."""
    fpr, tpr, _ = roc_curve(y_true, pred)
    J = (tpr - fpr).max()
    return (1 + J) / 2


def full_metrics(y_true, pred):
    return {
        'auc_binary': binary_auc(y_true, pred),
        'acc': accuracy_score(y_true, pred),
        'f1': f1_score(y_true, pred),
        'prec': precision_score(y_true, pred),
        'rec': recall_score(y_true, pred),
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
        L_correct = self.M * violation ** 2
        L_safe = (t_pred - center) ** 2
        return (L_correct + self.lam * L_safe).mean()


class MLPThreshold(nn.Module):
    def __init__(self, n_input=200, hidden=(256, 128)):
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
print("EXP 016: Train threshold models")
print("=" * 85)

# threshold dataset
X_features = np.load(os.path.join(THRESHOLD_DIR, 'X_features.npy'))
y_labels = np.load(os.path.join(THRESHOLD_DIR, 'y_labels.npy'))
p_ensemble = np.load(os.path.join(THRESHOLD_DIR, f'{ROUTER_KEY}_ensemble_probs.npy'))
t_median = np.load(os.path.join(THRESHOLD_DIR, f'{ROUTER_KEY}_t_star_median.npy'))
t_lower = np.load(os.path.join(THRESHOLD_DIR, f'{ROUTER_KEY}_t_star_lower.npy'))
t_upper = np.load(os.path.join(THRESHOLD_DIR, f'{ROUTER_KEY}_t_star_upper.npy'))

print(f"\n  X_features: {X_features.shape}")
print(f"  y_labels:   {y_labels.shape}")
print(f"  p_ensemble: {p_ensemble.shape}")
print(f"  t_median:   {t_median.shape}")

# meta-val data
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


# ====================== 2. GET ENSEMBLE PROBS ON META-VAL ======================

print(f"\n{'=' * 85}")
print("  Getting ensemble probabilities on meta-val")
print(f"{'=' * 85}")

router_bundle = joblib.load(os.path.join(MODELS_DIR, f'{ROUTER_KEY}_router.joblib'))
router = router_bundle['router']
router_model_names = router_bundle['model_names']

val_score_list = []
for name in router_model_names:
    safe_name = name.lower().replace(' ', '_')
    bundle = joblib.load(os.path.join(MODELS_DIR, f'{safe_name}.joblib'))
    fe = bundle['fe']
    model = bundle['model']
    X_val_fe = fe.transform(X_test_sc)
    val_score_list.append(model.predict_proba(X_val_fe))

val_matrix = np.column_stack(val_score_list)
p_val = router.predict_proba(val_matrix)[:, 1]

# baseline: global Youden's J threshold on ensemble
fpr, tpr, thr_arr = roc_curve(y_test, p_val)
J = tpr - fpr
global_best_thr = thr_arr[int(np.argmax(J))]
baseline_pred = (p_val >= global_best_thr).astype(int)
baseline_auc_bin = binary_auc(y_test, baseline_pred)

print(f"  p_val: mean={p_val.mean():.4f}  std={p_val.std():.4f}")
print(f"  Global best threshold: {global_best_thr:.4f}")
print(f"  Baseline binary AUC:   {baseline_auc_bin:.4f}")


# ====================== 3. TRAIN THRESHOLD MODELS ======================

results = {}

# ---- Test 1: Linear Regression + median ----
print(f"\n{'=' * 70}")
print(f"  Test 1: Linear Regression + median target")
print(f"{'=' * 70}")

t0 = time.time()
lr_model = LinearRegression()
lr_model.fit(X_features, t_median)
t_pred = lr_model.predict(X_test_sc)
pred = (p_val >= t_pred).astype(int)
m = full_metrics(y_test, pred)
results['LR_median'] = m
print(f"  AUC_bin={m['auc_binary']:.4f}  acc={m['acc']:.4f}  f1={m['f1']:.4f}  time={time.time()-t0:.1f}s")

# ---- Test 2: XGBoost Regressor + median ----
print(f"\n{'=' * 70}")
print(f"  Test 2: XGBoost Regressor + median target")
print(f"{'=' * 70}")

from xgboost import XGBRegressor

t0 = time.time()
xgb_med = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)
xgb_med.fit(X_features, t_median)
t_pred = xgb_med.predict(X_test_sc)
pred = (p_val >= t_pred).astype(int)
m = full_metrics(y_test, pred)
results['XGB_median'] = m
print(f"  AUC_bin={m['auc_binary']:.4f}  acc={m['acc']:.4f}  f1={m['f1']:.4f}  time={time.time()-t0:.1f}s")

# ---- Test 3: XGBoost Regressor + two-row ----
print(f"\n{'=' * 70}")
print(f"  Test 3: XGBoost Regressor + two-row augmentation")
print(f"{'=' * 70}")

t0 = time.time()
X_tw = np.vstack([X_features, X_features])
y_tw = np.concatenate([t_lower, t_upper])

xgb_tw = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)
xgb_tw.fit(X_tw, y_tw)
t_pred = xgb_tw.predict(X_test_sc)
pred = (p_val >= t_pred).astype(int)
m = full_metrics(y_test, pred)
results['XGB_tworow'] = m
print(f"  AUC_bin={m['auc_binary']:.4f}  acc={m['acc']:.4f}  f1={m['f1']:.4f}  time={time.time()-t0:.1f}s")

# ---- Test 4: MLP + median (MSE) ----
print(f"\n{'=' * 70}")
print(f"  Test 4: MLP + median target (MSE)")
print(f"{'=' * 70}")

t0 = time.time()
torch.manual_seed(42)
mlp_mse = MLPThreshold(n_input=X_features.shape[1])
opt = torch.optim.Adam(mlp_mse.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()

X_t = torch.FloatTensor(X_features)
y_t = torch.FloatTensor(t_median)
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
    t_pred = mlp_mse(torch.FloatTensor(X_test_sc)).numpy()
pred = (p_val >= t_pred).astype(int)
m = full_metrics(y_test, pred)
results['MLP_median'] = m
print(f"  AUC_bin={m['auc_binary']:.4f}  acc={m['acc']:.4f}  f1={m['f1']:.4f}  time={time.time()-t0:.1f}s")

# ---- Test 5: MLP + range-aware loss ----
print(f"\n{'=' * 70}")
print(f"  Test 5: MLP + range-aware loss")
print(f"{'=' * 70}")

t0 = time.time()
torch.manual_seed(42)
mlp_range = MLPThreshold(n_input=X_features.shape[1])
opt = torch.optim.Adam(mlp_range.parameters(), lr=0.001, weight_decay=1e-4)
range_loss = RangeAwareLoss(M=10.0, lam=0.1)

# need p and y alongside features for loss
X_t = torch.FloatTensor(X_features)
p_t = torch.FloatTensor(p_ensemble)
y_t = torch.FloatTensor(y_labels)
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
    t_pred = mlp_range(torch.FloatTensor(X_test_sc)).numpy()
pred = (p_val >= t_pred).astype(int)
m = full_metrics(y_test, pred)
results['MLP_range'] = m
print(f"  AUC_bin={m['auc_binary']:.4f}  acc={m['acc']:.4f}  f1={m['f1']:.4f}  time={time.time()-t0:.1f}s")

# ---- Test 6: MLP + two-row ----
print(f"\n{'=' * 70}")
print(f"  Test 6: MLP + two-row augmentation")
print(f"{'=' * 70}")

t0 = time.time()
torch.manual_seed(42)
mlp_tw = MLPThreshold(n_input=X_features.shape[1])
opt = torch.optim.Adam(mlp_tw.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()

X_t = torch.FloatTensor(X_tw)
y_t = torch.FloatTensor(y_tw)
dl = DataLoader(TensorDataset(X_t, y_t), batch_size=4096, shuffle=True)

for epoch in range(50):
    mlp_tw.train()
    for Xb, yb in dl:
        opt.zero_grad()
        loss = criterion(mlp_tw(Xb), yb)
        loss.backward()
        opt.step()

mlp_tw.eval()
with torch.no_grad():
    t_pred = mlp_tw(torch.FloatTensor(X_test_sc)).numpy()
pred = (p_val >= t_pred).astype(int)
m = full_metrics(y_test, pred)
results['MLP_tworow'] = m
print(f"  AUC_bin={m['auc_binary']:.4f}  acc={m['acc']:.4f}  f1={m['f1']:.4f}  time={time.time()-t0:.1f}s")


# ====================== 4. SUMMARY ======================

print(f"\n{'=' * 95}")
print("THRESHOLD MODEL COMPARISON")
print(f"{'=' * 95}")

rows = [{
    'model': 'baseline_global_thr',
    'auc_binary': baseline_auc_bin,
    'acc': accuracy_score(y_test, baseline_pred),
    'f1': f1_score(y_test, baseline_pred),
    'prec': precision_score(y_test, baseline_pred),
    'rec': recall_score(y_test, baseline_pred),
}]

for name, m in results.items():
    rows.append({'model': name, **m})

df = pd.DataFrame(rows).sort_values('auc_binary', ascending=False)
print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print(f"\n  Baseline (global threshold): AUC_bin={baseline_auc_bin:.4f}")
best = df.iloc[0]
print(f"  Best: {best['model']} AUC_bin={best['auc_binary']:.4f} "
      f"(delta={best['auc_binary'] - baseline_auc_bin:+.4f})")

df.to_csv('data/processed/threshold_data/threshold_model_comparison.csv', index=False)
print(f"\n  Saved: data/processed/threshold_data/threshold_model_comparison.csv")