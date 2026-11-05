# experiments/exp_020_threshold_improvements.py

"""
Experiment 20: 3 improvements on MLP_range threshold model.
Base: MLP + range-aware loss on OOF threshold dataset.

Idea 2: Blend threshold (α × t_global + (1-α) × t_model), sweep α
Idea 3: Margin re-ranking (p - t_model, find best cutoff)
Idea 1: Retrain on 200k (OOF train + meta-val), repeat 2 & 3

Submits each as separate file.
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

from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score

from src.preprocessing import DataSplitter
from src.submission import Submitter


# ====================== CONFIG ======================

MODELS_DIR = 'models'
THRESHOLD_DIR = 'data/processed/threshold_data'
CACHE_DIR = 'data/processed/ensemble_cache'

os.makedirs(CACHE_DIR, exist_ok=True)

ROUTER_KEY = 'lr'


# ====================== HELPERS ======================

def best_threshold(y_true, scores):
    fpr, tpr, thr_arr = roc_curve(y_true, scores)
    J = tpr - fpr
    idx = int(np.argmax(J))
    return thr_arr[idx], (1 + J[idx]) / 2


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


def train_mlp_range(X_train, t_star, p_train, y_train, epochs=50, lr=0.001):
    torch.manual_seed(42)
    model = MLPThreshold(n_input=X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = RangeAwareLoss(M=10.0, lam=0.1)

    X_t = torch.FloatTensor(X_train)
    p_t = torch.FloatTensor(p_train)
    y_t = torch.FloatTensor(y_train)
    dl = DataLoader(TensorDataset(X_t, p_t, y_t), batch_size=4096, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for Xb, pb, yb in dl:
            opt.zero_grad()
            t_pred = model(Xb)
            loss = loss_fn(t_pred, pb, yb)
            loss.backward()
            opt.step()

    return model


# ====================== 1. LOAD DATA ======================

print("=" * 85)
print("EXP 020: Threshold improvements (blend, margin, 200k)")
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
print("  Ensemble predictions")
print(f"{'=' * 85}")

p_val_path = os.path.join(CACHE_DIR, 'p_val.npy')
p_kaggle_path = os.path.join(CACHE_DIR, 'p_kaggle.npy')

if os.path.exists(p_val_path) and os.path.exists(p_kaggle_path):
    p_val = np.load(p_val_path)
    p_kaggle = np.load(p_kaggle_path)
    print(f"  Loaded cache: p_val={p_val.shape}  p_kaggle={p_kaggle.shape}")
else:
    router_bundle = joblib.load(os.path.join(MODELS_DIR, f'{ROUTER_KEY}_router.joblib'))
    meta_lr = router_bundle['router']
    model_names = router_bundle['model_names']

    val_score_list = []
    for name in model_names:
        safe_name = name.lower().replace(' ', '_')
        bundle = joblib.load(os.path.join(MODELS_DIR, f'{safe_name}.joblib'))
        val_score_list.append(bundle['model'].predict_proba(bundle['fe'].transform(X_test_sc)))
    p_val = meta_lr.predict_proba(np.column_stack(val_score_list))[:, 1]

    kaggle_score_list = []
    for name in model_names:
        safe_name = name.lower().replace(' ', '_')
        bundle = joblib.load(os.path.join(MODELS_DIR, f'{safe_name}.joblib'))
        kaggle_score_list.append(bundle['model'].predict_proba(bundle['fe'].transform(X_kaggle_sc)))
    p_kaggle = meta_lr.predict_proba(np.column_stack(kaggle_score_list))[:, 1]

    np.save(p_val_path, p_val)
    np.save(p_kaggle_path, p_kaggle)
    print(f"  Computed and cached")

print(f"  p_val:    mean={p_val.mean():.4f}  std={p_val.std():.4f}")
print(f"  p_kaggle: mean={p_kaggle.mean():.4f}  std={p_kaggle.std():.4f}")


# ====================== 3. BASELINE ======================

print(f"\n{'=' * 85}")
print("  ENSEMBLE PERFORMANCE (target)")
print(f"{'=' * 85}")

fpr, tpr, thr_arr = roc_curve(y_test, p_val)
J = tpr - fpr
auc_prob = auc(fpr, tpr)
idx = int(np.argmax(J))
global_best_thr = thr_arr[idx]
auc_bin_global = (1 + J[idx]) / 2

print(f"  AUC (prob)              = {auc_prob:.4f}")
print(f"  AUC (binary, global)    = {auc_bin_global:.4f}")
print(f"  Global threshold        = {global_best_thr:.4f}")
print(f"  Gap to close            = {auc_prob - auc_bin_global:.4f}")


# ====================== 4. LOAD OOF THRESHOLD DATASET ======================

print(f"\n{'=' * 85}")
print("  Loading OOF threshold dataset")
print(f"{'=' * 85}")

X_features = np.load(os.path.join(THRESHOLD_DIR, 'oof_X_features.npy'))
t_star_oof = np.load(os.path.join(THRESHOLD_DIR, 'oof_t_star_median.npy'))
y_oof = np.load(os.path.join(THRESHOLD_DIR, 'oof_y_labels.npy'))
p_oof = np.load(os.path.join(THRESHOLD_DIR, 'oof_ensemble_probs.npy'))

print(f"  X_features:   {X_features.shape}")
print(f"  t_star_oof:   {t_star_oof.shape}")
print(f"  y_oof:        {y_oof.shape}")
print(f"  p_oof:        {p_oof.shape}")


# ====================== 5. TRAIN MLP_RANGE ON 160K OOF ======================

print(f"\n{'=' * 85}")
print("  STEP 1: Train MLP_range on 160k OOF")
print(f"{'=' * 85}")

t0 = time.time()
mlp_oof = train_mlp_range(X_features, t_star_oof, p_oof, y_oof, epochs=50)
print(f"  Trained in {time.time()-t0:.1f}s")

mlp_oof.eval()
with torch.no_grad():
    t_val_oof = mlp_oof(torch.FloatTensor(X_test_sc)).numpy()
    t_kaggle_oof = mlp_oof(torch.FloatTensor(X_kaggle_sc)).numpy()

# also need p_oof on meta-val for range-aware loss in 200k retrain
# p_val already have

results = {}
all_preds = {}


# ====================== 6. IDEA 2: BLEND THRESHOLD (160k) ======================

print(f"\n{'=' * 85}")
print("  IDEA 2: Blend threshold (160k model)")
print(f"{'=' * 85}")

alphas = np.arange(0.0, 1.05, 0.05)
blend_results = []

for alpha in alphas:
    t_blend = alpha * global_best_thr + (1 - alpha) * t_val_oof
    pred = (p_val >= t_blend).astype(int)
    m = full_metrics(y_test, pred)
    m['alpha'] = alpha
    blend_results.append(m)
    print(f"  α={alpha:.2f}  AUC_bin={m['auc_binary']:.4f}  "
          f"f1={m['f1']:.4f}  prec={m['prec']:.4f}  rec={m['rec']:.4f}")

# find best alpha
best_blend = max(blend_results, key=lambda x: x['auc_binary'])
best_alpha = best_blend['alpha']

# generate kaggle predictions with best alpha
t_blend_kaggle = best_alpha * global_best_thr + (1 - best_alpha) * t_kaggle_oof
all_preds['blend_160k'] = (p_kaggle >= t_blend_kaggle).astype(int)
results['blend_160k'] = best_blend

print(f"\n  Best α={best_alpha:.2f}  AUC_bin={best_blend['auc_binary']:.4f}")


# ====================== 7. IDEA 3: MARGIN RE-RANKING (160k) ======================

print(f"\n{'=' * 85}")
print("  IDEA 3: Margin re-ranking (160k model)")
print(f"{'=' * 85}")

margin_val = p_val - t_val_oof
margin_cutoff, margin_auc = best_threshold(y_test, margin_val)
pred = (margin_val >= margin_cutoff).astype(int)
m = full_metrics(y_test, pred)
m['margin_cutoff'] = margin_cutoff
results['margin_160k'] = m

# kaggle
margin_kaggle = p_kaggle - t_kaggle_oof
all_preds['margin_160k'] = (margin_kaggle >= margin_cutoff).astype(int)

print(f"  Margin cutoff = {margin_cutoff:.4f}")
print(f"  AUC_bin={m['auc_binary']:.4f}  acc={m['acc']:.4f}  "
      f"f1={m['f1']:.4f}  prec={m['prec']:.4f}  rec={m['rec']:.4f}")


# ====================== 8. IDEA 1: RETRAIN ON 200K ======================

print(f"\n{'=' * 85}")
print("  IDEA 1: Retrain MLP_range on 200k (OOF 160k + meta-val 40k)")
print(f"{'=' * 85}")

# compute t* for meta-val using ensemble p_val
y_test_f = y_test.astype(float)
t_lower_test = np.where(y_test_f == 1, 0.0, p_val)
t_upper_test = np.where(y_test_f == 1, p_val, 1.0)
t_star_test = np.clip((t_lower_test + t_upper_test) / 2.0, 0.0, 1.0)

# combine train + meta-val
X_200k = np.vstack([X_features, X_test_sc])
t_star_200k = np.concatenate([t_star_oof, t_star_test])
p_200k = np.concatenate([p_oof, p_val])
y_200k = np.concatenate([y_oof, y_test_f])

print(f"  X_200k:     {X_200k.shape}")
print(f"  t_star_200k: {t_star_200k.shape}")

t0 = time.time()
mlp_200k = train_mlp_range(X_200k, t_star_200k, p_200k, y_200k, epochs=50)
print(f"  Trained in {time.time()-t0:.1f}s")

mlp_200k.eval()
with torch.no_grad():
    t_val_200k = mlp_200k(torch.FloatTensor(X_test_sc)).numpy()
    t_kaggle_200k = mlp_200k(torch.FloatTensor(X_kaggle_sc)).numpy()

# --- Idea 2 with 200k model ---
print(f"\n  Idea 2 (blend) with 200k model:")

blend_results_200k = []
for alpha in alphas:
    t_blend = alpha * global_best_thr + (1 - alpha) * t_val_200k
    pred = (p_val >= t_blend).astype(int)
    m = full_metrics(y_test, pred)
    m['alpha'] = alpha
    blend_results_200k.append(m)

best_blend_200k = max(blend_results_200k, key=lambda x: x['auc_binary'])
best_alpha_200k = best_blend_200k['alpha']

t_blend_kaggle_200k = best_alpha_200k * global_best_thr + (1 - best_alpha_200k) * t_kaggle_200k
all_preds['blend_200k'] = (p_kaggle >= t_blend_kaggle_200k).astype(int)
results['blend_200k'] = best_blend_200k

print(f"    Best α={best_alpha_200k:.2f}  AUC_bin={best_blend_200k['auc_binary']:.4f}")

# --- Idea 3 with 200k model ---
print(f"\n  Idea 3 (margin) with 200k model:")

margin_val_200k = p_val - t_val_200k
margin_cutoff_200k, _ = best_threshold(y_test, margin_val_200k)
pred = (margin_val_200k >= margin_cutoff_200k).astype(int)
m = full_metrics(y_test, pred)
m['margin_cutoff'] = margin_cutoff_200k
results['margin_200k'] = m

margin_kaggle_200k = p_kaggle - t_kaggle_200k
all_preds['margin_200k'] = (margin_kaggle_200k >= margin_cutoff_200k).astype(int)

print(f"    Margin cutoff = {margin_cutoff_200k:.4f}")
print(f"    AUC_bin={m['auc_binary']:.4f}  acc={m['acc']:.4f}  "
      f"f1={m['f1']:.4f}  prec={m['prec']:.4f}  rec={m['rec']:.4f}")


# ====================== 9. COMPARISON ======================

print(f"\n{'=' * 105}")
print("ALL RESULTS")
print(f"{'=' * 105}")
print(f"  Ensemble AUC (prob) = {auc_prob:.4f}  ← target")
print(f"  Baseline AUC (bin)  = {auc_bin_global:.4f}")

baseline_pred = (p_val >= global_best_thr).astype(int)
baseline_m = {
    'auc_binary': auc_bin_global,
    'acc': accuracy_score(y_test, baseline_pred),
    'f1': f1_score(y_test, baseline_pred),
    'prec': precision_score(y_test, baseline_pred),
    'rec': recall_score(y_test, baseline_pred),
}
all_results = {'baseline_global': baseline_m, **results}

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
    })

df = pd.DataFrame(rows).sort_values('auc_binary', ascending=False)
print(f"\n{df.to_string(index=False, float_format=lambda v: f'{v:.4f}')}")

best = df.iloc[0]
print(f"\n  Best: {best['approach']}  AUC_bin={best['auc_binary']:.4f}")
print(f"  Gap to ensemble AUC: {best['gap_to_auc_prob']:.4f}")

df.to_csv(os.path.join(THRESHOLD_DIR, 'exp020_comparison.csv'), index=False)


# ====================== 10. SUBMISSIONS ======================

print(f"\n{'=' * 85}")
print("  Submissions")
print(f"{'=' * 85}")

submitter = Submitter(submission_dir='submissions')

# baseline
baseline_kaggle_preds = (p_kaggle >= global_best_thr).astype(int)
submitter.submit_scores('exp020_baseline', baseline_kaggle_preds, test_ids,
                        threshold=global_best_thr)

# each idea
for name, preds in all_preds.items():
    submitter.submit_scores(f'exp020_{name}', preds, test_ids, threshold=None)

submitter.show_history()