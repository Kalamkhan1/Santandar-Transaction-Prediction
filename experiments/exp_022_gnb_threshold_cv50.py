# experiments/exp_022_gnb_threshold_cv50.py

"""
Experiment 22: Generalized GNB + Threshold Model (CV 50).
Stratified CV 50 for OOF probabilities (correct priors).
t_fold-corrected ranges. MLP + RangeAwareLoss.
Evaluates on meta-val. Single submission.
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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score

from src.preprocessing import DataSplitter
from src.models import GenGaussianNBModel
from src.submission import Submitter


# ====================== CONFIG ======================

N_FOLDS = 50
MAX_TRAIN = 20000  # proportional subsample per fold (keeps 90/10 ratio)
OUTPUT_DIR = 'data/processed/threshold_data'

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


# ====================== 1. LOAD DATA ======================

print("=" * 85)
print("EXP 022: Generalized GNB + Threshold Model (CV 50)")
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

print(f"\n  X_train_sc:  {X_train_sc.shape}  y_train: {y_train.shape}")
print(f"  X_test_sc:   {X_test_sc.shape}  y_test:   {y_test.shape}")


# ====================== 2. CV 50 → OOF PROBABILITIES ======================

print(f"\n{'=' * 85}")
print(f"  PHASE 1: CV {N_FOLDS} OOF probabilities (GenGaussianNB)")
print(f"{'=' * 85}")

n = len(y_train)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

p_oof = np.zeros(n)
t_fold_arr = np.zeros(n)

for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_train_sc, y_train)):
    t0 = time.time()

    X_fold_train = X_train_sc[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train_sc[val_idx]
    y_fold_val = y_train[val_idx]

    # proportional subsample (keeps 90/10 ratio → correct priors)
    rng = np.random.RandomState(42 + fold_i)
    if len(y_fold_train) > MAX_TRAIN:
        sub_idx = rng.choice(len(y_fold_train), size=MAX_TRAIN, replace=False)
    else:
        sub_idx = np.arange(len(y_fold_train))

    X_fit = X_fold_train[sub_idx]
    y_fit = y_fold_train[sub_idx]

    # verify class ratio preserved
    ratio = (y_fit == 1).mean()

    # train GenGaussianNB
    model = GenGaussianNBModel(seed=42)
    model.fit(X_fit, y_fit)

    # predict fold val
    p_fold = model.predict_proba(X_fold_val)
    p_oof[val_idx] = p_fold

    # t_fold = Youden's J on this fold's predictions
    t_fold, _ = best_threshold(y_fold_val, p_fold)
    t_fold_arr[val_idx] = t_fold

    elapsed = time.time() - t0
    if (fold_i + 1) % 5 == 0 or fold_i == 0:
        print(f"  Fold {fold_i+1:2d}/{N_FOLDS}:  "
              f"train={len(train_idx)}  val={len(val_idx)}  "
              f"ratio={ratio:.3f}  t_fold={t_fold:.4f}  ({elapsed:.1f}s)")

# summary
print(f"\n  OOF complete.")
print(f"  p_oof: mean={p_oof.mean():.4f}  std={p_oof.std():.4f}  "
      f"min={p_oof.min():.4f}  max={p_oof.max():.4f}")
print(f"  t_fold: mean={t_fold_arr.mean():.4f}  std={t_fold_arr.std():.4f}  "
      f"min={t_fold_arr.min():.4f}  max={t_fold_arr.max():.4f}")

# GNB AUC on OOF
fpr, tpr, thr_arr = roc_curve(y_train, p_oof)
auc_prob_oof = auc(fpr, tpr)
J = tpr - fpr
auc_bin_oof = (1 + J.max()) / 2
print(f"\n  GNB OOF: AUC_prob={auc_prob_oof:.4f}  AUC_bin={auc_bin_oof:.4f}")


# ====================== 3. COMPUTE THRESHOLD TARGETS ======================

print(f"\n{'=' * 85}")
print("  PHASE 2: Compute threshold targets (t_fold-corrected ranges)")
print(f"{'=' * 85}")

y = y_train.astype(float)
p = p_oof
t_fold = t_fold_arr

# model prediction
pred = (p >= t_fold).astype(int)
correct = (pred == y)

print(f"  Model correct: {correct.sum()}/{len(y)} ({correct.mean()*100:.1f}%)")
print(f"  Model wrong:   {(~correct).sum()}/{len(y)} ({(~correct).mean()*100:.1f}%)")

# when right: use p for range
# when wrong: use t_fold for range
t_lower = np.where(y == 1,
    0.0,
    np.where(correct, p, t_fold))

t_upper = np.where(y == 1,
    np.where(correct, p, t_fold),
    1.0)

t_star_median = (t_lower + t_upper) / 2.0

t_lower = np.clip(t_lower, 0.0, 1.0)
t_upper = np.clip(t_upper, 0.0, 1.0)
t_star_median = np.clip(t_star_median, 0.0, 1.0)

# verify
pred_at_tstar = (p >= t_star_median).astype(int)
correct_at_tstar = (pred_at_tstar == y).mean()

print(f"\n  Target stats:")
print(f"    y=1: mean={t_star_median[y==1].mean():.4f}  std={t_star_median[y==1].std():.4f}")
print(f"    y=0: mean={t_star_median[y==0].mean():.4f}  std={t_star_median[y==0].std():.4f}")
print(f"    Gap: {t_star_median[y==0].mean() - t_star_median[y==1].mean():.4f}")
print(f"    Verification at t*: {correct_at_tstar:.6f}")

# save
np.save(os.path.join(OUTPUT_DIR, 'gnb_cv50_X_features.npy'), X_train_sc)
np.save(os.path.join(OUTPUT_DIR, 'gnb_cv50_y_labels.npy'), y_train)
np.save(os.path.join(OUTPUT_DIR, 'gnb_cv50_oof_probs.npy'), p_oof)
np.save(os.path.join(OUTPUT_DIR, 'gnb_cv50_t_fold.npy'), t_fold_arr)
np.save(os.path.join(OUTPUT_DIR, 'gnb_cv50_t_star.npy'), t_star_median)
np.save(os.path.join(OUTPUT_DIR, 'gnb_cv50_t_lower.npy'), t_lower)
np.save(os.path.join(OUTPUT_DIR, 'gnb_cv50_t_upper.npy'), t_upper)

print(f"\n  Saved to {OUTPUT_DIR}/")


# ====================== 4. TRAIN THRESHOLD MODEL ======================

print(f"\n{'=' * 85}")
print("  PHASE 3: Train MLP + RangeAwareLoss threshold model")
print(f"{'=' * 85}")

t0 = time.time()
torch.manual_seed(42)
mlp = MLPThreshold(n_input=X_train_sc.shape[1])
opt = torch.optim.Adam(mlp.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = RangeAwareLoss(M=10.0, lam=0.1)

X_t = torch.FloatTensor(X_train_sc)
p_t = torch.FloatTensor(p_oof)
y_t = torch.FloatTensor(y_train.astype(float))
dl = DataLoader(TensorDataset(X_t, p_t, y_t), batch_size=4096, shuffle=True)

for epoch in range(50):
    mlp.train()
    epoch_loss = 0
    for Xb, pb, yb in dl:
        opt.zero_grad()
        t_pred = mlp(Xb)
        loss = loss_fn(t_pred, pb, yb)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1}: loss={epoch_loss/len(dl):.4f}")

print(f"  Trained in {time.time()-t0:.1f}s")

joblib.dump(mlp, os.path.join(OUTPUT_DIR, 'gnb_cv50_threshold_mlp.joblib'))
print(f"  Saved: {OUTPUT_DIR}/gnb_cv50_threshold_mlp.joblib")


# ====================== 5. GENGAUSSIAN ON FULL DATA → META-VAL + KAGGLE ======================

print(f"\n{'=' * 85}")
print("  PHASE 4: GenGaussianNB on full data → meta-val + kaggle")
print(f"{'=' * 85}")

# proportional subsample for speed (keeps 90/10 ratio)
rng = np.random.RandomState(42)
MAX_FULL = 20000
if len(y_train) > MAX_FULL:
    fit_idx = rng.choice(len(y_train), size=MAX_FULL, replace=False)
else:
    fit_idx = np.arange(len(y_train))

X_fit_full = X_train_sc[fit_idx]
y_fit_full = y_train[fit_idx]
print(f"  Fitting on {len(fit_idx)} samples (ratio: {(y_fit_full==1).mean():.3f})")

t0 = time.time()
gnb_full = GenGaussianNBModel(seed=42)
gnb_full.fit(X_fit_full, y_fit_full)
print(f"  Trained in {time.time()-t0:.1f}s")

# predict meta-val
t0 = time.time()
p_val = gnb_full.predict_proba(X_test_sc)
print(f"  Meta-val predicted ({time.time()-t0:.1f}s)")

# predict kaggle
t0 = time.time()
p_kaggle = gnb_full.predict_proba(X_kaggle_sc)
print(f"  Kaggle predicted ({time.time()-t0:.1f}s)")

print(f"  p_val:    mean={p_val.mean():.4f}  std={p_val.std():.4f}")
print(f"  p_kaggle: mean={p_kaggle.mean():.4f}  std={p_kaggle.std():.4f}")


# ====================== 6. EVALUATE ON META-VAL ======================

print(f"\n{'=' * 85}")
print("  PHASE 5: Evaluate on meta-val")
print(f"{'=' * 85}")

# baseline: global threshold on GNB p
fpr, tpr, thr_arr = roc_curve(y_test, p_val)
J = tpr - fpr
auc_prob_val = auc(fpr, tpr)
idx = int(np.argmax(J))
global_best_thr = thr_arr[idx]
baseline_pred = (p_val >= global_best_thr).astype(int)
baseline_m = full_metrics(y_test, baseline_pred)

print(f"  GNB AUC_prob (meta-val) = {auc_prob_val:.4f}")
print(f"  Baseline (global thr={global_best_thr:.4f}):")
print(f"    AUC_bin={baseline_m['auc_binary']:.4f}  acc={baseline_m['acc']:.4f}  "
      f"f1={baseline_m['f1']:.4f}  prec={baseline_m['prec']:.4f}  rec={baseline_m['rec']:.4f}")

# threshold model
mlp.eval()
with torch.no_grad():
    t_val = mlp(torch.FloatTensor(X_test_sc)).numpy()
    t_kaggle = mlp(torch.FloatTensor(X_kaggle_sc)).numpy()

pred_thresh = (p_val >= t_val).astype(int)
thresh_m = full_metrics(y_test, pred_thresh)

print(f"\n  Threshold model (per-sample t):")
print(f"    AUC_bin={thresh_m['auc_binary']:.4f}  acc={thresh_m['acc']:.4f}  "
      f"f1={thresh_m['f1']:.4f}  prec={thresh_m['prec']:.4f}  rec={thresh_m['rec']:.4f}")

# comparison
print(f"\n  Comparison:")
print(f"    Baseline:         AUC_bin={baseline_m['auc_binary']:.4f}")
print(f"    Threshold model:  AUC_bin={thresh_m['auc_binary']:.4f}")
print(f"    Delta:            {thresh_m['auc_binary'] - baseline_m['auc_binary']:+.4f}")
print(f"    Gap to AUC_prob:  {auc_prob_val - thresh_m['auc_binary']:.4f}")
print(f"    t_val:  mean={t_val.mean():.4f}  std={t_val.std():.4f}")
print(f"    t_kaggle: mean={t_kaggle.mean():.4f}  std={t_kaggle.std():.4f}")


# ====================== 7. SUBMIT ======================

print(f"\n{'=' * 85}")
print("  PHASE 6: Submission")
print(f"{'=' * 85}")

preds = (p_kaggle >= t_kaggle).astype(int)
print(f"  Pred positives: {preds.sum()} ({preds.mean()*100:.2f}%)")

submitter = Submitter(submission_dir='submissions')
submitter.submit_scores('exp022_gnb_cv50_threshold', preds, test_ids, threshold=None)
submitter.show_history()