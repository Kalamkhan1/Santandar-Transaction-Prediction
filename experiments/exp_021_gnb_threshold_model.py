# experiments/exp_021_gnb_threshold_model.py

"""
Experiment 21: Generalized GNB + Threshold Model.
Bagging 50 for OOF probabilities. t_fold-corrected ranges.
MLP + RangeAwareLoss for threshold prediction. Single submission.
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
from src.models import GenGaussianNBModel
from src.submission import Submitter


# ====================== CONFIG ======================

N_BAGS = 50
MAX_TRAIN_PER_CLASS = 5000  # subsample per class for GenGaussian speed
OUTPUT_DIR = 'data/processed/threshold_data'
CACHE_DIR = 'data/processed/ensemble_cache'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ====================== HELPERS ======================

def best_threshold(y_true, scores):
    fpr, tpr, thr_arr = roc_curve(y_true, scores)
    J = tpr - fpr
    idx = int(np.argmax(J))
    return thr_arr[idx]


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
print("EXP 021: Generalized GNB + Threshold Model (Bagging 50)")
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


# ====================== 2. BAGGING 50 → OOF PROBABILITIES ======================

print(f"\n{'=' * 85}")
print(f"  PHASE 1: Bagging {N_BAGS} OOF probabilities (GenGaussianNB)")
print(f"{'=' * 85}")

n = len(y_train)
rng_master = np.random.RandomState(42)

# accumulators
oof_sum = np.zeros(n)
oof_count = np.zeros(n)
tfold_sum = np.zeros(n)
tfold_count = np.zeros(n)

for bag_i in range(N_BAGS):
    t0 = time.time()

    # bootstrap sample
    rng = np.random.RandomState(42 + bag_i)
    bag_idx = rng.choice(n, size=n, replace=True)
    oob_mask = np.ones(n, dtype=bool)
    oob_mask[bag_idx] = False
    oob_idx = np.where(oob_mask)[0]

    if len(oob_idx) == 0:
        continue

    X_bag = X_train_sc[bag_idx]
    y_bag = y_train[bag_idx]

    # subsample for GenGaussian speed (per class)
    c0_idx = np.where(y_bag == 0)[0]
    c1_idx = np.where(y_bag == 1)[0]

    if len(c0_idx) > MAX_TRAIN_PER_CLASS:
        c0_sel = rng.choice(c0_idx, size=MAX_TRAIN_PER_CLASS, replace=False)
    else:
        c0_sel = c0_idx

    if len(c1_idx) > MAX_TRAIN_PER_CLASS:
        c1_sel = rng.choice(c1_idx, size=MAX_TRAIN_PER_CLASS, replace=False)
    else:
        c1_sel = c1_idx

    fit_idx = np.concatenate([c0_sel, c1_sel])
    X_fit = X_bag[fit_idx]
    y_fit = y_bag[fit_idx]

    # train GenGaussianNB
    model = GenGaussianNBModel(seed=42)
    model.fit(X_fit, y_fit)

    # predict OOB
    p_oob = model.predict_proba(X_train_sc[oob_idx])

    # t_fold = Youden's J on this bag's predictions (using training labels)
    t_fold = best_threshold(y_train[oob_idx], p_oob)

    # accumulate
    oof_sum[oob_idx] += p_oob
    oof_count[oob_idx] += 1
    tfold_sum[oob_idx] += t_fold
    tfold_count[oob_idx] += 1

    elapsed = time.time() - t0
    print(f"  Bag {bag_i+1:2d}/{N_BAGS}:  OOB={len(oob_idx):>6}  "
          f"t_fold={t_fold:.4f}  ({elapsed:.1f}s)")

# average
p_oof = np.divide(oof_sum, np.maximum(oof_count, 1))
t_oof = np.divide(tfold_sum, np.maximum(tfold_count, 1))

# check coverage
missing = (oof_count == 0).sum()
print(f"\n  OOF complete. Missing (never OOB): {missing}")
print(f"  p_oof: mean={p_oof.mean():.4f}  std={p_oof.std():.4f}  "
      f"min={p_oof.min():.4f}  max={p_oof.max():.4f}")
print(f"  t_oof: mean={t_oof.mean():.4f}  std={t_oof.std():.4f}  "
      f"min={t_oof.min():.4f}  max={t_oof.max():.4f}")

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
t_fold = t_oof

# model prediction
pred = (p >= t_fold).astype(int)
correct = (pred == y)

print(f"  Model correct: {correct.sum()}/{len(y)} ({correct.mean()*100:.1f}%)")
print(f"  Model wrong:   {(~correct).sum()}/{len(y)} ({(~correct).mean()*100:.1f}%)")

# when right: use p for range
# when wrong: use t_fold for range
t_lower = np.where(y == 1,
    np.where(correct, 0.0, 0.0),           # y=1 always starts at 0
    np.where(correct, p, t_fold))           # y=0: lower = p (right) or t_fold (wrong)

t_upper = np.where(y == 1,
    np.where(correct, p, t_fold),           # y=1: upper = p (right) or t_fold (wrong)
    np.where(correct, 1.0, 1.0))            # y=0 always ends at 1

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
np.save(os.path.join(OUTPUT_DIR, 'gnb_X_features.npy'), X_train_sc)
np.save(os.path.join(OUTPUT_DIR, 'gnb_y_labels.npy'), y_train)
np.save(os.path.join(OUTPUT_DIR, 'gnb_oof_probs.npy'), p_oof)
np.save(os.path.join(OUTPUT_DIR, 'gnb_t_fold.npy'), t_oof)
np.save(os.path.join(OUTPUT_DIR, 'gnb_t_star.npy'), t_star_median)
np.save(os.path.join(OUTPUT_DIR, 'gnb_t_lower.npy'), t_lower)
np.save(os.path.join(OUTPUT_DIR, 'gnb_t_upper.npy'), t_upper)

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

# save threshold model
joblib.dump(mlp, os.path.join(OUTPUT_DIR, 'gnb_threshold_mlp.joblib'))
print(f"  Saved: {OUTPUT_DIR}/gnb_threshold_mlp.joblib")


# ====================== 5. GENGAUSSIAN ON FULL DATA → KAGGLE ======================

print(f"\n{'=' * 85}")
print("  PHASE 4: GenGaussianNB on full data → kaggle")
print(f"{'=' * 85}")

# train on full 160k (with subsample for speed)
rng = np.random.RandomState(42)
c0_idx = np.where(y_train == 0)[0]
c1_idx = np.where(y_train == 1)[0]

if len(c0_idx) > 10000:
    c0_sel = rng.choice(c0_idx, size=10000, replace=False)
else:
    c0_sel = c0_idx

if len(c1_idx) > 10000:
    c1_sel = rng.choice(c1_idx, size=10000, replace=False)
else:
    c1_sel = c1_idx

fit_idx = np.concatenate([c0_sel, c1_sel])
X_fit_full = X_train_sc[fit_idx]
y_fit_full = y_train[fit_idx]

t0 = time.time()
gnb_full = GenGaussianNBModel(seed=42)
gnb_full.fit(X_fit_full, y_fit_full)
print(f"  GenGaussianNB trained on {len(fit_idx)} samples ({time.time()-t0:.1f}s)")

# predict kaggle
t0 = time.time()
p_kaggle = gnb_full.predict_proba(X_kaggle_sc)
print(f"  Kaggle predicted ({time.time()-t0:.1f}s)")
print(f"  p_kaggle: mean={p_kaggle.mean():.4f}  std={p_kaggle.std():.4f}")

# threshold model predicts t for kaggle
mlp.eval()
with torch.no_grad():
    t_kaggle = mlp(torch.FloatTensor(X_kaggle_sc)).numpy()

print(f"  t_kaggle: mean={t_kaggle.mean():.4f}  std={t_kaggle.std():.4f}")


# ====================== 6. SUBMIT ======================

print(f"\n{'=' * 85}")
print("  PHASE 5: Submission")
print(f"{'=' * 85}")

preds = (p_kaggle >= t_kaggle).astype(int)
print(f"  Pred positives: {preds.sum()} ({preds.mean()*100:.2f}%)")

submitter = Submitter(submission_dir='submissions')
submitter.submit_scores('exp021_gnb_threshold', preds, test_ids, threshold=None)
submitter.show_history()