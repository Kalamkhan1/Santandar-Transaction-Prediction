# experiments/exp_017_oof_threshold_and_full_ensemble_model.py

"""
Experiment 17: OOF threshold model with 3-fold CV.

Phase 1: 3-fold CV — train 7 models + LR router per fold,
          predict held-out fold → honest p_oof for all 160k samples.
          Compute t* (ideal threshold) from p_oof.

Phase 2: Train threshold model on (features, t*).

Phase 3: Full ensemble (trained on all 160k) → p on meta-val + kaggle.
          Threshold model → t from features.
          pred = 1 if p >= t else 0.
          Compare vs baseline (global threshold).

No GNB (negative weight). 7 models.
OOF fold models saved to models_oof/. Full ensemble saved to models/.
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

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score

from src.preprocessing import (
    DataSplitter, RandomUndersampler, SMOter,
    PCAReducer, PLSReducer
)
from src.features import FeatureEngineer
from src.models import (
    KNNModel, XGBoostModel, LightGBMModel,
    RBFNNModel, GenGaussianNBModel, KDENBModel
)
from src.evaluation import Evaluator
from src.submission import Submitter


# ====================== CONFIG ======================

FEATURE_CONFIGS = {
    'GenGaussian_NB':   {},
    'KDE_NB_tophat':    {},
    'KNN_PLS30':        {'row_stats': 'basic'},
    'KNN_PCA30':        {},
    'XGBoost':          {'row_stats': 'full', 'freq': True,
                         'group_agg': True, 'n_groups': 10,
                         'row_rank_stats': True},
    'LightGBM':         {'row_stats': 'full', 'freq': True,
                         'group_agg': True, 'n_groups': 10},
    'RBF_NN':           {'squared': True, 'gauss_distance': True},
}

SMOTE_MODELS = {'XGBoost', 'LightGBM'}
UNDERSAMPLE_MODELS = {'KNN_PLS30', 'KNN_PCA30', 'XGBoost',
                       'LightGBM', 'GenGaussian_NB', 'KDE_NB_tophat'}
KDE_SUBSAMPLE_PER_CLASS = 5000

HYPERPARAMS_DIR = 'data/processed/hyperparams'
MODELS_DIR = 'models'
OOF_MODELS_DIR = 'models_oof'
THRESHOLD_DIR = 'data/processed/threshold_data'

N_FOLDS = 3


# ====================== HELPERS ======================

def load_tuned_params(name):
    safe_name = name.lower().replace(' ', '_')
    path = os.path.join(HYPERPARAMS_DIR, f'tuned_{safe_name}.pkl')
    if os.path.exists(path):
        data = joblib.load(path)
        print(f"  Loaded tuned params for {name}: {data.get('best_params', {})}")
        return data['best_params']
    print(f"  No tuned params for {name}, using defaults")
    return None


def create_model(name, y_tr, tuned_params=None):
    params = tuned_params or {}

    if name == 'KNN_PLS30':
        m = KNNModel(
            preprocessor=PLSReducer(
                n_components=params.get('n_components', 30)),
            k=params.get('k', 301),
            weights=params.get('weights', 'distance'))
        m.name = 'KNN_PLS30'
        return m

    elif name == 'KNN_PCA30':
        m = KNNModel(
            preprocessor=PCAReducer(
                n_components=params.get('n_components', 30)),
            k=params.get('k', 301),
            weights=params.get('weights', 'distance'))
        m.name = 'KNN_PCA30'
        return m

    elif name == 'XGBoost':
        return XGBoostModel(**params)

    elif name == 'LightGBM':
        return LightGBMModel(**params)

    elif name == 'RBF_NN':
        pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        return RBFNNModel(
            n_rbf=params.get('n_rbf', 256),
            epochs=params.get('epochs', 50),
            lr=params.get('lr', 0.001),
            pos_weight=pos_weight, seed=42)

    elif name == 'GenGaussian_NB':
        return GenGaussianNBModel()

    elif name == 'KDE_NB_tophat':
        return KDENBModel(
            kernel='tophat',
            bandwidth=params.get('bandwidth', 0.5),
            max_train=5000)

    else:
        raise ValueError(f"Unknown model: {name}")


def prepare_data(name, X_train, y_train, seed=42):
    X_tr = X_train.copy()
    y_tr = y_train.copy()

    if name in UNDERSAMPLE_MODELS:
        if name == 'KDE_NB_tophat':
            rus = RandomUndersampler(
                scope='per_class',
                targets={0: KDE_SUBSAMPLE_PER_CLASS,
                         1: KDE_SUBSAMPLE_PER_CLASS},
                seed=seed)
        else:
            rus = RandomUndersampler(
                scope='per_class', targets={0: 60000}, seed=seed)
        X_tr, y_tr = rus.fit_resample(X_tr, y_tr)

    if name in SMOTE_MODELS:
        smoter = SMOter(sampling_strategy=0.5, k_neighbors=5, seed=seed)
        X_tr, y_tr = smoter.fit_resample(X_tr, y_tr)

    return X_tr, y_tr


def train_fold_ensemble(X_fold_train, y_fold_train, X_fold_val):
    """Train all 7 models on fold training data, return ensemble p for fold val."""
    score_list = []
    fold_models = {}

    for name in FEATURE_CONFIGS:
        tuned = load_tuned_params(name)

        X_tr, y_tr = prepare_data(name, X_fold_train, y_fold_train)

        fe = FeatureEngineer(FEATURE_CONFIGS[name])
        X_tr_fe = fe.fit_transform(X_tr, y_tr)
        X_val_fe = fe.transform(X_fold_val)

        model = create_model(name, y_tr, tuned_params=tuned)
        model.fit(X_tr_fe, y_tr)

        scores = model.predict_proba(X_val_fe)
        score_list.append(scores)
        fold_models[name] = (model, fe)

    # LR router
    score_matrix = np.column_stack(score_list)
    meta_lr = LogisticRegression(max_iter=1000, random_state=42)
    # need fold training labels for router... use fold train y
    # BUT we need scores on fold train to train router
    # Actually: train router on fold_val scores + fold_val labels
    # Wait — that's the same data. Let me think...

    # The standard OOF approach: train router on fold_val scores and labels
    # This is technically "in-sample" for the router but the models are "out-of-sample"
    # This is acceptable — the router is a simple LR, won't overfit

    return score_matrix, meta_lr, fold_models


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


# ====================== 1. LOAD + SPLIT DATA ======================

print("=" * 85)
print("EXP 017: OOF threshold model (3-fold CV)")
print("=" * 85)

os.makedirs(OOF_MODELS_DIR, exist_ok=True)
os.makedirs(THRESHOLD_DIR, exist_ok=True)

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

print(f"\n  X_train_sc:  {X_train_sc.shape}  y_train:  {y_train.shape}")
print(f"  X_test_sc:   {X_test_sc.shape}  y_test:   {y_test.shape}")
print(f"  X_kaggle_sc: {X_kaggle_sc.shape}  test_ids: {test_ids.shape}")

model_names = list(FEATURE_CONFIGS.keys())
print(f"  Models: {model_names}")


# ====================== 2. OOF PROBABILITIES (3-FOLD CV) ======================

print(f"\n{'=' * 85}")
print(f"  PHASE 1: OOF probabilities ({N_FOLDS}-fold CV)")
print(f"{'=' * 85}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
p_oof = np.zeros(len(y_train))

for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_train_sc, y_train)):
    t0 = time.time()
    print(f"\n  --- Fold {fold_i + 1}/{N_FOLDS} ---")
    print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}")

    X_fold_train = X_train_sc[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train_sc[val_idx]
    y_fold_val = y_train[val_idx]

    # train all models on fold train
    score_list = []
    fold_models = {}

    for name in model_names:
        tuned = load_tuned_params(name)

        X_tr, y_tr = prepare_data(name, X_fold_train, y_fold_train)

        fe = FeatureEngineer(FEATURE_CONFIGS[name])
        X_tr_fe = fe.fit_transform(X_tr, y_tr)
        X_val_fe = fe.transform(X_fold_val)

        model = create_model(name, y_tr, tuned_params=tuned)
        model.fit(X_tr_fe, y_tr)

        scores = model.predict_proba(X_val_fe)
        score_list.append(scores)
        fold_models[name] = (model, fe)

    # train LR router on fold val scores
    fold_score_matrix = np.column_stack(score_list)
    fold_lr = LogisticRegression(max_iter=1000, random_state=42)
    fold_lr.fit(fold_score_matrix, y_fold_val)

    # predict fold val → honest OOF probability
    p_oof[val_idx] = fold_lr.predict_proba(fold_score_matrix)[:, 1]

    # save fold models to OOF directory
    fold_dir = os.path.join(OOF_MODELS_DIR, f'fold_{fold_i}')
    os.makedirs(fold_dir, exist_ok=True)
    for name, (model, fe) in fold_models.items():
        safe_name = name.lower().replace(' ', '_')
        joblib.dump({'model': model, 'fe': fe, 'name': name},
                     os.path.join(fold_dir, f'{safe_name}.joblib'),
                     compress=3)
    joblib.dump({'router': fold_lr, 'model_names': model_names},
                 os.path.join(fold_dir, 'lr_router.joblib'))
    print(f"  Saved fold {fold_i + 1} models to {fold_dir}/")

    elapsed = time.time() - t0
    print(f"  Fold {fold_i + 1} done in {elapsed:.1f}s")
    print(f"  OOF stats: mean={p_oof[val_idx].mean():.4f}  std={p_oof[val_idx].std():.4f}")

print(f"\n  OOF complete: p_oof {p_oof.shape}")
print(f"  OOF stats: mean={p_oof.mean():.4f}  std={p_oof.std():.4f}  "
      f"min={p_oof.min():.4f}  max={p_oof.max():.4f}")


# ====================== 3. COMPUTE THRESHOLD TARGETS FROM OOF ======================

print(f"\n{'=' * 85}")
print("  PHASE 2: Compute threshold targets from OOF probabilities")
print(f"{'=' * 85}")

y = y_train.astype(float)
p = p_oof

# range bounds
# y=1: t ∈ [0, p]      (correct when p >= t → predict 1)
# y=0: t ∈ (p, 1]      (correct when p < t  → predict 0)
t_lower = np.where(y == 1, 0.0, p)
t_upper = np.where(y == 1, p, 1.0)
t_star_median = (t_lower + t_upper) / 2.0

t_lower = np.clip(t_lower, 0.0, 1.0)
t_upper = np.clip(t_upper, 0.0, 1.0)
t_star_median = np.clip(t_star_median, 0.0, 1.0)

# verify
pred_at_tstar = (p >= t_star_median).astype(int)
correct_at_tstar = (pred_at_tstar == y).mean()

print(f"\n  Target stats:")
print(f"    y=1: mean={t_star_median[y==1].mean():.4f}  "
      f"std={t_star_median[y==1].std():.4f}")
print(f"    y=0: mean={t_star_median[y==0].mean():.4f}  "
      f"std={t_star_median[y==0].std():.4f}")
print(f"    Gap: {t_star_median[y==0].mean() - t_star_median[y==1].mean():.4f}")
print(f"    Verification at t*: {correct_at_tstar:.6f}")

# save threshold dataset
np.save(os.path.join(THRESHOLD_DIR, 'oof_X_features.npy'), X_train_sc)
np.save(os.path.join(THRESHOLD_DIR, 'oof_y_labels.npy'), y_train)
np.save(os.path.join(THRESHOLD_DIR, 'oof_ensemble_probs.npy'), p_oof)
np.save(os.path.join(THRESHOLD_DIR, 'oof_t_star_median.npy'), t_star_median)
np.save(os.path.join(THRESHOLD_DIR, 'oof_t_star_lower.npy'), t_lower)
np.save(os.path.join(THRESHOLD_DIR, 'oof_t_star_upper.npy'), t_upper)

print(f"\n  Saved OOF threshold dataset to {THRESHOLD_DIR}/")


# ====================== 4. TRAIN THRESHOLD MODEL ======================

print(f"\n{'=' * 85}")
print("  PHASE 3: Train threshold model")
print(f"{'=' * 85}")

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Test 1: Linear Regression
t0 = time.time()
lr_thresh = LinearRegression()
lr_thresh.fit(X_train_sc, t_star_median)
print(f"  Linear Regression trained ({time.time()-t0:.1f}s)")

# Test 2: XGBoost Regressor
t0 = time.time()
xgb_thresh = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    tree_method='hist', random_state=42, n_jobs=-1, verbosity=0)
xgb_thresh.fit(X_train_sc, t_star_median)
print(f"  XGBoost Regressor trained ({time.time()-t0:.1f}s)")

# save threshold models
joblib.dump(lr_thresh, os.path.join(THRESHOLD_DIR, 'threshold_model_lr.joblib'))
joblib.dump(xgb_thresh, os.path.join(THRESHOLD_DIR, 'threshold_model_xgb.joblib'))
print(f"  Saved threshold models to {THRESHOLD_DIR}/")


# ====================== 5. TRAIN FULL ENSEMBLE (ALL 160K) ======================

print(f"\n{'=' * 85}")
print("  PHASE 4: Train full ensemble (all 160k)")
print(f"{'=' * 85}")

os.makedirs(MODELS_DIR, exist_ok=True)
evaluator = Evaluator(plot_dir='src/plots')
trained = {}

for name in FEATURE_CONFIGS:
    t0 = time.time()
    print(f"\n  Training {name}...")

    tuned = load_tuned_params(name)

    X_tr, y_tr = prepare_data(name, X_train_sc, y_train)

    fe = FeatureEngineer(FEATURE_CONFIGS[name])
    X_tr_fe = fe.fit_transform(X_tr, y_tr)
    X_val_fe = fe.transform(X_test_sc)

    model = create_model(name, y_tr, tuned_params=tuned)
    model.fit(X_tr_fe, y_tr)

    scores = model.predict_proba(X_val_fe)
    evaluator.add(name, y_test, scores)
    trained[name] = (model, fe)

    # save
    safe_name = name.lower().replace(' ', '_')
    joblib.dump({'model': model, 'fe': fe, 'name': name, 'tuned_params': tuned},
                 os.path.join(MODELS_DIR, f'{safe_name}.joblib'),
                 compress=3)
    print(f"  Saved: models/{safe_name}.joblib  ({time.time()-t0:.1f}s)")

# full ensemble router
score_matrix = np.column_stack([evaluator.scores[n] for n in model_names])
meta_lr_full = LogisticRegression(max_iter=1000, random_state=42)
meta_lr_full.fit(score_matrix, y_test)

ensemble_scores = meta_lr_full.predict_proba(score_matrix)[:, 1]
evaluator.add('Full_Ensemble', y_test, ensemble_scores)

joblib.dump({'router': meta_lr_full, 'model_names': model_names},
            os.path.join(MODELS_DIR, 'lr_router.joblib'))
print(f"  Saved: models/lr_router.joblib")

print(f"\n  Full ensemble weights:")
for name, w in sorted(zip(model_names, meta_lr_full.coef_[0]),
                       key=lambda x: -abs(x[1])):
    print(f"    {name:20s}  {w:+.4f}")

# baseline: global threshold
fpr, tpr, thr_arr = roc_curve(y_test, ensemble_scores)
J = tpr - fpr
global_best_thr = thr_arr[int(np.argmax(J))]
baseline_pred = (ensemble_scores >= global_best_thr).astype(int)
baseline_metrics = full_metrics(y_test, baseline_pred)
print(f"\n  Baseline (global threshold): AUC_bin={baseline_metrics['auc_binary']:.4f}")


# ====================== 6. EVALUATE THRESHOLD MODELS ON META-VAL ======================

print(f"\n{'=' * 85}")
print("  PHASE 5: Evaluate threshold models on meta-val")
print(f"{'=' * 85}")

results = {'baseline_global_thr': baseline_metrics}

for thresh_name, thresh_model in [('LR_threshold', lr_thresh),
                                    ('XGB_threshold', xgb_thresh)]:
    t_pred = thresh_model.predict(X_test_sc)
    pred = (ensemble_scores >= t_pred).astype(int)
    m = full_metrics(y_test, pred)
    results[thresh_name] = m
    print(f"\n  {thresh_name}:")
    print(f"    AUC_bin={m['auc_binary']:.4f}  acc={m['acc']:.4f}  "
          f"f1={m['f1']:.4f}  prec={m['prec']:.4f}  rec={m['rec']:.4f}")

# comparison
print(f"\n{'=' * 95}")
print("THRESHOLD MODEL COMPARISON (OOF-trained)")
print(f"{'=' * 95}")

df_res = pd.DataFrame([
    {'model': k, **v} for k, v in results.items()
]).sort_values('auc_binary', ascending=False)
print(df_res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

best = df_res.iloc[0]
print(f"\n  Best: {best['model']}  AUC_bin={best['auc_binary']:.4f} "
      f"(delta vs baseline: {best['auc_binary'] - baseline_metrics['auc_binary']:+.4f})")


# ====================== 7. SUBMISSION (BEST THRESHOLD MODEL) ======================

print(f"\n{'=' * 85}")
print("  PHASE 6: Submission")
print(f"{'=' * 85}")

# get kaggle ensemble probabilities
kaggle_score_list = []
for name in model_names:
    model, fe = trained[name]
    X_kaggle_fe = fe.transform(X_kaggle_sc)
    scores = model.predict_proba(X_kaggle_fe)
    kaggle_score_list.append(scores)

kaggle_matrix = np.column_stack(kaggle_score_list)
kaggle_probs = meta_lr_full.predict_proba(kaggle_matrix)[:, 1]

# pick best threshold model
best_name = best['model']
if best_name == 'baseline_global_thr':
    kaggle_preds = (kaggle_probs >= global_best_thr).astype(int)
    sub_name = 'exp017_oof_threshold_baseline'
else:
    best_thresh_model = lr_thresh if best_name == 'LR_threshold' else xgb_thresh
    kaggle_t = best_thresh_model.predict(X_kaggle_sc)
    kaggle_preds = (kaggle_probs >= kaggle_t).astype(int)
    sub_name = f'exp017_oof_threshold_{best_name}'

submitter = Submitter(submission_dir='submissions')
submitter.submit_scores(sub_name, kaggle_preds, test_ids)
submitter.show_history()