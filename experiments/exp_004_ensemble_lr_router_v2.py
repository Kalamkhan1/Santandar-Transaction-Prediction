
# experiments/exp_004_ensemble_lr_router_v2.py

"""
Experiment 4: 8 models, tuned data per model from exp_003 findings.
- SMOTE only for trees (XGBoost, LightGBM)
- KDE_NB_tophat gets subsample (fast, < 10 min)
- KNN, RBF_NN, NBs: no SMOTE
Ensemble with Logistic Regression router. Generates submission file.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression

from src.preprocessing import (
    DataSplitter, RandomUndersampler, SMOter,
    PCAReducer, PLSReducer
)
from src.features import FeatureEngineer
from src.models import (
    GNBModel, KNNModel, XGBoostModel, LightGBMModel,
    RBFNNModel, GenGaussianNBModel, KDENBModel
)
from src.evaluation import Evaluator
from src.submission import Submitter


# ====================== CONFIG ======================

FEATURE_CONFIGS = {
    'GNB':              {},
    'KNN_PLS30':        {},
    'KNN_PCA30':        {},
    'XGBoost':          {'squared': True, 'row_stats': 'full', 'freq': True,
                         'bins_target': True, 'interactions': True,
                         'group_agg': True, 'top_n': 20, 'n_groups': 10},
    'LightGBM':         {'squared': True, 'row_stats': 'full', 'freq': True,
                         'bins_target': True, 'interactions': True,
                         'group_agg': True, 'top_n': 20, 'n_groups': 10},
    'RBF_NN':           {'squared': True, 'row_stats': 'basic'},
    'GenGaussian_NB':   {},
    'KDE_NB_tophat':    {},
}

# only trees get SMOTE (from exp_003 results)
SMOTE_MODELS = {'XGBoost', 'LightGBM'}

# models that get undersampled training data
UNDERSAMPLE_MODELS = {'GNB', 'KNN_PLS30', 'KNN_PCA30', 'XGBoost',
                       'LightGBM', 'GenGaussian_NB', 'KDE_NB_tophat'}

# KDE gets extra subsample to stay fast
KDE_SUBSAMPLE_PER_CLASS = 5000


def create_model(name, y_tr):
    """Factory: create model instance based on name."""
    if name == 'GNB':
        return GNBModel(var_smoothing=1e-6)

    elif name == 'KNN_PLS30':
        m = KNNModel(
            preprocessor=PLSReducer(n_components=30),
            k=301, weights='distance')
        m.name = 'KNN_PLS30'
        return m

    elif name == 'KNN_PCA30':
        m = KNNModel(
            preprocessor=PCAReducer(n_components=30),
            k=301, weights='distance')
        m.name = 'KNN_PCA30'
        return m

    elif name == 'XGBoost':
        return XGBoostModel()

    elif name == 'LightGBM':
        return LightGBMModel()

    elif name == 'RBF_NN':
        pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        return RBFNNModel(
            n_rbf=256, epochs=50, lr=0.001,
            pos_weight=pos_weight, seed=42)

    elif name == 'GenGaussian_NB':
        return GenGaussianNBModel()

    elif name == 'KDE_NB_tophat':
        return KDENBModel(kernel='tophat', bandwidth=0.5, max_train=5000)

    else:
        raise ValueError(f"Unknown model: {name}")


# ====================== 1. LOAD + SPLIT DATA ======================

print("=" * 85)
print("EXP 004: 8-model ensemble (v2) with LR router")
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

print(f"\n  X_train_sc:  {X_train_sc.shape}  y_train:  {y_train.shape}")
print(f"  X_test_sc:   {X_test_sc.shape}  y_test:   {y_test.shape}")
print(f"  X_kaggle_sc: {X_kaggle_sc.shape}  test_ids: {test_ids.shape}")


# ====================== 2. TRAIN + EVALUATE EACH MODEL ======================

evaluator = Evaluator(plot_dir='src/plots')
trained = {}

for name in FEATURE_CONFIGS:
    t0 = time.time()
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    # --- training data ---
    X_tr = X_train_sc.copy()
    y_tr = y_train.copy()

    if name in UNDERSAMPLE_MODELS:
        if name == 'KDE_NB_tophat':
            # subsample both classes for speed
            rus = RandomUndersampler(
                scope='per_class',
                targets={0: KDE_SUBSAMPLE_PER_CLASS,
                         1: KDE_SUBSAMPLE_PER_CLASS},
                seed=42)
        else:
            # undersample class 0 only
            rus = RandomUndersampler(
                scope='per_class',
                targets={0: 60000},
                seed=42)
        X_tr, y_tr = rus.fit_resample(X_tr, y_tr)

    if name in SMOTE_MODELS:
        smoter = SMOter(sampling_strategy=0.5, k_neighbors=5, seed=42)
        X_tr, y_tr = smoter.fit_resample(X_tr, y_tr)

    print(f"  Training data: {X_tr.shape}  "
          f"class 0={int((y_tr==0).sum())}  class 1={int((y_tr==1).sum())}")

    # --- feature engineering ---
    fe = FeatureEngineer(FEATURE_CONFIGS[name])
    X_tr_fe = fe.fit_transform(X_tr, y_tr)
    X_val_fe = fe.transform(X_test_sc)

    # --- create + train model ---
    model = create_model(name, y_tr)
    model.fit(X_tr_fe, y_tr)

    # --- evaluate on meta-val ---
    scores = model.predict_proba(X_val_fe)
    evaluator.add(name, y_test, scores)
    trained[name] = (model, fe)

    elapsed = time.time() - t0
    m = evaluator.results[name]
    print(f"  AUC_prob={m['auc_prob']:.4f}  AUC_bin={m['auc_binary']:.4f}  "
          f"f1={m['f1']:.4f}  time={elapsed:.1f}s")


# ====================== 3. INDIVIDUAL MODEL SUMMARY ======================

print("\n" + "=" * 85)
print("INDIVIDUAL MODELS")
print("=" * 85)
evaluator.summarize()


# ====================== 4. LR ROUTER ======================

print("\n" + "=" * 85)
print("LR ROUTER (stacking ensemble)")
print("=" * 85)

model_names = list(FEATURE_CONFIGS.keys())
score_matrix = np.column_stack([evaluator.scores[n] for n in model_names])

meta_lr = LogisticRegression(max_iter=1000, random_state=42)
meta_lr.fit(score_matrix, y_test)

print(f"\n  Intercept: {meta_lr.intercept_[0]:+.4f}")
print(f"  Weights:")
for name, w in sorted(zip(model_names, meta_lr.coef_[0]),
                       key=lambda x: -abs(x[1])):
    print(f"    {name:20s}  weight={w:+.4f}")

ensemble_scores = meta_lr.predict_proba(score_matrix)[:, 1]
evaluator.add('LR_Router_Ensemble', y_test, ensemble_scores)

print("\n" + "=" * 85)
print("ALL MODELS + ENSEMBLE")
print("=" * 85)
evaluator.summarize()


# ====================== 5. PLOTS ======================

evaluator.plot_roc(filename='Evaluator_roc_exp004.png')
evaluator.plot_binary_auc(filename='Evaluator_binary_auc_exp004.png')
evaluator.plot_j_scores(filename='Evaluator_j_scores_exp004.png')


# ====================== 6. SUBMISSION ======================

print("\n" + "=" * 85)
print("SUBMISSION")
print("=" * 85)

submitter = Submitter(submission_dir='submissions')

kaggle_score_list = []
for name in model_names:
    model, fe = trained[name]
    X_kaggle_fe = fe.transform(X_kaggle_sc)
    scores = model.predict_proba(X_kaggle_fe)
    kaggle_score_list.append(scores)

kaggle_matrix = np.column_stack(kaggle_score_list)
kaggle_probs = meta_lr.predict_proba(kaggle_matrix)[:, 1]

best_thr = evaluator.results['LR_Router_Ensemble']['best_thr']

submitter.submit_scores(
    'exp004_lr_router_v2', kaggle_probs, test_ids, threshold=best_thr
)
submitter.show_history()