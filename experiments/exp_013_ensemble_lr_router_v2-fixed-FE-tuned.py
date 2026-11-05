# experiments/exp_013_ensemble_lr_router_v2-fixed-FE-tuned.py

"""
Experiment 13: 8 models with tuned hyperparams + per-model feature engineering.
SMOTE only for trees. Ensemble with Logistic Regression router.
Loads tuned params from data/processed/hyperparams/ if available.
Saves trained models to models/. Generates submission file.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import time
import joblib
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
UNDERSAMPLE_MODELS = {'GNB', 'KNN_PLS30', 'KNN_PCA30', 'XGBoost',
                       'LightGBM', 'GenGaussian_NB', 'KDE_NB_tophat'}
KDE_SUBSAMPLE_PER_CLASS = 5000

HYPERPARAMS_DIR = 'data/processed/hyperparams'
MODELS_DIR = 'models'


# ====================== LOAD TUNED PARAMS ======================

def load_tuned_params(name):
    """Load tuned params if available, return None otherwise."""
    safe_name = name.lower().replace(' ', '_')
    path = os.path.join(HYPERPARAMS_DIR, f'tuned_{safe_name}.pkl')
    if os.path.exists(path):
        data = joblib.load(path)
        print(f"  Loaded tuned params for {name}: {data.get('best_params', {})}")
        return data['best_params']
    print(f"  No tuned params for {name}, using defaults")
    return None


def create_model(name, y_tr, tuned_params=None):
    """Factory: create model with tuned params if available, else defaults."""
    params = tuned_params or {}

    if name == 'GNB':
        return GNBModel(
            var_smoothing=params.get('var_smoothing', 1e-6))

    elif name == 'KNN_PLS30':
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


# ====================== 1. LOAD + SPLIT DATA ======================

print("=" * 85)
print("EXP 013: 8-model ensemble (LR router) — tuned + model export")
print("=" * 85)

os.makedirs(MODELS_DIR, exist_ok=True)

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
all_tuned = {}

for name in FEATURE_CONFIGS:
    t0 = time.time()
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    # load tuned params
    tuned = load_tuned_params(name)
    all_tuned[name] = tuned

    # training data
    X_tr = X_train_sc.copy()
    y_tr = y_train.copy()

    if name in UNDERSAMPLE_MODELS:
        if name == 'KDE_NB_tophat':
            rus = RandomUndersampler(
                scope='per_class',
                targets={0: KDE_SUBSAMPLE_PER_CLASS,
                         1: KDE_SUBSAMPLE_PER_CLASS},
                seed=42)
        else:
            rus = RandomUndersampler(
                scope='per_class', targets={0: 60000}, seed=42)
        X_tr, y_tr = rus.fit_resample(X_tr, y_tr)

    if name in SMOTE_MODELS:
        smoter = SMOter(sampling_strategy=0.5, k_neighbors=5, seed=42)
        X_tr, y_tr = smoter.fit_resample(X_tr, y_tr)

    print(f"  Training data: {X_tr.shape}  "
          f"class 0={int((y_tr==0).sum())}  class 1={int((y_tr==1).sum())}")

    # feature engineering
    fe = FeatureEngineer(FEATURE_CONFIGS[name])
    X_tr_fe = fe.fit_transform(X_tr, y_tr)
    X_val_fe = fe.transform(X_test_sc)

    # create + train model
    model = create_model(name, y_tr, tuned_params=tuned)
    model.fit(X_tr_fe, y_tr)

    # evaluate on meta-val
    scores = model.predict_proba(X_val_fe)
    evaluator.add(name, y_test, scores)
    trained[name] = (model, fe)

    # save model + feature engineer
    safe_name = name.lower().replace(' ', '_')
    joblib.dump({'model': model, 'fe': fe, 'name': name,
                  'tuned_params': tuned},
                 os.path.join(MODELS_DIR, f'{safe_name}.joblib'))
    print(f"  Saved: models/{safe_name}.joblib")

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

# save router
joblib.dump({'router': meta_lr, 'model_names': model_names},
            os.path.join(MODELS_DIR, 'lr_router.joblib'))
print(f"  Saved: models/lr_router.joblib")

print("\n" + "=" * 85)
print("ALL MODELS + ENSEMBLE")
print("=" * 85)
evaluator.summarize()


# ====================== 5. PLOTS ======================

evaluator.plot_roc(filename='Evaluator_roc_exp013.png')
evaluator.plot_binary_auc(filename='Evaluator_binary_auc_exp013.png')
evaluator.plot_j_scores(filename='Evaluator_j_scores_exp013.png')


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
    'exp013_lr_router_v2-FE-tuned', kaggle_probs, test_ids, threshold=best_thr
)
submitter.show_history()