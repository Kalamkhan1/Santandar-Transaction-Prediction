# experiments/exp_012_hyperparameter_tuning.py

"""
Experiment 8: Hyperparameter tuning for all models using Optuna.
Bayesian search (TPE) on single validation set (locked meta-val).
Meta-val is 40k rows from structural split — representative, law of large numbers.
No CV — direct scoring on meta-val.

Models tuned: XGBoost, LightGBM, RBF_NN, KNN_PLS30, KNN_PCA30, KDE_NB_tophat, GNB
Skipped: GenGaussian_NB (no hyperparameters — scipy auto-fits)

Best params saved to data/processed/tuned_*.pkl
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import time

from src.preprocessing import (
    DataSplitter, RandomUndersampler, SMOter,
    PCAReducer, PLSReducer
)
from src.features import FeatureEngineer
from src.models import (
    GNBModel, KNNModel, XGBoostModel, LightGBMModel,
    RBFNNModel, GenGaussianNBModel, KDENBModel
)
from src.tuning import TuningPipeline


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
                         'group_agg': True, 'n_groups': 10,
                         'row_rank_stats': True},
    'RBF_NN':           {'squared': True, 'gauss_distance': True},
}

SMOTE_MODELS = {'XGBoost', 'LightGBM'}
UNDERSAMPLE_MODELS = {'GNB', 'KNN_PLS30', 'KNN_PCA30', 'XGBoost',
                       'LightGBM', 'GenGaussian_NB', 'KDE_NB_tophat'}
KDE_SUBSAMPLE_PER_CLASS = 5000

N_TRIALS = {
    'XGBoost':       60,
    'LightGBM':      60,
    'RBF_NN':        30,
    'KNN_PLS30':     25,
    'KNN_PCA30':     25,
    'KDE_NB_tophat': 15,
    'GNB':           15,
}


# ====================== HELPERS ======================

def prepare_data(name, X_train_sc, y_train):
    """Data prep per model (same as exp_004)."""
    X_tr = X_train_sc.copy()
    y_tr = y_train.copy()

    if name in UNDERSAMPLE_MODELS:
        if name == 'KDE_NB_tophat':
            rus = RandomUndersampler(
                scope='per_class',
                targets={0: KDE_SUBSAMPLE_PER_CLASS, 1: KDE_SUBSAMPLE_PER_CLASS},
                seed=42)
        else:
            rus = RandomUndersampler(
                scope='per_class', targets={0: 60000}, seed=42)
        X_tr, y_tr = rus.fit_resample(X_tr, y_tr)

    if name in SMOTE_MODELS:
        smoter = SMOter(sampling_strategy=0.5, k_neighbors=5, seed=42)
        X_tr, y_tr = smoter.fit_resample(X_tr, y_tr)

    return X_tr, y_tr


def apply_features(name, X_tr, y_tr, X_val):
    """Apply feature engineering, return transformed train and val."""
    fe = FeatureEngineer(FEATURE_CONFIGS.get(name, {}))
    X_tr_fe = fe.fit_transform(X_tr, y_tr)
    X_val_fe = fe.transform(X_val)
    return X_tr_fe, X_val_fe


# ====================== MODEL FACTORIES + SEARCH SPACES ======================

def make_gnb_factory(params):
    return GNBModel(var_smoothing=params['var_smoothing'])

def suggest_gnb(trial):
    return {'var_smoothing': trial.suggest_float('var_smoothing', 1e-11, 1e-2, log=True)}


def make_xgb_factory(params):
    return XGBoostModel(**params)

def suggest_xgb(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }


def make_lgbm_factory(params):
    return LightGBMModel(**params)

def suggest_lgbm(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
    }


def make_rbf_factory(params):
    return RBFNNModel(
        n_rbf=params['n_rbf'],
        epochs=params['epochs'],
        lr=params['lr'],
        pos_weight=params.get('pos_weight', 9.0),
        seed=42)

def suggest_rbf(trial):
    return {
        'n_rbf': trial.suggest_int('n_rbf', 128, 2048, step=128),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'epochs': trial.suggest_int('epochs', 50, 200, step=25),
    }


def make_knn_pls_factory(params):
    m = KNNModel(
        preprocessor=PLSReducer(n_components=params['n_components']),
        k=params['k'], weights=params['weights'])
    m.name = 'KNN_PLS30'
    return m

def suggest_knn_pls(trial):
    return {
        'k': trial.suggest_int('k', 50, 1000),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'n_components': trial.suggest_int('n_components', 5, 50),
    }


def make_knn_pca_factory(params):
    m = KNNModel(
        preprocessor=PCAReducer(n_components=params['n_components']),
        k=params['k'], weights=params['weights'])
    m.name = 'KNN_PCA30'
    return m

def suggest_knn_pca(trial):
    return {
        'k': trial.suggest_int('k', 50, 1000),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'n_components': trial.suggest_int('n_components', 5, 50),
    }


def make_kde_factory(params):
    return KDENBModel(
        kernel='tophat',
        bandwidth=params['bandwidth'],
        max_train=5000)

def suggest_kde(trial):
    return {
        'bandwidth': trial.suggest_float('bandwidth', 0.1, 2.0, log=True),
    }


TUNING_MODELS = {
    'XGBoost':       (make_xgb_factory, suggest_xgb),
    'LightGBM':      (make_lgbm_factory, suggest_lgbm),
    'RBF_NN':        (make_rbf_factory, suggest_rbf),
    'KNN_PLS30':     (make_knn_pls_factory, suggest_knn_pls),
    'KNN_PCA30':     (make_knn_pca_factory, suggest_knn_pca),
    'KDE_NB_tophat': (make_kde_factory, suggest_kde),
    'GNB':           (make_gnb_factory, suggest_gnb),
}


# ====================== 1. LOAD + SPLIT DATA ======================

print("=" * 85)
print("EXP 012: Hyperparameter tuning (Optuna, single validation set)")
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
print(f"  X_test_sc:  {X_test_sc.shape}  y_test:  {y_test.shape}")


# ====================== 2. TUNE EACH MODEL ======================

all_best_params = {}

for model_name in TUNING_MODELS:
    t0 = time.time()

    print(f"\n{'=' * 70}")
    print(f"  PREPARING: {model_name}")
    print(f"{'=' * 70}")

    # prepare data
    X_tr, y_tr = prepare_data(model_name, X_train_sc, y_train)
    print(f"  Training data: {X_tr.shape}  "
          f"class 0={int((y_tr==0).sum())}  class 1={int((y_tr==1).sum())}")

    # apply feature engineering
    X_tr_fe, X_val_fe = apply_features(model_name, X_tr, y_tr, X_test_sc)
    print(f"  Features: {X_tr_fe.shape[1]}")

    # get factory and suggest functions
    model_factory, suggest_params = TUNING_MODELS[model_name]

    # inject pos_weight for RBF_NN
    if model_name == 'RBF_NN':
        pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        def factory_with_pw(params, _pw=pos_weight, _mf=model_factory):
            params = dict(params)
            params['pos_weight'] = _pw
            return _mf(params)
        model_factory = factory_with_pw

    # create and run tuner
    tuner = TuningPipeline(seed=42, plot_dir='src/plots', scoring='auc_prob')
    tuner.add(
        name=model_name,
        model_factory=model_factory,
        suggest_params=suggest_params,
        n_trials=N_TRIALS[model_name],
    )
    tuner.run(X_tr_fe, y_tr, X_val_fe, y_test)

    # save results
    best_params = tuner.get_best_params()[model_name]
    all_best_params[model_name] = best_params

    safe_name = model_name.lower().replace(' ', '_')
    tuner.tuners[0].save(f'data/processed/tuned_{safe_name}.pkl')
    tuner.tuners[0].plot()

    elapsed = time.time() - t0
    print(f"\n  {model_name} done in {elapsed:.1f}s")


# ====================== 3. SUMMARY ======================

print(f"\n{'=' * 85}")
print("ALL TUNED MODELS — summary")
print(f"{'=' * 85}")

for model_name, params in all_best_params.items():
    print(f"\n  {model_name}:")
    for k, v in params.items():
        print(f"    {k:30s} = {v}")

print(f"\n  Saved to data/processed/tuned_*.pkl")