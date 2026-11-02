# experiments/exp_007_full_fe_ablation.py

"""
Experiment 7: Full feature engineering ablation.

Part 1: NB + KNN — test feature REPLACEMENT for NB (freq-only, x²-only, bins_target-only)
                    and feature ADDITION for KNN (raw+freq → PLS/PCA, raw+row_stats → PLS/PCA)
        Goal: can NB/KNN benefit from engineered features while staying uncorrelated?

Part 2: Trees + RBF_NN — test NEW feature techniques from exp_006 suggestions:
        freq_rounded (1 and 2 decimals), gauss_distance, row_rank_stats, poly_row_stats
        Goal: which new techniques improve the models that already benefit from FE?

No ensemble, no submission — pure comparison.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.preprocessing import (
    DataSplitter, RandomUndersampler, SMOter,
    PCAReducer, PLSReducer
)
from src.features import FeatureEngineer
from src.models import (
    GNBModel, KNNModel, XGBoostModel, LightGBMModel,
    RBFNNModel, GenGaussianNBModel, KDENBModel
)


# ====================== HELPERS ======================

def best_threshold_metrics(y_true, scores):
    from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
    fpr, tpr, thr_arr = roc_curve(y_true, scores)
    J = tpr - fpr
    idx = int(np.argmax(J))
    thr = thr_arr[idx]
    y_pred = (scores >= thr).astype(int)
    return {
        'auc_prob': auc(fpr, tpr),
        'auc_binary': (1 + tpr[idx] - fpr[idx]) / 2,
        'best_thr': thr,
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'prec': precision_score(y_true, y_pred),
        'rec': recall_score(y_true, y_pred),
    }


def create_model(name, y_tr):
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


def prepare_data(name, X_train_sc, y_train):
    """Data prep per model (same as exp_004)."""
    X_tr = X_train_sc.copy()
    y_tr = y_train.copy()

    undersample_models = {'GNB', 'KNN_PLS30', 'KNN_PCA30', 'XGBoost',
                          'LightGBM', 'GenGaussian_NB', 'KDE_NB_tophat'}
    smote_models = {'XGBoost', 'LightGBM'}

    if name in undersample_models:
        if name == 'KDE_NB_tophat':
            rus = RandomUndersampler(
                scope='per_class',
                targets={0: 5000, 1: 5000}, seed=42)
        else:
            rus = RandomUndersampler(
                scope='per_class', targets={0: 60000}, seed=42)
        X_tr, y_tr = rus.fit_resample(X_tr, y_tr)

    if name in smote_models:
        smoter = SMOter(sampling_strategy=0.5, k_neighbors=5, seed=42)
        X_tr, y_tr = smoter.fit_resample(X_tr, y_tr)

    return X_tr, y_tr


# ====================== 1. LOAD + SPLIT ======================

print("=" * 85)
print("EXP 007: Full feature engineering ablation (NB/KNN replacement + new techniques)")
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


# ====================== 2. PART 1: NB + KNN ======================

print("\n" + "=" * 85)
print("PART 1: NB + KNN — feature replacement / addition")
print("=" * 85)

PART1_TESTS = {
    # model_name: list of (test_label, feature_config, replace_raw)
    'GNB': [
        ('baseline_raw',    {},                         False),
        ('freq_only',       {'freq': True},             True),
        ('squared_only',    {'squared': True},          True),
    ],
    'GenGaussian_NB': [
        ('baseline_raw',    {},                         False),
        ('freq_only',       {'freq': True},             True),
        ('bins_target_only',{'bins_target': True},      True),
    ],
    'KDE_NB_tophat': [
        ('baseline_raw',    {},                         False),
        ('freq_only',       {'freq': True},             True),
        ('squared_only',    {'squared': True},          True),
    ],
    'KNN_PLS30': [
        ('raw',             {},                         False),
        ('raw+freq',        {'freq': True},             False),
        ('raw+row_stats',   {'row_stats': 'basic'},     False),
    ],
    'KNN_PCA30': [
        ('raw',             {},                         False),
        ('raw+freq',        {'freq': True},             False),
        ('raw+row_stats',   {'row_stats': 'basic'},     False),
    ],
}

part1_results = {}

for model_name, tests in PART1_TESTS.items():
    print(f"\n  --- {model_name} ---")

    X_tr, y_tr = prepare_data(model_name, X_train_sc, y_train)
    print(f"  Training data: {X_tr.shape}  "
          f"class 0={int((y_tr==0).sum())}  class 1={int((y_tr==1).sum())}")

    for test_label, feat_config, replace_raw in tests:
        t0 = time.time()

        fe = FeatureEngineer(feat_config, replace_raw=replace_raw)
        X_tr_fe = fe.fit_transform(X_tr, y_tr)
        X_val_fe = fe.transform(X_test_sc)

        model = create_model(model_name, y_tr)
        model.fit(X_tr_fe, y_tr)
        scores = model.predict_proba(X_val_fe)

        m = best_threshold_metrics(y_test, scores)
        part1_results[(model_name, test_label)] = m

        elapsed = time.time() - t0
        print(f"    {test_label:20s}  feats={X_tr_fe.shape[1]:>5}  "
              f"AUC_prob={m['auc_prob']:.4f}  AUC_bin={m['auc_binary']:.4f}  "
              f"f1={m['f1']:.4f}  time={elapsed:.1f}s")

# Part 1 summary
print(f"\n{'=' * 95}")
print("PART 1 RESULTS — NB + KNN feature ablation")
print(f"{'=' * 95}")

for model_name, tests in PART1_TESTS.items():
    print(f"\n  {model_name}:")
    baseline = part1_results[(model_name, tests[0][0])]['auc_binary']
    for test_label, _, _ in tests:
        m = part1_results[(model_name, test_label)]
        delta = m['auc_binary'] - baseline
        sign = '+' if delta > 0 else ''
        print(f"    {test_label:20s}  AUC_bin={m['auc_binary']:.4f}  "
              f"delta={sign}{delta:+.4f}")


# ====================== 3. PART 2: Trees + RBF_NN — new techniques ======================

print("\n" + "=" * 85)
print("PART 2: Trees + RBF_NN — new feature techniques")
print("=" * 85)

# baseline for trees: freq + row_stats (the proven winners)
TREE_BASELINE = {'freq': True, 'row_stats': 'full'}
RBF_BASELINE = {'squared': True}

# new techniques to test individually (added to baseline)
NEW_TECHNIQUES = {
    'baseline':              {},
    'freq_rounded_1':        {'freq_rounded': 1},
    'freq_rounded_2':        {'freq_rounded': 2},
    'gauss_distance':        {'gauss_distance': True},
    'row_rank_stats':        {'row_rank_stats': True},
    'poly_row_stats':        {'poly_row_stats': True},
    # best combos (adjusted after seeing individual results)
    'freq+gauss':            {'freq_rounded': 2, 'gauss_distance': True},
    'freq+rank':             {'freq_rounded': 2, 'row_rank_stats': True},
}

PART2_MODELS = {
    'XGBoost':  {'baseline_config': TREE_BASELINE},
    'LightGBM': {'baseline_config': TREE_BASELINE},
    'RBF_NN':   {'baseline_config': RBF_BASELINE},
}

part2_results = {}

for model_name, model_cfg in PART2_MODELS.items():
    print(f"\n  --- {model_name} ---")

    X_tr, y_tr = prepare_data(model_name, X_train_sc, y_train)
    print(f"  Training data: {X_tr.shape}  "
          f"class 0={int((y_tr==0).sum())}  class 1={int((y_tr==1).sum())}")

    baseline_config = model_cfg['baseline_config']

    for tech_label, tech_config in NEW_TECHNIQUES.items():
        t0 = time.time()

        # merge baseline + new technique
        if tech_label == 'baseline':
            full_config = dict(baseline_config)
        else:
            full_config = dict(baseline_config)
            full_config.update(tech_config)

        fe = FeatureEngineer(full_config)
        X_tr_fe = fe.fit_transform(X_tr, y_tr)
        X_val_fe = fe.transform(X_test_sc)

        model = create_model(model_name, y_tr)
        model.fit(X_tr_fe, y_tr)
        scores = model.predict_proba(X_val_fe)

        m = best_threshold_metrics(y_test, scores)
        part2_results[(model_name, tech_label)] = m

        elapsed = time.time() - t0
        print(f"    {tech_label:20s}  feats={X_tr_fe.shape[1]:>5}  "
              f"AUC_prob={m['auc_prob']:.4f}  AUC_bin={m['auc_binary']:.4f}  "
              f"f1={m['f1']:.4f}  time={elapsed:.1f}s")

# Part 2 summary
print(f"\n{'=' * 95}")
print("PART 2 RESULTS — new FE techniques on trees + RBF_NN")
print(f"{'=' * 95}")

for model_name in PART2_MODELS:
    print(f"\n  {model_name}:")
    baseline = part2_results[(model_name, 'baseline')]['auc_binary']
    baseline_prob = part2_results[(model_name, 'baseline')]['auc_prob']

    rows = []
    for tech_label in NEW_TECHNIQUES:
        m = part2_results[(model_name, tech_label)]
        rows.append({
            'technique': tech_label,
            'auc_prob': m['auc_prob'],
            'delta_prob': m['auc_prob'] - baseline_prob,
            'auc_binary': m['auc_binary'],
            'delta_bin': m['auc_binary'] - baseline,
            'f1': m['f1'],
        })

    df = pd.DataFrame(rows).sort_values('auc_binary', ascending=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  Baseline: AUC_bin={baseline:.4f}")


# ====================== 4. COMBINED COMPARISON ======================

print(f"\n{'=' * 105}")
print("COMBINED RESULTS — all models × all tests")
print(f"{'=' * 105}")

all_rows = []

for (model_name, test_label), m in part1_results.items():
    all_rows.append({
        'model': model_name,
        'test': test_label,
        'part': 'replacement',
        'auc_prob': m['auc_prob'],
        'auc_binary': m['auc_binary'],
        'f1': m['f1'],
    })

for (model_name, tech_label), m in part2_results.items():
    all_rows.append({
        'model': model_name,
        'test': tech_label,
        'part': 'new_technique',
        'auc_prob': m['auc_prob'],
        'auc_binary': m['auc_binary'],
        'f1': m['f1'],
    })

df_all = pd.DataFrame(all_rows)
df_all = df_all.sort_values(['model', 'auc_binary'], ascending=[True, False])
print(df_all.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

df_all.to_csv('data/processed/exp007_full_fe_ablation.csv', index=False)
print(f"\n  Saved: data/processed/exp007_full_fe_ablation.csv")


# ====================== 5. PLOTS ======================

prefix = 'exp007_fe_ablation'

# --- Part 1: NB + KNN bar charts ---
fig, axes = plt.subplots(1, 5, figsize=(25, 6), squeeze=False)
axes = axes.flatten()

for i, (model_name, tests) in enumerate(PART1_TESTS.items()):
    ax = axes[i]
    test_labels = [t[0] for t in tests]
    aucs = [part1_results[(model_name, t[0])]['auc_binary'] for t in tests]
    baseline_auc = aucs[0]
    colors = ['steelblue' if j == 0 else
              ('mediumseagreen' if aucs[j] > baseline_auc else 'crimson')
              for j in range(len(aucs))]

    bars = ax.bar(test_labels, aucs, color=colors,
                  edgecolor='black', linewidth=0.5)
    ax.axhline(y=baseline_auc, color='blue', linestyle='--', alpha=0.3,
               label=f'baseline={baseline_auc:.4f}')
    ax.set_title(model_name, fontsize=10)
    ax.set_ylabel('Binary AUC')
    ax.tick_params(axis='x', rotation=25, labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=7)

plt.suptitle('Part 1: NB + KNN — Feature Replacement / Addition', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join('src/plots', f'{prefix}_part1.png'), dpi=150)
plt.close()

# --- Part 2: trees + RBF bar charts ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, model_name in enumerate(PART2_MODELS):
    ax = axes[i]
    tech_labels = list(NEW_TECHNIQUES.keys())
    aucs = [part2_results[(model_name, t)]['auc_binary'] for t in tech_labels]
    baseline_auc = part2_results[(model_name, 'baseline')]['auc_binary']
    colors = ['steelblue' if t == 'baseline' else
              ('mediumseagreen' if a > baseline_auc else 'crimson')
              for t, a in zip(tech_labels, aucs)]

    bars = ax.bar(tech_labels, aucs, color=colors,
                  edgecolor='black', linewidth=0.5)
    ax.axhline(y=baseline_auc, color='blue', linestyle='--', alpha=0.3,
               label=f'baseline={baseline_auc:.4f}')
    ax.set_title(model_name, fontsize=11)
    ax.set_ylabel('Binary AUC')
    ax.tick_params(axis='x', rotation=35, labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=7)

plt.suptitle('Part 2: Trees + RBF_NN — New Feature Techniques', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join('src/plots', f'{prefix}_part2.png'), dpi=150)
plt.close()

print(f"\n  Plots saved: {prefix}_part1.png, {prefix}_part2.png")


# ====================== 6. RECOMMENDATIONS ======================

print(f"\n{'=' * 85}")
print("RECOMMENDATIONS — best config per model")
print(f"{'=' * 85}")

for model_name in PART1_TESTS:
    tests = PART1_TESTS[model_name]
    best_test = max(tests, key=lambda t: part1_results[(model_name, t[0])]['auc_binary'])
    m = part1_results[(model_name, best_test[0])]
    print(f"  {model_name:20s}  →  {best_test[0]:20s}  AUC_bin={m['auc_binary']:.4f}")

for model_name in PART2_MODELS:
    best_tech = max(NEW_TECHNIQUES.keys(),
                    key=lambda t: part2_results[(model_name, t)]['auc_binary'])
    m = part2_results[(model_name, best_tech)]
    print(f"  {model_name:20s}  →  {best_tech:20s}  AUC_bin={m['auc_binary']:.4f}")