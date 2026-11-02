# experiments/exp_003_smote_comparison.py

"""
Experiment 3: Test all 8 individual models BEFORE and AFTER SMOTE.
Shows exactly which models benefit from SMOTE and which get hurt.
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
from src.evaluation import Evaluator


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

UNDERSAMPLE_MODELS = {'GNB', 'KNN_PLS30', 'KNN_PCA30', 'XGBoost',
                       'LightGBM', 'GenGaussian_NB', 'KDE_NB_tophat'}


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
        return KDENBModel(kernel='tophat', bandwidth=0.5, max_train=15000)

    else:
        raise ValueError(f"Unknown model: {name}")


def prepare_data(name, X_train_sc, y_train, use_smote):
    """Prepare training data: undersample if needed, SMOTE if requested."""
    X_tr = X_train_sc.copy()
    y_tr = y_train.copy()

    if name in UNDERSAMPLE_MODELS:
        rus = RandomUndersampler(target_class0=60000, seed=42)
        X_tr, y_tr = rus.fit_resample(X_tr, y_tr)

    if use_smote:
        smoter = SMOter(sampling_strategy=0.5, k_neighbors=5, seed=42)
        X_tr, y_tr = smoter.fit_resample(X_tr, y_tr)

    return X_tr, y_tr


# ====================== 1. LOAD + SPLIT DATA ======================

print("=" * 85)
print("EXP 003: SMOTE vs No-SMOTE — individual model comparison")
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


# ====================== 2. WITHOUT SMOTE ======================

print("\n" + "=" * 85)
print("PASS 1: WITHOUT SMOTE")
print("=" * 85)

eval_no_smote = Evaluator(plot_dir='src/plots')

for name in FEATURE_CONFIGS:
    t0 = time.time()
    print(f"\n--- {name} (no SMOTE) ---")

    X_tr, y_tr = prepare_data(name, X_train_sc, y_train, use_smote=False)

    fe = FeatureEngineer(FEATURE_CONFIGS[name])
    X_tr_fe = fe.fit_transform(X_tr, y_tr)
    X_val_fe = fe.transform(X_test_sc)

    model = create_model(name, y_tr)
    model.fit(X_tr_fe, y_tr)

    scores = model.predict_proba(X_val_fe)
    eval_no_smote.add(name, y_test, scores)

    elapsed = time.time() - t0
    m = eval_no_smote.results[name]
    print(f"  AUC_prob={m['auc_prob']:.4f}  AUC_bin={m['auc_binary']:.4f}  "
          f"f1={m['f1']:.4f}  time={elapsed:.1f}s")

print("\n" + "=" * 85)
print("NO SMOTE RESULTS")
print("=" * 85)
eval_no_smote.summarize()


# ====================== 3. WITH SMOTE ======================

print("\n" + "=" * 85)
print("PASS 2: WITH SMOTE (all models)")
print("=" * 85)

eval_smote = Evaluator(plot_dir='src/plots')

for name in FEATURE_CONFIGS:
    t0 = time.time()
    print(f"\n--- {name} (with SMOTE) ---")

    X_tr, y_tr = prepare_data(name, X_train_sc, y_train, use_smote=True)

    fe = FeatureEngineer(FEATURE_CONFIGS[name])
    X_tr_fe = fe.fit_transform(X_tr, y_tr)
    X_val_fe = fe.transform(X_test_sc)

    model = create_model(name, y_tr)
    model.fit(X_tr_fe, y_tr)

    scores = model.predict_proba(X_val_fe)
    eval_smote.add(name, y_test, scores)

    elapsed = time.time() - t0
    m = eval_smote.results[name]
    print(f"  AUC_prob={m['auc_prob']:.4f}  AUC_bin={m['auc_binary']:.4f}  "
          f"f1={m['f1']:.4f}  time={elapsed:.1f}s")

print("\n" + "=" * 85)
print("SMOTE RESULTS")
print("=" * 85)
eval_smote.summarize()


# ====================== 4. COMPARISON ======================

print("\n" + "=" * 85)
print("SMOTE vs NO-SMOTE — SIDE BY SIDE")
print("=" * 85)

rows = []
for name in FEATURE_CONFIGS:
    no = eval_no_smote.results[name]
    sm = eval_smote.results[name]
    rows.append({
        'model': name,
        'no_smote_auc_prob': no['auc_prob'],
        'smote_auc_prob': sm['auc_prob'],
        'delta_auc_prob': sm['auc_prob'] - no['auc_prob'],
        'no_smote_auc_bin': no['auc_binary'],
        'smote_auc_bin': sm['auc_binary'],
        'delta_auc_bin': sm['auc_binary'] - no['auc_binary'],
        'no_smote_f1': no['f1'],
        'smote_f1': sm['f1'],
        'delta_f1': sm['f1'] - no['f1'],
    })

df_compare = pd.DataFrame(rows)
df_compare = df_compare.sort_values('delta_auc_bin', ascending=False)

print(f"\n{'model':<20} {'AUC_bin':>10} {'AUC_bin':>10} {'delta':>8}  "
      f"{'AUC_prob':>10} {'AUC_prob':>10} {'delta':>8}  {'delta F1':>8}")
print(f"{'':<20} {'no_smote':>10} {'smote':>10} {'':>8}  "
      f"{'no_smote':>10} {'smote':>10} {'':>8}")
print("-" * 100)

for _, row in df_compare.iterrows():
    sign_bin = '+' if row['delta_auc_bin'] > 0 else ''
    sign_prob = '+' if row['delta_auc_prob'] > 0 else ''
    sign_f1 = '+' if row['delta_f1'] > 0 else ''
    print(f"{row['model']:<20} "
          f"{row['no_smote_auc_bin']:>10.4f} {row['smote_auc_bin']:>10.4f} "
          f"{sign_bin}{row['delta_auc_bin']:>+8.4f}  "
          f"{row['no_smote_auc_prob']:>10.4f} {row['smote_auc_prob']:>10.4f} "
          f"{sign_prob}{row['delta_auc_prob']:>+8.4f}  "
          f"{sign_f1}{row['delta_f1']:>+8.4f}")

print(f"\n  Positive delta = SMOTE helped")
print(f"  Negative delta = SMOTE hurt")


# ====================== 5. PLOTS ======================

prefix = 'exp003_smote'

# --- 1. Side-by-side bar chart: AUC_binary ---
fig, ax = plt.subplots(figsize=(12, 6))
models = df_compare['model'].tolist()
x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, df_compare['no_smote_auc_bin'], width,
               label='No SMOTE', color='steelblue', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, df_compare['smote_auc_bin'], width,
               label='With SMOTE', color='coral', edgecolor='black', linewidth=0.5)

ax.set_ylabel('Binary AUC (J score)')
ax.set_title('SMOTE vs No-SMOTE — Binary AUC')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha='right')
ax.legend()
ax.grid(alpha=0.3, axis='y')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join('src/plots', f'{prefix}_binary_auc_comparison.png'), dpi=150)
plt.close()

# --- 2. Delta chart (what SMOTE changed) ---
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['mediumseagreen' if d > 0 else 'crimson' for d in df_compare['delta_auc_bin']]
bars = ax.barh(df_compare['model'], df_compare['delta_auc_bin'],
               color=colors, edgecolor='black', linewidth=0.5)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Change in Binary AUC (SMOTE - No SMOTE)')
ax.set_title("SMOTE Impact — positive = helped, negative = hurt")
ax.grid(alpha=0.3, axis='x')

for bar, val in zip(bars, df_compare['delta_auc_bin']):
    offset = 0.0003 if val >= 0 else -0.0003
    ha = 'left' if val >= 0 else 'right'
    ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
            f'{val:+.4f}', va='center', ha=ha, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join('src/plots', f'{prefix}_delta_chart.png'), dpi=150)
plt.close()



# ====================== 6. SAVE RESULTS ======================

df_compare.to_csv('data/processed/exp003_smote_comparison.csv', index=False)
print(f"\n  Results saved: data/processed/exp003_smote_comparison.csv")