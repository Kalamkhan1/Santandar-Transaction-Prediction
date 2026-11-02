# experiments/exp_006_feature_engineering_ablation.py

"""
Experiment 6: Ablation study — test each feature engineering type
individually to see which helps and which hurts.

Models tested: XGBoost, LightGBM, RBF_NN (the only ones with features).
Baseline = raw 200 features only. Each variant adds exactly one feature type.
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
    XGBoostModel, LightGBMModel, RBFNNModel
)
from src.evaluation import Evaluator


# ====================== CONFIG ======================

# individual feature types to test (one at a time)
ABLATION_CONFIGS = {
    'baseline':         {},
    'squared':          {'squared': True},
    'row_stats_basic':  {'row_stats': 'basic'},
    'row_stats_full':   {'row_stats': 'full'},
    'freq':             {'freq': True},
    'bins_target':      {'bins_target': True},
    'interactions':     {'interactions': True, 'top_n': 20},
    'group_agg':        {'group_agg': True, 'n_groups': 10},
    'all':              {'squared': True, 'row_stats': 'full', 'freq': True,
                         'bins_target': True, 'interactions': True,
                         'group_agg': True, 'top_n': 20, 'n_groups': 10},
}

# models to test
MODELS = {
    'XGBoost':          {'smote': True, 'undersample': True},
    'LightGBM':         {'smote': True, 'undersample': True},
    'RBF_NN':           {'smote': False, 'undersample': False},
}

SMOTE_MODELS = {name for name, cfg in MODELS.items() if cfg['smote']}


def create_model(name, y_tr):
    if name == 'XGBoost':
        return XGBoostModel()
    elif name == 'LightGBM':
        return LightGBMModel()
    elif name == 'RBF_NN':
        pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        return RBFNNModel(
            n_rbf=256, epochs=50, lr=0.001,
            pos_weight=pos_weight, seed=42)
    else:
        raise ValueError(f"Unknown model: {name}")


# ====================== 1. LOAD + SPLIT DATA ======================

print("=" * 85)
print("EXP 006: Feature engineering ablation")
print("=" * 85)

splitter = DataSplitter(
    data_dir='data/raw',
    processed_dir='data/processed',
    plot_dir='src/plots',
)
splitter.run()

X_train_sc = splitter.X_train_sc
X_test_sc = splitter.X_test_sc
y_train = splitter.y_train
y_test = splitter.y_test

print(f"\n  X_train_sc: {X_train_sc.shape}  y_train: {y_train.shape}")
print(f"  X_test_sc:  {X_test_sc.shape}  y_test:  {y_test.shape}")


# ====================== 2. RUN ABLATION ======================

all_results = {}  # (model, feature_set) -> metrics

for model_name, model_cfg in MODELS.items():
    print(f"\n{'=' * 85}")
    print(f"  MODEL: {model_name}")
    print(f"{'=' * 85}")

    # prepare training data (same for all feature sets)
    X_tr = X_train_sc.copy()
    y_tr = y_train.copy()

    if model_cfg['undersample']:
        rus = RandomUndersampler(
            scope='per_class',
            targets={0: 60000},
            seed=42)
        X_tr, y_tr = rus.fit_resample(X_tr, y_tr)

    if model_cfg['smote']:
        smoter = SMOter(sampling_strategy=0.5, k_neighbors=5, seed=42)
        X_tr, y_tr = smoter.fit_resample(X_tr, y_tr)

    print(f"  Training data: {X_tr.shape}  "
          f"class 0={int((y_tr==0).sum())}  class 1={int((y_tr==1).sum())}")

    for feat_name, feat_config in ABLATION_CONFIGS.items():
        t0 = time.time()

        fe = FeatureEngineer(feat_config)
        X_tr_fe = fe.fit_transform(X_tr, y_tr)
        X_val_fe = fe.transform(X_test_sc)

        n_features = X_tr_fe.shape[1]

        model = create_model(model_name, y_tr)
        model.fit(X_tr_fe, y_tr)

        scores = model.predict_proba(X_val_fe)

        # quick metrics (no Evaluator — just compute inline)
        from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
        fpr, tpr, thr_arr = roc_curve(y_test, scores)
        J = tpr - fpr
        idx = int(np.argmax(J))
        thr = thr_arr[idx]
        y_pred = (scores >= thr).astype(int)

        metrics = {
            'auc_prob': auc(fpr, tpr),
            'auc_binary': (1 + J[idx]) / 2,
            'best_thr': thr,
            'f1': f1_score(y_test, y_pred),
            'prec': precision_score(y_test, y_pred),
            'rec': recall_score(y_test, y_pred),
            'n_features': n_features,
        }

        elapsed = time.time() - t0
        all_results[(model_name, feat_name)] = metrics

        print(f"  {feat_name:20s}  feats={n_features:>5}  "
              f"AUC_prob={metrics['auc_prob']:.4f}  "
              f"AUC_bin={metrics['auc_binary']:.4f}  "
              f"f1={metrics['f1']:.4f}  "
              f"time={elapsed:.1f}s")


# ====================== 3. SUMMARY TABLES ======================

for model_name in MODELS:
    print(f"\n{'=' * 95}")
    print(f"  {model_name} — FEATURE ABLATION")
    print(f"{'=' * 95}")

    rows = []
    baseline_auc = all_results[(model_name, 'baseline')]['auc_binary']
    baseline_prob = all_results[(model_name, 'baseline')]['auc_prob']

    for feat_name in ABLATION_CONFIGS:
        m = all_results[(model_name, feat_name)]
        rows.append({
            'features': feat_name,
            'n_feat': m['n_features'],
            'auc_prob': m['auc_prob'],
            'delta_prob': m['auc_prob'] - baseline_prob,
            'auc_binary': m['auc_binary'],
            'delta_bin': m['auc_binary'] - baseline_auc,
            'f1': m['f1'],
        })

    df = pd.DataFrame(rows).sort_values('auc_binary', ascending=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  Baseline (raw 200): AUC_bin={baseline_auc:.4f}")
    print(f"  Positive delta = feature type helps, negative = hurts")


# ====================== 4. COMBINED COMPARISON ACROSS MODELS ======================

print(f"\n{'=' * 105}")
print("ALL MODELS × ALL FEATURE SETS — AUC_binary")
print(f"{'=' * 105}")

rows = []
for feat_name in ABLATION_CONFIGS:
    row = {'features': feat_name}
    for model_name in MODELS:
        m = all_results[(model_name, feat_name)]
        row[f'{model_name}_auc_bin'] = m['auc_binary']
        row[f'{model_name}_delta'] = m['auc_binary'] - all_results[(model_name, 'baseline')]['auc_binary']
    rows.append(row)

df_combined = pd.DataFrame(rows)
print(df_combined.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# save to CSV
df_combined.to_csv('data/processed/exp006_feature_ablation.csv', index=False)
print(f"\n  Saved: data/processed/exp006_feature_ablation.csv")


# ====================== 5. PLOTS ======================

prefix = 'exp006_feature_ablation'

for model_name in MODELS:
    # --- bar chart: AUC_bin per feature set ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    rows = []
    baseline_auc = all_results[(model_name, 'baseline')]['auc_binary']
    baseline_prob = all_results[(model_name, 'baseline')]['auc_prob']

    for feat_name in ABLATION_CONFIGS:
        m = all_results[(model_name, feat_name)]
        rows.append({
            'features': feat_name,
            'n_feat': m['n_features'],
            'auc_binary': m['auc_binary'],
            'delta_bin': m['auc_binary'] - baseline_auc,
            'auc_prob': m['auc_prob'],
            'delta_prob': m['auc_prob'] - baseline_prob,
        })
    df_plot = pd.DataFrame(rows).sort_values('delta_bin', ascending=True)

    # --- 1. AUC_bin bar chart ---
    colors = ['mediumseagreen' if d > 0 else 'crimson' for d in df_plot['delta_bin']]
    bars = axes[0].barh(df_plot['features'], df_plot['auc_binary'],
                        color=colors, edgecolor='black', linewidth=0.5)
    axes[0].axvline(x=baseline_auc, color='blue', linestyle='--', alpha=0.5,
                    label=f'baseline={baseline_auc:.4f}')
    axes[0].set_xlabel('Binary AUC')
    axes[0].set_title(f'{model_name} — AUC_bin per feature set')
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='x')
    for bar, val in zip(bars, df_plot['auc_binary']):
        axes[0].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                     f'{val:.4f}', va='center', fontsize=8)

    # --- 2. Delta bar chart ---
    colors = ['mediumseagreen' if d > 0 else 'crimson' for d in df_plot['delta_bin']]
    bars = axes[1].barh(df_plot['features'], df_plot['delta_bin'],
                        color=colors, edgecolor='black', linewidth=0.5)
    axes[1].axvline(x=0, color='black', linewidth=0.8)
    axes[1].set_xlabel('Change in Binary AUC (vs baseline)')
    axes[1].set_title(f'{model_name} — feature impact (green=hurts, red=helps)')
    axes[1].grid(alpha=0.3, axis='x')
    for bar, val in zip(bars, df_plot['delta_bin']):
        offset = 0.0003 if val >= 0 else -0.0003
        ha = 'left' if val >= 0 else 'right'
        axes[1].text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
                     f'{val:+.4f}', va='center', ha=ha, fontsize=8)

    plt.suptitle(f'{model_name} — Feature Engineering Ablation', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join('src/plots', f'{prefix}_{model_name}.png'), dpi=150)
    plt.close()

# --- combined heatmap ---
fig, ax = plt.subplots(figsize=(12, 6))
heatmap_data = []
feat_names = list(ABLATION_CONFIGS.keys())
model_names = list(MODELS.keys())

for model_name in model_names:
    row = []
    baseline_auc = all_results[(model_name, 'baseline')]['auc_binary']
    for feat_name in feat_names:
        delta = all_results[(model_name, feat_name)]['auc_binary'] - baseline_auc
        row.append(delta)
    heatmap_data.append(row)

heatmap_data = np.array(heatmap_data)
im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')

ax.set_xticks(range(len(feat_names)))
ax.set_xticklabels(feat_names, rotation=30, ha='right')
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels(model_names)

for i in range(len(model_names)):
    for j in range(len(feat_names)):
        val = heatmap_data[i, j]
        ax.text(j, i, f'{val:+.4f}', ha='center', va='center', fontsize=8)

plt.colorbar(im, label='Delta AUC_binary vs baseline')
plt.title('Feature Impact Heatmap (green = helps, red = hurts)')
plt.tight_layout()
plt.savefig(os.path.join('src/plots', f'{prefix}_heatmap.png'), dpi=150)
plt.close()

print(f"\n  Plots saved: {prefix}_*.png")