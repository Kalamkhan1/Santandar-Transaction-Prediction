
# src/evaluation.py

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score


class Evaluator:
    """
    Takes model predictions (scores) and true labels.
    Computes all metrics, finds best threshold, generates plots.

    Usage:
        ev = Evaluator(plot_dir='src/plots')

        # add models one by one
        ev.add('GNB', y_test, gnb_scores)
        ev.add('XGBoost', y_test, xgb_scores)

        # compute and print
        ev.summarize()

        # plots
        ev.plot_roc()
        ev.plot_binary_auc()

        # get results
        results_df = ev.get_results()
    """

    def __init__(self, plot_dir='src/plots'):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        self.results = {}    # name -> metrics dict
        self.scores = {}     # name -> score array
        self.y_true = None

    def add(self, name, y_true, scores):
        """Register a model's predictions."""
        y_true = np.asarray(y_true)
        scores = np.asarray(scores)
        self.y_true = y_true
        self.scores[name] = scores

        # compute metrics at Youden's J threshold
        fpr, tpr, thr_arr = roc_curve(y_true, scores)
        J = tpr - fpr
        idx = int(np.argmax(J))
        thr = thr_arr[idx]
        y_pred = (scores >= thr).astype(int)

        self.results[name] = {
            'auc_prob': auc(fpr, tpr),
            'auc_binary': (1 + J[idx]) / 2,
            'best_thr': thr,
            'acc': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'prec': precision_score(y_true, y_pred),
            'rec': recall_score(y_true, y_pred),
            'J_score': J[idx],
            'TPR': tpr[idx],
            'FPR': fpr[idx],
        }
        return self

    def get_results(self):
        """Return results as DataFrame, sorted by binary AUC."""
        df = pd.DataFrame(self.results).T
        df.index.name = 'model'
        df = df.sort_values('auc_binary', ascending=False)
        return df.reset_index()

    def summarize(self):
        """Print summary table."""
        df = self.get_results()
        print(f"\n{'=' * 110}")
        print("EVALUATION SUMMARY — ranked by binary AUC (Youden's J)")
        print(f"{'=' * 110}")
        print(df[['model', 'auc_prob', 'auc_binary', 'J_score', 'best_thr',
                   'acc', 'f1', 'prec', 'rec']].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"{'=' * 110}")
        return df

    def plot_roc(self, filename='Evaluator_roc_curves.png'):
        """ROC curves for all registered models."""
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.colormaps['tab10'].resampled(max(len(self.scores), 1))

        for i, (name, scores) in enumerate(self.scores.items()):
            fpr, tpr, _ = roc_curve(self.y_true, scores)
            roc_auc = self.results[name]['auc_prob']
            is_best = name == self.get_results().iloc[0]['model']
            lw = 2.5 if is_best else 1.5
            style = '-' if is_best else '--'
            ax.plot(fpr, tpr, style, color=cmap(i), lw=lw,
                    label=f'{name} (AUC={roc_auc:.4f})')

        ax.plot([0, 1], [0, 1], 'k:', label='chance')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves — All Models')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, filename), dpi=150)
        plt.close()
        print(f"  Plot saved: {self.plot_dir}/{filename}")
        return self

    def plot_binary_auc(self, filename='Evaluator_binary_auc.png'):
        """Binary AUC triangle diagram (J scores). Each model at its best threshold."""
        df = self.get_results()
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.colormaps['tab10'].resampled(max(len(self.scores), 1))

        for i, (_, row) in enumerate(df.iterrows()):
            name = row['model']
            tpr_bin = row['TPR']
            fpr_bin = row['FPR']
            is_best = (i == 0)
            lw = 2.5 if is_best else 1.5
            color = cmap(i)
            marker_size = 8 if is_best else 5

            xs = [0, fpr_bin, 1]
            ys = [0, tpr_bin, 1]
            ax.plot(xs, ys, '-o', color=color, lw=lw, markersize=marker_size,
                    alpha=0.8 if is_best else 0.5,
                    label=f"{name} (AUC_bin={row['auc_binary']:.4f})")

        ax.plot([0, 1], [0, 1], 'k:', label='chance')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title("Binary AUC (Youden's J) — All Models at Best Threshold")
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, filename), dpi=150)
        plt.close()
        print(f"  Plot saved: {self.plot_dir}/{filename}")
        return self

    def plot_j_scores(self, filename='Evaluator_j_scores.png'):
        """Horizontal bar chart of J scores for all models."""
        df = self.get_results().sort_values('J_score')

        fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.5)))
        colors = ['coral' if i == len(df) - 1 else 'steelblue'
                  for i in range(len(df))]
        bars = ax.barh(df['model'], df['J_score'], color=colors,
                       edgecolor='black', linewidth=0.5)
        ax.set_xlabel('J Score (TPR - FPR)')
        ax.set_title("J Scores — All Models (higher = better)")
        ax.set_xlim(0, 1)

        for bar, val in zip(bars, df['J_score']):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)

        ax.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, filename), dpi=150)
        plt.close()
        print(f"  Plot saved: {self.plot_dir}/{filename}")
        return self

    def plot_thresholds(self, filename='Evaluator_thresholds.png'):
        """Threshold sweep showing F1 vs threshold for all models."""
        fig, axes = plt.subplots(1, len(self.scores),
                                  figsize=(5 * max(len(self.scores), 1), 5),
                                  squeeze=False)

        for i, (name, scores) in enumerate(self.scores.items()):
            ax = axes[0][i]
            fpr, tpr, thr_arr = roc_curve(self.y_true, scores)

            # compute J and F1 at every threshold
            J = tpr - fpr
            y_pred_all = (scores[:, None] >= thr_arr[None, :])
            f1s = []
            for k in range(len(thr_arr)):
                yp = y_pred_all[:, k].astype(int)
                tp = ((yp == 1) & (self.y_true == 1)).sum()
                fp = ((yp == 1) & (self.y_true == 0)).sum()
                fn = ((yp == 0) & (self.y_true == 1)).sum()
                f1 = 2 * tp / (2 * tp + fp + fn + 1e-10)
                f1s.append(f1)
            f1s = np.array(f1s)

            best_J_idx = np.argmax(J)
            best_f1_idx = np.argmax(f1s)

            ax.plot(thr_arr, J, 'b-', alpha=0.7, label='J score')
            ax.plot(thr_arr, f1s, 'r-', alpha=0.7, label='F1')
            ax.axvline(x=thr_arr[best_J_idx], color='blue', linestyle='--',
                        alpha=0.5, label=f'best J thr={thr_arr[best_J_idx]:.3f}')
            ax.axvline(x=thr_arr[best_f1_idx], color='red', linestyle='--',
                        alpha=0.5, label=f'best F1 thr={thr_arr[best_f1_idx]:.3f}')
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Score')
            ax.set_title(name)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        plt.suptitle('Threshold Sweep — J score and F1 per model', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, filename), dpi=150)
        plt.close()
        print(f"  Plot saved: {self.plot_dir}/{filename}")
        return self