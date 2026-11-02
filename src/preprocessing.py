# src/preprocessing.py

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
class DataSplitter:
    """
    Loads raw data, finds best structural split, scales, saves to processed/.
    Global stats use all 400k rows (train + kaggle test).
    Class-conditional stats use train only (kaggle test has no labels).

    Usage (search for best seed):
        splitter = DataSplitter(data_dir='data/raw', processed_dir='data/processed')
        splitter.run()

    Usage (known seed, skip search):
        splitter = DataSplitter(data_dir='data/raw', processed_dir='data/processed', seed=43)
        splitter.run()
    """

    def __init__(self, data_dir='data/raw', processed_dir='data/processed',
                 plot_dir='src/plots', test_size=0.2,
                 seed=None, seed_range=(1, 160)):
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.plot_dir = plot_dir
        self.test_size = test_size
        self.seed = seed
        self.seed_range = seed_range

        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.X_all = None
        self.y_all = None
        self.X_kaggle_test = None
        self.test_ids = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_sc = None
        self.X_test_sc = None
        self.scaler = None
        self.split_info = None
        self._global_X = None

    # ---------- loading ----------

    def load(self):
        train_path = os.path.join(self.data_dir, 'train.csv')
        test_path = os.path.join(self.data_dir, 'test.csv')

        data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        self.X_all = data.drop(columns=['target', 'ID_code']).values
        self.y_all = data['target'].values
        self.X_kaggle_test = test_data.drop(columns=['ID_code']).values
        self.test_ids = test_data['ID_code'].values

        # all 400k rows for global marginal stats
        self._global_X = np.vstack([self.X_all, self.X_kaggle_test])

        print(f"Full data:     {self.X_all.shape}")
        print(f"Kaggle test:   {self.X_kaggle_test.shape}")
        print(f"Global (all):  {self._global_X.shape}")
        print(f"Positive rate: {self.y_all.mean():.5f}")
        return self

    # ---------- scoring ----------

    @staticmethod
    def _score_split(X, y, X_tr, X_te, y_tr, y_te, global_X=None):
        """
        Score how well split preserves data structure. Lower = better.

        Marginal stats (means, stds): compared against global_X (400k rows).
        Class-conditional stats: compared against X (train only, has labels).
        """
        gX = global_X if global_X is not None else X

        global_mean = gX.mean(axis=0)
        global_std = gX.std(axis=0) + 1e-8

        ratio_diff = abs(y_tr.mean() - y_te.mean())

        tr_mean_diff = np.abs(X_tr.mean(axis=0) - global_mean) / global_std
        te_mean_diff = np.abs(X_te.mean(axis=0) - global_mean) / global_std
        feat_mean_diff = max(tr_mean_diff.mean(), te_mean_diff.mean())

        tr_std_ratio = np.abs(X_tr.std(axis=0) / global_std - 1)
        te_std_ratio = np.abs(X_te.std(axis=0) / global_std - 1)
        feat_std_diff = max(tr_std_ratio.mean(), te_std_ratio.mean())

        # class-conditional: from training data only (kaggle test has no labels)
        c0_global = X[y == 0].mean(axis=0)
        c1_global = X[y == 1].mean(axis=0)

        c0_tr_diff = np.abs(X_tr[y_tr == 0].mean(axis=0) - c0_global).mean() / global_std.mean()
        c0_te_diff = np.abs(X_te[y_te == 0].mean(axis=0) - c0_global).mean() / global_std.mean()
        c1_tr_diff = np.abs(X_tr[y_tr == 1].mean(axis=0) - c1_global).mean() / global_std.mean()
        c1_te_diff = np.abs(X_te[y_te == 1].mean(axis=0) - c1_global).mean() / global_std.mean()

        total = (ratio_diff * 100 + feat_mean_diff + feat_std_diff +
                 c0_tr_diff + c0_te_diff + c1_tr_diff + c1_te_diff)

        details = {
            'ratio_diff': ratio_diff,
            'feat_mean_diff': feat_mean_diff,
            'feat_std_diff': feat_std_diff,
            'c0_diff': max(c0_tr_diff, c0_te_diff),
            'c1_diff': max(c1_tr_diff, c1_te_diff),
            'total': total
        }
        return total, details

    # ---------- split with known seed (fast) ----------

    def _split_with_seed(self):
        """Split using a known seed. No search needed."""
        print(f"\n===== Using known seed: {self.seed} (skipping search) =====")

        X_tr, X_te, y_tr, y_te = train_test_split(
            self.X_all, self.y_all,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=self.y_all
        )
        total, details = self._score_split(
            self.X_all, self.y_all, X_tr, X_te, y_tr, y_te,
            global_X=self._global_X
        )

        print(f"  Seed:  {self.seed}")
        print(f"  Score: {total:.5f}")
        for k, v in details.items():
            print(f"    {k:20s} = {v:.5f}")

        self.X_train = X_tr
        self.X_test = X_te
        self.y_train = y_tr
        self.y_test = y_te
        self.split_info = {
            'score': total, 'seed': self.seed,
            'details': details,
            'X_tr': X_tr, 'X_te': X_te, 'y_tr': y_tr, 'y_te': y_te
        }
        return self

    # ---------- split search (slow) ----------

    def find_best_split(self):
        if self.seed is not None:
            return self._split_with_seed()

        start, end = self.seed_range
        best = {'score': float('inf')}
        self.split_history = []

        print(f"\n===== Searching for best structural split =====")
        for seed in range(start, end):
            X_tr, X_te, y_tr, y_te = train_test_split(
                self.X_all, self.y_all,
                test_size=self.test_size,
                random_state=seed,
                stratify=self.y_all
            )
            total, details = self._score_split(
                self.X_all, self.y_all, X_tr, X_te, y_tr, y_te,
                global_X=self._global_X
            )
            self.split_history.append({'seed': seed, **details})

            if total < best['score']:
                best = {
                    'score': total, 'seed': seed, 'details': details,
                    'X_tr': X_tr, 'X_te': X_te, 'y_tr': y_tr, 'y_te': y_te
                }

        n_tried = end - start
        print(f"  Tried {n_tried} seeds")
        print(f"  Best seed: {best['seed']}")
        print(f"  Score:     {best['score']:.5f}")
        for k, v in best['details'].items():
            print(f"    {k:20s} = {v:.5f}")

        self.X_train = best['X_tr']
        self.X_test = best['X_te']
        self.y_train = best['y_tr']
        self.y_test = best['y_te']
        self.split_info = best
        return self

    # ---------- scaling ----------

    def scale(self):
        self.scaler = StandardScaler()
        self.X_train_sc = self.scaler.fit_transform(self.X_train)
        self.X_test_sc = self.scaler.transform(self.X_test)

        print(f"\n===== LOCKED SPLIT =====")
        print(f"  X_train: {self.X_train_sc.shape}  "
              f"y_train: {self.y_train.shape}  "
              f"(pos rate: {self.y_train.mean():.5f})")
        print(f"  X_test:  {self.X_test_sc.shape}  "
              f"y_test:  {self.y_test.shape}  "
              f"(pos rate: {self.y_test.mean():.5f})")
        return self

    # ---------- saving ----------

    def save(self):
        """Save processed datasets to processed_dir."""
        np.savez_compressed(
            os.path.join(self.processed_dir, 'split.npz'),
            X_train=self.X_train, X_test=self.X_test,
            y_train=self.y_train, y_test=self.y_test,
            X_train_sc=self.X_train_sc, X_test_sc=self.X_test_sc
        )
        np.savez_compressed(
            os.path.join(self.processed_dir, 'kaggle_test.npz'),
            X_kaggle_test=self.X_kaggle_test,
            test_ids=self.test_ids
        )
        np.savez_compressed(
            os.path.join(self.processed_dir, 'scaler.npz'),
            mean=self.scaler.mean_,
            scale=self.scaler.scale_
        )
        print(f"\n  Saved to {self.processed_dir}/")
        print(f"    split.npz        (train/test, scaled + raw)")
        print(f"    kaggle_test.npz  (test data + IDs)")
        print(f"    scaler.npz       (mean, scale)")
        return self

    # ---------- plots ----------

    def plot_diagnostics(self):
        prefix = 'DataSplitter'
        plot_dir = self.plot_dir

        # --- 1. Split quality across seeds (only if search was run) ---
        if hasattr(self, 'split_history') and self.split_history:
            fig, ax = plt.subplots(figsize=(10, 5))
            df_hist = pd.DataFrame(self.split_history)
            ax.plot(df_hist['seed'], df_hist['total'], 'b-o', markersize=3, alpha=0.7)
            ax.axvline(x=self.split_info['seed'], color='red', linestyle='--',
                       label=f"best seed={self.split_info['seed']}")
            ax.set_xlabel('Seed')
            ax.set_ylabel('Structural deviation score')
            ax.set_title('Split quality across seeds (lower = better)')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{prefix}_split_quality.png'), dpi=150)
            plt.close()

        # --- 2. Score components ---
        fig, ax = plt.subplots(figsize=(8, 5))
        components = ['ratio_diff', 'feat_mean_diff', 'feat_std_diff', 'c0_diff', 'c1_diff']
        values = [self.split_info['details'][c] for c in components]
        colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple', 'goldenrod']
        bars = ax.bar(components, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Deviation')
        ax.set_title(f'Best split (seed={self.split_info["seed"]}): deviation by component')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        ax.tick_params(axis='x', rotation=25)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{prefix}_score_components.png'), dpi=150)
        plt.close()

        # --- 3. Feature means (vs 400k global) ---
        fig, ax = plt.subplots(figsize=(7, 7))
        global_mean = self._global_X.mean(axis=0)
        ax.scatter(global_mean, self.X_train.mean(axis=0), s=8, alpha=0.4,
                   label='train', color='blue')
        ax.scatter(global_mean, self.X_test.mean(axis=0), s=8, alpha=0.4,
                   label='test', color='red')
        lims = [global_mean.min(), global_mean.max()]
        ax.plot(lims, lims, 'k--', alpha=0.3)
        ax.set_xlabel('Global feature mean (400k)')
        ax.set_ylabel('Split feature mean')
        ax.set_title('Feature means: train/test vs global (400k)')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{prefix}_feature_means.png'), dpi=150)
        plt.close()

        # --- 4. Feature stds (vs 400k global) ---
        fig, ax = plt.subplots(figsize=(7, 7))
        global_std = self._global_X.std(axis=0)
        ax.scatter(global_std, self.X_train.std(axis=0), s=8, alpha=0.4,
                   label='train', color='blue')
        ax.scatter(global_std, self.X_test.std(axis=0), s=8, alpha=0.4,
                   label='test', color='red')
        lims = [global_std.min(), global_std.max()]
        ax.plot(lims, lims, 'k--', alpha=0.3)
        ax.set_xlabel('Global feature std (400k)')
        ax.set_ylabel('Split feature std')
        ax.set_title('Feature stds: train/test vs global (400k)')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{prefix}_feature_stds.png'), dpi=150)
        plt.close()

        # --- 5. Class ratio comparison ---
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = ['Full data', 'Train', 'Test']
        pos_rates = [self.y_all.mean(), self.y_train.mean(), self.y_test.mean()]
        neg_rates = [1 - p for p in pos_rates]
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, neg_rates, width, label='Class 0', color='steelblue')
        ax.bar(x + width/2, pos_rates, width, label='Class 1', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Proportion')
        ax.set_title('Class ratio: full vs train vs test')
        ax.legend()
        for i, (n, p) in enumerate(zip(neg_rates, pos_rates)):
            ax.text(i - width/2, n + 0.01, f'{n:.4f}', ha='center', fontsize=9)
            ax.text(i + width/2, p + 0.01, f'{p:.4f}', ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{prefix}_class_ratio.png'), dpi=150)
        plt.close()

        print(f"  Plots saved to {plot_dir}/ with prefix '{prefix}_'")
        return self

    # ---------- full pipeline ----------

    def run(self):
        self.load()
        self.find_best_split()
        self.scale()
        self.save()
        self.plot_diagnostics()
        print("\n  DataSplitter done. Variables ready:")
        print(f"    X_train_sc, X_test_sc, y_train, y_test")
        print(f"    X_kaggle_test, test_ids")
        return self

class RandomUndersampler:
    """
    Random undersampler with configurable scope and class targets.

    Options:
        scope='per_class'   — sample specific classes to specific counts
        scope='population'  — sample entire population to target (proportional)

        seed_search=True    — search seeds, pick best population match
        seed_search=False   — use given seed directly

    Usage:
        # per-class: sample class 0 to 60k, class 1 untouched
        rus = RandomUndersampler(scope='per_class', targets={0: 60000}, seed=42)
        X_out, y_out = rus.fit_resample(X, y)

        # per-class: sample BOTH classes
        rus = RandomUndersampler(scope='per_class', targets={0: 60000, 1: 7500}, seed=42)
        X_out, y_out = rus.fit_resample(X, y)

        # population: sample entire data to 50k rows
        rus = RandomUndersampler(scope='population', target=50000, seed=42)
        X_out, y_out = rus.fit_resample(X, y)

        # with seed search
        rus = RandomUndersampler(scope='per_class', targets={0: 60000}, seed_search=True)
        X_out, y_out = rus.fit_resample(X, y)

        rus.verify(X, y)
        rus.plot(X, y)
    """

    def __init__(self, scope='per_class', targets=None, target=None,
                 seed=42, seed_search=False, n_seeds=100, plot_dir='src/plots'):
        self.scope = scope
        self.targets = targets or {}   # {class_label: n_samples}
        self.target = target           # for population scope
        self.seed = seed
        self.seed_search = seed_search
        self.n_seeds = n_seeds
        self.plot_dir = plot_dir

        self.selected_idx = {}    # class -> indices
        self.X_sampled = {}
        self._search_history = None

    def _sample_random(self, n, size):
        rng = np.random.RandomState(self.seed)
        return rng.choice(n, size=size, replace=False)

    def _sample_random_search(self, X, size):
        """Search seeds, pick sample best matching population."""
        pop_mean = X.mean(axis=0)
        pop_std = X.std(axis=0) + 1e-8

        self._search_history = []
        best = {'score': float('inf')}

        for seed in range(1, 1 + self.n_seeds):
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(X), size=size, replace=False)
            X_sam = X[idx]

            mean_dev = np.abs(X_sam.mean(axis=0) - pop_mean) / pop_std
            std_ratio = np.abs(X_sam.std(axis=0) / pop_std - 1)

            quantile_devs = []
            for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
                pop_q = np.quantile(X, q, axis=0)
                sam_q = np.quantile(X_sam, q, axis=0)
                quantile_devs.append(np.abs(sam_q - pop_q) / pop_std)

            q_dev = np.mean([d.mean() for d in quantile_devs])
            total = mean_dev.mean() + std_ratio.mean() + q_dev

            self._search_history.append({'seed': seed, 'total': total})

            if total < best['score']:
                best = {'score': total, 'seed': seed, 'idx': idx}

        print(f"  Seed search: best seed={best['seed']}  score={best['score']:.6f}")
        self.seed = best['seed']
        return best['idx']

    def fit_resample(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.scope == 'per_class':
            # check that targets are valid
            if not self.targets:
                raise ValueError("targets required for per_class scope. "
                                 "e.g. targets={0: 60000} or {0: 60000, 1: 7500}")

            parts_X = []
            parts_y = []

            for cls in sorted(self.targets.keys()):
                X_cls = X[y == cls]
                n_target = self.targets[cls]

                print(f"  Class {cls}: {len(X_cls)} -> {n_target}")

                if n_target >= len(X_cls):
                    # no sampling needed
                    idx = np.arange(len(X_cls))
                elif self.seed_search:
                    idx = self._sample_random_search(X_cls, n_target)
                else:
                    idx = self._sample_random(len(X_cls), n_target)

                self.selected_idx[cls] = idx
                self.X_sampled[cls] = X_cls[idx]

                parts_X.append(self.X_sampled[cls])
                parts_y.append(np.full(n_target, cls))

            # add untouched classes
            sampled_classes = set(self.targets.keys())
            all_classes = np.unique(y)
            for cls in all_classes:
                if cls not in sampled_classes:
                    X_cls = X[y == cls]
                    print(f"  Class {cls}: {len(X_cls)} (untouched)")
                    self.X_sampled[cls] = X_cls
                    parts_X.append(X_cls)
                    parts_y.append(np.full(len(X_cls), cls))

            X_out = np.vstack(parts_X)
            y_out = np.concatenate(parts_y)

        elif self.scope == 'population':
            if self.target is None:
                raise ValueError("target required for population scope. "
                                 "e.g. target=50000")

            print(f"  Population: {X.shape} -> {self.target}")

            if self.seed_search:
                idx = self._sample_random_search(X, self.target)
            else:
                idx = self._sample_random(len(X), self.target)

            self.selected_idx['population'] = idx
            self.X_sampled['population'] = X[idx]

            X_out = X[idx]
            y_out = y[idx]

        else:
            raise ValueError(f"Unknown scope: {self.scope}")

        # shuffle
        rng = np.random.RandomState(self.seed)
        shuffle_idx = rng.permutation(len(X_out))
        X_out = X_out[shuffle_idx]
        y_out = y_out[shuffle_idx]

        n0 = (y_out == 0).sum()
        n1 = (y_out == 1).sum()
        print(f"  Result: {X_out.shape}, class 0 = {n0}, class 1 = {n1}, "
              f"pos rate = {y_out.mean():.5f}")

        return X_out, y_out

    def verify(self, X_original, y_original):
        """Check distribution preservation per sampled class."""
        X = np.asarray(X_original)
        y = np.asarray(y_original)

        results = {}
        for cls, X_sam in self.X_sampled.items():
            if cls == 'population':
                X_pop = X
            else:
                X_pop = X[y == cls]

            pop_mean = X_pop.mean(axis=0)
            sam_mean = X_sam.mean(axis=0)
            pop_std = X_pop.std(axis=0) + 1e-8

            mean_dev = np.abs(sam_mean - pop_mean) / pop_std
            std_ratio = (X_sam.std(axis=0) + 1e-8) / pop_std

            label = f'class_{cls}' if cls != 'population' else 'population'
            print(f"\n  Distribution preservation ({label}):")
            print(f"    Mean deviation: max={mean_dev.max():.4f}  mean={mean_dev.mean():.4f}")
            print(f"    Std ratio:      min={std_ratio.min():.4f}  max={std_ratio.max():.4f}  mean={std_ratio.mean():.4f}")

            results[label] = {
                'mean_dev_max': mean_dev.max(),
                'mean_dev_mean': mean_dev.mean(),
                'std_ratio_min': std_ratio.min(),
                'std_ratio_max': std_ratio.max(),
                'std_ratio_mean': std_ratio.mean(),
            }

        return results

    def plot(self, X_original, y_original):
        prefix = f'RandomUndersampler_{self.scope}'
        X = np.asarray(X_original)
        y = np.asarray(y_original)

        has_search = self._search_history is not None
        n_classes = len(self.X_sampled)
        n_plots = n_classes * 3 + (1 if has_search else 0)

        fig, axes = plt.subplots(1, n_plots,
                                  figsize=(5 * n_plots, 4), squeeze=False)
        axes = axes.flatten()
        plot_idx = 0

        for cls, X_sam in self.X_sampled.items():
            if cls == 'population':
                X_pop = X
                label = 'Population'
            else:
                X_pop = X[y == cls]
                label = f'Class {cls}'

            pop_mean = X_pop.mean(axis=0)
            sam_mean = X_sam.mean(axis=0)
            pop_std = X_pop.std(axis=0) + 1e-8
            sam_std = X_sam.std(axis=0) + 1e-8
            mean_dev = np.abs(sam_mean - pop_mean) / pop_std

            # means
            axes[plot_idx].scatter(pop_mean, sam_mean, s=8, alpha=0.5)
            lims = [pop_mean.min(), pop_mean.max()]
            axes[plot_idx].plot(lims, lims, 'k--', alpha=0.3)
            axes[plot_idx].set_xlabel('Population mean')
            axes[plot_idx].set_ylabel('Sample mean')
            axes[plot_idx].set_title(f'{label} — means')
            plot_idx += 1

            # stds
            axes[plot_idx].scatter(pop_std, sam_std, s=8, alpha=0.5)
            lims = [pop_std.min(), pop_std.max()]
            axes[plot_idx].plot(lims, lims, 'k--', alpha=0.3)
            axes[plot_idx].set_xlabel('Population std')
            axes[plot_idx].set_ylabel('Sample std')
            axes[plot_idx].set_title(f'{label} — stds')
            plot_idx += 1

            # deviation
            axes[plot_idx].hist(mean_dev, bins=50, color='steelblue',
                                edgecolor='black', linewidth=0.5)
            axes[plot_idx].set_xlabel('Mean deviation (std units)')
            axes[plot_idx].set_ylabel('Feature count')
            axes[plot_idx].set_title(f'{label} — deviation')
            plot_idx += 1

        # seed search
        if has_search:
            df_h = pd.DataFrame(self._search_history)
            axes[plot_idx].plot(df_h['seed'], df_h['total'], 'b-o', markersize=3, alpha=0.7)
            axes[plot_idx].axvline(x=self.seed, color='red', linestyle='--',
                                    label=f'best={self.seed}')
            axes[plot_idx].set_xlabel('Seed')
            axes[plot_idx].set_ylabel('Deviation')
            axes[plot_idx].set_title('Seed search')
            axes[plot_idx].legend()
            axes[plot_idx].grid(alpha=0.3)

        plt.suptitle(f'Random Undersampling ({self.scope})', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'{prefix}.png'), dpi=150)
        plt.close()

        print(f"  Plot saved: {self.plot_dir}/{prefix}.png")
        return self

class SMOter:
    """
    Apply SMOTE to oversample minority class with synthetic points.

    Usage:
        smoter = SMOter(sampling_strategy=0.5, k_neighbors=5, seed=42)
        X_smote, y_smote = smoter.fit_resample(X_train, y_train)
        smoter.plot(X_train, y_train, X_smote, y_smote)
    """

    def __init__(self, sampling_strategy=0.5, k_neighbors=5, seed=42,
                 plot_dir='src/plots'):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.seed = seed
        self.plot_dir = plot_dir

    def fit_resample(self, X, y):
        """
        Apply SMOTE. Returns (X_resampled, y_resampled).

        sampling_strategy:
            0.5  -> class 1 becomes 50% of class 0
            1.0  -> balanced 50/50
            float -> ratio N1/N0 after resampling
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_before_0 = (y == 0).sum()
        n_before_1 = (y == 1).sum()
        print(f"Before SMOTE: class 0 = {n_before_0}, class 1 = {n_before_1} "
              f"(ratio: {n_before_1/n_before_0:.3f})")

        # only apply SMOTE to class 1
        smote = SMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.seed
        )
        X_out, y_out = smote.fit_resample(X, y)

        n_after_0 = (y_out == 0).sum()
        n_after_1 = (y_out == 1).sum()
        n_synthetic = n_after_1 - n_before_1

        print(f"After SMOTE:  class 0 = {n_after_0}, class 1 = {n_after_1} "
              f"(ratio: {n_after_1/n_after_0:.3f})")
        print(f"Synthetic points generated: {n_synthetic}")

        return X_out, y_out

    def plot(self, X_before, y_before, X_after, y_after):
        """Save before/after class distribution and feature comparison plots."""
        prefix = 'SMOter'

        X_before_c1 = X_before[np.asarray(y_before) == 1]
        X_after_c1 = X_after[np.asarray(y_after) == 1]
        synthetic = X_after_c1[len(X_before_c1):]  # only the new synthetic points

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        # --- 1. Class counts before/after ---
        labels = ['Before', 'After']
        c0_vals = [(y_before == 0).sum(), (y_after == 0).sum()]
        c1_vals = [(y_before == 1).sum(), (y_after == 1).sum()]
        x = np.arange(2)
        width = 0.35
        axes[0].bar(x - width/2, c0_vals, width, label='Class 0', color='steelblue')
        axes[0].bar(x + width/2, c1_vals, width, label='Class 1', color='coral')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_ylabel('Count')
        axes[0].set_title('Class counts before/after SMOTE')
        axes[0].legend()
        for i, (n, p) in enumerate(zip(c0_vals, c1_vals)):
            axes[0].text(i - width/2, n + 500, f'{n}', ha='center', fontsize=9)
            axes[0].text(i + width/2, p + 500, f'{p}', ha='center', fontsize=9)

        # --- 2. Feature means: real vs synthetic class 1 ---
        real_mean = X_before_c1.mean(axis=0)
        synth_mean = synthetic.mean(axis=0) if len(synthetic) > 0 else real_mean
        axes[1].scatter(real_mean, synth_mean, s=8, alpha=0.5)
        lims = [real_mean.min(), real_mean.max()]
        axes[1].plot(lims, lims, 'k--', alpha=0.3)
        axes[1].set_xlabel('Real class-1 mean')
        axes[1].set_ylabel('Synthetic class-1 mean')
        axes[1].set_title('Feature means: real vs synthetic')
        axes[1].grid(alpha=0.3)

        # --- 3. Feature stds: real vs synthetic class 1 ---
        real_std = X_before_c1.std(axis=0) + 1e-8
        synth_std = synthetic.std(axis=0) + 1e-8 if len(synthetic) > 0 else real_std
        axes[2].scatter(real_std, synth_std, s=8, alpha=0.5)
        lims = [real_std.min(), real_std.max()]
        axes[2].plot(lims, lims, 'k--', alpha=0.3)
        axes[2].set_xlabel('Real class-1 std')
        axes[2].set_ylabel('Synthetic class-1 std')
        axes[2].set_title('Feature stds: real vs synthetic')
        axes[2].grid(alpha=0.3)

        plt.suptitle('SMOTE — Synthetic Sample Quality', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'{prefix}_distribution.png'), dpi=150)
        plt.close()

        print(f"  Plot saved: {self.plot_dir}/{prefix}_distribution.png")
        return self



from sklearn.cluster import MiniBatchKMeans


class ClusterBalancedSubsampler:
    """
    Create a balanced subsample: equal class 0 and class 1.
    Class 0 (large): random sample.
    Class 1 (small): K-means cluster-based proportional sample (preserves 200D sub-groups).

    Usage:
        cbs = ClusterBalancedSubsampler(target_per_class=7500, n_clusters=50, seed=42)
        X_balanced, y_balanced = cbs.fit_resample(X_train_sc, y_train)
        cbs.verify(X_train_sc, y_train)
        cbs.plot()
    """

    def __init__(self, target_per_class=7500, n_clusters=50, seed=42,
                 plot_dir='src/plots'):
        self.target_per_class = target_per_class
        self.n_clusters = n_clusters
        self.seed = seed
        self.plot_dir = plot_dir

        self.c0_selected_idx = None
        self.c1_selected_idx = None
        self.c1_cluster_ratio = None
        self.c1_clusters_with_zero = None

    @staticmethod
    def _cluster_balanced_subsample(X, n_target, n_clusters, seed):
        """K-means cluster-based proportional subsample. Preserves 200D sub-groups."""
        if len(X) <= n_target:
            return np.arange(len(X))

        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed,
                              batch_size=2048)
        labels = km.fit_predict(X)

        rng = np.random.RandomState(seed)
        selected = []
        for c in range(n_clusters):
            cluster_idx = np.where(labels == c)[0]
            if len(cluster_idx) == 0:
                continue
            n_keep = max(1, int(round(len(cluster_idx) * n_target / len(X))))
            n_keep = min(n_keep, len(cluster_idx))
            chosen = rng.choice(cluster_idx, size=n_keep, replace=False)
            selected.append(chosen)

        return np.concatenate(selected)

    def fit_resample(self, X, y):
        """
        Create balanced subsample. Returns (X_balanced, y_balanced).
        Both classes get exactly target_per_class points.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        rng = np.random.RandomState(self.seed)
        c0_idx = np.where(y == 0)[0]
        c1_idx = np.where(y == 1)[0]

        print(f"Training classes: class 0 = {len(c0_idx)}, class 1 = {len(c1_idx)}")

        # class 0 (large): random sample
        self.c0_selected_idx = rng.choice(
            c0_idx, size=self.target_per_class, replace=False
        )

        # class 1 (small): cluster-based proportional sample
        X_c1 = X[c1_idx]
        c1_local_idx = self._cluster_balanced_subsample(
            X_c1,
            n_target=self.target_per_class,
            n_clusters=self.n_clusters,
            seed=self.seed
        )
        self.c1_selected_idx = c1_idx[c1_local_idx]

        print(f"  Class 0: {len(self.c0_selected_idx)} "
              f"(random from {len(c0_idx)})")
        print(f"  Class 1: {len(self.c1_selected_idx)} "
              f"(cluster-based from {len(c1_idx)})")

        # combine and shuffle
        combined_idx = np.concatenate(
            [self.c0_selected_idx, self.c1_selected_idx]
        )
        shuffle_idx = rng.permutation(len(combined_idx))
        combined_idx = combined_idx[shuffle_idx]

        X_out = X[combined_idx]
        y_out = y[combined_idx]

        print(f"\n  X_balanced: {X_out.shape}")
        print(f"  y_balanced: {y_out.shape}  "
              f"(pos rate: {y_out.mean():.5f})")

        return X_out, y_out

    def verify(self, X_original, y_original):
        """Check class 1 sub-group preservation across clusters."""
        X = np.asarray(X_original)
        y = np.asarray(y_original)
        c1_idx = np.where(y == 1)[0]

        X_c1_orig = X[c1_idx]
        X_c1_sub = X[self.c1_selected_idx]

        # cluster original class 1
        km_check = MiniBatchKMeans(
            n_clusters=self.n_clusters, random_state=self.seed, batch_size=2048
        )
        labels_orig = km_check.fit_predict(X_c1_orig)
        labels_sub = km_check.predict(X_c1_sub)

        orig_counts = np.bincount(labels_orig, minlength=self.n_clusters)
        sub_counts = np.bincount(labels_sub, minlength=self.n_clusters)
        ratio = sub_counts / np.maximum(orig_counts, 1)
        n_zero = (sub_counts == 0).sum()

        self.c1_cluster_ratio = ratio
        self.c1_clusters_with_zero = n_zero

        print(f"\n  Class 1 sub-group check ({self.n_clusters} clusters):")
        print(f"    ratio min={ratio.min():.3f}  max={ratio.max():.3f}  "
              f"mean={ratio.mean():.3f}")
        print(f"    clusters with 0 samples: {n_zero}")

        return {
            'ratio_min': ratio.min(),
            'ratio_max': ratio.max(),
            'ratio_mean': ratio.mean(),
            'clusters_with_zero': n_zero,
        }

    def plot(self):
        """Save sub-group preservation plot for class 1."""
        prefix = 'ClusterBalancedSubsampler'

        if self.c1_cluster_ratio is None:
            print("  No verification data. Call verify() first.")
            return self

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # --- 1. Cluster ratio per cluster ---
        x = np.arange(len(self.c1_cluster_ratio))

        axes[0].bar(x, self.c1_cluster_ratio, color='steelblue',
                    edgecolor='black', linewidth=0.5)
        axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5,
                        label='ideal (1.0)')
        axes[0].set_xlabel('Cluster index')
        axes[0].set_ylabel('Sub-sample ratio')
        axes[0].set_title('Class 1 sub-group representation per cluster')
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis='y')

        # --- 2. Distribution of ratios ---
        axes[1].hist(self.c1_cluster_ratio, bins=20, color='steelblue',
                     edgecolor='black', linewidth=0.5)
        axes[1].axvline(x=1.0, color='red', linestyle='--', alpha=0.5,
                        label='ideal (1.0)')
        axes[1].set_xlabel('Sub-sample ratio')
        axes[1].set_ylabel('Cluster count')
        axes[1].set_title('Distribution of cluster representation ratios')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.suptitle('Cluster-Balanced Subsampling — Class 1 Preservation', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'{prefix}_clusters.png'), dpi=150)
        plt.close()

        print(f"  Plot saved: {self.plot_dir}/{prefix}_clusters.png")
        return self


from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

class PCAReducer:
    """
    PCA dimensionality reduction. Fit on train, transform on test.

    Usage:
        pca = PCAReducer(n_components=30)
        X_train_pca = pca.fit_transform(X_train, y)
        X_test_pca  = pca.transform(X_test)
        pca.plot(X_train_pca, y)
    """

    def __init__(self, n_components=30, plot_dir='src/plots'):
        self.n_components = n_components
        self.plot_dir = plot_dir
        self.model = None

    def fit(self, X, y=None):
        self.model = PCA(n_components=self.n_components)
        self.model.fit(X)
        return self

    def transform(self, X):
        return self.model.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def plot(self, X_pca=None, y=None):
        """Save PCA diagnostic plots."""
        prefix = 'PCAReducer'
        ev = self.model.explained_variance_ratio_
        cum_ev = np.cumsum(ev)

        # --- 1. Scree plot + cumulative variance ---
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(range(1, len(ev) + 1), ev, color='steelblue', alpha=0.7,
                label='individual')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title(f'PCA Scree Plot ({self.n_components} components)')

        ax2 = ax1.twinx()
        ax2.plot(range(1, len(cum_ev) + 1), cum_ev, 'r-o', markersize=4,
                 label='cumulative')
        ax2.axhline(y=cum_ev[-1], color='green', linestyle='--', alpha=0.5,
                     label=f'kept = {cum_ev[-1]*100:.2f}%')
        ax2.set_ylabel('Cumulative Explained Variance')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'{prefix}_scree.png'), dpi=150)
        plt.close()

        # --- 2. 2D projection colored by class ---
        if X_pca is not None and y is not None and X_pca.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(8, 7))
            y = np.asarray(y)
            mask0 = (y == 0)
            mask1 = (y == 1)
            ax.scatter(X_pca[mask0, 0], X_pca[mask0, 1],
                       s=3, alpha=0.3, color='steelblue', label=f'class 0 ({mask0.sum()})')
            ax.scatter(X_pca[mask1, 0], X_pca[mask1, 1],
                       s=5, alpha=0.5, color='coral', label=f'class 1 ({mask1.sum()})')
            ax.set_xlabel(f'PC1 ({ev[0]*100:.2f}%)')
            ax.set_ylabel(f'PC2 ({ev[1]*100:.2f}%)')
            ax.set_title('PCA Projection — PC1 vs PC2')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f'{prefix}_projection.png'), dpi=150)
            plt.close()

        print(f"  Plots saved: {self.plot_dir}/{prefix}_*.png")
        return self


class PLSReducer:
    """
    PLS-DA dimensionality reduction (supervised).
    Fit on train with labels, transform on test.

    Usage:
        pls = PLSReducer(n_components=30)
        X_train_pls = pls.fit_transform(X_train, y_train)
        X_test_pls  = pls.transform(X_test)
        pls.plot(X_train_pls, y_train)
    """

    def __init__(self, n_components=30, plot_dir='src/plots'):
        self.n_components = n_components
        self.plot_dir = plot_dir
        self.model = None

    def fit(self, X, y):
        y = np.asarray(y).reshape(-1, 1)
        self.model = PLSRegression(n_components=self.n_components, scale=False)
        self.model.fit(X, y)
        return self

    def transform(self, X):
        return self.model.transform(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def plot(self, X_pls=None, y=None):
        """Save PLS diagnostic plots."""
        prefix = 'PLSReducer'

        # --- 1. X variance explained per component ---
        x_scores = self.model.x_scores_
        ev = x_scores.var(axis=0)
        ev_ratio = ev / ev.sum()
        cum_ev = np.cumsum(ev_ratio)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.bar(range(1, len(ev_ratio) + 1), ev_ratio, color='steelblue',
                alpha=0.7, label='individual')
        ax1.set_xlabel('PLS Component')
        ax1.set_ylabel('Explained X-Variance Ratio')
        ax1.set_title(f'PLS Component Variance ({self.n_components} components)')

        ax2 = ax1.twinx()
        ax2.plot(range(1, len(cum_ev) + 1), cum_ev, 'r-o', markersize=4,
                 label='cumulative')
        ax2.axhline(y=cum_ev[-1], color='green', linestyle='--', alpha=0.5,
                     label=f'kept = {cum_ev[-1]*100:.2f}%')
        ax2.set_ylabel('Cumulative Explained Variance')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'{prefix}_variance.png'), dpi=150)
        plt.close()

        # --- 2. 2D projection colored by class ---
        if X_pls is not None and y is not None and X_pls.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(8, 7))
            y = np.asarray(y)
            mask0 = (y == 0)
            mask1 = (y == 1)
            ax.scatter(X_pls[mask0, 0], X_pls[mask0, 1],
                       s=3, alpha=0.3, color='steelblue', label=f'class 0 ({mask0.sum()})')
            ax.scatter(X_pls[mask1, 0], X_pls[mask1, 1],
                       s=5, alpha=0.5, color='coral', label=f'class 1 ({mask1.sum()})')
            ax.set_xlabel('PLS Component 1')
            ax.set_ylabel('PLS Component 2')
            ax.set_title('PLS-DA Projection — Component 1 vs 2')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f'{prefix}_projection.png'), dpi=150)
            plt.close()

        # --- 3. Component weights (which features matter per component) ---
        weights = self.model.x_weights_  # (n_features, n_components)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # top 20 feature weights for component 1
        w1 = weights[:, 0]
        top_idx = np.argsort(-np.abs(w1))[:20]
        axes[0].barh(range(len(top_idx)), w1[top_idx][::-1],
                     color=['coral' if v > 0 else 'steelblue' for v in w1[top_idx][::-1]])
        axes[0].set_yticks(range(len(top_idx)))
        axes[0].set_yticklabels([f'feat_{i}' for i in top_idx][::-1], fontsize=8)
        axes[0].set_xlabel('Weight')
        axes[0].set_title('Top 20 feature weights — PLS Component 1')

        # top 20 feature weights for component 2
        w2 = weights[:, 1]
        top_idx = np.argsort(-np.abs(w2))[:20]
        axes[1].barh(range(len(top_idx)), w2[top_idx][::-1],
                     color=['coral' if v > 0 else 'steelblue' for v in w2[top_idx][::-1]])
        axes[1].set_yticks(range(len(top_idx)))
        axes[1].set_yticklabels([f'feat_{i}' for i in top_idx][::-1], fontsize=8)
        axes[1].set_xlabel('Weight')
        axes[1].set_title('Top 20 feature weights — PLS Component 2')

        plt.suptitle('PLS Feature Weights', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'{prefix}_weights.png'), dpi=150)
        plt.close()

        print(f"  Plots saved: {self.plot_dir}/{prefix}_*.png")
        return self



import joblib


class SaverLoader:
    """
    Save and load processed datasets and scalers to/from processed/.
    All outputs go as .npz or .joblib for fast reload.

    Usage:
        sl = SaverLoader(processed_dir='data/processed')
        sl.save('undersampled', X, y)
        sl.save_scaler('main', scaler)
        X, y = sl.load('undersampled')
        scaler = sl.load_scaler('main')
    """

    def __init__(self, processed_dir='data/processed'):
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def save(self, name, X, y, X_test=None, y_test=None, test_ids=None):
        """
        Save a dataset. Always saves X and y.
        Optionally saves X_test, y_test, test_ids alongside.

        Files created:
            data/processed/{name}.npz
        """
        arrays = {'X': X, 'y': y}
        if X_test is not None:
            arrays['X_test'] = X_test
        if y_test is not None:
            arrays['y_test'] = y_test
        if test_ids is not None:
            arrays['test_ids'] = test_ids

        path = os.path.join(self.processed_dir, f'{name}.npz')
        np.savez_compressed(path, **arrays)

        shapes = ', '.join(f"{k}={v.shape}" for k, v in arrays.items())
        print(f"  Saved: {path}  ({shapes})")
        return self

    def load(self, name):
        """
        Load a dataset. Returns dict with all saved arrays.
        Keys: 'X', 'y', optionally 'X_test', 'y_test', 'test_ids'.
        """
        path = os.path.join(self.processed_dir, f'{name}.npz')
        data = np.load(path, allow_pickle=True)

        result = {k: data[k] for k in data.files}
        shapes = ', '.join(f"{k}={v.shape}" for k, v in result.items())
        print(f"  Loaded: {path}  ({shapes})")
        return result

    def save_scaler(self, name, scaler):
        """
        Save a fitted StandardScaler as joblib.
        Files created:
            data/processed/scaler_{name}.joblib
        """
        path = os.path.join(self.processed_dir, f'scaler_{name}.joblib')
        joblib.dump(scaler, path)
        print(f"  Saved: {path}")
        return self

    def load_scaler(self, name):
        """Load a fitted StandardScaler."""
        path = os.path.join(self.processed_dir, f'scaler_{name}.joblib')
        scaler = joblib.load(path)
        print(f"  Loaded: {path}")
        return scaler

    def save_pickle(self, name, obj):
        """Save any Python object (configs, model params, etc.)."""
        path = os.path.join(self.processed_dir, f'{name}.pkl')
        joblib.dump(obj, path)
        print(f"  Saved: {path}")
        return self

    def load_pickle(self, name):
        """Load any Python object."""
        path = os.path.join(self.processed_dir, f'{name}.pkl')
        obj = joblib.load(path)
        print(f"  Loaded: {path}")
        return obj

    def list_files(self):
        """List all processed files."""
        files = sorted(os.listdir(self.processed_dir))
        print(f"\n  Files in {self.processed_dir}/:")
        for f in files:
            size = os.path.getsize(os.path.join(self.processed_dir, f))
            print(f"    {f:40s}  {size/1024:.1f} KB")
        return files




