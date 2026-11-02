# src/features.py

import numpy as np
from scipy.stats import skew

class FeatureEngineer:
    """
    Config-driven feature engineering. One class, all feature types.

    Config keys:
        squared:        bool — add x_i^2 for all features
        row_stats:      None | 'basic' (mean,std) | 'full' (7 stats)
        freq:           bool — frequency encoding per feature
        freq_rounded:   None | int — frequency of values rounded to N decimals
        bins_target:    bool — quantile binning + target encoding
        interactions:   bool — pairwise products of top-N features
        group_agg:      bool — group-wise aggregation stats
        gauss_distance: bool — per-feature distance from mean/std (z-score per feature)
        row_rank_stats: bool — rank-based row statistics (entropy, min_rank, max_rank, mean_rank, std_rank)
        poly_row_stats: bool — polynomial row stats (row_mean^2, row_std^2, row_mean*row_std)
        top_n:          int  — number of features for interactions (default 20)
        n_groups:       int  — number of groups for group_agg (default 10)
        replace_raw:    bool — if True, output ONLY engineered features (no raw 200)

    Usage:
        # raw + engineered
        fe = FeatureEngineer({'freq': True, 'row_stats': 'full'})
        X_train_fe = fe.fit_transform(X_train, y_train)

        # only engineered (for NB models)
        fe = FeatureEngineer({'freq': True}, replace_raw=True)
        X_train_fe = fe.fit_transform(X_train, y_train)
    """

    def __init__(self, config=None, replace_raw=False):
        self.config = config or {}
        self.replace_raw = replace_raw
        self._fitted = False

        # learned mappings
        self._freq_maps = None
        self._freq_rounded_maps = None
        self._bin_edges = None
        self._target_maps = None
        self._top_features = None
        self._group_indices = None

    # ====================== FIT ======================

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape

        # --- freq encoding maps ---
        if self.config.get('freq'):
            self._freq_maps = []
            for j in range(d):
                vals, counts = np.unique(X[:, j], return_counts=True)
                self._freq_maps.append(dict(zip(vals, counts / n)))

        # --- freq_rounded maps ---
        round_digits = self.config.get('freq_rounded')
        if round_digits is not None:
            self._freq_rounded_maps = []
            X_rounded = np.round(X, round_digits)
            for j in range(d):
                vals, counts = np.unique(X_rounded[:, j], return_counts=True)
                self._freq_rounded_maps.append(dict(zip(vals, counts / n)))

        # --- binning + target encoding maps ---
        if self.config.get('bins_target') and y is not None:
            y_arr = np.asarray(y).astype(float)
            n_bins = 10
            self._bin_edges = []
            self._target_maps = []
            for j in range(d):
                edges = np.quantile(X[:, j], np.linspace(0, 1, n_bins + 1))
                edges = np.unique(edges)
                self._bin_edges.append(edges)
                bin_idx = np.clip(
                    np.searchsorted(edges, X[:, j], side='right') - 1,
                    0, len(edges) - 2
                )
                target_map = {}
                for b in range(len(edges) - 1):
                    mask = bin_idx == b
                    if mask.sum() > 0:
                        target_map[b] = y_arr[mask].mean()
                    else:
                        target_map[b] = y_arr.mean()
                self._target_maps.append(target_map)

        # --- top-N features for interactions ---
        if self.config.get('interactions'):
            top_n = self.config.get('top_n', 20)
            if y is not None:
                y_arr = np.asarray(y)
                scores = np.zeros(d)
                for j in range(d):
                    X0 = X[y_arr == 0, j]
                    X1 = X[y_arr == 1, j]
                    if len(X0) > 1 and len(X1) > 1:
                        var_between = (X0.mean() - X1.mean()) ** 2
                        var_within = X0.var() + X1.var() + 1e-10
                        scores[j] = var_between / var_within
                self._top_features = np.argsort(-scores)[:top_n]
            else:
                self._top_features = np.arange(min(top_n, d))

        # --- group indices for group_agg ---
        if self.config.get('group_agg'):
            n_groups = self.config.get('n_groups', 10)
            rng = np.random.RandomState(42)
            all_idx = rng.permutation(d)
            self._group_indices = np.array_split(all_idx, n_groups)

        self._fitted = True
        return self

    # ====================== TRANSFORM ======================

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        X = np.asarray(X, dtype=np.float64)
        features = [] if self.replace_raw else [X]

        if self.config.get('squared'):
            features.append(self._squared(X))

        row_stats = self.config.get('row_stats')
        if row_stats == 'basic':
            features.append(self._row_stats(X, mode='basic'))
        elif row_stats == 'full':
            features.append(self._row_stats(X, mode='full'))

        if self.config.get('freq'):
            features.append(self._freq_encode(X))

        if self.config.get('freq_rounded') is not None:
            features.append(self._freq_rounded_encode(X))

        if self.config.get('bins_target'):
            features.append(self._bins_target_encode(X))

        if self.config.get('interactions'):
            features.append(self._interactions(X))

        if self.config.get('group_agg'):
            features.append(self._group_aggregation(X))

        if self.config.get('gauss_distance'):
            features.append(self._gauss_distance(X))

        if self.config.get('row_rank_stats'):
            features.append(self._row_rank_stats(X))

        if self.config.get('poly_row_stats'):
            features.append(self._poly_row_stats(X))

        return np.hstack(features)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    # ====================== PRIVATE METHODS ======================

    def _squared(self, X):
        """x_i^2 for all features. Count: d"""
        return X ** 2

    def _row_stats(self, X, mode='full'):
        """Row-wise statistics across all features."""
        mu = X.mean(axis=1, keepdims=True)
        sig = X.std(axis=1, keepdims=True) + 1e-8

        row_mean = X.mean(axis=1, keepdims=True)
        row_std = X.std(axis=1, keepdims=True)

        if mode == 'basic':
            return np.hstack([row_mean, row_std])

        row_skew = (((X - mu) / sig) ** 3).mean(axis=1, keepdims=True)
        row_min = X.min(axis=1, keepdims=True)
        row_max = X.max(axis=1, keepdims=True)
        row_sum = X.sum(axis=1, keepdims=True)
        row_outlier_count = ((np.abs((X - mu) / sig) > 2).sum(axis=1)).reshape(-1, 1)

        return np.hstack([row_mean, row_std, row_skew, row_min,
                          row_max, row_sum, row_outlier_count])

    def _freq_encode(self, X):
        """Frequency encoding per feature. Count: d"""
        n, d = X.shape
        out = np.empty((n, d), dtype=np.float64)
        for j in range(d):
            mapping = self._freq_maps[j]
            out[:, j] = np.array([mapping.get(v, 0.0) for v in X[:, j]])
        return out

    def _freq_rounded_encode(self, X):
        """Frequency encoding on rounded values. Count: d"""
        n, d = X.shape
        round_digits = self.config['freq_rounded']
        X_rounded = np.round(X, round_digits)
        out = np.empty((n, d), dtype=np.float64)
        for j in range(d):
            mapping = self._freq_rounded_maps[j]
            out[:, j] = np.array([mapping.get(v, 0.0) for v in X_rounded[:, j]])
        return out

    def _bins_target_encode(self, X):
        """Quantile binning + target encoding per feature. Count: d"""
        n, d = X.shape
        out = np.empty((n, d), dtype=np.float64)
        for j in range(d):
            edges = self._bin_edges[j]
            bin_idx = np.clip(
                np.searchsorted(edges, X[:, j], side='right') - 1,
                0, len(edges) - 2
            )
            target_map = self._target_maps[j]
            out[:, j] = np.array([target_map.get(b, 0.5) for b in bin_idx])
        return out

    def _interactions(self, X):
        """Pairwise products of top-N features. Count: n_top*(n_top-1)/2"""
        cols = self._top_features
        X_top = X[:, cols]
        n_top = X_top.shape[1]
        products = []
        for i in range(n_top):
            for j in range(i + 1, n_top):
                products.append((X_top[:, i] * X_top[:, j]).reshape(-1, 1))
        return np.hstack(products)

    def _group_aggregation(self, X):
        """Group-wise stats (mean, std, min, max) per group. Count: n_groups*4"""
        cols_list = []
        for grp_idx in self._group_indices:
            X_grp = X[:, grp_idx]
            cols_list.append(np.column_stack([
                X_grp.mean(axis=1),
                X_grp.std(axis=1),
                X_grp.min(axis=1),
                X_grp.max(axis=1),
            ]))
        return np.hstack(cols_list)

    def _gauss_distance(self, X):
        """
        Per-feature standardized deviation: |x_i - mean_i| / std_i.
        'How far from typical' for each feature.
        Count: d
        """
        mu = X.mean(axis=1, keepdims=True)
        sig = X.std(axis=1, keepdims=True) + 1e-8
        z = np.abs((X - mu) / sig)
        return z

    def _row_rank_stats(self, X):
        """
        Rank each row's values, then compute rank-based statistics.
        Captures shape/profile of each row independent of scale.
        Count: 5 (entropy, min_rank, max_rank, mean_rank, std_rank)
        """
        n, d = X.shape
        # rank each row's values (0 to 1)
        ranks = np.empty((n, d), dtype=np.float64)
        for i in range(n):
            ranks[i] = np.argsort(np.argsort(X[i])) / (d - 1)

        # entropy of rank distribution (binned into 20 bins)
        n_bins = 20
        bin_idx = np.clip((ranks * n_bins).astype(int), 0, n_bins - 1)
        entropies = np.zeros(n)
        for i in range(n):
            counts = np.bincount(bin_idx[i], minlength=n_bins).astype(np.float64)
            probs = counts / d
            probs = probs[probs > 0]
            entropies[i] = -np.sum(probs * np.log(probs + 1e-10))

        min_rank = ranks.min(axis=1, keepdims=True)
        max_rank = ranks.max(axis=1, keepdims=True)
        mean_rank = ranks.mean(axis=1, keepdims=True)
        std_rank = ranks.std(axis=1, keepdims=True)

        return np.column_stack([entropies, min_rank, max_rank,
                                mean_rank, std_rank])

    def _poly_row_stats(self, X):
        """
        Polynomial row statistics.
        Count: 3 (row_mean^2, row_std^2, row_mean * row_std)
        """
        row_mean = X.mean(axis=1, keepdims=True)
        row_std = X.std(axis=1, keepdims=True)

        return np.column_stack([
            row_mean ** 2,
            row_std ** 2,
            row_mean * row_std,
        ])