
# src/models.py

import numpy as np
import warnings
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import gennorm
from sklearn.neighbors import KernelDensity


# ====================== SHARED EVALUATION ======================

def _best_threshold_metrics(y_true, scores):
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


class BaseModel:
    """Interface: fit(X, y) -> self, predict_proba(X) -> 1D scores, evaluate(X_val, y_val) -> dict."""
    name = 'base'

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def evaluate(self, X_val, y_val):
        scores = self.predict_proba(X_val)
        metrics = _best_threshold_metrics(y_val, scores)
        metrics['name'] = self.name
        return metrics


# ====================== 1. GNB ======================

class GNBModel(BaseModel):
    name = 'GNB'

    def __init__(self, var_smoothing=1e-6):
        self.var_smoothing = var_smoothing
        self.model = None

    def fit(self, X, y):
        self.model = GaussianNB(var_smoothing=self.var_smoothing)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


# ====================== 2. KNN with preprocessor ======================

class KNNModel(BaseModel):
    """KNN with optional preprocessor (PCA or PLS)."""

    def __init__(self, preprocessor=None, k=301, weights='distance'):
        self.preprocessor = preprocessor
        self.k = k
        self.weights = weights
        self.model = None
        self.name = f'KNN_{type(preprocessor).__name__}' if preprocessor else 'KNN'

    def fit(self, X, y):
        if self.preprocessor is not None:
            X_proc = self.preprocessor.fit_transform(X, y)
        else:
            X_proc = X
        self.model = KNeighborsClassifier(
            n_neighbors=self.k, weights=self.weights, n_jobs=-1)
        self.model.fit(X_proc, y)
        return self

    def predict_proba(self, X):
        if self.preprocessor is not None:
            X_proc = self.preprocessor.transform(X)
        else:
            X_proc = X
        return self.model.predict_proba(X_proc)[:, 1]


# ====================== 3. XGBoost ======================

class XGBoostModel(BaseModel):
    name = 'XGBoost'

    def __init__(self, **params):
        self.params = params or dict(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
        )
        self.model = None

    def fit(self, X, y):
        self.model = XGBClassifier(
            **self.params,
            tree_method='hist', eval_metric='auc',
            n_jobs=-1, random_state=42, verbosity=0)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


# ====================== 4. LightGBM ======================

class LightGBMModel(BaseModel):
    name = 'LightGBM'

    def __init__(self, **params):
        self.params = params or dict(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
        )
        self.model = None

    def fit(self, X, y):
        self.model = LGBMClassifier(
            **self.params,
            is_unbalance=True,
            n_jobs=-1, random_state=42, verbose=-1)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


# ====================== 5. RBF Neural Network ======================
class _RBFLayer(nn.Module):
    def __init__(self, n_input, n_rbf, X_init):
        super().__init__()
        idx = np.random.RandomState(42).choice(len(X_init), n_rbf, replace=False)
        self.centers = nn.Parameter(torch.FloatTensor(X_init[idx]))
        with torch.no_grad():
            d = torch.cdist(self.centers, self.centers)
            med = d.median().item()
        self.log_sigma = nn.Parameter(torch.ones(n_rbf) * np.log(med / 2 + 1e-8))

    def forward(self, x):
        d2 = torch.cdist(x, self.centers, p=2).pow(2)
        sigma2 = torch.exp(2 * self.log_sigma).unsqueeze(0)
        return torch.exp(-d2 / (2 * sigma2))


class _RBFNet(nn.Module):
    """Module-level class so pickle can find it."""
    def __init__(self, n_input, n_rbf, X_init):
        super().__init__()
        self.rbf = _RBFLayer(n_input, n_rbf, X_init)
        self.output = nn.Linear(n_rbf, 1)

    def forward(self, x):
        return self.output(self.rbf(x)).squeeze(-1)


class RBFNNModel(BaseModel):
    name = 'RBF_NN'

    def __init__(self, n_rbf=256, epochs=50, lr=0.001, batch_size=2048,
                 pos_weight=9.0, seed=42):
        self.n_rbf = n_rbf
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.pos_weight = pos_weight
        self.seed = seed
        self.model = None

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y)

        self.model = _RBFNet(X.shape[1], self.n_rbf, X)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight]))
        dl = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            for Xb, yb in dl:
                optimizer.zero_grad()
                loss = criterion(self.model(Xb), yb)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(self.model(torch.FloatTensor(X))).numpy()
        return scores

# ====================== 6. GenGaussian NB ======================

class GenGaussianNBModel(BaseModel):
    name = 'GenGaussian_NB'

    def __init__(self, seed=42):
        self.seed = seed
        self.classes_ = None
        self.priors_ = None
        self.params_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.priors_ = np.array([max((y == c).mean(), 1e-10) for c in self.classes_])
        self.params_ = {}
        for c in self.classes_:
            Xc = X[y == c]
            self.params_[c] = []
            for j in range(X.shape[1]):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        p, loc, scale = gennorm.fit(Xc[:, j])
                    p = max(p, 0.1)
                    scale = max(scale, 1e-6)
                except Exception:
                    p = 2.0
                    loc = Xc[:, j].mean()
                    scale = max(Xc[:, j].std(), 1e-6)
                self.params_[c].append((p, loc, scale))
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        n = len(X)
        log_joint = np.zeros((n, len(self.classes_)))
        for ci, c in enumerate(self.classes_):
            for j in range(X.shape[1]):
                p, loc, scale = self.params_[c][j]
                log_joint[:, ci] += gennorm.logpdf(X[:, j], p, loc=loc, scale=scale)
            log_joint[:, ci] += np.log(self.priors_[ci])
        log_joint = np.nan_to_num(log_joint, nan=-1e10, posinf=1e10, neginf=-1e10)
        log_joint -= log_joint.max(axis=1, keepdims=True)
        probs = np.exp(log_joint)
        probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-10, None)
        return probs[:, list(self.classes_).index(1)]


# ====================== 7. KDE-NB (tophat) ======================

class KDENBModel(BaseModel):
    name = 'KDE_NB_tophat'

    def __init__(self, kernel='tophat', bandwidth=0.5, max_train=5000, seed=42):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.max_train = max_train
        self.seed = seed
        self.classes_ = None
        self.priors_ = None
        self._sorted_train = {}

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.priors_ = np.array([max((y == c).mean(), 1e-10) for c in self.classes_])
        self._sorted_train = {}
        rng = np.random.RandomState(self.seed)
        for c in self.classes_:
            Xc = X[y == c]
            if len(Xc) > self.max_train:
                idx = rng.choice(len(Xc), self.max_train, replace=False)
                Xc = Xc[idx]
            # store sorted values per feature for fast searchsorted
            self._sorted_train[c] = np.sort(Xc, axis=0)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        n = len(X)
        h = self.bandwidth
        log_joint = np.zeros((n, len(self.classes_)))
        for ci, c in enumerate(self.classes_):
            sorted_c = self._sorted_train[c]
            k = len(sorted_c)
            for j in range(X.shape[1]):
                col = sorted_c[:, j]
                # count training points within bandwidth h of each test point
                lo = np.searchsorted(col, X[:, j] - h, side='left')
                hi = np.searchsorted(col, X[:, j] + h, side='right')
                counts = (hi - lo).astype(np.float64)
                # tophat KDE: P(x) = count / (2*k*h)
                log_joint[:, ci] += np.log(counts / (2 * k * h) + 1e-10)
            log_joint[:, ci] += np.log(self.priors_[ci])
        log_joint = np.nan_to_num(log_joint, nan=-1e10, posinf=1e10, neginf=-1e10)
        log_joint -= log_joint.max(axis=1, keepdims=True)
        probs = np.exp(log_joint)
        probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-10, None)
        return probs[:, list(self.classes_).index(1)]