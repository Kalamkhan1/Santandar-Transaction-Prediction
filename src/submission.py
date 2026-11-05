# src/submission.py

import os
import numpy as np
import pandas as pd
from datetime import datetime


class Submitter:
    """
    Generate submission files from model predictions on Kaggle test set.

    Usage:
        # single model
        sub = Submitter(submission_dir='submissions')
        sub.submit('gnb_v1', model, X_kaggle_sc, test_ids, threshold=0.25)

        # ensemble (average of multiple model probabilities)
        sub.submit_ensemble('ens_8_v1', [model1, model2, ...],
                            X_kaggle_sc, test_ids, threshold=0.20)

        # ensemble with custom weights (from LR router)
        sub.submit_weighted('ens_router_v1', [model1, ...],
                            X_kaggle_sc, test_ids, weights=lr.coef_[0],
                            threshold=0.08)
    """

    def __init__(self, submission_dir='submissions'):
        self.submission_dir = submission_dir
        os.makedirs(self.submission_dir, exist_ok=True)
        self.history = []

    def _save(self, name, preds, test_ids, metadata=None):
        """Save predictions as submission CSV."""
        sub = pd.DataFrame({'ID_code': test_ids, 'target': preds.astype(int)})
        filename = f'{name}.csv'
        filepath = os.path.join(self.submission_dir, filename)
        sub.to_csv(filepath, index=False)

        n_pos = preds.sum()
        pct = n_pos / len(preds) * 100

        record = {
            'name': name,
            'file': filename,
            'n_rows': len(sub),
            'n_positive': int(n_pos),
            'pct_positive': pct,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **(metadata or {})
        }
        self.history.append(record)

        print(f"\n  Submission saved: {filepath}")
        print(f"    rows:          {len(sub)}")
        print(f"    pred positives: {n_pos} ({pct:.2f}%)")
        return filepath

    def submit(self, name, model, X_kaggle, test_ids, threshold):
        """
        Single model submission.
        Applies threshold to probabilities → binary predictions.
        """
        scores = model.predict_proba(X_kaggle)
        preds = (scores >= threshold).astype(int)

        return self._save(name, preds, test_ids, {
            'model_type': 'single',
            'model_name': model.name if hasattr(model, 'name') else 'unknown',
            'threshold': threshold,
        })

    def submit_ensemble(self, name, models, X_kaggle, test_ids, threshold):
        """
        Ensemble submission (simple average of probabilities).
        All models must have predict_proba.
        """
        all_scores = np.zeros(len(X_kaggle))
        model_names = []

        for m in models:
            scores = m.predict_proba(X_kaggle)
            all_scores += scores
            model_names.append(m.name if hasattr(m, 'name') else 'unknown')

        all_scores /= len(models)
        preds = (all_scores >= threshold).astype(int)

        return self._save(name, preds, test_ids, {
            'model_type': 'ensemble_avg',
            'models': model_names,
            'n_models': len(models),
            'threshold': threshold,
        })

    def submit_weighted(self, name, models, X_kaggle, test_ids,
                        weights, threshold):
        """
        Weighted ensemble submission (from LR router or manual weights).
        weights: 1D array matching len(models).
        """
        all_scores = np.zeros(len(X_kaggle))
        model_names = []

        for m, w in zip(models, weights):
            scores = m.predict_proba(X_kaggle)
            all_scores += w * scores
            model_names.append(m.name if hasattr(m, 'name') else 'unknown')

        preds = (all_scores >= threshold).astype(int)

        return self._save(name, preds, test_ids, {
            'model_type': 'ensemble_weighted',
            'models': model_names,
            'weights': weights.tolist() if hasattr(weights, 'tolist') else list(weights),
            'n_models': len(models),
            'threshold': threshold,
        })

    def submit_scores(self, name, scores, test_ids, threshold=None):
        """Submit raw scores. If already binary, threshold is just metadata."""
        preds = np.asarray(scores)
        if threshold is not None:
            preds = (preds >= threshold).astype(int)
        else:
            preds = preds.astype(int)
        return self._save(name, preds, test_ids, {
            'model_type': 'raw_scores',
            'threshold': threshold,
        })

    def show_history(self):
        """Print all submissions made so far."""
        if not self.history:
            print("  No submissions yet.")
            return
        df = pd.DataFrame(self.history)
        print(f"\n{'=' * 80}")
        print("SUBMISSION HISTORY")
        print(f"{'=' * 80}")
        print(df.to_string(index=False))
        return df