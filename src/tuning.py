# src/tuning.py

import os
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import roc_curve, auc


class OptunaTuner:
    """
    Bayesian hyperparameter tuning with Optuna.
    Pure methodology — no model-specific logic.

    The experiment provides:
        1. model_factory: function(params) -> model instance (with fit/predict_proba)
        2. suggest_params: function(trial) -> dict of params

    Usage:
        tuner = OptunaTuner(
            name='XGBoost',
            model_factory=lambda p: XGBoostModel(**p),
            suggest_params=lambda trial: {
                'n_estimators': trial.suggest_int('n_estimators', 300, 5000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                ...
            },
            n_trials=50
        )
        best_params = tuner.fit(X_train, y_train, X_val, y_val)
        tuner.plot()
        tuner.save('data/processed/tuned_xgboost.pkl')
    """

    def __init__(self, name, model_factory, suggest_params,
                 n_trials=50, seed=42, plot_dir='src/plots',
                 timeout=None, scoring='auc_prob'):
        """
        name:             label for logs and plots
        model_factory:    function(params_dict) -> model with fit(X,y) and predict_proba(X)
        suggest_params:   function(optuna_trial) -> dict of params
        n_trials:         number of Optuna trials
        scoring:          'auc_prob' (probability AUC) or 'auc_binary' (Youden's J)
        """
        self.name = name
        self.model_factory = model_factory
        self.suggest_params = suggest_params
        self.n_trials = n_trials
        self.seed = seed
        self.plot_dir = plot_dir
        self.timeout = timeout
        self.scoring = scoring

        self.study = None
        self.best_params = None
        self.best_score = None
        self.trial_history = None

    def _evaluate(self, y_true, scores):
        """Score predictions. Returns metric value."""
        fpr, tpr, thr_arr = roc_curve(y_true, scores)

        if self.scoring == 'auc_prob':
            return auc(fpr, tpr)
        elif self.scoring == 'auc_binary':
            J = tpr - fpr
            return (1 + J.max()) / 2
        else:
            raise ValueError(f"Unknown scoring: {self.scoring}")

    def _objective(self, trial):
        """Optuna objective: fit on train, score on validation."""
        params = self.suggest_params(trial)
        model = self.model_factory(params)

        model.fit(self._X_train, self._y_train)
        scores = model.predict_proba(self._X_val)

        return self._evaluate(self._y_val, scores)
    
    def _progress_callback(self, study, trial):
        if trial.number % 5 == 0 or trial.number == 0:
            best = study.best_value if study.best_value else 0
            print(f"    trial {trial.number + 1}/{self.n_trials}: "
                f"value={trial.value:.4f}  best={best:.4f}")
            
    def fit(self, X_train, y_train, X_val, y_val):
        """
        Run Optuna tuning. Train on train set, score on validation set.
        Returns best params dict.
        """
        self._X_train = np.asarray(X_train)
        self._y_train = np.asarray(y_train)
        self._X_val = np.asarray(X_val)
        self._y_val = np.asarray(y_val)

        print(f"\n{'=' * 70}")
        print(f"  TUNING: {self.name}  ({self.n_trials} trials)")
        print(f"  Train: {self._X_train.shape}  Val: {self._X_val.shape}")
        print(f"  Scoring: {self.scoring}")
        print(f"{'=' * 70}")

        t0 = time.time()

        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
            callbacks=[self._progress_callback]
        )

        elapsed = time.time() - t0
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        print(f"\n  Best {self.scoring}: {self.best_score:.4f}  ({elapsed:.1f}s)")
        print(f"  Best params:")
        for k, v in self.best_params.items():
            print(f"    {k:30s} = {v}")

        return self.best_params

    def get_trial_history(self):
        """Return trial history as list of dicts."""
        if self.study is None:
            return []
        rows = []
        for t in self.study.trials:
            rows.append({
                'trial': t.number,
                'value': t.value,
                'state': t.state.name,
                **(t.params or {})
            })
        self.trial_history = rows
        return rows

    def plot(self):
        """Save optimization history and parameter importance plots."""
        prefix = f'OptunaTuner_{self.name}'

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # --- 1. Optimization history ---
        trials = self.study.trials
        values = [t.value for t in trials if t.value is not None]
        best_so_far = np.maximum.accumulate(values)

        axes[0].plot(range(len(values)), values, 'b.', alpha=0.3, label='trial')
        axes[0].plot(range(len(values)), best_so_far, 'r-', lw=2, label='best so far')
        axes[0].axhline(y=self.best_score, color='green', linestyle='--',
                        label=f'best={self.best_score:.4f}')
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel(self.scoring)
        axes[0].set_title(f'{self.name} — Optimization History')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # --- 2. Parameter importance ---
        try:
            importance = optuna.importance.get_param_importances(self.study)
            names = list(importance.keys())
            vals = list(importance.values())
            axes[1].barh(names, vals, color='steelblue',
                         edgecolor='black', linewidth=0.5)
            axes[1].set_xlabel('Importance')
            axes[1].set_title(f'{self.name} — Parameter Importance')
            axes[1].grid(alpha=0.3, axis='x')
        except Exception:
            axes[1].text(0.5, 0.5, 'Not enough trials\nfor importance',
                         ha='center', va='center', fontsize=12)
            axes[1].set_title(f'{self.name} — Parameter Importance')

        plt.suptitle(f'Hyperparameter Tuning: {self.name}', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f'{prefix}.png'), dpi=150)
        plt.close()

        print(f"  Plot saved: {self.plot_dir}/{prefix}.png")
        return self

    def save(self, path):
        """Save best params to pickle."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {
            'name': self.name,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'scoring': self.scoring,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  Saved: {path}")
        return self

    @staticmethod
    def load(path):
        """Load saved params from pickle."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded: {path}")
        print(f"  {data['name']}: {data['scoring']}={data['best_score']:.4f}")
        return data


class TuningPipeline:
    """
    Run multiple tuners sequentially. Collect results, save summary.

    Usage:
        pipeline = TuningPipeline()
        pipeline.add('XGBoost', model_factory, suggest_params, n_trials=50)
        pipeline.add('LightGBM', model_factory, suggest_params, n_trials=50)
        pipeline.run(X_train, y_train, X_val, y_val)
        pipeline.summary()
        pipeline.plot_all()
        pipeline.save_all('data/processed/')
    """

    def __init__(self, seed=42, plot_dir='src/plots', scoring='auc_prob'):
        self.tuners = []
        self.seed = seed
        self.plot_dir = plot_dir
        self.scoring = scoring

    def add(self, name, model_factory, suggest_params,
            n_trials=50, timeout=None):
        """Register a model for tuning."""
        tuner = OptunaTuner(
            name=name,
            model_factory=model_factory,
            suggest_params=suggest_params,
            n_trials=n_trials,
            seed=self.seed,
            plot_dir=self.plot_dir,
            timeout=timeout,
            scoring=self.scoring
        )
        self.tuners.append(tuner)
        return self

    def run(self, X_train, y_train, X_val, y_val):
        """Run all tuners sequentially."""
        for tuner in self.tuners:
            tuner.fit(X_train, y_train, X_val, y_val)
        return self

    def summary(self):
        """Print summary table of all tuned models."""
        print(f"\n{'=' * 85}")
        print(f"TUNING SUMMARY — scored by {self.scoring}")
        print(f"{'=' * 85}")

        rows = []
        for tuner in self.tuners:
            rows.append({
                'model': tuner.name,
                self.scoring: tuner.best_score,
                'n_trials': tuner.n_trials,
                'params': tuner.best_params,
            })

        rows.sort(key=lambda r: r[self.scoring], reverse=True)

        for r in rows:
            print(f"\n  {r['model']}:  {self.scoring}={r[self.scoring]:.4f}")
            for k, v in r['params'].items():
                print(f"    {k:30s} = {v}")

        print(f"\n{'=' * 85}")
        return rows

    def plot_all(self):
        """Generate plots for all tuners."""
        for tuner in self.tuners:
            tuner.plot()
        return self

    def save_all(self, dir_path):
        """Save all best params to separate pickles."""
        for tuner in self.tuners:
            safe_name = tuner.name.lower().replace(' ', '_')
            tuner.save(os.path.join(dir_path, f'tuned_{safe_name}.pkl'))
        return self

    def get_best_params(self):
        """Return dict of {model_name: best_params}."""
        return {t.name: t.best_params for t in self.tuners}