"""
Model loader — handles backward compatibility for models trained with:
- sklearn 0.24.2  (current: 1.x)
- statsmodels     (not installed; mocked for OLS results)
- pandas < 2.0    (Int64Index removed in 2.0)
"""
import sys, types, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 1. Patch pandas backward compat ──────────────────────────────────────────
if not hasattr(pd.core.indexes, "numeric"):
    _num = types.ModuleType("pandas.core.indexes.numeric")
    _num.Int64Index   = pd.Index
    _num.Float64Index = pd.Index
    _num.UInt64Index  = pd.Index
    sys.modules["pandas.core.indexes.numeric"] = _num

# ── 2. Minimal statsmodels OLS stub ─────────────────────────────────────────
class _OLSResults:
    """Minimal stub for statsmodels OLSResults — only .predict() needed."""
    def __init__(self):
        self.params = None
        self.pvalues = None
        self.rsquared = None
        self.coef_ = None
        self.intercept_ = None

    def predict(self, X):
        return np.asarray(X) @ np.asarray(self.coef_) + self.intercept_


class _DummySM:
    """Catch-all stub for any statsmodels class we don't need."""
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return self
    def __getattr__(self, k):
        return _DummySM() if k not in ("coef_", "intercept_", "results") else None
    def __setattr__(self, k, v): object.__setattr__(self, k, v)


# ── 3. Custom unpickler ───────────────────────────────────────────────────────
from sklearn import linear_model
from sklearn.base import BaseEstimator, RegressorMixin


class LogisticRegression_with_p_values:
    """Matches the class used when saving pd_model.sav and lgd_model_stage_1.sav."""
    def __init__(self, *args, **kwargs):
        self.model = linear_model.LogisticRegression(*args, **kwargs)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LinearRegressionWithPValues(BaseEstimator, RegressorMixin):
    """Matches the class used when saving lgd_model_stage_2.sav and ead.sav."""
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.results = None

    def predict(self, X):
        coef = np.asarray(self.coef_)
        return np.asarray(X) @ coef + self.intercept_


class _CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "LogisticRegression_with_p_values":
            return LogisticRegression_with_p_values
        if name == "LinearRegressionWithPValues":
            return LinearRegressionWithPValues
        if module.startswith("statsmodels"):
            # Return a dummy for anything statsmodels — we only need coef_ / intercept_
            return _DummySM
        return super().find_class(module, name)


def load_model(path: str):
    with open(path, "rb") as f:
        return _CustomUnpickler(f).load()
