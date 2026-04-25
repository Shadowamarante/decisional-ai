# decisional_ai_v0_9_full_v4.py
# ============================================================
# decisional_ai — Decision-first AI
# Single-file distribution (baseline v0.9.0)
#
# v4 patch focus:
# - Make id_column case-insensitive in batch_decide/DecisionLogger.
#   Accepts 'id', 'ID', 'Id', etc.
# - Keep v3 fix: prevent sklearn feature-name mismatch by predicting only on fit-time features.
# - Keep v0.2 capabilities: decision_curve, best_threshold, export artifacts.
# - Keep v0.5 ops: DecisionLogger, decide_routes_from_best, batch_decide.
#
# Dependencies:
#   numpy, pandas, scikit-learn, joblib
# Optional:
#   matplotlib (plots), shap (explain)
# ============================================================

from __future__ import annotations

__version__ = "0.9.0"

__all__ = [
    "__version__",
    # Public API
    "Flow",
    "DecisionConfig",
    "DecisionEngine",
    "DecisionReport",
    "DecisionChoice",
    # v0.2 add-ons
    "decision_curve",
    "best_threshold",
    # Governance
    "AuditEngine",
    "ModelCard",
    "profile_dataset",
    "psi",
    "drift_report",
    "fairness_by_group",
    # Explain
    "permutation_importance",
    "shap_explain_if_available",
    # Calibration
    "calibration_report",
    "plot_calibration",
    # Optimization
    "optimize_triage",
    # Ops
    "DecisionLogger",
    "decide_routes_from_best",
    "batch_decide",
    # Analysis
    "simulate_policies",
    "pareto_frontier",
    "plot_frontier",
]

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import hashlib
import json
import platform

import numpy as np
import pandas as pd

import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# Helpers
# ============================================================


def _resolve_column_case_insensitive(df: pd.DataFrame, name: str | None) -> str | None:
    """Resolve column name in a case-insensitive way (also ignores leading/trailing spaces).

    Examples accepted:
      - name='id' matches columns: 'id', 'ID', ' Id ', 'id '

    If multiple matches exist after normalization, raises ValueError.
    """
    if name is None:
        return None

    # exact match first
    if name in df.columns:
        return name

    key = str(name).strip().lower()
    if not key:
        return None

    matches = [c for c in df.columns if str(c).strip().lower() == key]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"id_column '{name}' is ambiguous after normalization (matches: {matches})")
    return matches[0]


def _select_feature_frame(flow, X: pd.DataFrame, *, id_column: str | None = None, target: str | None = None) -> pd.DataFrame:
    """Return X restricted to the feature columns used at fit time.

    Prevents sklearn feature-name mismatch when operational frames include extra columns
    (e.g., an 'id' column).

    Priority:
      1) flow.feature_names_ (captured at fit time)
      2) drop target and id_column (case-insensitive) if present
    """
    X2 = X
    if target is not None:
        tcol = _resolve_column_case_insensitive(X2, target)
        if tcol is not None:
            X2 = X2.drop(columns=[tcol])

    cols = getattr(flow, "feature_names_", None)
    if cols:
        cols = [c for c in cols if c in X2.columns]
        return X2[cols]

    if id_column is not None:
        icol = _resolve_column_case_insensitive(X2, id_column)
        if icol is not None:
            return X2.drop(columns=[icol])

    return X2


# ============================================================
# 1) Decision configuration
# ============================================================


@dataclass
class DecisionConfig:
    cost_fp: float
    cost_fn: float

    capacity: int | None = None
    cost_review: float = 0.0

    # simple default grid (v0.2 style)
    threshold_grid: list[float] = field(default_factory=lambda: [round(i / 100, 2) for i in range(10, 91, 10)])

    min_precision: float | None = None
    min_recall: float | None = None

    policy: str = "auto"  # auto|cost|capacity|triage|hybrid

    triage_low: float | None = None
    triage_high: float | None = None
    triage_review_capacity: int | None = None

    calibration_method: str | None = None


# ============================================================
# 2) Metrics and cost
# ============================================================


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "positives": int((y_pred == 1).sum()),
        "base_rate": float(np.mean(y_true)),
    }


def passes_constraints(metrics: dict, *, min_precision=None, min_recall=None) -> bool:
    if min_precision is not None and metrics["precision"] < min_precision:
        return False
    if min_recall is not None and metrics["recall"] < min_recall:
        return False
    return True


def cost_from_metrics(metrics: dict, cost_fp: float, cost_fn: float, *, cost_review: float = 0.0, reviewed: int = 0) -> float:
    return float(metrics["fp"] * cost_fp + metrics["fn"] * cost_fn + reviewed * cost_review)


# ============================================================
# 3) Decision report types
# ============================================================


@dataclass
class DecisionChoice:
    strategy: str
    threshold: float | None
    k: int | None
    metrics: dict


@dataclass
class DecisionReport:
    by_threshold: list[dict]
    by_topk: list[dict]
    by_triage: list[dict]
    best: DecisionChoice

    def to_frame(self) -> pd.DataFrame:
        rows = (self.by_threshold or []) + (self.by_topk or []) + (self.by_triage or [])
        return pd.DataFrame(rows) if rows else pd.DataFrame()


# ============================================================
# 4) Decision engine (+ v0.2 methods)
# ============================================================


class DecisionEngine:
    def __init__(self, config: DecisionConfig):
        self.config = config

    def _triage_enabled(self) -> bool:
        c = self.config
        return (
            c.triage_low is not None
            and c.triage_high is not None
            and c.triage_review_capacity is not None
            and 0.0 <= float(c.triage_low) < float(c.triage_high) <= 1.0
        )

    def _effective_policy(self) -> str:
        c = self.config
        if c.policy == "auto":
            if self._triage_enabled():
                return "triage"
            if c.capacity:
                return "hybrid"
            return "cost"
        return c.policy

    # v0.2
    def decision_curve(self, y_true, y_proba) -> pd.DataFrame:
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        p1 = y_proba[:, 1]

        rows = []
        for t in self.config.threshold_grid:
            y_pred = (p1 >= t).astype(int)
            m = compute_metrics(y_true, y_pred)
            cost = cost_from_metrics(m, self.config.cost_fp, self.config.cost_fn)
            rows.append({
                "threshold": float(t),
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "cost": float(cost),
            })
        return pd.DataFrame(rows)

    # v0.2
    def best_threshold(self, y_true, y_proba) -> dict:
        df = self.decision_curve(y_true, y_proba)
        if df.empty:
            return {"threshold": None, "expected_cost": float("inf")}

        if self.config.min_precision is not None:
            df = df[df["precision"] >= float(self.config.min_precision)]
        if self.config.min_recall is not None:
            df = df[df["recall"] >= float(self.config.min_recall)]

        if df.empty:
            return {"threshold": None, "expected_cost": float("inf")}

        row = df.loc[df["cost"].idxmin()]
        return {"threshold": float(row["threshold"]), "expected_cost": float(row["cost"])}

    def evaluate(self, y_true, y_proba) -> DecisionReport:
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        assert set(np.unique(y_true)).issubset({0, 1}), "y must be binary"
        assert y_proba.ndim == 2 and y_proba.shape[1] == 2, "predict_proba must have shape (n,2)"

        p1 = y_proba[:, 1]
        by_threshold = self._eval_thresholds(y_true, p1)
        by_topk = self._eval_topk(y_true, p1) if self.config.capacity else []
        by_triage = self._eval_triage(y_true, p1) if self._triage_enabled() else []
        best = self._choose_best(by_threshold, by_topk, by_triage)
        return DecisionReport(by_threshold=by_threshold, by_topk=by_topk, by_triage=by_triage, best=best)

    def _eval_thresholds(self, y_true, p1) -> list[dict]:
        rows = []
        for t in self.config.threshold_grid:
            y_pred = (p1 >= t).astype(int)
            m = compute_metrics(y_true, y_pred)
            cost = cost_from_metrics(m, self.config.cost_fp, self.config.cost_fn)
            rows.append({
                "strategy": "threshold",
                "threshold": float(t),
                "k": None,
                "reviewed": 0,
                **m,
                "cost": float(cost),
                "constraints_ok": passes_constraints(m, min_precision=self.config.min_precision, min_recall=self.config.min_recall),
            })
        return rows

    def _eval_topk(self, y_true, p1) -> list[dict]:
        k = int(self.config.capacity)
        if k <= 0:
            return []
        idx = np.argsort(-p1)
        y_pred = np.zeros_like(y_true)
        y_pred[idx[:k]] = 1
        thr_equiv = float(p1[idx[k - 1]]) if k - 1 < len(idx) else 0.0
        m = compute_metrics(y_true, y_pred)
        cost = cost_from_metrics(m, self.config.cost_fp, self.config.cost_fn)
        return [{
            "strategy": "topk",
            "threshold": thr_equiv,
            "k": k,
            "reviewed": 0,
            **m,
            "cost": float(cost),
            "constraints_ok": passes_constraints(m, min_precision=self.config.min_precision, min_recall=self.config.min_recall),
        }]

    def _eval_triage(self, y_true, p1) -> list[dict]:
        c = self.config
        low = float(c.triage_low)
        high = float(c.triage_high)
        cap = int(c.triage_review_capacity)

        auto_pos = p1 >= high
        auto_neg = p1 <= low
        gray = ~(auto_pos | auto_neg)

        gray_idx = np.where(gray)[0]
        reviewed = min(cap, len(gray_idx))

        y_pred = np.zeros_like(y_true)
        y_pred[auto_pos] = 1

        if reviewed > 0:
            order = np.argsort(-p1[gray_idx])
            chosen = gray_idx[order[:reviewed]]
            y_pred[chosen] = 1

        m = compute_metrics(y_true, y_pred)
        cost = cost_from_metrics(m, c.cost_fp, c.cost_fn, cost_review=float(c.cost_review), reviewed=int(reviewed))

        return [{
            "strategy": "triage",
            "threshold": None,
            "k": None,
            "triage_low": low,
            "triage_high": high,
            "review_capacity": cap,
            "reviewed": int(reviewed),
            "auto_positive": int(auto_pos.sum()),
            "auto_negative": int(auto_neg.sum()),
            "gray_zone": int(gray.sum()),
            **m,
            "cost": float(cost),
            "constraints_ok": passes_constraints(m, min_precision=c.min_precision, min_recall=c.min_recall),
        }]

    def _choose_best(self, by_threshold: list[dict], by_topk: list[dict], by_triage: list[dict]) -> DecisionChoice:
        policy = self._effective_policy()
        candidates: list[dict] = []

        if policy in {"cost", "hybrid"}:
            candidates += [r for r in by_threshold if r["constraints_ok"]]
        if policy in {"capacity", "hybrid"} and self.config.capacity:
            candidates += [r for r in by_topk if r["constraints_ok"]]
        if policy in {"triage", "hybrid"} and self._triage_enabled():
            candidates += [r for r in by_triage if r["constraints_ok"]]

        if not candidates:
            candidates = by_threshold + by_topk + by_triage

        best_row = min(candidates, key=lambda r: r["cost"])
        return DecisionChoice(
            strategy=best_row["strategy"],
            threshold=best_row.get("threshold"),
            k=best_row.get("k"),
            metrics={k: best_row[k] for k in best_row.keys() if k != "constraints_ok"},
        )


# Functional aliases (v0.2)

def decision_curve(engine: DecisionEngine, y_true, y_proba) -> pd.DataFrame:
    return engine.decision_curve(y_true, y_proba)


def best_threshold(engine: DecisionEngine, y_true, y_proba) -> dict:
    return engine.best_threshold(y_true, y_proba)


# ============================================================
# 5) Model adapters
# ============================================================


class SklearnAdapter:
    def __init__(self, name: str, **kwargs):
        self.name = name
        if name == "logistic":
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(max_iter=500, **kwargs)
        elif name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(**kwargs)
        else:
            raise ValueError("Unsupported sklearn model")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self):
        return self.model.get_params(deep=True) if hasattr(self.model, "get_params") else {}


def resolve_model(model):
    if isinstance(model, str):
        if model in {"logistic", "random_forest"}:
            return SklearnAdapter(model)
        raise ValueError(f"Model '{model}' not supported")
    return model


# ============================================================
# 6) Governance
# ============================================================


def profile_dataset(X: pd.DataFrame) -> dict:
    prof = {
        "rows": int(len(X)),
        "cols": int(X.shape[1]),
        "schema": {c: str(X[c].dtype) for c in X.columns},
        "missing_rate": {c: float(X[c].isna().mean()) for c in X.columns},
    }
    return prof


class AuditEngine:
    def __init__(self, *, tags: dict | None = None):
        self.metadata: dict[str, Any] = {}
        self.tags = tags or {}

    def register_training(self, X, y, model, *, decision_config: dict | None = None, version: str = "N/A"):
        self.metadata.setdefault("tags", {}).update(self.tags)
        self.metadata["version"] = version
        self.metadata["trained_at_utc"] = datetime.utcnow().isoformat()
        self.metadata["environment"] = {"python": platform.python_version(), "platform": platform.platform()}
        self.metadata["decision_config"] = decision_config or {}
        self.metadata["training"] = {
            "rows": int(len(X)),
            "features": list(getattr(X, "columns", [])),
            "data_hash": self._hash_frame(X),
            "target_hash": self._hash_series(y),
            "dataset_profile": profile_dataset(X),
        }
        self.metadata["model"] = {"repr": repr(model), "class": model.__class__.__name__}

    def register_evaluation(self, X, y, y_proba, decision_report: DecisionReport):
        self.metadata.setdefault("evaluations", [])
        self.metadata["evaluations"].append({
            "evaluated_at_utc": datetime.utcnow().isoformat(),
            "rows": int(len(X)),
            "data_hash": self._hash_frame(X),
            "best": asdict(decision_report.best),
        })

    def generate_report(self, audience: str = "tecnico") -> dict:
        return {"audience": audience, "metadata": self.metadata}

    def full_audit(self) -> str:
        return json.dumps(self.metadata, indent=2, ensure_ascii=False)

    # v0.2 export
    def export(self, path: str | Path, *, decision_curve_df: pd.DataFrame | None = None) -> dict:
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)

        if decision_curve_df is not None:
            (folder / "decision_curve.json").write_text(decision_curve_df.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")

        audit_txt = self.full_audit()
        (folder / "audit.json").write_text(audit_txt, encoding="utf-8")
        (folder / "metadata.json").write_text(audit_txt, encoding="utf-8")
        return {"status": "exported", "path": str(folder)}

    def register_model(self) -> dict:
        return {"status": "registered", "metadata": self.metadata}

    def save(self, folder: str | Path, model) -> dict:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, folder / "model.joblib")
        (folder / "audit.json").write_text(self.full_audit(), encoding="utf-8")
        return {"status": "saved", "path": str(folder)}

    def load_model(self, folder: str | Path):
        folder = Path(folder)
        model = joblib.load(folder / "model.joblib")
        audit_path = folder / "audit.json"
        if audit_path.exists():
            self.metadata = json.loads(audit_path.read_text(encoding="utf-8"))
        return model

    def _hash_frame(self, X):
        try:
            csv = X.to_csv(index=False)
        except Exception:
            csv = str(X)
        return hashlib.sha256(csv.encode()).hexdigest()

    def _hash_series(self, y):
        try:
            txt = y.to_csv(index=False) if hasattr(y, "to_csv") else ",".join(map(str, list(y)))
        except Exception:
            txt = str(y)
        return hashlib.sha256(txt.encode()).hexdigest()


class ModelCard:
    def to_markdown(self, report: dict) -> str:
        meta = report.get("metadata", {})
        lines = [
            "# Model Card — decisional_ai",
            f"- Versao: {meta.get('version', 'N/A')}",
            f"- Treinado em (UTC): {meta.get('trained_at_utc', 'N/A')}",
            f"- Tags: {meta.get('tags', {})}",
        ]
        return "\n".join(lines)


# ============================================================
# 7) Drift (PSI)
# ============================================================


def psi(expected: pd.Series, actual: pd.Series, *, bins: int = 10) -> float:
    expected = expected.dropna()
    actual = actual.dropna()
    if expected.empty or actual.empty:
        return 0.0

    if pd.api.types.is_numeric_dtype(expected):
        q = np.linspace(0, 1, bins + 1)
        cuts = np.unique(np.quantile(expected, q))
        if len(cuts) <= 2:
            return 0.0
        e = pd.cut(expected, cuts, include_lowest=True).value_counts(normalize=True)
        a = pd.cut(actual, cuts, include_lowest=True).value_counts(normalize=True)
        a = a.reindex(e.index).fillna(0)
    else:
        e = expected.astype("object").value_counts(normalize=True)
        a = actual.astype("object").value_counts(normalize=True)
        idx = e.index.union(a.index)
        e = e.reindex(idx).fillna(0)
        a = a.reindex(idx).fillna(0)

    eps = 1e-6
    e = np.clip(e.values, eps, 1)
    a = np.clip(a.values, eps, 1)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(X_ref: pd.DataFrame, X_new: pd.DataFrame, *, bins: int = 10) -> dict:
    cols = [c for c in X_ref.columns if c in X_new.columns]
    out = {c: psi(X_ref[c], X_new[c], bins=bins) for c in cols}
    vals = list(out.values())
    return {"psi": out, "summary": {"mean": float(np.mean(vals)) if vals else 0.0, "max": float(np.max(vals)) if vals else 0.0}}


# ============================================================
# 8) Fairness
# ============================================================


def fairness_by_group(y_true, y_pred, group) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)

    out = {}
    for g in np.unique(group):
        mask = group == g
        out[str(g)] = compute_metrics(y_true[mask], y_pred[mask])

    def gap(metric):
        vals = [v.get(metric, 0.0) for v in out.values()]
        return float(max(vals) - min(vals)) if vals else 0.0

    return {"by_group": out, "gaps": {"precision_gap": gap("precision"), "recall_gap": gap("recall"), "f1_gap": gap("f1")}}


# ============================================================
# 9) Explain
# ============================================================


def permutation_importance(model, X, y, *, scorer: Callable | None = None, n_repeats: int = 5, random_state: int = 42) -> dict:
    rng = np.random.default_rng(random_state)
    proba = model.predict_proba(X)

    if scorer is None:
        baseline = float(np.mean((proba[:, 1] >= 0.5).astype(int) == np.asarray(y)))

        def scorer_fn(yt, yp):
            return float(np.mean((yp[:, 1] >= 0.5).astype(int) == yt))
    else:
        baseline = float(scorer(np.asarray(y), proba))
        scorer_fn = scorer

    cols = list(getattr(X, "columns", range(X.shape[1])))
    importances = {}

    for c in cols:
        scores = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            if hasattr(Xp, "iloc"):
                Xp[c] = Xp[c].iloc[perm].values
            else:
                Xp[:, c] = Xp[perm, c]
            scores.append(scorer_fn(np.asarray(y), model.predict_proba(Xp)))
        importances[str(c)] = float(baseline - float(np.mean(scores)))

    importances = dict(sorted(importances.items(), key=lambda kv: -kv[1]))
    return {"baseline": baseline, "importances": importances}


def shap_explain_if_available(model, X, *, max_samples: int = 200) -> dict:
    try:
        import shap  # type: ignore
        Xs = X.sample(min(max_samples, len(X)), random_state=42) if hasattr(X, "sample") else X
        explainer = shap.Explainer(getattr(model, "model", model), Xs)
        values = explainer(Xs)
        return {"available": True, "type": str(type(explainer)), "values": values}
    except Exception:
        return {"available": False}


# ============================================================
# 10) Calibration
# ============================================================


def calibration_report(y_true, y_proba, *, n_bins: int = 10) -> dict:
    from sklearn.metrics import brier_score_loss
    from sklearn.calibration import calibration_curve

    y_true = np.asarray(y_true)
    p1 = np.asarray(y_proba)[:, 1]
    brier = float(brier_score_loss(y_true, p1))
    frac_pos, mean_pred = calibration_curve(y_true, p1, n_bins=n_bins, strategy="uniform")
    return {"brier": brier, "n_bins": int(n_bins), "curve": {"mean_pred": [float(x) for x in mean_pred], "frac_pos": [float(x) for x in frac_pos]}}


def plot_calibration(y_true, y_proba, *, n_bins: int = 10, title: str = "Curva de Calibracao"):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError("matplotlib is required for plot_calibration") from e

    rep = calibration_report(y_true, y_proba, n_bins=n_bins)
    mp = rep["curve"]["mean_pred"]
    fp = rep["curve"]["frac_pos"]

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(mp, fp, marker="o")
    ax.set_xlabel("Probabilidade prevista (media por bin)")
    ax.set_ylabel("Fracao de positivos")
    ax.set_title(title)
    return fig


# ============================================================
# 11) Optimize triage
# ============================================================


def optimize_triage(
    y_true,
    y_proba,
    *,
    grid: list[float] | None = None,
    review_capacity: int = 100,
    cost_fp: float = 1.0,
    cost_fn: float = 1.0,
    cost_review: float = 0.0,
    min_precision: float | None = None,
    min_recall: float | None = None,
) -> dict:
    y_true = np.asarray(y_true)
    p1 = np.asarray(y_proba)[:, 1]

    if grid is None:
        grid = [round(i / 100, 2) for i in range(5, 96, 5)]

    best = None
    for low in grid:
        for high in grid:
            if not (0 <= low < high <= 1):
                continue
            auto_pos = p1 >= high
            auto_neg = p1 <= low
            gray = ~(auto_pos | auto_neg)

            gray_idx = np.where(gray)[0]
            reviewed = min(int(review_capacity), len(gray_idx))

            y_pred = np.zeros_like(y_true)
            y_pred[auto_pos] = 1

            if reviewed > 0:
                order = np.argsort(-p1[gray_idx])
                chosen = gray_idx[order[:reviewed]]
                y_pred[chosen] = 1

            m = compute_metrics(y_true, y_pred)
            if not passes_constraints(m, min_precision=min_precision, min_recall=min_recall):
                continue

            cost = cost_from_metrics(m, cost_fp, cost_fn, cost_review=cost_review, reviewed=reviewed)
            row = {"triage_low": float(low), "triage_high": float(high), "review_capacity": int(review_capacity), "reviewed": int(reviewed), **m, "cost": float(cost)}
            if best is None or row["cost"] < best["cost"]:
                best = row

    return best or {"triage_low": 0.1, "triage_high": 0.8, "review_capacity": int(review_capacity), "reviewed": 0, "cost": float("inf")}


# ============================================================
# 12) Analysis
# ============================================================


def simulate_policies(y_true, y_proba, *, cfg: DecisionConfig) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    p1 = np.asarray(y_proba)[:, 1]
    engine = DecisionEngine(cfg)
    rep = engine.evaluate(y_true, np.column_stack([1 - p1, p1]))
    return rep.to_frame()


def pareto_frontier(df: pd.DataFrame, *, x: str = "cost", y: str = "recall") -> pd.DataFrame:
    if df.empty:
        return df
    d = df.dropna(subset=[x, y]).sort_values(by=[x, y], ascending=[True, False])
    best_y = -np.inf
    keep = []
    for _, r in d.iterrows():
        if r[y] > best_y:
            keep.append(True)
            best_y = r[y]
        else:
            keep.append(False)
    return d.loc[keep]


def plot_frontier(df: pd.DataFrame, *, x: str = "cost", y: str = "recall", title: str = "Fronteira de Pareto"):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError("matplotlib is required for plot_frontier") from e

    d = df.dropna(subset=[x, y])
    front = pareto_frontier(d, x=x, y=y)

    fig, ax = plt.subplots()
    ax.scatter(d[x], d[y], alpha=0.3, label="candidatos")
    if not front.empty:
        ax.plot(front[x], front[y], marker="o", label="pareto")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend()
    return fig


# ============================================================
# 13) Flow
# ============================================================


class _CalibratedWrapper:
    def __init__(self, calibrated_model):
        self.model = calibrated_model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self):
        return self.model.get_params(deep=True) if hasattr(self.model, "get_params") else {}


class Flow:
    def __init__(self, model, decision: DecisionConfig | None = None, *, tags: dict | None = None):
        if decision is None:
            raise ValueError("DecisionConfig é obrigatório")

        self.model = resolve_model(model)
        self.decision = decision
        self.tags = tags or {}

        self.audit_engine = AuditEngine(tags=self.tags)
        self.decision_engine = DecisionEngine(decision)
        self.model_card = ModelCard()

        self.fitted = False
        self.last_report: DecisionReport | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y=None, target: str | None = None):
        if target:
            tcol = _resolve_column_case_insensitive(X, target)
            y = X[tcol]
            X = X.drop(columns=[tcol])

        # capture feature names used at fit time
        self.feature_names_ = list(getattr(X, "columns", []))

        self.model.fit(X, y)
        self.audit_engine.register_training(X, y, self.model, decision_config=asdict(self.decision), version=__version__)
        self.fitted = True
        return self

    def calibrate(self, X: pd.DataFrame, y=None, *, target: str | None = None, method: str = "sigmoid", cv: int | str = 5):
        if target:
            tcol = _resolve_column_case_insensitive(X, target)
            y = X[tcol]
            X = X.drop(columns=[tcol])

        from sklearn.calibration import CalibratedClassifierCV

        base = getattr(self.model, "model", self.model)
        if self.fitted and cv != "prefit":
            cv = "prefit"

        cal = CalibratedClassifierCV(base, method=method, cv=cv)
        cal.fit(X, y)

        self.model = _CalibratedWrapper(cal)
        self.decision.calibration_method = method
        self.fitted = True
        return self

    def evaluate(self, X: pd.DataFrame, y=None, target: str | None = None) -> DecisionReport:
        if not self.fitted:
            raise RuntimeError("Modelo ainda não treinado")

        if target:
            tcol = _resolve_column_case_insensitive(X, target)
            y = X[tcol]
            X = X.drop(columns=[tcol])

        X_pred = _select_feature_frame(self, X)
        y_proba = self.model.predict_proba(X_pred)

        report = self.decision_engine.evaluate(y, y_proba)
        self.last_report = report
        self.audit_engine.register_evaluation(X_pred, y, y_proba, report)
        return report

    def impact_report(self, decision_report: DecisionReport) -> dict:
        best = decision_report.best
        m = best.metrics
        return {
            "version": __version__,
            "best_strategy": best.strategy,
            "cost_total": float(m.get("cost", 0.0)),
            "reviewed": int(m.get("reviewed", 0)),
            "capacity": self.decision.capacity,
            "triage_review_capacity": self.decision.triage_review_capacity,
            "auto_positive": int(m.get("auto_positive", 0)),
            "auto_negative": int(m.get("auto_negative", 0)),
            "gray_zone": int(m.get("gray_zone", 0)),
            "metrics": {
                "accuracy": float(m.get("accuracy", 0.0)),
                "precision": float(m.get("precision", 0.0)),
                "recall": float(m.get("recall", 0.0)),
                "f1": float(m.get("f1", 0.0)),
            },
        }

    def export(self, path: str | Path = "artifacts") -> dict:
        curve_df = None
        if self.last_report is not None:
            curve_df = pd.DataFrame(self.last_report.by_threshold)
        return self.audit_engine.export(path, decision_curve_df=curve_df)


# ============================================================
# 14) Ops: routing + logging + batch
# ============================================================


def _safe_hash(val: Any) -> str:
    return hashlib.sha256(str(val).encode("utf-8")).hexdigest()


def decide_routes_from_best(p1: np.ndarray, best: DecisionChoice, cfg: DecisionConfig) -> dict:
    p1 = np.asarray(p1)
    n = len(p1)

    if best.strategy == "triage":
        low = float(best.metrics.get("triage_low", cfg.triage_low))
        high = float(best.metrics.get("triage_high", cfg.triage_high))
        cap = int(best.metrics.get("review_capacity", cfg.triage_review_capacity))

        auto_pos = p1 >= high
        auto_neg = p1 <= low
        gray = ~(auto_pos | auto_neg)

        gray_idx = np.where(gray)[0]
        reviewed = min(cap, len(gray_idx))

        y_pred = np.zeros(n, dtype=int)
        y_pred[auto_pos] = 1

        reviewed_mask = np.zeros(n, dtype=bool)
        if reviewed > 0:
            order = np.argsort(-p1[gray_idx])
            chosen = gray_idx[order[:reviewed]]
            y_pred[chosen] = 1
            reviewed_mask[chosen] = True

        route = np.full(n, "auto_neg", dtype=object)
        route[auto_pos] = "auto_pos"
        route[reviewed_mask] = "review"
        route[gray & ~reviewed_mask] = "defer"
        return {"y_pred": y_pred, "route": route, "reviewed_mask": reviewed_mask}

    if best.strategy == "topk":
        k = int(best.metrics.get("k", cfg.capacity or 0))
        idx = np.argsort(-p1)
        y_pred = np.zeros(n, dtype=int)
        y_pred[idx[:k]] = 1
        route = np.where(y_pred == 1, "topk", "below_topk")
        return {"y_pred": y_pred, "route": route, "reviewed_mask": np.zeros(n, dtype=bool)}

    thr = float(best.metrics.get("threshold", best.threshold or 0.5))
    y_pred = (p1 >= thr).astype(int)
    route = np.where(y_pred == 1, "threshold_pos", "threshold_neg")
    return {"y_pred": y_pred, "route": route, "reviewed_mask": np.zeros(n, dtype=bool)}


class DecisionLogger:
    def __init__(self, *, id_column: str | None = None):
        self.id_column = id_column

    def build_log(self, X: pd.DataFrame, p1: np.ndarray, applied: dict, *, best: DecisionChoice) -> pd.DataFrame:
        df = pd.DataFrame({
            "proba_1": p1.astype(float),
            "y_pred": applied["y_pred"].astype(int),
            "route": applied["route"],
        }, index=X.index)

        icol = _resolve_column_case_insensitive(X, self.id_column)
        if icol is not None:
            df["id_hash"] = X[icol].apply(_safe_hash)
        else:
            df["id_hash"] = [None] * len(df)

        df["strategy"] = best.strategy
        df["threshold"] = best.threshold
        return df


def batch_decide(flow: Flow, X: pd.DataFrame, *, id_column: str | None = None, best_override: DecisionChoice | None = None) -> pd.DataFrame:
    if best_override is None:
        if flow.last_report is None:
            raise RuntimeError("batch_decide requires flow.evaluate() first (or provide best_override)")
        best = flow.last_report.best
    else:
        best = best_override

    # resolve id column in a case-insensitive way
    id_col_resolved = _resolve_column_case_insensitive(X, id_column)

    # predict only on fit-time feature set
    X_pred = _select_feature_frame(flow, X, id_column=id_col_resolved)
    proba = flow.model.predict_proba(X_pred)
    p1 = proba[:, 1]

    applied = decide_routes_from_best(p1, best, flow.decision)
    logger = DecisionLogger(id_column=id_col_resolved)
    log_df = logger.build_log(X, p1, applied, best=best)

    # ✅ FIX DEFINITIVO: se id_col_resolved existe, força id_hash a partir da própria coluna
    if id_col_resolved is not None and id_col_resolved in X.columns:
        log_df["id_hash"] = X[id_col_resolved].astype("object").apply(_safe_hash).astype("object")

    return log_df


# ============================================================
# Examples
# ============================================================


