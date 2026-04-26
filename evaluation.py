from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))


def precision_at_recall_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    recall_target: float = 0.80,
) -> float:
    """Precision at the highest threshold that achieves recall >= recall_target.

    Returns 0.0 if the class has no positives or recall_target is unreachable.
    """
    if y_true.sum() == 0:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = recall >= recall_target
    if not mask.any():
        return 0.0
    return float(precision[mask].max())


def tune_per_clause_thresholds(
    logits: np.ndarray,
    labels: np.ndarray,
    id_to_clause: dict[int, str],
    thresholds: np.ndarray | None = None,
    min_positives_for_full_sweep: int = 5,
    rare_class_default: float = 0.5,
) -> dict[str, float]:
    """Find the threshold per clause that maximises per-clause F1 on the given logits.

    For clauses with fewer than `min_positives_for_full_sweep` positives in `labels`,
    the per-clause F1 sweep on val is dominated by single-example noise (1-3 positives
    can yield F1=1.0 at an arbitrarily low threshold that does not generalise). For
    these rare clauses the function returns `rare_class_default` directly without
    sweeping. The default 0.5 is the Bayes-optimal sigmoid decision boundary under an
    uninformative prior and the value used by the CUAD baselines (Hendrycks et al.,
    2021) and Legal-BERT (Chalkidis et al., 2020); the fallback-for-low-support-classes
    approach follows Yang (1999) "A study of thresholding strategies for text
    categorization" and Lewis (1995, SIGIR), which both recommend a global default
    in place of per-class tuning when class support is too small for stable estimation.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    probs = _sigmoid(logits)
    best: dict[str, float] = {}

    for clause_id, clause_name in id_to_clause.items():
        y_true = labels[:, clause_id].astype(int)
        n_pos = int(y_true.sum())

        if n_pos < min_positives_for_full_sweep:
            best[clause_name] = float(rare_class_default)
            continue

        y_score = probs[:, clause_id]
        best_f1 = -1.0
        best_t = 0.5
        for t in thresholds:
            preds = (y_score >= t).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        best[clause_name] = best_t
    return best


def compute_aggregate_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: dict[int | str, float],
    id_to_clause: dict[int, str] | None = None,
) -> dict[str, float]:
    """Macro/micro F1, precision, recall using per-clause thresholds.

    thresholds keys may be clause_name (str) or clause_id (int).
    """
    probs = _sigmoid(logits)
    num_labels = logits.shape[1]
    preds = np.zeros_like(probs, dtype=int)

    for i in range(num_labels):
        key = id_to_clause[i] if (id_to_clause and i in id_to_clause) else i
        t = thresholds.get(key, 0.5)
        preds[:, i] = (probs[:, i] >= t).astype(int)

    int_labels = labels.astype(int)
    return {
        "macro_f1":        float(f1_score(int_labels, preds, average="macro",  zero_division=0)),
        "micro_f1":        float(f1_score(int_labels, preds, average="micro",  zero_division=0)),
        "micro_precision": float(precision_score(int_labels, preds, average="micro", zero_division=0)),
        "micro_recall":    float(recall_score(int_labels, preds, average="micro",    zero_division=0)),
    }


def compute_per_clause_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: dict[str, float],
    id_to_clause: dict[int, str],
    n_positive_train: dict[int, int],
) -> pd.DataFrame:
    """Per-clause breakdown: precision, recall, F1, P@80R, AUPR, threshold, n_positive_train.

    Returns a DataFrame sorted by F1 descending with one row per clause type.
    """
    probs = _sigmoid(logits)
    rows = []
    for clause_id, clause_name in id_to_clause.items():
        y_true = labels[:, clause_id].astype(int)
        y_score = probs[:, clause_id]
        t = thresholds.get(clause_name, 0.5)
        preds = (y_score >= t).astype(int)

        rows.append({
            "clause_type":            clause_name,
            "threshold":              round(t, 2),
            "precision":              float(precision_score(y_true, preds, zero_division=0)),
            "recall":                 float(recall_score(y_true, preds, zero_division=0)),
            "f1":                     float(f1_score(y_true, preds, zero_division=0)),
            "precision_at_80_recall": precision_at_recall_threshold(y_true, y_score, 0.80),
            "aupr":                   float(average_precision_score(y_true, y_score)) if y_true.sum() > 0 else 0.0,
            "support_test":           int(y_true.sum()),
            "n_positive_train":       n_positive_train.get(clause_id, 0),
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def plot_confusion_matrix(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: dict[str, float],
    id_to_clause: dict[int, str],
    title: str = "Predicted vs True Clause Co-occurrence",
    save_path: str | None = None,
) -> None:
    """Heatmap: for each true-positive clause (row), what rate is each clause predicted (col)?

    Rows are indexed by true positive samples for each clause. Values show prediction rates.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    probs = _sigmoid(logits)
    num_labels = logits.shape[1]
    clause_names = [id_to_clause[i] for i in range(num_labels)]
    preds = np.zeros_like(probs, dtype=int)
    for i in range(num_labels):
        t = thresholds.get(clause_names[i], 0.5)
        preds[:, i] = (probs[:, i] >= t).astype(int)

    int_labels = labels.astype(int)
    matrix = np.zeros((num_labels, num_labels), dtype=float)
    for true_idx in range(num_labels):
        true_positive_mask = int_labels[:, true_idx] == 1
        if true_positive_mask.sum() == 0:
            continue
        for pred_idx in range(num_labels):
            matrix[true_idx, pred_idx] = preds[true_positive_mask, pred_idx].mean()

    # Abbreviate long names: keep first word + first letter of each subsequent word
    def _abbrev(name: str, max_len: int = 22) -> str:
        if len(name) <= max_len:
            return name
        words = name.split()
        return words[0] + " " + "".join(w[0] + "." for w in words[1:])

    short_names = [_abbrev(n) for n in clause_names]
    side = max(18, num_labels * 0.62)
    fig, ax = plt.subplots(figsize=(side, side * 0.88))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        matrix,
        xticklabels=short_names,
        yticklabels=short_names,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax,
        annot_kws={"size": 6.5},
        linewidths=0.25,
        linecolor="#F0F0F0",
        cbar_kws={"label": "Prediction rate", "shrink": 0.55, "pad": 0.02},
    )
    ax.set_xlabel("Predicted clause", labelpad=10, fontsize=10)
    ax.set_ylabel("True clause (positive samples only)", labelpad=10, fontsize=10)
    ax.set_title(title, pad=14, fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7.5)
    plt.yticks(rotation=0, fontsize=7.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_precision_recall_curves(
    logits_dict: dict[str, np.ndarray],
    labels_dict: dict[str, np.ndarray],
    save_path: str | None = None,
) -> None:
    """Macro-averaged precision-recall curve for each model.

    For each model, per-label PR curves are interpolated onto a shared recall
    axis and averaged, giving a threshold-independent view of ranking quality.
    """
    import matplotlib.pyplot as plt

    recall_base = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    colors = plt.cm.tab10.colors

    for idx, (model_name, logits) in enumerate(logits_dict.items()):
        labels = labels_dict[model_name]
        probs = _sigmoid(logits)
        n_labels = logits.shape[1]
        interp_precisions = []

        for j in range(n_labels):
            y_true = labels[:, j].astype(int)
            if y_true.sum() == 0:
                continue
            prec, rec, _ = precision_recall_curve(y_true, probs[:, j])
            # Interpolate onto common recall grid (curve sorted ascending)
            interp_precisions.append(np.interp(recall_base, rec[::-1], prec[::-1]))

        if not interp_precisions:
            continue
        mean_prec = np.mean(interp_precisions, axis=0)
        aupr = float(average_precision_score(labels.astype(int), probs, average="macro"))
        ax.plot(recall_base, mean_prec,
                label=f"{model_name} (AUPR={aupr:.3f})",
                color=colors[idx % len(colors)], linewidth=1.8)

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Macro-Averaged Precision–Recall Curves — All Models",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_model_comparison(
    results: dict[str, pd.DataFrame],
    metric: str = "f1",
    save_path: str | None = None,
) -> None:
    """Heatmap: models (rows) × clause types (columns) for a given metric.

    A heatmap is far more readable than a grouped bar chart when there are
    many clause types: colour immediately shows which model wins on which clause,
    and the layout never overlaps.
    """
    if not results:
        raise ValueError("results dict is empty — nothing to plot")

    import matplotlib.pyplot as plt
    import seaborn as sns

    model_names = list(results.keys())
    combined = None
    for model_name, df in results.items():
        sub = df[["clause_type", metric]].rename(columns={metric: model_name})
        combined = sub if combined is None else combined.merge(sub, on="clause_type", how="outer")

    # Rows = models, columns = clause types (transposed from the merged frame)
    combined = combined.set_index("clause_type").fillna(0)
    heat = combined.T   # shape: (n_models, n_clauses)

    # Sort clause columns by mean F1 descending so the easiest clauses are on the left
    heat = heat[heat.mean(axis=0).sort_values(ascending=False).index]

    n_models  = len(model_names)
    n_clauses = heat.shape[1]
    fig_w = max(16, n_clauses * 0.52)
    fig_h = max(4,  n_models  * 0.9 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap="YlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        annot_kws={"size": 7.5},
        linewidths=0.3,
        linecolor="#EEEEEE",
        cbar_kws={"label": metric.upper(), "shrink": 0.6, "pad": 0.01},
    )
    ax.set_xlabel("Clause Type", labelpad=10, fontsize=10)
    ax.set_ylabel("Model", labelpad=8, fontsize=10)
    ax.set_title(
        f"Per-Clause {metric.upper()} — All Models",
        pad=14, fontsize=12, fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right", fontsize=7.5)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
