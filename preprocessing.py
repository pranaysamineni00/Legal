from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# ── Clause filtering ──────────────────────────────────────────────────────────

def filter_clauses(
    cuad_df: pd.DataFrame,
    min_positives: int = 20,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop clause types with fewer than min_positives positive examples.

    Returns (filtered_df, excluded_dict) where excluded_dict maps
    clause_type → positive_count for every excluded type.
    """
    positive_counts = (
        cuad_df[cuad_df["has_answer"]]
        .groupby("clause_type")
        .size()
    )
    excluded = {
        clause: int(count)
        for clause, count in positive_counts.items()
        if count < min_positives
    }
    # Also exclude clause types with zero positives (not in positive_counts at all)
    all_types = set(cuad_df["clause_type"].unique())
    for clause in all_types:
        if clause not in positive_counts.index:
            excluded[clause] = 0

    keep = all_types - set(excluded.keys())
    filtered = cuad_df[cuad_df["clause_type"].isin(keep)].copy()

    if excluded:
        print(f"Excluded {len(excluded)} clause types (below min_positives={min_positives}):")
        for c, n in sorted(excluded.items(), key=lambda x: x[1]):
            print(f"  {c}: {n} positives")

    return filtered, excluded


def plot_clause_frequency(cuad_df: pd.DataFrame, save_path: str | None = None) -> pd.DataFrame:
    """Horizontal bar chart: positive rate per clause type, colour-coded by rate quartile."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    summary = (
        cuad_df.groupby("clause_type")["has_answer"]
        .agg(positive_rate="mean", positive_count="sum", total="count")
        .sort_values("positive_rate", ascending=True)   # ascending so top is at chart top
        .reset_index()
    )

    # Colour bars by quartile: dark blue (high) → light blue (low)
    cmap = mpl.colormaps.get_cmap("Blues")
    norm = mpl.colors.Normalize(vmin=summary["positive_rate"].min(),
                                vmax=summary["positive_rate"].max())
    bar_colors = [cmap(norm(v) * 0.75 + 0.2) for v in summary["positive_rate"]]

    fig, ax = plt.subplots(figsize=(14, max(9, len(summary) * 0.32)))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("white")

    bars = ax.barh(summary["clause_type"], summary["positive_rate"],
                   color=bar_colors, edgecolor="white", linewidth=0.6, height=0.72)

    # Value labels on each bar
    for bar, val in zip(bars, summary["positive_rate"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", ha="left", fontsize=7.5, color="#333333")

    ax.axvline(x=0.2, color="#E07B3F", linestyle="--", linewidth=1.4,
               alpha=0.85, label="20% rate threshold")
    ax.set_title("Positive Rate per Clause Type — CUAD", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Positive rate", fontsize=10)
    ax.set_ylabel("")
    ax.set_xlim(0, min(1.0, summary["positive_rate"].max() * 1.18))
    ax.tick_params(axis="y", labelsize=8.5)
    ax.xaxis.grid(True, alpha=0.45, color="#E4E4E4")
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.85)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
    return summary


# ── Clause mappings ───────────────────────────────────────────────────────────

def build_clause_mappings(cuad_df: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    """Build bidirectional clause name ↔ integer ID mappings."""
    clause_names = sorted(cuad_df["clause_type"].unique().tolist())
    clause_to_id = {name: idx for idx, name in enumerate(clause_names)}
    id_to_clause = {idx: name for name, idx in clause_to_id.items()}
    return clause_to_id, id_to_clause


# ── Contract record building ──────────────────────────────────────────────────

def build_contract_records(cuad_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Aggregate clause answer spans per contract into a list of contract records."""
    records: list[dict[str, Any]] = []
    for contract_title, group in cuad_df.groupby("contract_title", sort=False):
        contract_text = group["contract_text"].iloc[0]
        clause_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row in group.itertuples(index=False):
            for answer_text, answer_start in zip(row.answer_texts, row.answer_starts):
                answer_end = int(answer_start) + len(answer_text)
                clause_spans[row.clause_type].append((int(answer_start), answer_end))
        records.append({
            "contract_title": contract_title,
            "contract_text": contract_text,
            "clause_spans": dict(clause_spans),
        })
    return records


def split_contract_records(
    contract_records: list[dict[str, Any]],
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Split contract records into train/val/test sets at the contract level."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")
    titles = [r["contract_title"] for r in contract_records]
    title_to_record = {r["contract_title"]: r for r in contract_records}
    train_titles, temp_titles = train_test_split(titles, train_size=train_size, random_state=seed)
    val_ratio = val_size / (val_size + test_size)
    val_titles, test_titles = train_test_split(temp_titles, train_size=val_ratio, random_state=seed)
    return (
        [title_to_record[t] for t in train_titles],
        [title_to_record[t] for t in val_titles],
        [title_to_record[t] for t in test_titles],
    )


def sample_contracts(
    contract_records: list[dict[str, Any]],
    frac: float = 1.0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Return a reproducible random sample of ``frac`` of the given contract records.

    Sampling is performed *before* the train/val/test split so every split
    receives a proportional share of the reduced pool.  The methodology
    (chunking, label assignment, split ratios) is identical to the full run —
    only the number of contracts changes.

    Args:
        contract_records: Full list of contract records.
        frac: Fraction of contracts to keep (0 < frac <= 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Shuffled-and-subsampled list of contract records.
    """
    if frac >= 1.0:
        return contract_records
    n = max(1, round(len(contract_records) * frac))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(contract_records), size=n, replace=False)
    sampled = [contract_records[i] for i in sorted(indices)]
    print(f"sample_contracts: using {n}/{len(contract_records)} contracts ({frac:.0%})")
    return sampled


# ── Chunk examples ────────────────────────────────────────────────────────────

def _span_overlaps_chunk(span_start: int, span_end: int, chunk_start: int, chunk_end: int) -> bool:
    return max(span_start, chunk_start) < min(span_end, chunk_end)


def build_chunk_examples(
    contract_records: list[dict[str, Any]],
    clause_to_id: dict[str, int],
    tokenizer: Any,
    max_length: int = 512,
    stride: int = 128,
) -> list[dict[str, Any]]:
    """Tokenize contracts with sliding window and assign multi-label vectors per chunk."""
    chunk_examples: list[dict[str, Any]] = []
    num_labels = len(clause_to_id)

    for record in contract_records:
        contract_text = record["contract_text"]
        chunked = tokenizer(
            contract_text,
            max_length=max_length,
            stride=stride,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        for chunk_index in range(len(chunked["input_ids"])):
            offset_mapping = chunked["offset_mapping"][chunk_index]
            valid_offsets = [(s, e) for s, e in offset_mapping if e > s]
            if not valid_offsets:
                continue
            chunk_char_start = valid_offsets[0][0]
            chunk_char_end = valid_offsets[-1][1]
            labels = np.zeros(num_labels, dtype=np.float32)
            for clause_name, spans in record["clause_spans"].items():
                if clause_name not in clause_to_id:
                    continue
                clause_id = clause_to_id[clause_name]
                if any(_span_overlaps_chunk(s, e, chunk_char_start, chunk_char_end) for s, e in spans):
                    labels[clause_id] = 1.0
            chunk_example: dict[str, Any] = {
                "contract_title": record["contract_title"],
                "chunk_index": chunk_index,
                "chunk_char_start": chunk_char_start,
                "chunk_char_end": chunk_char_end,
                "chunk_text": contract_text[chunk_char_start:chunk_char_end],
                "input_ids": chunked["input_ids"][chunk_index],
                "attention_mask": chunked["attention_mask"][chunk_index],
                "labels": labels.tolist(),
            }
            if "token_type_ids" in chunked:
                chunk_example["token_type_ids"] = chunked["token_type_ids"][chunk_index]
            chunk_examples.append(chunk_example)
    return chunk_examples


# ── Dataset class ─────────────────────────────────────────────────────────────

class MultiLabelChunkDataset(Dataset):
    """PyTorch Dataset wrapping a list of chunk examples."""

    def __init__(self, chunk_examples: list[dict[str, Any]]) -> None:
        self.chunk_examples = chunk_examples

    def __len__(self) -> int:
        return len(self.chunk_examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ex = self.chunk_examples[index]
        item = {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.float32),
        }
        if "token_type_ids" in ex:
            item["token_type_ids"] = torch.tensor(ex["token_type_ids"], dtype=torch.long)
        return item


# ── Sample & class weights ────────────────────────────────────────────────────

def compute_sample_weights(
    chunk_examples: list[dict[str, Any]],
    negative_weight: float = 0.1,
) -> list[float]:
    """Per-sample loss weights: all-negative chunks get negative_weight, others get 1.0.

    This is the feature-level downweighting from the CUAD paper — reduces the
    outsized influence of the majority all-negative chunks during training.
    """
    weights = []
    for ex in chunk_examples:
        is_all_negative = all(label_val == 0.0 for label_val in ex["labels"])
        weights.append(negative_weight if is_all_negative else 1.0)
    return weights


def compute_pos_weight(
    chunk_examples: list[dict[str, Any]],
    rare_threshold: int = 30,
    rare_boost: float = 2.0,
    max_weight: float = 50.0,
) -> torch.Tensor:
    """Per-label pos_weight tensor for BCEWithLogitsLoss: neg_count / pos_count.

    Clause types with fewer than rare_threshold positive chunks receive a rare_boost
    multiplier before capping, giving the loss function stronger signal on tail classes
    without changing the weighting formula for common clause types.
    max_weight raised to 50 (from 10) so natural neg/pos ratios for mid-rare classes
    are not suppressed.
    """
    label_matrix = np.asarray([ex["labels"] for ex in chunk_examples], dtype=np.float32)
    positive_counts = label_matrix.sum(axis=0)
    negative_counts = len(label_matrix) - positive_counts
    weights = np.where(positive_counts > 0, negative_counts / np.maximum(positive_counts, 1.0), 1.0)
    weights = np.where(positive_counts < rare_threshold, weights * rare_boost, weights)
    weights = np.clip(weights, 0, max_weight)
    return torch.tensor(weights, dtype=torch.float32)


# ── Contract-level aggregation ────────────────────────────────────────────────

def aggregate_contract_predictions(chunk_long_df: pd.DataFrame) -> pd.DataFrame:
    """Max-probability rollup across chunks per (contract_title, clause_type).

    Input DataFrame must have columns: contract_title, clause_type, score, chunk_index.
    Returns DataFrame with columns: contract_title, clause_type, max_score, best_chunk_index.
    """
    agg = (
        chunk_long_df
        .sort_values("score", ascending=False)
        .groupby(["contract_title", "clause_type"], sort=False)
        .first()
        .rename(columns={"score": "max_score", "chunk_index": "best_chunk_index"})
        .reset_index()
    )
    return agg[["contract_title", "clause_type", "max_score", "best_chunk_index"]]


def build_chunk_to_contract_map(chunk_examples: list[dict[str, Any]]) -> dict[int, str]:
    """Map chunk list index → contract_title for tracing predictions back to contracts."""
    return {i: ex["contract_title"] for i, ex in enumerate(chunk_examples)}


# ── Convenience: prepare all splits ──────────────────────────────────────────

def prepare_chunked_splits(
    cuad_df: pd.DataFrame,
    model_name: str = "bert-base-uncased",
    max_length: int = 512,
    stride: int = 128,
    seed: int = 42,
    sample_frac: float = 1.0,
) -> dict[str, Any]:
    """End-to-end helper: build clause mappings, split contracts, tokenize, return all artifacts.

    Note: call filter_clauses(cuad_df) before passing in if you want to exclude
    low-frequency clause types. This function uses whatever clause types are present in cuad_df.

    Args:
        sample_frac: Fraction of contracts to use (0 < frac <= 1.0).  Values
            below 1.0 invoke ``sample_contracts`` before splitting, giving a
            representative subset that runs proportionally faster.  Set to 1.0
            (default) for the full dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    clause_to_id, id_to_clause = build_clause_mappings(cuad_df)
    contract_records = build_contract_records(cuad_df)
    if sample_frac < 1.0:
        contract_records = sample_contracts(contract_records, frac=sample_frac, seed=seed)
    train_records, val_records, test_records = split_contract_records(contract_records, seed=seed)
    train_ex = build_chunk_examples(train_records, clause_to_id, tokenizer, max_length, stride)
    val_ex   = build_chunk_examples(val_records,   clause_to_id, tokenizer, max_length, stride)
    test_ex  = build_chunk_examples(test_records,  clause_to_id, tokenizer, max_length, stride)
    train_sample_weights = compute_sample_weights(train_ex)
    pos_weight_tensor = compute_pos_weight(train_ex)
    return {
        "tokenizer": tokenizer,
        "clause_to_id": clause_to_id,
        "id_to_clause": id_to_clause,
        "train_records": train_records, "val_records": val_records, "test_records": test_records,
        "train_examples": train_ex, "val_examples": val_ex, "test_examples": test_ex,
        "train_dataset": MultiLabelChunkDataset(train_ex),
        "val_dataset":   MultiLabelChunkDataset(val_ex),
        "test_dataset":  MultiLabelChunkDataset(test_ex),
        "train_sample_weights": train_sample_weights,
        "pos_weight": pos_weight_tensor,
    }
