import pytest
import pandas as pd

from preprocessing import filter_clauses, build_chunk_examples, build_contract_records, split_contract_records, compute_sample_weights, compute_pos_weight, aggregate_contract_predictions, build_chunk_to_contract_map

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_cuad_df():
    """Minimal synthetic CUAD DataFrame for testing."""
    rows = []
    for contract_idx in range(10):
        title = f"Contract_{contract_idx}"
        for clause in ["Governing Law", "Non-Compete", "Rare Clause"]:
            has_answer = (clause != "Rare Clause") or (contract_idx == 0)
            rows.append({
                "contract_title": title,
                "clause_type": clause,
                "contract_text": f"text of {title}",
                "has_answer": has_answer,
                "answer_texts": ["span"] if has_answer else [],
                "answer_starts": [5] if has_answer else [],
                "answer_count": 1 if has_answer else 0,
            })
    return pd.DataFrame(rows)


def test_filter_clauses_drops_below_threshold():
    df = _make_cuad_df()
    filtered, excluded = filter_clauses(df, min_positives=5)
    assert "Rare Clause" in excluded
    assert "Rare Clause" not in filtered["clause_type"].values


def test_filter_clauses_keeps_above_threshold():
    df = _make_cuad_df()
    filtered, excluded = filter_clauses(df, min_positives=5)
    assert "Governing Law" in filtered["clause_type"].values
    assert "Non-Compete" in filtered["clause_type"].values


def test_filter_clauses_excluded_is_dict_with_counts():
    df = _make_cuad_df()
    _, excluded = filter_clauses(df, min_positives=5)
    assert isinstance(excluded, dict)
    assert excluded["Rare Clause"] == 1


def test_build_chunk_examples_labels_positive_chunk():
    """A chunk overlapping a known answer span must have that clause labelled 1."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
    clause_to_id = {"Governing Law": 0, "Non-Compete": 1}

    records = [
        {
            "contract_title": "C1",
            "contract_text": "This agreement is governed by the laws of New York.",
            "clause_spans": {"Governing Law": [(18, 46)]},
        }
    ]
    examples = build_chunk_examples(records, clause_to_id, tokenizer, max_length=64, stride=16)
    assert len(examples) >= 1
    # At least one chunk should have Governing Law = 1
    governing_law_hits = [e for e in examples if e["labels"][0] == 1.0]
    assert len(governing_law_hits) >= 1
    # Non-Compete should be 0 everywhere
    assert all(e["labels"][1] == 0.0 for e in examples)


def test_split_contract_records_80_10_10():
    rows = []
    for i in range(100):
        rows.append({
            "contract_title": f"C{i}", "clause_type": "Governing Law",
            "contract_text": "text", "has_answer": True,
            "answer_texts": ["t"], "answer_starts": [0], "answer_count": 1,
        })
    df = pd.DataFrame(rows)
    records = build_contract_records(df)
    train, val, test = split_contract_records(records, seed=42)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10


def test_split_contract_records_raises_on_bad_sizes():
    from preprocessing import split_contract_records
    with pytest.raises(ValueError):
        split_contract_records([], train_size=0.8, val_size=0.1, test_size=0.5)


def test_compute_sample_weights_all_negative_gets_low_weight():
    examples = [
        {"labels": [0.0, 0.0, 0.0]},  # all-negative
        {"labels": [1.0, 0.0, 0.0]},  # has positive
        {"labels": [0.0, 1.0, 0.0]},  # has positive
    ]
    weights = compute_sample_weights(examples, negative_weight=0.1)
    assert weights[0] == pytest.approx(0.1)
    assert weights[1] == pytest.approx(1.0)
    assert weights[2] == pytest.approx(1.0)


def test_compute_pos_weight_shape_and_values():
    # 4 examples, 2 labels: label 0 has 1 positive, label 1 has 3 positives
    examples = [
        {"labels": [1.0, 1.0]},
        {"labels": [0.0, 1.0]},
        {"labels": [0.0, 1.0]},
        {"labels": [0.0, 0.0]},
    ]
    weights = compute_pos_weight(examples)
    assert weights.shape == (2,)
    # Both labels have <30 positives, so rare_boost (×2) is applied before clipping at 50×.
    # label 0: (3 neg / 1 pos) × 2 = 6.0
    assert float(weights[0]) == pytest.approx(6.0)
    # label 1: (1 neg / 3 pos) × 2 = 2/3
    assert float(weights[1]) == pytest.approx(2.0 / 3.0, rel=1e-3)


def test_aggregate_contract_predictions_takes_max():
    chunk_df = pd.DataFrame([
        {"contract_title": "C1", "clause_type": "Governing Law", "score": 0.3, "chunk_index": 0},
        {"contract_title": "C1", "clause_type": "Governing Law", "score": 0.9, "chunk_index": 1},
        {"contract_title": "C1", "clause_type": "Non-Compete",   "score": 0.2, "chunk_index": 0},
    ])
    result = aggregate_contract_predictions(chunk_df)
    assert result.loc[result["clause_type"] == "Governing Law", "max_score"].iloc[0] == pytest.approx(0.9)
    assert result.loc[result["clause_type"] == "Non-Compete",   "max_score"].iloc[0] == pytest.approx(0.2)


def test_build_chunk_to_contract_map_maps_index_to_title():
    examples = [
        {"contract_title": "ContractA", "labels": [1.0]},
        {"contract_title": "ContractA", "labels": [0.0]},
        {"contract_title": "ContractB", "labels": [1.0]},
    ]
    mapping = build_chunk_to_contract_map(examples)
    assert mapping[0] == "ContractA"
    assert mapping[1] == "ContractA"
    assert mapping[2] == "ContractB"
    assert len(mapping) == 3
