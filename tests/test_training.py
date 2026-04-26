import pandas as pd
import pytest


def test_model_artifacts_dataclass_fields():
    from training import ModelArtifacts
    import numpy as np
    ma = ModelArtifacts(
        model_name="Test",
        model=None,
        tokenizer=None,
        best_threshold=0.5,
        val_metrics={"micro_f1": 0.7},
        history=pd.DataFrame(),
        id_to_clause={0: "A"},
        val_logits=np.zeros((5, 1)),
        val_labels=np.zeros((5, 1)),
    )
    assert ma.model_name == "Test"
    assert ma.best_threshold == pytest.approx(0.5)


def test_train_tfidf_lr_smoke():
    from training import train_tfidf_lr, ModelArtifacts
    import pandas as pd
    import numpy as np

    rows = []
    for i in range(60):
        rows.append({
            "contract_title": f"C{i}", "clause_type": "Governing Law",
            "contract_text": f"agreement governed by law number {i}",
            "has_answer": i % 3 != 0,
            "answer_texts": ["governed"] if i % 3 != 0 else [],
            "answer_starts": [10] if i % 3 != 0 else [],
            "answer_count": 1 if i % 3 != 0 else 0,
        })
    for i in range(60):
        rows.append({
            "contract_title": f"C{i}", "clause_type": "Non-Compete",
            "contract_text": f"agreement governed by law number {i}",
            "has_answer": i % 2 == 0,
            "answer_texts": ["compete"] if i % 2 == 0 else [],
            "answer_starts": [5] if i % 2 == 0 else [],
            "answer_count": 1 if i % 2 == 0 else 0,
        })
    df = pd.DataFrame(rows)

    from preprocessing import build_clause_mappings, build_contract_records, split_contract_records
    clause_to_id, id_to_clause = build_clause_mappings(df)
    records = build_contract_records(df)
    train_r, val_r, _ = split_contract_records(records, seed=42)

    train_titles = {r["contract_title"] for r in train_r}
    val_titles   = {r["contract_title"] for r in val_r}
    train_df = df[df["contract_title"].isin(train_titles)]
    val_df   = df[df["contract_title"].isin(val_titles)]

    artifacts = train_tfidf_lr(train_df, val_df, id_to_clause)
    assert isinstance(artifacts, ModelArtifacts)
    assert artifacts.model_name == "TF-IDF + LR"
    assert artifacts.val_logits.shape[1] == 2


def test_train_bert_cuad_smoke():
    """Smoke test: 1 epoch, 2 batches — verifies output shape and types."""
    from training import train_bert_cuad, ModelArtifacts
    from preprocessing import prepare_chunked_splits
    import pandas as pd
    import numpy as np

    rows = []
    for i in range(60):
        for clause in ["Governing Law", "Non-Compete"]:
            rows.append({
                "contract_title": f"C{i}", "clause_type": clause,
                "contract_text": (
                    f"This agreement number {i} is governed by the laws of jurisdiction {i % 5}. "
                    f"The parties agree to comply with all applicable laws. "
                    f"Non-compete restrictions apply for a period of two years."
                ),
                "has_answer": i % 2 == 0,
                "answer_texts": ["governed"] if i % 2 == 0 else [],
                "answer_starts": [14] if i % 2 == 0 else [],
                "answer_count": 1 if i % 2 == 0 else 0,
            })
    df = pd.DataFrame(rows)
    splits = prepare_chunked_splits(df, model_name="distilbert-base-uncased",
                                    max_length=64, stride=16, seed=42)

    artifacts = train_bert_cuad(
        train_dataset=splits["train_dataset"],
        val_dataset=splits["val_dataset"],
        train_examples=splits["train_examples"],
        val_examples=splits["val_examples"],
        model_name="distilbert-base-uncased",
        tokenizer=splits["tokenizer"],
        id_to_clause=splits["id_to_clause"],
        epochs=1,
        batch_size=4,
        max_train_batches=2,
        max_val_batches=2,
    )
    assert isinstance(artifacts, ModelArtifacts)
    assert artifacts.model_name == "BERT (CUAD)"
    assert artifacts.val_logits.shape[1] == 2
    assert artifacts.val_labels.shape[1] == 2


def test_train_bert_ledgar_cuad_smoke():
    """Smoke: phase 1 LEDGAR fine-tune + phase 2 CUAD fine-tune with minimal batches."""
    HFDataset = pytest.importorskip("datasets").Dataset
    ClassLabel = pytest.importorskip("datasets").ClassLabel

    from training import train_bert_ledgar_cuad, ModelArtifacts
    from preprocessing import prepare_chunked_splits
    import pandas as pd
    import numpy as np

    # Minimal synthetic LEDGAR-like dataset (10 categories to keep it light)
    ledgar_data = {
        "text": [f"provision text example number {i}" for i in range(40)],
        "label": [i % 10 for i in range(40)],
    }
    mock_ledgar = {"train": HFDataset.from_dict(ledgar_data)}
    mock_ledgar["train"] = mock_ledgar["train"].cast_column(
        "label", ClassLabel(num_classes=10, names=[str(i) for i in range(10)])
    )

    rows = []
    for i in range(60):
        for clause in ["Governing Law", "Non-Compete"]:
            rows.append({
                "contract_title": f"C{i}", "clause_type": clause,
                "contract_text": (
                    f"This contract number {i} is governed under jurisdiction {i % 5}. "
                    f"The parties agree to all applicable terms and conditions."
                ),
                "has_answer": i % 2 == 0,
                "answer_texts": ["governed"] if i % 2 == 0 else [],
                "answer_starts": [13] if i % 2 == 0 else [],
                "answer_count": 1 if i % 2 == 0 else 0,
            })
    df = pd.DataFrame(rows)
    splits = prepare_chunked_splits(df, model_name="distilbert-base-uncased",
                                    max_length=64, stride=16, seed=42)

    artifacts = train_bert_ledgar_cuad(
        ledgar_dataset=mock_ledgar,
        train_dataset=splits["train_dataset"],
        val_dataset=splits["val_dataset"],
        train_examples=splits["train_examples"],
        val_examples=splits["val_examples"],
        model_name="distilbert-base-uncased",
        tokenizer=splits["tokenizer"],
        id_to_clause=splits["id_to_clause"],
        ledgar_epochs=1, ledgar_max_batches=2,
        cuad_epochs=1,   cuad_max_train_batches=2, cuad_max_val_batches=2,
        batch_size=4,
    )
    assert isinstance(artifacts, ModelArtifacts)
    assert artifacts.model_name == "BERT (LEDGAR→CUAD)"
    assert artifacts.val_logits.shape[1] == 2


def test_train_legal_bert_cuad_smoke():
    from training import train_bert_cuad
    from preprocessing import prepare_chunked_splits
    import pandas as pd
    rows = []
    for i in range(60):
        rows.append({"contract_title": f"C{i}", "clause_type": "Governing Law",
                     "contract_text": f"This legal agreement number {i} is governed by the applicable laws of the relevant jurisdiction.",
                     "has_answer": i%2==0,
                     "answer_texts": ["governed"] if i%2==0 else [],
                     "answer_starts": [38] if i%2==0 else [], "answer_count": 1 if i%2==0 else 0})
    df = pd.DataFrame(rows)
    splits = prepare_chunked_splits(df, model_name="distilbert-base-uncased",
                                    max_length=64, stride=16, seed=42)
    # Use distilbert as stand-in for legal-bert in smoke test
    artifacts = train_bert_cuad(
        splits["train_dataset"], splits["val_dataset"], splits["train_examples"],
        "distilbert-base-uncased", splits["tokenizer"], splits["id_to_clause"],
        val_examples=splits["val_examples"],
        epochs=1, batch_size=4, max_train_batches=2, max_val_batches=2,
        artifact_name="Legal-BERT (CUAD)",
    )
    assert artifacts.model_name == "Legal-BERT (CUAD)"


def test_init_longformer_from_legal_bert_weight_shapes():
    from training import init_longformer_from_legal_bert
    # Uses distilbert-base-uncased as a stand-in to avoid downloading large models in tests
    # Just verify the function runs and returns a model with correct output size
    model = init_longformer_from_legal_bert(
        num_labels=3,
        legal_bert_name="distilbert-base-uncased",
        longformer_name="allenai/longformer-base-4096",
    )
    assert model.config.num_labels == 3
