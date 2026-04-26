from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from preprocessing import MultiLabelChunkDataset, compute_pos_weight


def _tfidf_proba_col(est, X):
    """Return class-1 probability; DummyClassifier may only have one class."""
    proba = est.predict_proba(X)
    if len(est.classes_) == 1:
        return np.full(X.shape[0], float(est.classes_[0]))
    return proba[:, 1]


class _TfIdfPipeline:
    """Wraps a TF-IDF vectorizer + per-label estimators for inference.

    predict_proba returns raw probabilities in [0, 1] — NOT log-odds.
    Convert to log-odds before passing to sigmoid-based evaluation if needed:
        p = np.clip(pipeline.predict_proba(texts), 1e-7, 1-1e-7)
        logits = np.log(p / (1 - p))
    """

    def __init__(self, vec, ests):
        self.vectorizer = vec
        self.estimators_ = ests

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return np.column_stack([_tfidf_proba_col(e, X) for e in self.estimators_])


@dataclass
class ModelArtifacts:
    """Unified container returned by every training function."""
    model_name: str
    model: Any
    tokenizer: Any
    best_threshold: float
    val_metrics: dict[str, float]
    history: pd.DataFrame
    id_to_clause: dict[int, str]
    val_logits: np.ndarray
    val_labels: np.ndarray


def choose_device() -> torch.device:
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))


def _aggregate_to_contract_level(
    chunk_logits: np.ndarray,
    chunk_labels: np.ndarray,
    examples: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Max-probability rollup: chunk predictions → one row per contract.

    examples must be aligned with chunk_logits/chunk_labels (same order, val loader
    must use shuffle=False). For each (contract, clause) pair the maximum sigmoid
    probability across all chunks is taken; labels are OR-ed (max) across chunks.
    Returns contract-level logits as log-odds so they are compatible with _sigmoid.
    """
    seen: dict[str, int] = {}
    contract_order: list[str] = []
    for ex in examples:
        t = ex["contract_title"]
        if t not in seen:
            seen[t] = len(contract_order)
            contract_order.append(t)

    n_contracts = len(contract_order)
    n_labels = chunk_logits.shape[1]
    contract_max_probs = np.zeros((n_contracts, n_labels), dtype=np.float32)
    contract_labels = np.zeros((n_contracts, n_labels), dtype=np.float32)

    chunk_probs = _sigmoid(chunk_logits)
    for i, ex in enumerate(examples):
        ci = seen[ex["contract_title"]]
        contract_max_probs[ci] = np.maximum(contract_max_probs[ci], chunk_probs[i])
        contract_labels[ci] = np.maximum(contract_labels[ci], chunk_labels[i])

    p = np.clip(contract_max_probs, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)), contract_labels


def _tune_global_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Find the global threshold maximising micro-F1; return (threshold, metrics_dict)."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        preds = (_sigmoid(logits) >= t).astype(int)
        f1 = float(f1_score(labels.astype(int), preds, average="micro", zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    preds = (_sigmoid(logits) >= best_t).astype(int)
    return best_t, {
        "micro_f1":        best_f1,
        "micro_precision": float(precision_score(labels.astype(int), preds, average="micro", zero_division=0)),
        "micro_recall":    float(recall_score(labels.astype(int), preds, average="micro",    zero_division=0)),
    }


def collect_logits_and_labels(
    model: Any,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a DataLoader and return (logits, labels) as numpy arrays."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            labels = batch["labels"].cpu().numpy()
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**inputs).logits.cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels)
    return np.vstack(all_logits), np.vstack(all_labels)


def _run_training_loop(
    model: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_examples: list[dict],
    val_examples: list[dict],
    device: torch.device,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> tuple[Any, pd.DataFrame, float, dict, np.ndarray, np.ndarray]:
    """Shared training loop for all transformer models.

    Returns (model, history_df, best_threshold, best_val_metrics, best_val_logits, best_val_labels).
    val_logits and val_labels in the return value are contract-level (max-probability rollup),
    matching the granularity used in Section 4 test evaluation.
    Applies pos_weight (BCEWithLogitsLoss) and per-sample downweighting for all-negative chunks.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    effective = len(train_loader) if max_train_batches is None else min(len(train_loader), max_train_batches)
    total_steps = max(1, epochs * max(1, effective))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * warmup_ratio), total_steps)

    pos_weight = compute_pos_weight(train_examples).to(device)
    # reduction="none" so we can apply per-sample weights manually
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    # Mixed precision: ~1.5-2x speedup on CUDA (T4/V100/A100). No-op on CPU/MPS.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_state: dict | None = None
    best_t = 0.5
    best_metrics: dict[str, float] = {"micro_f1": -1.0, "macro_f1": -1.0}
    best_val_logits: np.ndarray | None = None
    best_val_labels: np.ndarray | None = None
    history_rows: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0

        for bi, batch in enumerate(train_loader):
            if max_train_batches is not None and bi >= max_train_batches:
                break
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(**inputs).logits
                # Per-sample down-weighting: all-negative chunks get weight 0.1.
                # Computed directly from the batch labels so it is correct under shuffle.
                is_all_negative = (labels == 0).all(dim=1)
                batch_sw = torch.where(is_all_negative,
                                       torch.full((labels.shape[0],), 0.1, device=device),
                                       torch.ones(labels.shape[0], device=device))
                loss = (loss_fn(logits, labels).mean(dim=1) * batch_sw).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.item())
            seen += 1

        val_chunk_logits, val_chunk_labels = collect_logits_and_labels(model, val_loader, device, max_val_batches)
        # Roll up to contract level so thresholds are tuned on the same granularity
        # as Section 4 test evaluation (max-probability rollup per contract).
        # Slice val_examples to match the collected rows when max_val_batches truncates.
        val_ex_subset = val_examples[:len(val_chunk_logits)]
        val_logits, val_labels = _aggregate_to_contract_level(val_chunk_logits, val_chunk_labels, val_ex_subset)
        t, metrics = _tune_global_threshold(val_logits, val_labels)
        # Threshold is tuned by micro-F1 (more stable signal), but the best epoch
        # checkpoint is selected by macro-F1 — the primary reported metric — so
        # that the saved model is optimal for equal per-clause evaluation.
        from sklearn.metrics import f1_score as _f1
        _preds = (_sigmoid(val_logits) >= t).astype(int)
        metrics["macro_f1"] = float(_f1(val_labels.astype(int), _preds, average="macro", zero_division=0))
        history_rows.append({"epoch": epoch, "train_loss": total_loss / max(1, seen),
                              "val_threshold": t, **metrics})
        print(f"  Epoch {epoch}/{epochs}: loss={total_loss / max(1, seen):.4f}  "
              f"val micro_F1={metrics['micro_f1']:.4f}  val macro_F1={metrics['macro_f1']:.4f}",
              flush=True)

        if metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = metrics
            best_t = t
            best_state = copy.deepcopy(model.state_dict())
            best_val_logits = val_logits
            best_val_labels = val_labels

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, pd.DataFrame(history_rows), best_t, best_metrics, best_val_logits, best_val_labels


def train_tfidf_lr(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    id_to_clause: dict[int, str],
) -> ModelArtifacts:
    """TF-IDF + multi-output Logistic Regression baseline (contract-level, not chunk-level).

    Uses full contract text per contract (one row per contract after groupby).
    Returns ModelArtifacts with logits stored as log-odds for compatibility
    with sigmoid-based evaluation functions.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier

    name_to_id = {v: k for k, v in id_to_clause.items()}

    def _build_contract_matrix(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
        texts, label_rows = [], []
        for title, group in df.groupby("contract_title"):
            # All rows for a contract share the same full contract text; take first.
            texts.append(group["contract_text"].iloc[0])
            row = np.zeros(len(id_to_clause), dtype=float)
            for r in group.itertuples(index=False):
                if r.has_answer and r.clause_type in name_to_id:
                    row[name_to_id[r.clause_type]] = 1.0
            label_rows.append(row)
        return texts, np.array(label_rows)

    train_texts, train_labels = _build_contract_matrix(train_df)
    val_texts,   val_labels   = _build_contract_matrix(val_df)

    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_texts)
    X_val   = vectorizer.transform(val_texts)

    # Fit a per-label classifier. Labels with only one class in the training split
    # (all-positive or all-negative) cannot be fit by LR — use DummyClassifier instead.
    estimators = []
    for i in range(train_labels.shape[1]):
        col = train_labels[:, i]
        if len(np.unique(col)) < 2:
            est = DummyClassifier(strategy="most_frequent")
        else:
            est = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, solver="lbfgs")
        est.fit(X_train, col)
        estimators.append(est)

    # Collect class-1 probabilities; convert to log-odds for uniform interface with _sigmoid
    eps = 1e-7
    val_probs = np.column_stack([_tfidf_proba_col(est, X_val) for est in estimators])
    p = np.clip(val_probs, eps, 1 - eps)
    val_logits = np.log(p / (1 - p))

    best_t, val_metrics = _tune_global_threshold(val_logits, val_labels)

    pipeline = _TfIdfPipeline(vectorizer, estimators)

    print(f"TF-IDF + LR → val micro_F1={val_metrics['micro_f1']:.4f}, threshold={best_t:.2f}")
    return ModelArtifacts(
        model_name="TF-IDF + LR",
        model=pipeline,
        tokenizer=None,
        best_threshold=best_t,
        val_metrics=val_metrics,
        history=pd.DataFrame(),
        id_to_clause=id_to_clause,
        val_logits=val_logits,
        val_labels=val_labels,
    )


def train_bert_cuad(
    train_dataset: MultiLabelChunkDataset,
    val_dataset: MultiLabelChunkDataset,
    train_examples: list[dict],
    model_name: str,
    tokenizer: Any,
    id_to_clause: dict[int, str],
    val_examples: list[dict] | None = None,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    device: torch.device | None = None,
    artifact_name: str = "BERT (CUAD)",
) -> ModelArtifacts:
    """Fine-tune a BERT-family model directly on CUAD multi-label chunks.

    The artifact_name parameter lets Legal-BERT reuse this function with a
    different name (see train_legal_bert_cuad in Task 11).
    """
    device = device or choose_device()

    label2id = {v: k for k, v in id_to_clause.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id_to_clause),
        id2label=id_to_clause,
        label2id=label2id,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    ).to(device)

    _pin = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=_pin, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=_pin, persistent_workers=True)

    model, history, best_t, metrics, val_logits, val_labels = _run_training_loop(
        model, train_loader, val_loader, train_examples, val_examples or [], device,
        epochs, learning_rate, weight_decay, warmup_ratio,
        max_train_batches, max_val_batches,
    )

    print(f"{artifact_name} → val micro_F1={metrics['micro_f1']:.4f}, threshold={best_t:.2f}")
    return ModelArtifacts(
        model_name=artifact_name,
        model=model,
        tokenizer=tokenizer,
        best_threshold=best_t,
        val_metrics=metrics,
        history=history,
        id_to_clause=id_to_clause,
        val_logits=val_logits,
        val_labels=val_labels,
    )


def train_bert_ledgar_cuad(
    ledgar_dataset: Any,
    train_dataset: MultiLabelChunkDataset,
    val_dataset: MultiLabelChunkDataset,
    train_examples: list[dict],
    model_name: str,
    tokenizer: Any,
    id_to_clause: dict[int, str],
    val_examples: list[dict] | None = None,
    ledgar_epochs: int = 3,
    ledgar_max_batches: int | None = None,
    cuad_epochs: int = 3,
    cuad_max_train_batches: int | None = None,
    cuad_max_val_batches: int | None = None,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    device: torch.device | None = None,
) -> ModelArtifacts:
    """Two-phase training: (1) fine-tune on LEDGAR multi-class, (2) transfer to CUAD multi-label.

    Phase 1 uses LEDGAR labels to warm the model on legal language.
    Phase 2 strips the LEDGAR classification head, attaches a new multi-label head,
    and fine-tunes on CUAD using the shared _run_training_loop.
    """
    from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader

    device = device or choose_device()

    # ── Phase 1: LEDGAR fine-tuning ───────────────────────────────────────────
    print("Phase 1: domain-adapting on LEDGAR...")
    ledgar_train = ledgar_dataset["train"]
    n_ledgar_labels = ledgar_train.features["label"].num_classes

    ledgar_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_ledgar_labels,
    ).to(device)

    def _tokenize_ledgar(batch: dict) -> dict:
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=512
        )

    ledgar_tok = ledgar_train.map(_tokenize_ledgar, batched=True)
    ledgar_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    class _LedgarDataset(TorchDataset):
        def __init__(self, hf_ds):
            self.ds = hf_ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, i):
            item = self.ds[i]
            return {
                "input_ids":      item["input_ids"],
                "attention_mask": item["attention_mask"],
                "labels":         item["label"],
            }

    _pin = device.type == "cuda"
    ledgar_loader = TorchDataLoader(
        _LedgarDataset(ledgar_tok), batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=_pin, persistent_workers=True,
    )
    optimizer_p1 = torch.optim.AdamW(ledgar_model.parameters(), lr=learning_rate, weight_decay=0.01)
    ce_loss = torch.nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler_p1 = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(1, ledgar_epochs + 1):
        ledgar_model.train()
        total_loss, seen = 0.0, 0
        for bi, batch in enumerate(ledgar_loader):
            if ledgar_max_batches is not None and bi >= ledgar_max_batches:
                break
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            optimizer_p1.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = ledgar_model(**inputs).logits
                loss = ce_loss(logits, labels)
            scaler_p1.scale(loss).backward()
            scaler_p1.step(optimizer_p1)
            scaler_p1.update()
            total_loss += float(loss.item())
            seen += 1
        print(f"  LEDGAR epoch {epoch}: loss={total_loss / max(1, seen):.4f}")

    # ── Phase 2: transfer backbone to CUAD ───────────────────────────────────
    print("Phase 2: transferring backbone to CUAD multi-label task...")
    # Extract backbone weights before freeing the LEDGAR model to save VRAM
    backbone_state = {
        k: v for k, v in ledgar_model.state_dict().items()
        if not k.startswith("classifier") and not k.startswith("pre_classifier")
    }

    del ledgar_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    label2id = {v: k for k, v in id_to_clause.items()}
    cuad_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id_to_clause),
        id2label=id_to_clause,
        label2id=label2id,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    missing, unexpected = cuad_model.load_state_dict(backbone_state, strict=False)
    print(f"  Transferred {len(backbone_state)} layers; "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    cuad_model = cuad_model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=_pin, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=_pin, persistent_workers=True)

    model, history, best_t, metrics, val_logits, val_labels = _run_training_loop(
        cuad_model, train_loader, val_loader, train_examples, val_examples or [], device,
        epochs=cuad_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_train_batches=cuad_max_train_batches,
        max_val_batches=cuad_max_val_batches,
    )

    print(f"BERT (LEDGAR→CUAD) → val micro_F1={metrics['micro_f1']:.4f}, threshold={best_t:.2f}")
    return ModelArtifacts(
        model_name="BERT (LEDGAR→CUAD)",
        model=model,
        tokenizer=tokenizer,
        best_threshold=best_t,
        val_metrics=metrics,
        history=history,
        id_to_clause=id_to_clause,
        val_logits=val_logits,
        val_labels=val_labels,
    )


def train_legal_bert_cuad(
    train_dataset: MultiLabelChunkDataset,
    val_dataset: MultiLabelChunkDataset,
    train_examples: list[dict],
    tokenizer: Any,
    id_to_clause: dict[int, str],
    val_examples: list[dict] | None = None,
    model_name: str = "nlpaueb/legal-bert-base-uncased",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    device: torch.device | None = None,
) -> ModelArtifacts:
    """Fine-tune Legal-BERT on CUAD (reuses train_bert_cuad with fixed artifact name)."""
    return train_bert_cuad(
        train_dataset, val_dataset, train_examples,
        model_name, tokenizer, id_to_clause,
        val_examples=val_examples,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        max_train_batches=max_train_batches, max_val_batches=max_val_batches,
        device=device, artifact_name="Legal-BERT (CUAD)",
    )


def init_longformer_from_legal_bert(
    num_labels: int,
    legal_bert_name: str = "nlpaueb/legal-bert-base-uncased",
    longformer_name: str = "allenai/longformer-base-4096",
) -> Any:
    """Initialise a LongformerForSequenceClassification with Legal-BERT backbone weights.

    Follows Mamakas et al. 2022:
    - Shared encoder layer weights copied by name
    - Position embeddings tiled from 512 → 4096
    - Global attention projections initialised from BERT attention weights
    """
    from transformers import AutoModel, LongformerForSequenceClassification, LongformerConfig

    print(f"Loading Legal-BERT backbone from {legal_bert_name}...")
    bert_model = AutoModel.from_pretrained(legal_bert_name)
    bert_state = bert_model.state_dict()

    print(f"Loading Longformer config from {longformer_name}...")
    lf_config = LongformerConfig.from_pretrained(longformer_name)
    lf_config.num_labels = num_labels
    lf_config.problem_type = "multi_label_classification"

    print("Initialising Longformer with Longformer-base-4096 weights...")
    lf_model = LongformerForSequenceClassification.from_pretrained(
        longformer_name, config=lf_config, ignore_mismatched_sizes=True
    )
    lf_state = lf_model.state_dict()
    new_state: dict[str, torch.Tensor] = {}

    for lf_key in lf_state:
        # Map longformer key → bert key (strip "longformer." prefix → "")
        bert_key = lf_key.replace("longformer.", "")

        if "position_embeddings" in lf_key:
            # Extend 512 positions → 4096 by tiling
            bert_pos_key = "embeddings.position_embeddings.weight"
            if bert_pos_key in bert_state:
                bert_pos = bert_state[bert_pos_key]           # [512, 768]
                target_size = lf_state[lf_key].shape[0]       # typically 4098
                repeats = (target_size // bert_pos.shape[0]) + 1
                extended = bert_pos.repeat(repeats, 1)[:target_size]
                new_state[lf_key] = extended
                continue
            new_state[lf_key] = lf_state[lf_key]

        elif "query_global" in lf_key or "key_global" in lf_key or "value_global" in lf_key:
            # Global attention projections: initialise from BERT's attention weights
            base = lf_key.replace("query_global", "query").replace("key_global", "key").replace("value_global", "value")
            bert_equiv = base.replace("longformer.", "")
            if bert_equiv in bert_state and bert_state[bert_equiv].shape == lf_state[lf_key].shape:
                new_state[lf_key] = bert_state[bert_equiv].clone()
            else:
                new_state[lf_key] = lf_state[lf_key]

        elif bert_key in bert_state and bert_state[bert_key].shape == lf_state[lf_key].shape:
            new_state[lf_key] = bert_state[bert_key]

        else:
            new_state[lf_key] = lf_state[lf_key]

    lf_model.load_state_dict(new_state)
    transferred = sum(1 for k in lf_state if new_state[k] is not lf_state[k])
    print(f"Transferred {transferred}/{len(lf_state)} weight tensors from Legal-BERT to Longformer.")
    return lf_model


def train_longformer_cuad(
    train_dataset: MultiLabelChunkDataset,
    val_dataset: MultiLabelChunkDataset,
    train_examples: list[dict],
    tokenizer: Any,
    id_to_clause: dict[int, str],
    val_examples: list[dict] | None = None,
    model_name: str = "allenai/longformer-base-4096",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    device: torch.device | None = None,
) -> ModelArtifacts:
    """Fine-tune Longformer-base-4096 on CUAD multi-label chunks."""
    device = device or choose_device()
    label2id = {v: k for k, v in id_to_clause.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id_to_clause),
        id2label=id_to_clause,
        label2id=label2id,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    ).to(device)

    # Gradient checkpointing: recomputes activations on the backward pass instead
    # of storing them, trading ~15% extra compute for ~4x less activation memory.
    # Critical for Longformer at 4096 tokens — without this the T4/A100 runs OOM.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    _pin = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=_pin, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=_pin, persistent_workers=True)

    model, history, best_t, metrics, val_logits, val_labels = _run_training_loop(
        model, train_loader, val_loader, train_examples, val_examples or [], device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    print(f"Longformer (CUAD) → val micro_F1={metrics['micro_f1']:.4f}, threshold={best_t:.2f}")
    return ModelArtifacts(
        model_name="Longformer (CUAD)",
        model=model, tokenizer=tokenizer,
        best_threshold=best_t, val_metrics=metrics, history=history,
        id_to_clause=id_to_clause, val_logits=val_logits, val_labels=val_labels,
    )


def train_legalbert_longformer_cuad(
    train_dataset: MultiLabelChunkDataset,
    val_dataset: MultiLabelChunkDataset,
    train_examples: list[dict],
    tokenizer: Any,
    id_to_clause: dict[int, str],
    val_examples: list[dict] | None = None,
    legal_bert_name: str = "nlpaueb/legal-bert-base-uncased",
    longformer_name: str = "allenai/longformer-base-4096",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    device: torch.device | None = None,
) -> ModelArtifacts:
    """Legal-BERT warm-started Longformer fine-tuned on CUAD (Mamakas et al. 2022)."""
    device = device or choose_device()
    model = init_longformer_from_legal_bert(
        num_labels=len(id_to_clause),
        legal_bert_name=legal_bert_name,
        longformer_name=longformer_name,
    ).to(device)

    # Gradient checkpointing: same rationale as train_longformer_cuad — prevents
    # OOM on 4096-token sequences without changing the training methodology.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    _pin = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=_pin, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=_pin, persistent_workers=True)

    model, history, best_t, metrics, val_logits, val_labels = _run_training_loop(
        model, train_loader, val_loader, train_examples, val_examples or [], device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
    )
    print(f"Legal-BERT→Longformer → val micro_F1={metrics['micro_f1']:.4f}, threshold={best_t:.2f}")
    return ModelArtifacts(
        model_name="Legal-BERT→Longformer (CUAD)",
        model=model, tokenizer=tokenizer,
        best_threshold=best_t, val_metrics=metrics, history=history,
        id_to_clause=id_to_clause, val_logits=val_logits, val_labels=val_labels,
    )


def train_longformer_ledgar_cuad(
    ledgar_dataset: Any,
    train_dataset: MultiLabelChunkDataset,
    val_dataset: MultiLabelChunkDataset,
    train_examples: list[dict],
    tokenizer: Any,
    id_to_clause: dict[int, str],
    val_examples: list[dict] | None = None,
    longformer_name: str = "allenai/longformer-base-4096",
    ledgar_epochs: int = 2,
    ledgar_max_batches: int | None = None,
    ledgar_batch_size: int = 32,
    cuad_epochs: int = 3,
    cuad_max_train_batches: int | None = None,
    cuad_max_val_batches: int | None = None,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    device: torch.device | None = None,
) -> ModelArtifacts:
    """Two-phase training: (1) fine-tune Longformer-base on LEDGAR, (2) transfer to CUAD.

    Unlike train_legalbert_longformer_cuad, this does NOT copy Legal-BERT weights with
    512→4096 position-embedding tiling. It domain-adapts Longformer's native position
    embeddings on LEDGAR, keeping all 4096 positions properly trained.
    Both phases use local-only attention (no global_attention_mask), consistent with
    train_longformer_cuad and the CUAD test evaluation in Section 4.
    """
    from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader

    device = device or choose_device()
    _pin = device.type == "cuda"

    # ── Phase 1: LEDGAR fine-tuning ───────────────────────────────────────────
    print("Phase 1: domain-adapting Longformer on LEDGAR...", flush=True)
    ledgar_train = ledgar_dataset["train"]
    n_ledgar_labels = ledgar_train.features["label"].num_classes

    ledgar_model = AutoModelForSequenceClassification.from_pretrained(
        longformer_name, num_labels=n_ledgar_labels,
    ).to(device)
    ledgar_model.config.use_cache = False
    ledgar_model.gradient_checkpointing_enable()

    def _tokenize_ledgar(batch: dict) -> dict:
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=512
        )

    print(f"  Tokenizing {len(ledgar_train):,} LEDGAR examples...", flush=True)
    ledgar_tok = ledgar_train.map(_tokenize_ledgar, batched=True, num_proc=4)
    ledgar_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    print(f"  Tokenization complete.", flush=True)

    class _LFLedgarDataset(TorchDataset):
        def __init__(self, hf_ds):
            self.ds = hf_ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, i):
            item = self.ds[i]
            return {
                "input_ids":      item["input_ids"],
                "attention_mask": item["attention_mask"],
                "labels":         item["label"],
            }

    # ledgar_batch_size decouples Phase 1 batch size from Phase 2:
    # Phase 1 uses 512-token sequences so a much larger batch is safe on H100.
    ledgar_loader = TorchDataLoader(
        _LFLedgarDataset(ledgar_tok), batch_size=ledgar_batch_size, shuffle=True,
        num_workers=2, pin_memory=_pin, persistent_workers=True,
    )
    optimizer_p1 = torch.optim.AdamW(ledgar_model.parameters(), lr=learning_rate, weight_decay=0.01)
    ce_loss = torch.nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler_p1 = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(1, ledgar_epochs + 1):
        ledgar_model.train()
        total_loss, seen = 0.0, 0
        for bi, batch in enumerate(ledgar_loader):
            if ledgar_max_batches is not None and bi >= ledgar_max_batches:
                break
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            optimizer_p1.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = ledgar_model(**inputs).logits
                loss = ce_loss(logits, labels)
            scaler_p1.scale(loss).backward()
            scaler_p1.step(optimizer_p1)
            scaler_p1.update()
            total_loss += float(loss.item())
            seen += 1
        print(f"  LEDGAR epoch {epoch}: loss={total_loss / max(1, seen):.4f}", flush=True)

    # ── Phase 2: transfer backbone to CUAD ───────────────────────────────────
    print("Phase 2: transferring Longformer backbone to CUAD multi-label task...", flush=True)
    backbone_state = {
        k: v for k, v in ledgar_model.state_dict().items()
        if not k.startswith("classifier")
    }
    del ledgar_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    label2id = {v: k for k, v in id_to_clause.items()}
    cuad_model = AutoModelForSequenceClassification.from_pretrained(
        longformer_name,
        num_labels=len(id_to_clause),
        id2label=id_to_clause,
        label2id=label2id,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    missing, unexpected = cuad_model.load_state_dict(backbone_state, strict=False)
    print(f"  Transferred {len(backbone_state)} layers; "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    cuad_model = cuad_model.to(device)
    cuad_model.config.use_cache = False
    cuad_model.gradient_checkpointing_enable()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=_pin, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=_pin, persistent_workers=True)

    model, history, best_t, metrics, val_logits, val_labels = _run_training_loop(
        cuad_model, train_loader, val_loader, train_examples, val_examples or [], device,
        epochs=cuad_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_train_batches=cuad_max_train_batches,
        max_val_batches=cuad_max_val_batches,
    )
    print(f"Longformer (LEDGAR→CUAD) → val micro_F1={metrics['micro_f1']:.4f}, threshold={best_t:.2f}")
    return ModelArtifacts(
        model_name="Longformer (LEDGAR→CUAD)",
        model=model, tokenizer=tokenizer,
        best_threshold=best_t, val_metrics=metrics, history=history,
        id_to_clause=id_to_clause, val_logits=val_logits, val_labels=val_labels,
    )
