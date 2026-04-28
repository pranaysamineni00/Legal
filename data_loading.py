from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download


# CUAD questions are templated as: '... related to "<Clause Name>" that should ...'
# The clause name is the substring inside the first pair of double quotes, used
# as a fallback when the qa["id"] does not carry the canonical "<title>__<clause>" form.
_QUESTION_CLAUSE_RE = re.compile(r'related to "([^"]+)"')


def _extract_clause_type(qa: dict[str, Any]) -> str:
    """Extract clause_type from a CUAD `qa` entry.

    Primary: split qa["id"] on the standard "<title>__<clause>" delimiter.
    Fallback: parse the clause name from qa["question"] (every CUAD question
    contains 'related to "<Clause Name>"'). Raising rather than returning a
    silent default avoids polluting the clause taxonomy with ill-formed ids.
    """
    qa_id = qa.get("id", "")
    if "__" in qa_id:
        return qa_id.split("__", 1)[-1]
    match = _QUESTION_CLAUSE_RE.search(qa.get("question", ""))
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract clause_type from CUAD qa entry: id={qa_id!r}")


def _parse_cuad_json(cuad: dict[str, Any]) -> pd.DataFrame:
    """Convert raw CUAD SQuAD-format dict to a flat clause-level DataFrame."""
    rows = []
    for doc in cuad["data"]:
        for paragraph in doc["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                answers = qa.get("answers", [])
                rows.append(
                    {
                        "contract_title": doc["title"],
                        "clause_type": _extract_clause_type(qa),
                        "question": qa["question"],
                        "contract_text": context,
                        "has_answer": bool(len(answers) > 0),
                        "answer_texts": [a["text"] for a in answers],
                        "answer_starts": [a["answer_start"] for a in answers],
                        "answer_count": len(answers),
                    }
                )
    return pd.DataFrame(rows)


def load_cuad(data_dir: str = "data/cuad") -> pd.DataFrame:
    """Download (if needed) and parse CUAD into a flat clause-level DataFrame."""
    raw_json_path = Path(data_dir) / "CUAD_v1" / "CUAD_v1.json"
    raw_json_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_json_path.exists():
        downloaded = hf_hub_download(
            repo_id="theatticusproject/cuad",
            repo_type="dataset",
            filename="CUAD_v1/CUAD_v1.json",
            local_dir=data_dir,
        )
        raw_json_path = Path(downloaded)

    with raw_json_path.open() as f:
        cuad = json.load(f)

    df = _parse_cuad_json(cuad)

    print(f"CUAD loaded: {df['contract_title'].nunique():,} contracts, "
          f"{df['clause_type'].nunique()} clause types, "
          f"{len(df):,} rows, "
          f"positive rate {df['has_answer'].mean():.2%}")
    return df


def load_ledgar(cache_dir: str = "data/ledgar") -> Any:
    """Download and return the LEDGAR dataset from lex_glue."""
    from datasets import load_dataset
    dataset = load_dataset("coastalcph/lex_glue", "ledgar", cache_dir=cache_dir)
    n_train = len(dataset["train"])
    n_labels = dataset["train"].features["label"].num_classes
    print(f"LEDGAR loaded: {n_train:,} train examples, {n_labels} categories")
    return dataset
