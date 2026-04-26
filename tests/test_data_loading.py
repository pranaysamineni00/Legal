def test_parse_cuad_json_returns_required_columns():
    mock_cuad = {
        "data": [
            {
                "title": "Contract A",
                "paragraphs": [
                    {
                        "context": "This agreement shall be governed by...",
                        "qas": [
                            {
                                "id": "Contract A__Governing Law",
                                "question": "What law governs?",
                                "answers": [{"text": "governed by", "answer_start": 25}],
                                "is_impossible": False,
                            }
                        ],
                    }
                ],
            }
        ]
    }

    # Direct construction test — call internal builder
    from data_loading import _parse_cuad_json
    df = _parse_cuad_json(mock_cuad)
    required = {"contract_title", "clause_type", "question", "contract_text",
                "has_answer", "answer_texts", "answer_starts", "answer_count"}
    assert required.issubset(set(df.columns))
    assert len(df) == 1
    assert df.iloc[0]["clause_type"] == "Governing Law"
    assert df.iloc[0]["has_answer"] == True
