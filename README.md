# Legal Clause Classification

BT5153 Group Project

## Purpose

This project builds a machine learning system trained on legal contracts to classify individual clauses into **high**, **medium**, and **low** risk categories, supported by relevant excerpts from the document and plain-English summaries of each excerpt. The target user is a small business owner: the goal is to help people without in-house legal teams or formal legal training make sense of contracts they are asked to sign, by combining ML-based clause classification with AI-generated explanations.

---

## 1. Training the Model (Notebook)

We recommend running the training notebook ([legal_clause_classification.ipynb](legal_clause_classification.ipynb)) on your own before using the demo, so the results are fully reproducible end to end.

- **Environment:** The notebook was developed and run on **Google Colab** with a GPU runtime. Training on CPU is not practical — fine-tuning the transformer models takes a long time without acceleration.
- **Hardware used:** We used an **NVIDIA H100 GPU**. End-to-end training time was approximately **1 hour 30 minutes** on this hardware. Smaller GPUs (e.g. T4, A100) will work but will take longer.
- **Reproducibility settings (final run):** `DEV_MODE=False`, `TRAIN_EPOCHS=5`, `MIN_POSITIVES=20`, `seed=42`, full CUAD (510 contracts → 38 retained clause types after the minimum-positive filter). These are the values committed in Cell 23 of the notebook and the basis of the macro-F1 numbers reported in Section 4.
- **Saving trained models:** During training we mounted **Google Drive** from within Colab and gave the notebook permission to write to it. All trained model checkpoints were saved to Google Drive so they persist after the Colab runtime is recycled. If you re-run the notebook yourself, you will be prompted to authorize Google Drive access in the same way.
- **Note on saved execution counts:** The notebook was finalized after a sequence of partial reruns; some setup cells (Cell 1 environment bootstrap, Cell 2 `load_cuad`, Cell 23 chunking, Cell 25 force-retrain, Cell 26 checkpoint gate, Cell 44 deploy) display `execution_count = None` even though the cells with non-trivial outputs (EDA, training, evaluation) are stamped 5–26 in run order. The saved cell outputs reflect the canonical end-to-end run; the unstamped cells are infrastructure cells that were skipped on the final pass to avoid re-tokenizing or re-downloading.
- **Selecting the best model:** After training completes, download **two files** from your Google Drive checkpoint folder:
  1. The best-performing checkpoint (e.g. `Legal-BERT_(CUAD).pt`) → place it inside the [models/](models/) folder. The demo expects [models/Legal-BERT_(CUAD).pt](models/Legal-BERT_(CUAD).pt) by default; if you save under a different name, update the model path accordingly.
  2. **`s4_outputs.pkl`** → place it at the project root, next to [classifier.py](classifier.py). This file carries the per-clause sigmoid thresholds tuned on the validation set in Section 4 (e.g. Document Name = 0.05, Cap On Liability = 0.90). Without it, the deployed app falls back to a flat `t = 0.5` for every clause and does **not** reproduce the macro-F1 reported in the notebook — it will over-flag most clauses and under-detect Document Name / Parties.

---

## 2. Running the Demo Website Locally

The demo loads the best-performing model from training — in our case **Legal-BERT** — and uses it to classify clauses from uploaded legal documents.

### Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements_web.txt
   ```
   *(For training, the notebook self-installs its own dependencies in its first cell. If you prefer to install them locally instead, run `pip install -r requirements.txt`.)*
2. Make sure the trained model file (e.g. `Legal-BERT_(CUAD).pt`) is present inside [models/](models/).
3. Create a `.env` file in the project root and add your **OpenAI API key**:
   ```
   OPENAI_API_KEY=sk-...
   ```
   This is required because the demo makes an LLM call to **OpenAI GPT-4o-mini** to produce the supporting excerpts and plain-English summaries. If you want to recreate the exact demo we built, you must supply your own OpenAI API key here.

### Launch

Run the dashboard script — this is the entry point for the local demo:

```bash
python run_dashboard.py
```

This starts the web app locally and opens it in your browser.

### Using the Demo

- **Upload any legal document** through the web interface. The demo accepts standard contract-style documents.
- The trained Legal-BERT model then **classifies each clause** in the document into high / medium / low risk.
- For each classified clause, an **LLM call to GPT-4o-mini** is made to:
  - return the **supporting excerpt(s)** from the original document, and
  - generate a **plain-English summary** of that excerpt so a non-legal reader can understand what it actually means.
- The classifications, excerpts, and summaries are displayed back in the dashboard.

### A Note on the Training Data

The model was trained on the **CUAD (Contract Understanding Atticus Dataset)** — 510 expert-annotated commercial contracts sourced from EDGAR public filings, covering 41 clause types. The classifier is not restricted to that corpus — it is fully capable of working on **any legal document** you upload, so feel free to try it on real contracts of your own.

---

## 3. What You Need to Reproduce This End-to-End

To run the project exactly as intended as a third party, you will need:

- A Google account with access to **Google Colab** and **Google Drive** (for training).
- A GPU runtime in Colab (H100 recommended; A100/T4 also workable with longer training time).
- Python 3 locally, with the packages from [requirements_web.txt](requirements_web.txt) installed.
- The trained model checkpoint downloaded from Drive and placed in [models/](models/).
- An **OpenAI API key** with access to **GPT-4o-mini**, stored in a local `.env` file as `OPENAI_API_KEY`.
- A modern web browser to interact with the dashboard launched by [run_dashboard.py](run_dashboard.py).

Once those pieces are in place, the workflow is: **train in Colab → save to Drive → drop model into `/models` → run `run_dashboard.py` → upload a document in the browser**.
