# Fintech Customer Support Copilot

A course project for **Recent areas of artificial intelligence**. It fine-tunes DistilBERT to classify banking support questions, compares it with a TF-IDF baseline, and connects the classifier to a small FAQ and transaction lookup prototype.

The support workflow uses nine authored help entries and 30 synthetic card and transfer records. It does not connect to bank accounts. The benchmark result measures intent classification; the smaller FAQ and workflow checks are reported separately.

## What it does

1. Predicts one of 77 BANKING77 intents with fine-tuned DistilBERT.
2. Retrieves relevant help text with TF-IDF.
3. Checks a supplied reference against a local SQLite database for four supported intents. Python validates the intent and reference, and the query is read-only.
4. When a supported reference is present, the app can send a constrained function call through NVIDIA NIM. A second NVIDIA NIM model can draft a reply from retrieved help text.

Supported lookup intents are `declined_card_payment`, `pending_transfer`, `transfer_not_received_by_recipient`, and `lost_or_stolen_card`. Other BANKING77 intents are classified, but do not have a complete support workflow.

The Streamlit app defaults to the fine-tuned checkpoint at [Hugging Face](https://huggingface.co/nmtan2001/fintech-support-distilbert). With `NVIDIA_API_KEY` set, the app uses `mistralai/mistral-large-4-1t-a52b` for function calls and `nvidia/nemotron-3.5-bolt-e4b` to draft replies. The Python code validates each function call and reference before querying SQLite. Without the key, the app uses a deterministic FAQ response.

## Results

BANKING77 contains 13,083 English questions across 77 intents. The project uses 9,002 training examples, 1,001 stratified validation examples, and the untouched official test set of 3,080 examples.

| Classifier | Test accuracy | Test Macro-F1 |
| --- | ---: | ---: |
| TF-IDF + Logistic Regression | 85.49% | 85.36% |
| Fine-tuned DistilBERT | 92.44% | 92.44% |

On the official test split, DistilBERT improves accuracy by 6.95 percentage points and Macro-F1 by 7.08 points. The run used seed 42.

The FAQ retriever reached Recall@1 of 90.9% and Recall@3 of 100% on 22 hand-labeled queries. A deterministic workflow check passed 37 of 48 authored cases (77.1%) with the baseline classifier and the LLM disabled. These small checks do not measure live LLM quality or performance on real customer traffic.

## Run the Streamlit app

Use Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

The app loads a local model from `outputs/distilbert/model/` when present. Otherwise, it downloads the checkpoint named by `HF_MODEL_ID`, which defaults to the published model above. If the DistilBERT dependencies are missing and `outputs/baseline/model.joblib` exists, the app uses the baseline classifier.

Set optional configuration in your environment or in `.streamlit/secrets.toml`:

```toml
HF_MODEL_ID = "nmtan2001/fintech-support-distilbert"
# HF_TOKEN = "your Hugging Face token, if needed"
# HF_MODEL_REVISION = "main"
# NVIDIA_API_KEY = "your NVIDIA API key"
```

Keep API keys out of Git. With no NVIDIA key, intent classification, FAQ retrieval, and the deterministic response still work.

## Run the API

Install the API dependencies in the same environment, then start FastAPI:

```bash
python -m pip install -e ".[api]"
export HF_MODEL_ID=nmtan2001/fintech-support-distilbert
uvicorn api:app --reload
```

The API reads `HF_MODEL_ID` from the environment. If you do not set it, the API needs a local classifier in `outputs/`.

The health endpoint is `GET /health`. Send a support question to `POST /support`:

```bash
curl -X POST http://127.0.0.1:8000/support \
  -H 'Content-Type: application/json' \
  -d '{"question":"Why was my TX-CARD-001 card payment declined?"}'
```

The request can also include `record_reference`, `min_confidence`, `use_llm`, and `llm_select_tool`. NVIDIA calls require `NVIDIA_API_KEY`; set `use_llm` to `true` to generate an answer, and set both `use_llm` and `llm_select_tool` to `true` to ask NVIDIA NIM to select the allowed lookup function.

## Train and evaluate

Download the official dataset and run the baseline:

```bash
python prepare_data.py
python baseline.py
```

Fine-tuning requires a CUDA GPU. The project workflow uses a P100 through `krun`:

```bash
krun run --gpu P100 --requirements requirements-kaggle.txt train.py
```

Run the retrieval and deterministic workflow evaluations with:

```bash
python evaluate_retrieval.py
python evaluate_smoke.py --model baseline
```

The test suite uses pytest:

```bash
python -m pip install -e ".[dev,api]" httpx
python -m pytest
```

## Project files

- `fintech_support/` contains classification, retrieval, grounding, and read-only lookup code.
- `app.py` runs the Streamlit app; `api.py` defines the FastAPI endpoints.
- `data/faq/demo_policies.json` contains the authored help entries.
- `data/transactions/demo_records.json` contains the synthetic records used to initialize SQLite.
- `data/evaluation/` contains the hand-labeled retrieval and workflow cases.
- `prepare_data.py`, `baseline.py`, and `train.py` prepare data and train the classifiers.
- `outputs/` stores generated models, metrics, and the local SQLite database. It is excluded from Git.

## Data and references

BANKING77 is available from the [PolyAI dataset page](https://huggingface.co/datasets/PolyAI/banking77) under CC BY 4.0. Casanueva et al. introduced the dataset in ["Efficient Intent Detection with Dual Sentence Encoders"](https://aclanthology.org/2020.nlp4convai-1.5/).

The FAQ text and lookup records are project fixtures. They are not policies or account data from a financial institution. The tool only reads the local SQLite database and cannot perform transfers, refunds, or card actions.

## Links

- Code: [github.com/nmtan2001/fintech-support-copilot](https://github.com/nmtan2001/fintech-support-copilot)
- Live app: [Streamlit demo](https://fintech-support-copilot-wrcv8mcbksylppx5oba8dm.streamlit.app)
