<div align="center">

# BERT Sentiment Analysis API

Real-time sentiment analysis powered by a fine-tuned transformer model and served via **FastAPI**.

<p>
<img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
</p>

</div>

---

## Overview

A production-style NLP microservice that classifies the sentiment of any input text (positive / neutral / negative) in real time. Built around **Cardiff NLP's `twitter-roberta-base-sentiment-latest`** model from Hugging Face and served behind a lightweight **FastAPI** layer with a `POST /predict` endpoint.

The repo also ships with a **Jupyter notebook** that walks through the same model interactively — single-text analysis, batch inference, and detailed per-class confidence scores.

## Features

- `POST /predict` endpoint returning sentiment label + confidence score
- Interactive Jupyter notebook for exploring the model (`notebooks/`)
- Clean `BERTSentimentAnalyzer` class with single-text, batch, and detailed-analysis modes
- GPU-aware (uses CUDA if available, falls back to CPU)
- Tested with Postman; low-latency responses on CPU

## Tech Stack

| Layer | Tools |
|---|---|
| Model | `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa) |
| ML framework | PyTorch, Hugging Face Transformers |
| API | FastAPI + Uvicorn |
| Tooling | uv, pydantic |

## Project Structure

```
BERT-Sentiment-API/
├── app.py                        # FastAPI server
├── main.py                       # CLI entry point
├── src/
│   └── sentiment_analyzer.py     # BERTSentimentAnalyzer class
├── notebooks/
│   └── text_classification_interactive.ipynb
├── examples/
│   └── test_api.py               # Sample client for the /predict endpoint
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/sravankaviti/BERT-Sentiment-API.git
cd BERT-Sentiment-API
```

### 2. Install dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or with plain pip:

```bash
pip install -r requirements.txt
```

### 3. Run the API

```bash
uvicorn app:app --reload
```

Server starts at `http://127.0.0.1:8000`. Interactive Swagger docs: `http://127.0.0.1:8000/docs`.

## Usage

### cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This model is absolutely fantastic!"}'
```

### Python client

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "I love how clean this API is."}
)
print(response.json())
# {'input_text': '...', 'label': 'positive', 'score': 0.987}
```

### CLI

```bash
python main.py "The service was slow and unresponsive."
```

## API Reference

### `GET /`

Health check.

```json
{ "message": "Sentiment API is running successfully!" }
```

### `POST /predict`

**Request body**

```json
{ "text": "string" }
```

**Response**

```json
{
  "input_text": "The food was great.",
  "label": "positive",
  "score": 0.978
}
```

## Notebook

Open `notebooks/text_classification_interactive.ipynb` to explore the model interactively. The notebook covers:

- Loading the model with SSL-safe fallbacks
- Single-text sentiment analysis
- Batch processing for multiple inputs
- Detailed analysis returning confidence across all classes

## Model Notes

This project uses **`cardiffnlp/twitter-roberta-base-sentiment-latest`** — a RoBERTa model fine-tuned on ~124M tweets and labeled for three-class sentiment (positive, negative, neutral). It generalizes well to short social-style text and product reviews.

Swap in any other Hugging Face sentiment model by changing `model_name` in `src/sentiment_analyzer.py`.

## License

MIT — free to use and modify.

## Author

**Kaviti Sravan** — B.Tech CSE @ Vellore Institute of Technology
[LinkedIn](https://www.linkedin.com/in/kaviti-sravan-15a6442b0/) · [GitHub](https://github.com/sravankaviti)
