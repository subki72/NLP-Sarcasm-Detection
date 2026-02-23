```markdown
# Production-Ready Sarcasm Detection API

## Overview
This repository contains an end-to-end Natural Language Processing (NLP) pipeline for detecting sarcasm in text. The model is fine-tuned on the `distilbert-base-uncased` architecture and deployed as a RESTful API using **FastAPI**.

This project demonstrates the transition from a research environment (Jupyter Notebook) to a structured, production-ready Software Engineering architecture.

## Project Structure
```text
Sarcasm Analysis NLP/
│
├── data/                      # Raw datasets (JSON)
├── models/                    # Stored model weights and checkpoints
├── notebooks/                 # Jupyter notebooks for EDA and initial training
├── src/                       # Production source code
│   ├── config.py              # Centralized configuration and paths
│   ├── inference.py           # OOP-based model inference class
│   └── main.py                # FastAPI application
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

```

## Model Performance
The model was evaluated on a test set of news headlines:

* **Accuracy:** 92.24%
* **F1-Score:** 0.9192

*Note on Domain Shift: The model performs exceptionally well on news-style text but may struggle with casual conversational sarcasm due to the nature of the training dataset.*


## Installation & Setup
1. **Clone the repository:**
```bash
git clone [https://github.com/yourusername/sarcasm-detection-api.git](https://github.com/yourusername/sarcasm-detection-api.git)
cd sarcasm-detection-api
```

2. **Create a virtual environment (Conda/Venv):**
```bash
conda create -n sarcasm_detect python=3.10
conda activate sarcasm_detect
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Running the API
To start the FastAPI server locally, run:

```bash
uvicorn src.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.


## Usage
### Swagger UI (Interactive Testing)
Navigate to `http://127.0.0.1:8000/docs` in your browser to interactively test the `/predict` endpoint.

### Example API Request (cURL)
```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Man Finally Finishes Reading Terms and Conditions Agreement"
}'

```

### Example Response
```json
{
  "status": "success",
  "data": {
    "text": "Man Finally Finishes Reading Terms and Conditions Agreement",
    "prediction": "SARCASTIC",
    "confidence": 99.48,
    "is_sarcastic": true
  }
}

```

