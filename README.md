# Production-Ready NLP Sarcasm Detection

## Overview

This repository contains an end-to-end Natural Language Processing (NLP) pipeline for detecting sarcasm in text. The system utilizes a fine-tuned **DistilBERT** architecture (`distilbert-base-uncased`) and is deployed as a RESTful service using the **FastAPI** framework.

The project demonstrates the transition from experimental research in Jupyter Notebooks to a structured, modular, and production-ready software architecture.

## Project Structure

```text
Sarcasm Analysis NLP/
│
├── data/               # Local only: Store datasets here (e.g., Sarcasm_Headlines_Dataset_v2.json)
├── models/             # Local only: Store fine-tuned model weights and checkpoints
├── notebooks/          # Exploratory Data Analysis (EDA) and training experiments
├── src/                # Core production source code
│   ├── config.py       # Global configuration and path management
│   ├── inference.py    # OOP-based prediction engine
│   └── main.py         # FastAPI web application layer
├── requirements.txt    # List of required Python libraries
└── README.md           # Project documentation and setup guide

```

---

## Model Performance

The model was evaluated on a held-out test set of news headlines:

| Metric | Value |
| --- | --- |
| **Accuracy** | 92.24% |
| **F1-Score** | 0.9192 |

> **Note on Domain Shift**: While the model exhibits high performance on news headlines, its predictive accuracy may vary on casual or conversational text due to the specific linguistic characteristics of the training dataset.

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/subki72/NLP-Sarcasm-Detection.git
cd NLP-Sarcasm-Detection

```
> **Note**: Please ensure you create the data/ and models/ directories locally and place the dataset/trained weights inside them as specified in the project structure

### 2. Environment Management

It is recommended to use a dedicated environment to manage dependencies:

```bash
conda create -n sarcasm_detect python=3.10
conda activate sarcasm_detect

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

---

## Running the Application

To launch the FastAPI server locally, execute the following command from the project root:

```bash
uvicorn src.main:app --reload

```

The server will initialize at `http://127.0.0.1:8000`.

---

## Usage

### Interactive API Documentation

FastAPI provides an automated Swagger UI for testing. Access it at:
`http://127.0.0.1:8000/docs`

### Example Request (cURL)

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Man Finally Finishes Reading Terms and Conditions Agreement"
}'

```

### Expected Response

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
