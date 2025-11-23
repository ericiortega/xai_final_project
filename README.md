# Explainable Sentiment Analyzer (BERT + SHAP)

An interactive Streamlit app for **interpretable sentiment analysis**.  
Users can:

- **Analyze a single review** and see a word-level SHAP heatmap.
- **Upload a CSV of many reviews** to score sentiment in bulk and download results.

Built for **AIPI 590: Emerging Trends in Explainable AI**.

---

## Demo Features

### Single review mode
- Input any sentence or review.
- Get:
  - Sentiment label (positive / neutral / negative)
  - Positivity score (0–100)
  - Model confidence
  - **SHAP explanation** showing which tokens push the prediction up or down
  - Top influential words plot

### Batch CSV mode
- Upload a CSV with **one row per review**.
- Select the column containing review text.
- Optionally skip a "note" first row.
- Score up to *N* rows for safety.
- Download a results CSV.

---

## Repository Structure

```
xai_final_project/
├── app/
│   ├── __init__.py
│   ├── app.py                # Streamlit app entrypoint
│   └── utils/                # (optional) helpers
├── models/
│   ├── __init__.py
│   ├── bert_model.py         # BERT sentiment model + scoring helpers
│   ├── shap_explainer.py     # SHAP token attribution
│   ├── parse_sentence.py     # (project-specific helper)
│   └── pca_sentiment.py      # (project-specific helper)
├── data/
│   └── sample_reviews.csv    # Example file for batch mode
├── requirements.txt
└── README.md
```

---

## Sample CSV Format

Your CSV should look like this:

- **Header row required**
- **One review per row**
- Contains at least one **text column** (example: `review`)

Example columns:
`review_id, review, product, rating`

See: `data/sample_reviews.csv` for a working example.

If your file includes a non‑data note row as the first *data* line (after header), enable  
**“Skip first data row”** in the UI.

---

## Setup (Run Locally)

### 1) Clone the repo
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2) Create a virtual environment (recommended)

**Mac/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

Your `requirements.txt` should include:

```txt
streamlit>=1.32
transformers>=4.38
torch>=2.1
shap>=0.45
numpy>=1.24
pandas>=2.0
plotly>=5.18
scikit-learn>=1.3
```

### 4) Run the Streamlit app
From the repository root:

```bash
streamlit run app/app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`).  
Open it in your browser.

---

## Notes / Troubleshooting

### Model download on first run
The first time you run the app, HuggingFace will download the BERT model.  
This may take a minute and requires internet access.

### If you get torch / tokenizer errors
Try upgrading pip + reinstalling:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Large CSVs
Batch mode limits rows for speed and safety.  
Use the slider to increase/decrease max rows analyzed.

---

## How SHAP Explanations Work (High level)

- The BERT model outputs a positive probability `p_pos`.
- SHAP computes **token-level contributions**:
  - **Green tokens** push sentiment positive.
  - **Orange tokens** push sentiment negative.
- The heatmap is rendered inline so users can interpret *why* the model predicted what it did.

---

## Authors

- **Eric Ortega**
- **Diya Mirji**

---