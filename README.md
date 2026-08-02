# 🌾 Smart Harvest AI

AI-powered crop-yield prediction and plant-disease screening for Indian agriculture.

## Features

- XGBoost crop-yield prediction using the trained feature set and time-based evaluation.
- TensorFlow Lite plant-disease classification for PlantVillage leaf images.
- Pinecone RAG farming assistant using local MiniLM embeddings and Groq Llama 3.3.
- SQLAlchemy persistence for yield predictions, disease predictions, and successful chatbot interactions.
- SQLite support for local development and managed PostgreSQL support for production deployment.
- Server-rendered web UI at `/yield` and `/disease`, with an accessible floating chatbot on every page.
- REST APIs for health, model options, yield prediction, disease prediction, persisted history, chat, weather, and regional defaults.

## Model performance

### Crop-yield regression

| Model | Train R² | Test R² | MAE | RMSE |
|---|---:|---:|---:|---:|
| XGBoost regression pipeline | 89.29% | 82.68% | 1.3817 t/ha | 2.9512 t/ha |

The production yield model is an `XGBRegressor` with one-hot categorical preprocessing and engineered numeric features. Metrics use a time-based split: 1997–2015 for training and 2016–2020 for testing. The train/test R² gap is 6.61 percentage points. `Production` is excluded because it would introduce target leakage. MAE is an average error estimate, not a guaranteed prediction interval.

### Plant-disease classification

| Model | Accuracy | Weighted precision | Weighted recall | Weighted F1 | Validation images |
|---|---:|---:|---:|---:|---:|
| MobileNetV3Small transfer learning → TensorFlow Lite | 81.58% | 83.70% | 81.58% | 81.62% | 4,127 |

These results were measured by running the deployed TensorFlow Lite model on the seeded 20% PlantVillage validation split (seed 123) across 15 classes. Macro precision, recall, and F1 are 79.78%, 83.06%, and 80.26%, respectively. See the complete per-class report and raw confusion-matrix values in [`models/disease_metrics.json`](models/disease_metrics.json), the rendered matrix in [`models/disease_confusion_matrix.png`](models/disease_confusion_matrix.png), and the reproducible evaluator in [`backend/ml/evaluate_disease_model.py`](backend/ml/evaluate_disease_model.py).

## Project structure

```text
smart_harvest.ai/
├── backend/
│   ├── app_v2.py
│   └── frontend.py
├── routes/
│   ├── __init__.py
│   └── chatbot.py
├── utils/
│   ├── __init__.py
│   └── rag_helper.py
├── static/
│   ├── css/chatbot-widget.css
│   └── js/chatbot-widget.js
├── farming_docs/
│   ├── all_farming_topics.json
│   └── all_diseases.json
├── frontend/
│   ├── base.html
│   ├── home.html
│   ├── yield.html
│   ├── disease.html
│   └── dashboard.html
├── models/                         # deployment inference artifacts only
│   ├── yield_model/
│   ├── disease_model.tflite
│   └── class_labels.json
├── notebooks/                      # yield and disease-model reference notebooks
│   ├── crop_yield_eda_feature_engineering_xgboost.ipynb
│   └── plant_disease_model_reference.ipynb
├── crop_yield.csv                  # compact yield retraining dataset
├── ingest_knowledge_base.py        # one-time local Pinecone ingestion
├── tests/test_app.py
├── requirements.txt
└── Procfile
```

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 setup.py
python3 backend/app_v2.py
```

Open `http://localhost:5001`. For production, use the existing [`Procfile`](Procfile) command with Gunicorn.

### Database configuration

The application persists yield predictions, disease predictions, and successful chatbot interactions. It uses SQLAlchemy and creates the schema automatically on startup. This supports local development with SQLite and production deployment with managed PostgreSQL:

- Local: omit `DATABASE_URL`; the database is created at `instance/smart_harvest.db`.
- Render/production: provision a managed PostgreSQL database and set its `DATABASE_URL` environment variable. The app converts Render's `postgres://` format automatically.
- Do not use SQLite for production because ephemeral service filesystems can lose data during redeploys or restarts.
- Existing tables are preserved; schema creation is idempotent. Use the `/api/health` response to verify `database_connected`.

### Render deployment

#### 1. Push the application

Commit all source changes, including `utils/database.py`, `requirements.txt`, and this README, then push the branch connected to Render. Never commit `.env`, database files, or real credentials.

#### 2. Provision PostgreSQL

Create a managed PostgreSQL database in Render. In the web service's environment settings, add `DATABASE_URL` using the database's **Internal Database URL**. The application accepts both `postgres://` and `postgresql://` URLs and configures the Psycopg driver automatically.

#### 3. Configure the web service

Create or update the Python web service with:

```text
Build command: pip install -r requirements.txt
Start command:  use the repository Procfile
Health check:   /api/health
```

Configure these environment variables in Render:

| Variable | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | Yes for persistent production data | Render PostgreSQL Internal Database URL |
| `SECRET_KEY` | Yes | Long, random Flask session-signing secret |
| `GROQ_API_KEY` | Yes for chatbot | Groq API credential |
| `PINECONE_API_KEY` | Yes for chatbot | Pinecone API credential |
| `PINECONE_INDEX_NAME` | Yes for chatbot | Name of the ingested 384-dimensional index |

Do not upload `.env` or place secrets in GitHub. The Procfile intentionally runs one threaded worker because XGBoost, TensorFlow Lite, and the lazily loaded embedding model are memory-heavy. Request recycling limits long-term native-library memory growth, while the 180-second timeout allows for a cold first chatbot request.

#### 4. Verify the deployment

After deployment, request `/api/health`. A correctly connected production database reports:

```json
{
  "status": "ok",
  "database_connected": true,
  "database_backend": "postgresql"
}
```

The response also reports yield and disease model availability. Submit a yield prediction and then request `/api/recent?limit=1` to verify that the record was persisted. Tables are created automatically on startup; no separate initialization command is needed for the current schema.

> **Important:** If `DATABASE_URL` is missing, the service falls back to SQLite and can start successfully, but records can be lost whenever Render replaces or restarts the service instance. PostgreSQL is therefore required for durable production storage.

### Chatbot configuration and ingestion

Create a Pinecone index with **384 dimensions** and the **cosine** metric. Configure these environment variables locally and in Render (never commit their values):

```bash
export PINECONE_API_KEY="your-key"
export PINECONE_INDEX_NAME="smart-harvest-knowledge"
export GROQ_API_KEY="your-key"
```

Run the ingestion once on your local machine after creating the index:

```bash
python3 ingest_knowledge_base.py
```

The ingestion script preserves each JSON entry as one labeled chunk, generates normalized `sentence-transformers/all-MiniLM-L6-v2` embeddings, and uploads text and metadata to Pinecone. Do **not** add this command to Render's start or build command unless you intentionally want to re-ingest the index.

> **Render memory note:** `tensorflow-cpu` and `sentence-transformers` (which brings PyTorch) together can exceed a 512 MB instance limit. Use at least a paid Starter instance for all features, or split disease inference and chatbot retrieval into separate services. The embedding model is CPU-only and loaded lazily, so normal Flask startup does not load PyTorch immediately.

The deployed yield model is the XGBoost pipeline trained from [`crop_yield.csv`](crop_yield.csv) and stored under `models/yield_model/`. The candidate notebook [`notebooks/crop_yield_eda_feature_engineering_xgboost.ipynb`](notebooks/crop_yield_eda_feature_engineering_xgboost.ipynb) is the source of truth for retraining. `Production` is excluded as target leakage.

## Data sources and attribution

### Crop-yield data

The file [`crop_yield.csv`](crop_yield.csv) is attributed to the following dataset published on Mendeley Data:

> V, Ramesh; P, Kumaresan (2025), “Stacked Ensemble Model for Accurate Crop Yield Prediction Using Machine Learning Techniques”, Mendeley Data, V2, doi: [10.17632/ncw2vbcgnk.2](https://doi.org/10.17632/ncw2vbcgnk.2).

According to the publisher’s description, the dataset contains 19,689 records covering 27 Indian states and 3 Union Territories from 1997–2020, with 55 crops and the fields Crop, Season, Crop_Year, State, Annual_Rainfall, Area, Production, Pesticide, Fertilizer, and Yield. The dataset is listed under the [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/). This project uses the publisher-provided data for research and prediction, credits the authors, and excludes `Production` from the yield model because it can create target leakage when predicting `Yield`.

The dataset’s publisher and DOI provide an authoritative provenance record. This README does not independently certify the accuracy or completeness of the measurements; users should consult the original record, version, licence, and citation requirements before redistribution or commercial use.

### Plant-disease images

The disease model was trained from the PlantVillage image dataset using the transfer-learning pipeline in [`backend/ml/train_disease_model.py`](backend/ml/train_disease_model.py). It uses an ImageNet-pretrained MobileNetV3Small backbone, global average pooling, a 128-unit dense layer, dropout, and a 15-class softmax output. Training uses a seeded 80/20 training-validation split, class weighting, image augmentation, frozen-backbone training followed by fine-tuning of the final 40 backbone layers, and 192 × 192 RGB inputs. The deployed model covers healthy and diseased pepper, potato, and tomato leaves; the exact output mapping is stored in [`models/class_labels.json`](models/class_labels.json). The notebook [`notebooks/plant_disease_model_reference.ipynb`](notebooks/plant_disease_model_reference.ipynb) provides a reference view of the architecture, deployed TensorFlow Lite input/output contract, labels, validation metrics, confusion matrix, and single-image inference flow.

The deployed artifact was evaluated without retraining on the reproducible 20% validation split. It achieved 81.58% accuracy, 83.70% weighted precision, 81.58% weighted recall, and 81.62% weighted F1 across 4,127 images. The repository includes the complete evaluation in [`models/disease_metrics.json`](models/disease_metrics.json), the confusion matrix in [`models/disease_confusion_matrix.png`](models/disease_confusion_matrix.png), and the evaluation script in [`backend/ml/evaluate_disease_model.py`](backend/ml/evaluate_disease_model.py). The large source image directory remains excluded from deployment. The original PlantVillage publication is:

> Hughes, D. P., & Salathé, M. (2015), “An open access repository of images on plant health to enable the development of mobile disease diagnostics”, arXiv:1511.08060. [doi:10.48550/arXiv.1511.08060](https://doi.org/10.48550/arXiv.1511.08060).

See the [PlantVillage project repository](https://github.com/spMohanty/PlantVillage-Dataset) and its stated dataset terms before obtaining or redistributing the images. Disease predictions are screening guidance and are not a substitute for expert diagnosis.

### Weather data

Live rainfall and forecast values are retrieved at runtime from [Open-Meteo](https://open-meteo.com/) and its [API documentation](https://open-meteo.com/en/docs). Normal Mode uses the selected state’s centroid for a recent 30-day archive lookup, displays the observed monthly total, and annualizes it for the model because the yield model expects annual rainfall. Open-Meteo attribution and applicable provider data licences should be retained when displaying or redistributing weather results.

### Farming and disease knowledge base

The chatbot knowledge base is the project-maintained content in [`farming_docs/`](farming_docs). It contains structured farming-topic and disease summaries used for Pinecone retrieval; these files are not presented as a replacement for official agricultural guidance. The application uses [Pinecone](https://www.pinecone.io/) for vector retrieval, `all-MiniLM-L6-v2` for local embeddings, and [Groq](https://groq.com/) for response generation. These are service/model providers rather than additional agricultural source datasets.

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Yield, disease model, and database status |
| GET | `/api/options` | Dataset states, crops, and seasons |
| POST | `/api/predict/yield` | JSON yield prediction |
| POST | `/api/predict/disease` | Multipart upload with field `image` |
| POST | `/chat` | RAG farming assistant; JSON message and optional history |
| GET | `/api/state-rainfall?state=` | Latest observed 30-day state rainfall and annualized model input |
| GET | `/api/weather?lat=&lon=` | Open-Meteo seven-day rainfall |
| GET | `/api/location-data?lat=&lon=` | Regional climate and soil defaults |
| GET | `/api/recent` | Persisted recent predictions |

### Yield request

```bash
curl -X POST http://localhost:5001/api/predict/yield \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "karnataka",
    "crop": "Rice",
    "crop_year": 2020,
    "season": "Kharif",
    "area": 10,
    "annual_rainfall": 1200,
    "fertilizer": 1444.9,
    "pesticide": 2.7
  }'
```

The response includes `raw_prediction`, the non-negative display prediction, and an MAE-based reference range. Unknown or missing categorical values return HTTP 400 instead of silently mapping to a fallback category. The current model test R² is approximately 0.827 and MAE is approximately 1.382 yield units.

Farmer Mode fetches the selected state's latest observed 30-day rainfall. The UI shows that monthly total and annualizes it before prediction because the model was trained with an annual-rainfall feature. If the weather service is unavailable, the dataset median of 1,247.6 mm is used.

### Disease request

```bash
curl -X POST http://localhost:5001/api/predict/disease \
  -F image=@/path/to/leaf.jpg
```

The service accepts PNG/JPEG uploads up to 8 MB, resizes images to 192×192 RGB, and returns the primary prediction plus the top three classes. Results are screening guidance only and should be confirmed by an agronomist.

### Chat request

```bash
curl -X POST http://localhost:5001/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"How do I manage tomato early blight?","history":[]}'
```

A successful response contains `reply` and deduplicated `sources`. The server limits history to the latest 12 valid user/assistant messages, retrieves the top three Pinecone passages, and applies a 30-second Groq read timeout. Missing configuration and upstream failures return structured JSON errors without stopping Flask.

The global widget is loaded by these shared-template includes:

```html
<link rel="stylesheet" href="{{ url_for('static', filename='css/chatbot-widget.css') }}"><script src="{{ url_for('static', filename='js/chatbot-widget.js') }}" defer></script>
```

## Testing

```bash
python3 -m compileall -q backend routes utils ingest_knowledge_base.py setup.py tests
pytest -q tests/test_app.py
```

Current validation result: **9 tests passed**. The suite covers application health, database connectivity, persisted yield history, chatbot persistence, model endpoints, validation, and upstream-service failure handling.

## Production readiness checklist

- [x] Gunicorn production command is defined in `Procfile`.
- [x] SQLAlchemy initializes the database schema safely at startup.
- [x] SQLite is available for zero-configuration local development.
- [x] PostgreSQL and Psycopg are included for Render deployment.
- [x] Predictions and successful chatbot interactions are persisted.
- [x] Database status is exposed through `/api/health`.
- [x] Local database files and `.env` are excluded from Git.
- [ ] A managed PostgreSQL database must be created in the target Render account.
- [ ] Production environment variables must be added in the Render dashboard.
- [ ] `/api/health` and `/api/recent` must be checked after the final deployment.

## Technology

Python, Flask, Gunicorn, SQLAlchemy, PostgreSQL, SQLite, XGBoost, scikit-learn, pandas, NumPy, TensorFlow Lite, Pillow, Pinecone, sentence-transformers, Groq, Jinja2, vanilla JavaScript, and Open-Meteo.
