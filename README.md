# Weather Prediction (Flask + LSTM)

End‑to‑end weather temperature prediction system built around:
- Open‑Meteo API (hourly data) for ingestion
- Univariate LSTM model to forecast temperature (target: `temperature_2m`)
- Flask web app (UI + REST API) for inference
- Interactive Chart.js line chart of the previous 10 hours
- Reproducible artifacts and Dockerfile for deployment

---

## Table of Contents
- Overview
- How It Works (Architecture)
- Features
- Project Structure
- Requirements
- Setup (Windows/macOS/Linux)
- Configuration
- Training Pipeline
- Running the Web App
- REST API Usage
- Frontend Chart (Last 10 Hours)
- Artifacts
- Retraining the Model
- Docker
- Troubleshooting
- FAQ
- Acknowledgements
- License

---

## Overview
This project forecasts air temperature using an LSTM neural network trained on hourly weather data from the Open‑Meteo API. The Flask application exposes both a browser UI and a REST API to make predictions for any latitude/longitude you provide. The result page displays the prediction and a landscape line chart of the last 10 completed hourly temperature readings from the same location.

## How It Works (Architecture)
- Ingestion: Pull hourly observations from Open‑Meteo for selected variables (including `temperature_2m`).
- Validation: Ensure the ingested data meets the expected schema (columns and types as defined in `schema.yml`).
- Transformation: Prepare data for modeling (cleaning, ordering, splitting). In the current setup, the LSTM is trained as a univariate forecaster on `temperature_2m` using MinMax scaling for that target only.
- Training: Build and train an LSTM model on sequences of scaled `temperature_2m` values.
- Evaluation: Produce metrics and save results to `artifacts/model_evaluation`.
- Serving: A Flask app loads the saved artifacts (model and scaler) and serves predictions + a chart of recent observations.

## Features
- Open‑Meteo hourly data ingestion with retry + caching
- Schema‑based validation
- LSTM univariate forecasting for `temperature_2m`
- Saved artifacts: `model.h5`, `scaler.pkl`
- Flask UI with lat/long input and a responsive Chart.js line chart
- REST API with `/health` and `/api/predict` endpoints
- Dockerfile for containerized runs

## Project Structure
```
Weather_prediction/
├─ app.py                       # Flask app (UI + REST API)
├─ main.py                      # Orchestrates the full ML pipeline
├─ requirements.txt             # Python dependencies
├─ params.yml                   # Model/pipeline hyperparameters
├─ schema.yml                   # Expected columns/types for validation
├─ config/
│  └─ config.yml                # Central config for paths & stages
├─ src/Predict_weather/
│  ├─ components/               # Ingestion, transformation, trainer, etc.
│  ├─ config/
│  ├─ constants/
│  ├─ entity/
│  ├─ pipeline/                 # Stage entrypoints
│  └─ utils/
├─ artifacts/                   # Outputs from each pipeline stage
│  ├─ data_ingestion/
│  ├─ data_transformation/
│  ├─ data_validation/
│  ├─ model_trainer/            # model.h5, scaler.pkl
│  └─ model_evaluation/
├─ templates/
│  ├─ index.html                # Input form
│  └─ result.html               # Prediction + last 10 hours chart
├─ Dockerfile
└─ README.md
```

## Requirements
- Python 3.9–3.11 (recommended)
- OS: Windows, macOS, or Linux
- Packages (see `requirements.txt`) and a compatible TensorFlow build



## Setup (Windows/macOS/Linux)
Choose either venv or conda.

venv (cross‑platform):
- python -m venv .venv
- Windows: .venv\Scripts\activate
- macOS/Linux: source .venv/bin/activate
- pip install --upgrade pip
- pip install -r requirements.txt
- pip install "numpy<2"
- pip install tensorflow-cpu==2.10.0

Conda (example):
- conda create -y -p ./venv python=3.10
- conda activate ./venv
- pip install -r requirements.txt
- pip install "numpy<2"
- pip install tensorflow-cpu==2.10.0

Verify environment:
- python -c "import sys, numpy as np, pandas as pd; print(sys.executable); print(np.__version__, pd.__version__)"

## Configuration
Key files:
- `config/config.yml`: global paths and stage toggles
- `params.yml`: hyperparameters (e.g., sequence length, epochs, batch size)
- `schema.yml`: expected columns and dtypes used by validation

Flask app knobs (in `app.py`):
- Open‑Meteo caching via requests‑cache (default TTL 5 min):
  - cache_session = requests_cache.CachedSession('.cache', expire_after=300)
- App port and host in `app.run(...)`.

## Training Pipeline
Run the full pipeline:
- python main.py

What it does:
1) Data Ingestion: fetches hourly data from Open‑Meteo.
2) Data Validation: checks columns/types against `schema.yml`.
3) Data Transformation: prepares series for modeling.
4) Model Training: trains an LSTM using only `temperature_2m` scaled via `MinMaxScaler` (univariate forecasting).
5) Model Evaluation: computes metrics and optionally logs artifacts.

Outputs:
- `artifacts/model_trainer/model.h5`
- `artifacts/model_trainer/scaler.pkl`
- evaluation files under `artifacts/model_evaluation/`

## Running the Web App
Start the server:
- python app.py

Open the UI:
- http://localhost:5000

Endpoints:
- UI home: `/`
- Health: `/health`
- Predict (form): `/predict` (POST)
- Predict (API): `/api/predict` (POST JSON)

Environment gotcha (Windows/Conda): if you created a conda env at `./venv` you can run:
- conda run -p .\venv python app.py

## REST API Usage
Request:
```
POST /api/predict
Content-Type: application/json
{
  "latitude": 28.6139,
  "longitude": 77.2090
}
```
Success response:
```
{
  "success": true,
  "prediction": "24.36°C",
  "latitude": 28.6139,
  "longitude": 77.209,
  "history": {
    "labels": ["2025-09-05T14:00:00Z", ...],
    "values": [28.9, 27.6, ...]
  }
}
```
PowerShell example:
- $json = '{"latitude":28.6139,"longitude":77.2090}'
- Invoke-RestMethod -Uri http://localhost:5000/api/predict -Method Post -ContentType 'application/json' -Body $json | ConvertTo-Json -Depth 5

curl example:
- curl -s -X POST http://localhost:5000/api/predict -H "Content-Type: application/json" -d '{"latitude":28.6139,"longitude":77.2090}'

## Frontend Chart (Last 10 Hours)
- The result page displays the previous 10 completed hourly values from Open‑Meteo for the selected coordinates.
- Labels are formatted for readability (HH:MM, day shown when the date changes).
- By default, the app maintains a 5‑minute cache of API responses to reduce calls. You can adjust `expire_after` in `app.py`.
- Open‑Meteo returns hourly values for completed hours; the last point represents the most recent completed hour, not the exact current minute.

## Artifacts
- `artifacts/data_ingestion/`: Raw/ingested data (if persisted)
- `artifacts/data_validation/`: Validation outputs
- `artifacts/data_transformation/`: Processed series/splits
- `artifacts/model_trainer/`: `model.h5` and `scaler.pkl`
- `artifacts/model_evaluation/`: Metrics, plots, and logs

## Retraining the Model
1) Optionally adjust hyperparameters in `params.yml` (e.g., sequence length, epochs, batch size).
2) Run `python main.py`.
3) Confirm that updated `model.h5` and `scaler.pkl` are present in `artifacts/model_trainer/`.
4) Restart the Flask app to load the new artifacts.

## Docker
Build and run locally:
- docker build -t weather-prediction .
- docker run -p 5000:5000 --name weather-prediction weather-prediction

Open http://localhost:5000.

Tips:
- If you retrain outside the container, copy artifacts into the image or mount a volume at runtime.
- Ensure your TensorFlow/NumPy versions are compatible inside the image.

## Troubleshooting
- NumPy 2.x error (e.g., "A module compiled using NumPy 1.x cannot be run in NumPy 2.x"):
  - pip install "numpy<2"
  - Verify which Python is running: `python -c "import sys; print(sys.executable)"`
- ImportError: TensorFlow not found when loading the model:
  - `pip install tensorflow-cpu==2.10.0` (or another build that matches your OS/Python)
- Wrong environment used by Flask:
  - If using conda with a path env: `conda run -p .\venv python app.py` (Windows) or `conda run -p ./venv python app.py` (POSIX)
- Port already in use:
  - Change port in `app.py` (e.g., `app.run(..., port=5001)`), or free port 5000
- Open‑Meteo rate limiting/stale data:
  - The app uses a 5‑minute cache. Decrease TTL for fresher data, but avoid hammering the API.
- Prediction errors about feature names:
  - The scaler is fit on `temperature_2m` only. Ensure you pass a single column shaped `(-1, 1)` when transforming.

## FAQ
- Why don’t I see the exact current minute on the chart?
  - Open‑Meteo provides hourly values for the last completed hour. The app plots these 10 completed hours.
- Can I forecast other variables?
  - Yes, but you’ll need to retrain the pipeline and adjust the app to target a different variable (and scaler).
- Can I use multivariate inputs?
  - This version trains a univariate LSTM. You can extend the pipeline to multivariate by scaling multiple features and changing the model input shape.

## Acknowledgements
- [Open‑Meteo](https://open-meteo.com/) for free weather data
- pandas, NumPy, scikit‑learn, TensorFlow/Keras, Flask, Chart.js

## License
This project is licensed under the terms of the repository `LICENSE` file.