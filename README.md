# Kalshi Signal

A paper trading signal dashboard for Kalshi BTC 15-minute markets.
Uses ensemble agreement between market probability and an independent XGBoost model to generate high-confidence trading signals.

## Setup

1. Install dependencies:
   `pip install -r requirements.txt`

2. Copy environment file:
   `cp .env.example .env`

3. Train the model (requires `kalshi_btc15m_dataset_30k.csv`):
   `python train_raw_model.py`

4. Run locally:
   `python run.py`

5. Run with Gunicorn:
   `gunicorn "app:create_app()" -c gunicorn.conf.py`

## Environment consistency

Always train and serve with the same Python environment.

- Check runtime interpreter and sklearn:
  - `python -c "import sys, sklearn; print(sys.executable); print(sklearn.__version__)"`
- Run that command in both:
  - the terminal where you start Flask (`python run.py`)
  - the terminal where you train (`python train_raw_model.py`)

The sklearn version must match in both environments. If it differs, retrain the model using the Flask runtime environment.

## Deploy to Render

1. Create new Web Service
2. Connect your repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn "app:create_app()" -c gunicorn.conf.py`
5. Add environment variables from `.env.example`
6. Note: `.pkl` model file must be committed or built at deploy time

## Project Structure

```text
KalshiPaperApp/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── db_helpers.py
│   ├── extensions.py
│   ├── feature_engineering.py
│   ├── kalshi_client.py
│   ├── model_loader.py
│   ├── models.py
│   ├── resolver.py
│   ├── scheduler.py
│   ├── signal_engine.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   └── dashboard.py
│   ├── static/
│   │   ├── css/
│   │   │   └── main.css
│   │   └── js/
│   │       ├── analytics.js
│   │       ├── main.js
│   │       └── settings.js
│   └── templates/
│       ├── analytics.html
│       ├── base.html
│       ├── dashboard.html
│       └── settings.html
├── .env.example
├── .gitignore
├── Procfile
├── gunicorn.conf.py
├── requirements.txt
├── run.py
└── train_raw_model.py
```

## Signal Logic

Generates a PAPER BUY YES signal when:
- `p_market >= YES_CUTOFF` (default 0.65)
- `p_raw >= YES_CUTOFF` (default 0.65)
- `seconds_to_close` is within the configured time window

Based on ensemble disagreement analysis showing strong YES agreement between market probability and raw-feature model produces 91-95% accuracy historically.
