# Kalshi Signal (BTCPred)

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

## Deploy to Render (recommended)

1. Push code to GitHub (include `raw_feature_model.pkl`).
2. Go to [render.com](https://render.com), create account, connect GitHub repo.
3. Click **New +** -> **Web Service** -> select your repo.
4. Render auto-detects `render.yaml` and creates:
   - Web service (Flask app)
   - PostgreSQL database (free tier)
5. Add environment variables in Render dashboard:
   - `FLASK_ENV=production`
   - `SECRET_KEY=[random string]`
   - (`DATABASE_URL` is set automatically from the database)
6. Deploy. App runs at your Render URL.
7. To retrain model:
   - Run `python train_raw_model.py` locally
   - `git add raw_feature_model.pkl`
   - `git commit -m "retrain model"`
   - `git push`
   - Render auto-redeploys

## Important notes for cloud deployment

- Commit updated `raw_feature_model.pkl` after retraining so Render deploys the latest model.
- The free PostgreSQL on Render expires after 90 days.
- Upgrade to paid for long-term persistent database usage.
- The scheduler runs continuously on the cloud server.
- Dashboard is accessible from any browser via the Render URL.
- Paper trades and signals persist across restarts with PostgreSQL.

## Memory requirements

The free Render tier can run out of memory under sustained dashboard and scheduler load.
Recommended plans:
- Starter (~$7/mo) for dedicated web memory baseline.
- Standard (~$25/mo) for larger headroom and burst tolerance.

Main memory consumers:
- Growing signal/trade history in the database query layer.
- Kalshi API response payloads and in-process caches.
- Gunicorn workers/threads and chart-heavy dashboard polling.

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
