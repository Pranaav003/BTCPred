import os

workers = 1
threads = 4
worker_class = "gthread"
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
accesslog = "-"
errorlog = "-"
loglevel = "info"
preload_app = False
