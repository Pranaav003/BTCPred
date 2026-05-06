import os

workers = 1
# Fewer concurrent request threads lowers peak RSS on ~512MB plans (pandas + skinny rows + overlaps).
threads = 2
worker_class = "gthread"
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
# Allow ~35s scheduler polls + headroom; large exports use streaming—keep workers from hanging hours.
timeout = 60
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
accesslog = "-"
errorlog = "-"
loglevel = "info"
preload_app = False
