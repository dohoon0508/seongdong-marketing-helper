web: gunicorn app:app \
  --worker-class gthread \
  --workers 1 --threads 4 \
  --timeout 90 --keep-alive 10 \
  --max-requests 100 --max-requests-jitter 20 \
  --log-level info
