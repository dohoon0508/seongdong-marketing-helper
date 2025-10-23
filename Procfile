web: gunicorn app:app \
  --worker-class gthread \
  --workers 1 --threads 2 \
  --timeout 60 --keep-alive 5 \
  --log-level info
