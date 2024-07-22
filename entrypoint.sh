#!/bin/bash

# Apply database migrations
python manage.py migrate

# Start the Gunicorn server
exec gunicorn myapp.wsgi:application --bind 0.0.0.0:8006 --timeout 300 --workers 4
