#!/usr/bin/env sh
echo "Starting server"
cd /opt/app
/usr/bin/uvicorn mnapi.main:app --reload --host 0.0.0.0 --port 8000
