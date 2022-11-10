#!/bin/bash

cd /app
uvicorn --host=0.0.0.0 predict:app --reload --reload-dir=/app --port 9696