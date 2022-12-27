FROM python:3.8.6-slim-buster

COPY api /api
COPY models/model.joblib models/model.joblib
COPY titanic /titanic
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port 80
