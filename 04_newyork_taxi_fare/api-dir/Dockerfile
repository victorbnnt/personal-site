FROM python:3.8.6-slim-buster

COPY api /api
COPY models/model.joblib models/model.joblib
COPY NYCTaxiFare /NYCTaxiFare
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

CMD uvicorn api.fast:app --host 0.0.0.0 --port 80
