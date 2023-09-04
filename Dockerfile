FROM python:3.10.6-buster

COPY cte/api /cte/api
COPY cte/ml_logic /cte/ml_logic
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .

WORKDIR /cte/ml_logic

CMD uvicorn cte.api.api:app --host 0.0.0.0 --port $PORT
