FROM python:3.10.6-buster

COPY cte cte
COPY saved_models saved_models
# COPY cte/ml_logic /cte/ml_logic

COPY setup.py setup.py
COPY docker_requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .

# WORKDIR /model_deployment

CMD uvicorn cte.api.api:app --host 0.0.0.0 --port $PORT
