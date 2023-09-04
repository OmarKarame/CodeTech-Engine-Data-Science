FROM python:3.10.6-buster
COPY cte/api /cte/api
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn cte.api.api:app --host 0.0.0.0
