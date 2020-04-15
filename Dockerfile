FROM python:3.8-alpine

RUN pip3 install pydub

COPY . /app

CMD python /app/main.py