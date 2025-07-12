FROM python:3.10-slim
WORKDIR /app
COPY . /app

RUN apt-get update -y

RUN apt-get install -y antiword libxml2-utils poppler-utils libreoffice

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]