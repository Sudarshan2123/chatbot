FROM python:3.9-bookworm


RUN apt-get update && apt-get upgrade -y
RUN apt-get install --no-install-recommends -y openssl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY  . /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

