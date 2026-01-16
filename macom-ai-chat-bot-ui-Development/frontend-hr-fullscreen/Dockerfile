FROM python:3.11-slim

RUN apt-get update && apt-get upgrade -y
RUN apt-get install --no-install-recommends -y openssl


WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]