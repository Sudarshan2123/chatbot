FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get upgrade -y openssl && \
    rm -rf /var/lib/apt/lists/*


# RUN mkdir -p /app/chroma_DB /app/Sqllite/


WORKDIR /app

COPY . /app
RUN pip install --upgrade pip
RUN pip install  -r requirements.txt


EXPOSE 5050


ENV HOST=0.0.0.0
ENV PORT=5050



CMD ["python", "app.py"]
HEALTHCHECK CMD curl --fail http://localhost:5000/health || exit 1
