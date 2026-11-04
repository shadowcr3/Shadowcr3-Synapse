# Synapse server image. NOT built or run in the environment this release was prepared in; see docs/OPERATIONS.md.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 synapse
WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY neural_network ./neural_network
COPY training ./training
COPY app ./app
COPY frontend ./frontend
COPY configs ./configs
COPY models/synapse_v1.json models/synapse_v1.npz ./models/
COPY data/mnist ./data/mnist
RUN mkdir -p /app/models/candidates && chown -R synapse:synapse /app/models /app/data

USER synapse
EXPOSE 5000
# User drawings (data/) and candidate models (models/) live in volumes so they survive restarts.
VOLUME ["/app/data", "/app/models"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=2).status == 200 else 1)"
# Bind inside the container; publish only on loopback unless you add authentication in front:
#   docker run -p 127.0.0.1:5000:5000 synapse            (or add --read-only to the command for inference only)
CMD ["python", "-m", "app.server", "--host", "0.0.0.0", "--port", "5000"]
