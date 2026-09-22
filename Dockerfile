# .dockerignore
# __pycache__/
# *.pyc
# .env
# .git/
# data/ (Don't copy huge datasets into the image!)
# .venv/
# venv/

# WHY DOCKER:
# Docker ensures our app runs the exact same way on your laptop, the testing server,
# and production. It packages the OS dependencies, Python runtime, and our code together.

# Use a slim, stable Python base image.
# MENTOR NOTE: Never use 'latest' in production. Always pin to a specific version (e.g., 3.11-slim)
# to guarantee reproducibility. Slim images reduce the attack surface and download size.
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy only the requirements first
# WHY: Docker layers are cached. If we only change our code, Docker will use the cached
# layer for installed dependencies, making builds much faster.
COPY requirements.txt .

# Install dependencies
# WHY --no-cache-dir: Keeps the image size small by not caching pip packages.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the necessary source code directories
# MENTOR NOTE: Be explicit about what you copy. Don't just `COPY . .` as you might
# accidentally include secrets or massive dataset files.
COPY src/ src/
COPY api/ api/
COPY models/ models/

# Document the port the container will listen on
EXPOSE 8000

# Command to run the application
# We use uvicorn to run the FastAPI app. --host 0.0.0.0 binds it to all network interfaces.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
