# Updated Dockerfiles for Python 3.13

## Why Different Docker Files?

In modern microservices architecture, we separate different components into their own containers for several reasons:

### 🏗️ **Separation of Concerns**
- **API Container**: Handles ML predictions and API endpoints
- **Streamlit Container**: Provides web UI for users
- **MLflow Container**: Manages experiment tracking and model registry

### 🎯 **Benefits of Separate Containers**
1. **Independent Scaling**: Scale API separately from UI based on demand
2. **Different Resource Requirements**: API needs more CPU, UI needs less
3. **Independent Deployment**: Update API without affecting UI
4. **Security Isolation**: API can run with restricted permissions
5. **Technology Flexibility**: Different containers can use different base images

---

## Updated Docker Files

### 1. API Dockerfile (`docker/api.Dockerfile`)
*Purpose: Serves the FastAPI ML prediction API*

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Install poetry
RUN pip install poetry

# Configure poetry: don't create virtual environment in container
RUN poetry config virtualenvs.create false

# Install only production dependencies
RUN poetry install --only main --no-dev

# Copy source code
COPY src/ ./src/

# Create models directory and copy trained model
RUN mkdir -p ./models
COPY models/ ./models/

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (Cloud Run will use PORT environment variable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use environment variable for port (required for Cloud Run)
CMD ["sh", "-c", "python src/api/main.py"]
```

**Why this container?**
- Runs the ML prediction API
- Needs scikit-learn, FastAPI, and model files
- Optimized for handling HTTP requests
- Can scale independently based on API demand

---

### 2. Streamlit Dockerfile (`docker/streamlit.Dockerfile`)
*Purpose: Serves the user-facing web interface*

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Install poetry
RUN pip install poetry

# Configure poetry
RUN poetry config virtualenvs.create false

# Install only production dependencies
RUN poetry install --only main --no-dev

# Copy streamlit app
COPY streamlit_app/ ./streamlit_app/

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8501

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run streamlit with proper configuration for containers
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]
```

**Why this container?**
- Serves the user interface
- Needs Streamlit, Plotly for visualizations
- Different scaling needs than API
- Can be updated independently

---

### 3. MLflow Dockerfile (`docker/mlflow.Dockerfile`)
*Purpose: Runs MLflow tracking server*

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow and dependencies
RUN pip install mlflow psycopg2-binary boto3

# Create mlruns directory
RUN mkdir -p /app/mlruns

# Create a non-root user
RUN useradd -m -u 1000 mlflowuser && chown -R mlflowuser:mlflowuser /app
USER mlflowuser

# Expose MLflow port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "file:///app/mlruns"]
```

**Why this container?**
- Manages experiment tracking
- Stores model artifacts and metrics
- Can persist data to external storage
- Independent lifecycle from API/UI

---

## Updated Docker Compose (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  # ML API Service
  api:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models  # Mount models directory
    environment:
      - PYTHONPATH=/app
      - PORT=8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # Streamlit Web Interface
  streamlit:
    build:
      context: .
      dockerfile: docker/streamlit.Dockerfile
    ports:
      - "8501:8501"
    depends_on:
      api:
        condition: service_healthy
    environment:
      - PYTHONPATH=/app
      - API_URL=http://api:8000  # Use service name for internal communication
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # MLflow Tracking Server
  mlflow:
    build:
      context: .
      dockerfile: docker/mlflow.Dockerfile
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/app/mlruns  # Persist experiment data
      - ./models:/app/models  # Access to model artifacts
    environment:
      - MLFLOW_BACKEND_STORE_URI=file:///app/mlruns
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

networks:
  default:
    name: ml-house-prediction-network
```

---

## Cloud-Optimized Dockerfile (for Google Cloud Run)

For Cloud Run deployment, use this optimized version:

```dockerfile
# docker/api-cloud.Dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Install poetry
RUN pip install poetry

# Configure poetry
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --only main --no-dev

# Copy source code and models
COPY src/ ./src/
COPY models/ ./models/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Cloud Run uses PORT environment variable
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Use PORT environment variable for Cloud Run compatibility
CMD ["sh", "-c", "cd src && python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
```

---

## Container Communication Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   MLflow        │
│   Container     │────│   Container     │────│   Container     │
│   Port: 8501    │    │   Port: 8000    │    │   Port: 5000    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                │
                     ┌─────────────────┐
                     │   Shared        │
                     │   Volumes       │
                     │   - models/     │
                     │   - mlruns/     │
                     └─────────────────┘
```

---

## Build and Run Commands

### Build Individual Containers
```bash
# Build API container
docker build -f docker/api.Dockerfile -t ml-house-api .

---> build requirements.txt +++
run docker build -f docker/api.Dockerfile -t ml-house-api .
then
docker run -p 8000:8000 ml-house-api


# Build Streamlit container
docker build -f docker/streamlit.Dockerfile -t ml-house-streamlit .

# Build MLflow container
docker build -f docker/mlflow.Dockerfile -t ml-house-mlflow .
```

### Run with Docker Compose
```bash
# Build and run all containers
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop all containers
docker-compose down
```

### Individual Container Commands
```bash
# Run API container only
docker run -p 8000:8000 -v $(pwd)/models:/app/models ml-house-api

# Run Streamlit container only
docker run -p 8501:8501 ml-house-streamlit

# Run MLflow container only
docker run -p 5000:5000 -v $(pwd)/mlruns:/app/mlruns ml-house-mlflow
```

---

## Production Considerations

### 1. **Multi-Stage Builds** (Optional Optimization)
```dockerfile
# Multi-stage build for smaller production images
FROM python:3.13-slim as builder
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --only main --no-dev

FROM python:3.13-slim as production
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY src/ ./src/
COPY models/ ./models/
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["python", "src/api/main.py"]
```

### 2. **Security Best Practices**
- ✅ Non-root user in all containers
- ✅ Minimal base images (slim variants)
- ✅ Health checks for monitoring
- ✅ Proper port exposure
- ✅ Environment variable configuration

### 3. **Resource Optimization**
- ✅ No dev dependencies in production
- ✅ Clean package manager cache
- ✅ Optimized layer caching
- ✅ Minimal system dependencies

This containerization strategy gives you a professional, scalable, and maintainable ML system! 🚀