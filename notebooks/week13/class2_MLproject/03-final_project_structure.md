# Final ML Project Structure

After completing the tutorial, your project will have this comprehensive structure:

```
ml-house-prediction/
├── README.md                           # Project documentation
├── .gitignore                          # Git ignore file
├── .env                                # Environment variables (local)
├── .gcloudignore                       # Google Cloud ignore file
├── pyproject.toml                      # Poetry dependency management
├── poetry.lock                         # Poetry lock file
├── docker-compose.yml                 # Multi-container orchestration
├── Procfile                           # Heroku deployment config
├── runtime.txt                        # Python version specification
├── cloudbuild.yaml                    # Google Cloud Build config
│
├── src/                               # Source code directory
│   ├── __init__.py
│   │
│   ├── api/                           # FastAPI application
│   │   ├── __init__.py
│   │   └── main.py                    # API endpoints and FastAPI app
│   │
│   ├── models/                        # ML model code
│   │   ├── __init__.py
│   │   ├── model.py                   # ML model class definition
│   │   ├── train.py                   # Training script
│   │   └── house_price_model.joblib   # Trained model file
│   │
│   ├── data/                          # Data handling
│   │   ├── __init__.py
│   │   ├── data_loader.py             # Data loading utilities
│   │   └── california_housing.csv     # Processed dataset
│   │
│   └── utils/                         # Utility functions
│       └── __init__.py
│
├── streamlit_app/                     # Streamlit web interface
│   ├── app.py                         # Main Streamlit application
│   └── requirements.txt               # Streamlit-specific dependencies (optional)
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_model_development.ipynb     # Model development (optional)
│   └── 03_model_evaluation.ipynb      # Model evaluation (optional)
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── test_api.py                    # API endpoint tests
│   ├── test_model.py                  # Model functionality tests
│   └── test_data_loader.py            # Data loading tests
│
├── docker/                            # Docker configurations
│   ├── api.Dockerfile                 # API container definition
│   ├── streamlit.Dockerfile           # Streamlit container definition
│   └── mlflow.Dockerfile              # MLflow container (optional)
│
├── .github/                           # GitHub Actions
│   └── workflows/
│       ├── deploy.yml                 # Deployment workflow
│       ├── test.yml                   # Testing workflow
│       └── docker-build.yml           # Docker build workflow
│
├── mlruns/                            # MLflow experiment tracking
│   ├── 0/                             # Default experiment
│   └── 1/                             # Custom experiments
│
├── data/                              # Data directory (gitignored except samples)
│   ├── raw/                           # Raw data files
│   ├── processed/                     # Processed data files
│   └── california_housing.csv         # Main dataset
│
├── models/                            # Model artifacts (gitignored)
│   ├── house_price_model.joblib       # Main trained model
│   ├── scaler.joblib                  # Feature scaler
│   └── model_metadata.json            # Model version info
│
├── logs/                              # Application logs (gitignored)
│   ├── api.log
│   ├── training.log
│   └── streamlit.log
│
└── docs/                              # Documentation (optional)
    ├── api_documentation.md
    ├── deployment_guide.md
    └── model_documentation.md
```

## Key Files Breakdown

### **Core Application Files**

**`src/api/main.py`** (150+ lines)
- FastAPI application with health checks
- ML prediction endpoints (single + batch)
- Request/response models with Pydantic
- Error handling and logging

**`src/models/model.py`** (120+ lines)
- `HousePricePredictor` class
- sklearn pipeline with preprocessing
- MLflow integration for experiment tracking
- Model training, evaluation, and persistence

**`src/data/data_loader.py`** (80+ lines)
- Data loading from sklearn datasets
- Train/test splitting functionality
- Data preprocessing utilities
- CSV export capabilities

**`streamlit_app/app.py`** (200+ lines)
- Interactive web interface
- Real-time prediction forms
- Data visualizations with Plotly
- API integration and error handling

### **Configuration Files**

**`pyproject.toml`**
```toml
[tool.poetry]
name = "ml-house-prediction"
version = "0.1.0"
description = "ML House Price Prediction Project"

[tool.poetry.dependencies]
python = "^3.9"
scikit-learn = "^1.3.0"
pandas = "^2.0.0"
fastapi = "^0.104.0"
streamlit = "^1.28.0"
mlflow = "^2.8.0"
# ... more dependencies
```

**`docker-compose.yml`**
- Multi-service orchestration
- API, Streamlit, and MLflow services
- Volume mounts and networking
- Health checks and dependencies

### **Docker Files**

**`docker/api.Dockerfile`**
- Python 3.9 slim base image
- Poetry dependency management
- Non-root user security
- Health checks and optimization

**`docker/streamlit.Dockerfile`**
- Streamlit-optimized container
- Port configuration for Cloud Run
- Environment variable handling

### **CI/CD and Deployment**

**`.github/workflows/deploy.yml`**
- Automated testing pipeline
- Docker image building
- Cloud deployment automation
- Environment-specific configurations

**`cloudbuild.yaml`** (Google Cloud)
- Cloud Build configuration
- Container registry integration
- Cloud Run deployment steps

### **Testing Infrastructure**

**`tests/test_api.py`**
- FastAPI endpoint testing
- Request/response validation
- Error handling verification

**`tests/test_model.py`**
- Model training/prediction tests
- Data pipeline validation
- Performance benchmarking

### **MLflow Tracking**

**`mlruns/` directory structure:**
```
mlruns/
├── 0/                                 # Default experiment
│   └── meta.yaml
├── 1/                                 # House price prediction experiment
│   ├── meta.yaml
│   └── [run-id]/                      # Individual experiment runs
│       ├── artifacts/
│       │   └── model/                 # Stored model artifacts
│       ├── metrics/                   # Training metrics
│       ├── params/                    # Hyperparameters
│       └── tags/                      # Run metadata
└── models/                            # Model registry
    └── house-price-model/
        └── version-1/
```

## Project Size and Complexity

**Total Files:** ~25-30 files
**Lines of Code:** ~1,000-1,500 lines
**Docker Images:** 2-3 containers
**Dependencies:** ~15-20 Python packages

## What Students Will Learn

### **Technical Skills**
1. **Python packaging** with Poetry
2. **API development** with FastAPI
3. **ML pipeline** with scikit-learn
4. **Experiment tracking** with MLflow
5. **Containerization** with Docker
6. **Web development** with Streamlit
7. **Cloud deployment** with Google Cloud Run
8. **CI/CD** with GitHub Actions

### **Best Practices**
1. **Project structure** and organization
2. **Environment management** and reproducibility
3. **Testing** and quality assurance
4. **Documentation** and code clarity
5. **Security** and production readiness
6. **Monitoring** and observability
7. **Version control** and collaboration

### **Production Concepts**
1. **Microservices architecture**
2. **API design patterns**
3. **Scalable ML deployment**
4. **Infrastructure as Code**
5. **DevOps workflows**
6. **Cost optimization**
7. **Security best practices**

## Deployment Outputs

After deployment, students will have:

1. **Live API endpoint:** `https://ml-house-api-xyz.run.app`
2. **Interactive web app:** `https://ml-house-streamlit-xyz.run.app`
3. **MLflow dashboard:** Running on Cloud Run or locally
4. **API documentation:** Available at `/docs` endpoint
5. **Monitoring dashboard:** Google Cloud Console
6. **Source code repository:** GitHub with CI/CD

This structure provides a **complete, production-ready ML system** that students can showcase in portfolios, use as a reference for future projects, and understand modern ML engineering practices.