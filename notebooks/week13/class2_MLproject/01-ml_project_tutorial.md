# Complete ML Project Tutorial: From Development to Deployment
*A 2-hour hands-on workshop building a production-ready ML project*

## Overview
In this tutorial, we'll build a complete machine learning project that predicts house prices using the California Housing dataset. We'll use modern best practices and tools to create a production-ready system.

### What We'll Build
- A machine learning model for house price prediction
- RESTful API using FastAPI
- Experiment tracking with MLflow
- Containerized application with Docker
- Streamlit web interface
- Deployment to a public cloud platform

### Tech Stack
- **Python**: Programming language
- **Poetry**: Dependency management
- **FastAPI**: API framework
- **MLflow**: Experiment tracking
- **Docker**: Containerization
- **Streamlit**: Web interface
- **Scikit-learn**: Machine learning

---

## Part 1: Project Setup (20 minutes)

### Step 1: Initialize the Project Structure

Create the project directory and navigate into it:
```bash
mkdir ml-house-prediction
cd ml-house-prediction
```

Create the following directory structure:
```bash
mkdir -p src/{api,models,data,utils}
mkdir -p streamlit_app
mkdir -p notebooks
mkdir -p tests
mkdir -p docker
mkdir -p .github/workflows
touch README.md .gitignore .env
```

Your project structure should look like this:
```
ml-house-prediction/
├── src/
│   ├── api/
│   ├── models/
│   ├── data/
│   └── utils/
├── streamlit_app/
├── notebooks/
├── tests/
├── docker/
├── .github/workflows/
├── README.md
├── .gitignore
└── .env
```

### Step 2: Initialize Poetry

Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Initialize Poetry in your project:
```bash
poetry init --name ml-house-prediction --version 0.1.0 --description "ML House Price Prediction Project" --author "Your Name <your.email@example.com>" --python "^3.9" --no-interaction
```

Add dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
# Core ML dependencies
poetry add scikit-learn pandas numpy matplotlib seaborn

# API dependencies
poetry add fastapi uvicorn python-multipart

# Experiment tracking
poetry add mlflow

# Streamlit for web interface
poetry add streamlit

# Utilities
poetry add python-dotenv pydantic

# Development dependencies
poetry add --group dev pytest black isort flake8 jupyter
```

#### toml file

[tool.poetry]
name = "ml-house-prediction"
version = "0.1.0"
description = "ML full project"
authors = ["andre <andre.deoliveiragomes@ovomcare.com>"]
readme = "README.md"
package-mode = false

[tool.poetry.dependencies]
# Set this to match your current venv Python version.
# If you're on 3.12, the next line is a safe default:
python = ">=3.10,<3.13"

requests = ">=2.32.5,<3.0.0"
scikit-learn = ">=1.3.0,<2.0.0"
pandas = ">=2.1.0,<3.0.0"
numpy = ">=1.26.0,<3.0.0"
matplotlib = ">=3.8.0,<4.0.0"
seaborn = ">=0.13.0,<0.14.0"
fastapi = ">=0.110.0,<1.0.0"
uvicorn = ">=0.27.0,<1.0.0"
python-multipart = ">=0.0.20,<0.0.21"
mlflow = ">=2.12.0,<3.0.0"
streamlit = ">=1.32.0,<2.0.0"
python-dotenv = ">=1.0.0,<2.0.0"
pydantic = ">=2.6.0,<3.0.0"

[tool.poetry.group.dev.dependencies]
pytest = ">=8.0.0,<9.0.0"
black = ">=24.0.0,<26.0.0"
isort = ">=5.13.0,<7.0.0"
flake8 = ">=7.0.0,<8.0.0"
jupyter = ">=1.0.0,<2.0.0"

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"


Activate the virtual environment:
```bash
poetry shell
```

### Step 3: Create .gitignore

Create `.gitignore` file:
```bash
cat > .gitignore << 'EOF'

EOF
```

---

## Part 2: Data Preparation and EDA (25 minutes)

### Step 4: Create Data Loading Module

Create `src/data/data_loader.py`:
```python
```

### Step 5: Create Exploratory Data Analysis Notebook

Create `notebooks/01_eda.ipynb`:
```bash
jupyter notebook
```

In the notebook, add this content:
```python
# Cell 1
import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data.data_loader import DataLoader

plt.style.use('seaborn-v0_8')
%matplotlib inline

# Cell 2
# Load data
loader = DataLoader()
df = loader.load_california_housing()
print(f"Dataset shape: {df.shape}")
df.head()

# Cell 3
# Basic statistics
print("Dataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# Cell 4
# Visualizations
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for i, col in enumerate(df.columns):
    axes[i].hist(df[col], bins=50, alpha=0.7)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Cell 5
# Correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# Cell 6
# Save processed data
loader.save_data('../data/california_housing.csv')
```

Run the notebook:
```bash
cd notebooks
jupyter notebook 01_eda.ipynb
```

---

## Part 3: Model Development (30 minutes)

### Step 6: Create Model Training Module

Create `src/models/model.py`:
```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import os
from typing import Dict, Any
import mlflow
import mlflow.sklearn

class HousePricePredictor:
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.pipeline = None
        self.is_trained = False
        
    def create_pipeline(self):
        """Create ML pipeline with preprocessing and model"""
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            ))
        ])
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Train the model"""
        if self.pipeline is None:
            self.create_pipeline()
            
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("n_estimators", self.n_estimators)
            mlflow.log_param("random_state", self.random_state)
            
            # Train model
            self.pipeline.fit(X_train, y_train)
            self.is_trained = True
            
            # Make predictions on training set
            y_pred = self.pipeline.predict(X_train)
            
            # Calculate metrics
            metrics = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred)),
                'train_mae': mean_absolute_error(y_train, y_pred),
                'train_r2': r2_score(y_train, y_pred)
            }
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(self.pipeline, "model")
            
            print(f"Training completed. RMSE: {metrics['train_rmse']:.2f}")
            return metrics
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model on test set"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
            
        y_pred = self.pipeline.predict(X_test)
        
        metrics = {
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_r2': r2_score(y_test, y_pred)
        }
        
        # Log test metrics to MLflow
        with mlflow.start_run():
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        return self.pipeline.predict(X)
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        joblib.dump(self.pipeline, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        self.pipeline = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")
```

### Step 7: Create Training Script

Create `src/models/train.py`:
```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import mlflow
from data.data_loader import DataLoader
from models.model import HousePricePredictor

def main():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("house-price-prediction")
    
    # Load data
    print("Loading data...")
    loader = DataLoader()
    loader.load_california_housing()
    X_train, X_test, y_train, y_test = loader.get_train_test_split()
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Initialize and train model
    print("Training model...")
    model = HousePricePredictor(n_estimators=100)
    train_metrics = model.train(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    test_metrics = model.evaluate(X_test, y_test)
    
    print("Training Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save_model("models/house_price_model.joblib")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
```

### Step 8: Start MLflow and Train Model

Open a new terminal and start MLflow server:
```bash
mlflow server --host 127.0.0.1 --port 5000
```

In your main terminal, train the model:
```bash
cd src
python models/train.py
```

Visit http://localhost:5000 to see MLflow UI and your experiment results.

---

## Part 4: API Development (25 minutes)

### Step 9: Create FastAPI Application

Create `src/api/main.py`:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import List
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices using ML",
    version="1.0.0"
)

# Global variable to store the model
model = None

class HouseFeatures(BaseModel):
    MedInc: float  # Median income
    HouseAge: float  # House age
    AveRooms: float  # Average rooms
    AveBedrms: float  # Average bedrooms
    Population: float  # Population
    AveOccup: float  # Average occupancy
    Latitude: float  # Latitude
    Longitude: float  # Longitude

class PredictionResponse(BaseModel):
    predicted_price: float
    features_used: dict

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model
    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "src", "models", "house_price_model.joblib")
    
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.get("/")
async def root():
    return {"message": "House Price Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: HouseFeatures):
    """Predict house price based on features"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to DataFrame
        feature_dict = features.dict()
        df = pd.DataFrame([feature_dict])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        return PredictionResponse(
            predicted_price=round(prediction, 2),
            features_used=feature_dict
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(features_list: List[HouseFeatures]):
    """Predict prices for multiple houses"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        features_data = [f.dict() for f in features_list]
        df = pd.DataFrame(features_data)
        
        # Make predictions
        predictions = model.predict(df)
        
        results = []
        for i, prediction in enumerate(predictions):
            results.append({
                "house_id": i,
                "predicted_price": round(prediction, 2),
                "features": features_data[i]
            })
        
        return {"predictions": results}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 10: Test the API

Start the FastAPI server:
```bash
cd src
python api/main.py
```

Visit http://localhost:8000/docs to see the interactive API documentation.

Test the API with curl:
```bash

cURL, which stands for client URL, is a command line tool that developers use to transfer data to and from a server. At the most fundamental, cURL lets you talk to a server by specifying the location (in the form of a URL) and the data you want to send.


curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "MedInc": 5.0,
  "HouseAge": 10.0,
  "AveRooms": 6.0,
  "AveBedrms": 1.2,
  "Population": 3000.0,
  "AveOccup": 3.0,
  "Latitude": 34.0,
  "Longitude": -118.0
}'
```

---

## Part 5: Streamlit Web Interface (15 minutes)

### Step 11: Create Streamlit App

Create `streamlit_app/app.py`:
```python
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 California House Price Predictor")
st.markdown("Predict house prices using machine learning based on various features of California housing data.")

# Sidebar for inputs
st.sidebar.header("House Features")

# Input fields
med_inc = st.sidebar.slider("Median Income (in 10k$)", 0.5, 15.0, 5.0, 0.1)
house_age = st.sidebar.slider("House Age (years)", 1.0, 50.0, 10.0, 1.0)
ave_rooms = st.sidebar.slider("Average Rooms", 2.0, 10.0, 6.0, 0.1)
ave_bedrms = st.sidebar.slider("Average Bedrooms", 0.5, 3.0, 1.2, 0.1)
population = st.sidebar.slider("Population", 100.0, 10000.0, 3000.0, 100.0)
ave_occup = st.sidebar.slider("Average Occupancy", 1.0, 10.0, 3.0, 0.1)
latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 34.0, 0.1)
longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -118.0, 0.1)

# API endpoint
API_URL = "http://localhost:8000"

# Prediction button
if st.sidebar.button("Predict Price", type="primary"):
    # Prepare data
    features = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }
    
    try:
        # Make API call
        response = requests.post(f"{API_URL}/predict", json=features)
        
        if response.status_code == 200:
            result = response.json()
            predicted_price = result["predicted_price"]
            
            # Display result
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Predicted Price",
                    value=f"${predicted_price:,.2f}",
                    delta=None
                )
            
            with col2:
                price_per_sqft = predicted_price / (ave_rooms * 200)  # Rough estimate
                st.metric(
                    label="Price per Room",
                    value=f"${price_per_sqft:,.2f}",
                    delta=None
                )
            
            with col3:
                affordability = "High" if predicted_price < 200000 else "Medium" if predicted_price < 400000 else "Low"
                st.metric(
                    label="Affordability",
                    value=affordability,
                    delta=None
                )
            
            # Feature importance visualization
            st.subheader("Input Features Summary")
            feature_df = pd.DataFrame(list(features.items()), columns=['Feature', 'Value'])
            
            fig = px.bar(feature_df, x='Feature', y='Value', title="Input Feature Values")
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Location visualization
            st.subheader("Property Location")
            map_data = pd.DataFrame({
                'lat': [latitude],
                'lon': [longitude],
                'price': [predicted_price]
            })
            st.map(map_data)
            
        else:
            st.error(f"API Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the API server is running on http://localhost:8000")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Information section
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("This app predicts California house prices using a Random Forest model trained on the California Housing dataset.")

# Main content when no prediction is made
if not st.sidebar.button("Predict Price", type="primary", key="dummy"):
    st.markdown("### How to use this app:")
    st.markdown("1. Adjust the feature values in the sidebar")
    st.markdown("2. Click 'Predict Price' to get a prediction")
    st.markdown("3. View the results and visualizations")
    
    # Sample data visualization
    st.subheader("California Housing Dataset Overview")
    
    # Create sample data for demonstration
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Latitude': np.random.uniform(32, 42, 1000),
        'Longitude': np.random.uniform(-125, -114, 1000),
        'Price': np.random.uniform(50000, 800000, 1000)
    })
    
    fig = px.scatter(sample_data, x='Longitude', y='Latitude', color='Price',
                    title="Sample California Housing Prices by Location",
                    color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)
```

### Step 12: Run Streamlit App

Make sure your FastAPI server is still running, then start Streamlit:
```bash
cd streamlit_app
streamlit run app.py
```

Visit http://localhost:8501 to use the web interface.

---

#### end local development ####
##### prepare deployment phase #####

## Part 6: Containerization with Docker (15 minutes)

### Step 13: Create Dockerfiles

Create `docker/api.Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Install poetry
RUN pip install poetry

# Configure poetry
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --only main

# Copy source code
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["python", "src/api/main.py"]
```

Create `docker/streamlit.Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Install poetry
RUN pip install poetry

# Configure poetry
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --only main

# Copy streamlit app
COPY streamlit_app/ ./streamlit_app/

# Expose port
EXPOSE 8501

# Command to run streamlit
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.address", "0.0.0.0"]
```

### Step 14: Create Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./src/models:/app/src/models
    environment:
      - PYTHONPATH=/app
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  streamlit:
    build:
      context: .
      dockerfile: docker/streamlit.Dockerfile
    ports:
      - "8501:8501"
    depends_on:
      - api
    environment:
      - PYTHONPATH=/app
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  mlflow:
    image: python:3.9-slim
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/app/mlruns
    working_dir: /app
    command: >
      bash -c "pip install mlflow &&
               mlflow server --host 0.0.0.0 --port 5000"
```

### Step 15: Build and Run with Docker

Build and run the containers:
```bash
docker-compose up --build
```

Your application will be available at:
- API: http://localhost:8000
- Streamlit: http://localhost:8501
- MLflow: http://localhost:5000

---

## Part 7: Cloud Deployment Preparation (10 minutes)

### Step 16: Create Deployment Configuration

Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy ML Application

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install Poetry
      uses: snok/install-poetry@v1
    
    - name: Install dependencies
      run: poetry install
    
    - name: Run tests
      run: poetry run pytest tests/
    
    - name: Build Docker images
      run: |
        docker build -f docker/api.Dockerfile -t ml-api .
        docker build -f docker/streamlit.Dockerfile -t ml-streamlit .
    
    # Add your cloud provider deployment steps here
    # For example, for Google Cloud Run:
    # - name: Deploy to Cloud Run
    #   run: |
    #     gcloud run deploy ml-api --image ml-api --platform managed --region us-central1
```

### Step 17: Create Basic Tests

Create `tests/test_api.py`:
```python
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "House Price Prediction API" in response.json()["message"]

def test_health():
    response = client.get("/health")
    # May fail if model is not loaded, but that's expected in test environment
    assert response.status_code in [200, 503]

def test_predict():
    test_data = {
        "MedInc": 5.0,
        "HouseAge": 10.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.2,
        "Population": 3000.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0
    }
    response = client.post("/predict", json=test_data)
    # May fail if model is not loaded
    assert response.status_code in [200, 503]
```

Run tests:
```bash
poetry run pytest tests/ -v
```

---

## Part 8: Cloud Deployment Options (10 minutes)

### Option 1: Heroku Deployment

Create `Procfile`:
```
web: python src/api/main.py
```

Create `runtime.txt`:
```
python-3.9.18
```

Deploy commands:
```bash
# Install Heroku CLI and login
heroku create your-ml-app
heroku container:login
heroku container:push web --app your-ml-app
heroku container:release web --app your-ml-app
```

### Option 2: Google Cloud Run

Create `cloudbuild.yaml`:
```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'docker/api.Dockerfile', '-t', 'gcr.io/$PROJECT_ID/ml-api', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/ml-api']
  - name: 'gcr.io/cloud-builders/gcloud'
    args: ['run', 'deploy', 'ml-api', '--image', 'gcr.io/$PROJECT_ID/ml-api', '--platform', 'managed', '--region', 'us-central1']
```

Deploy commands:
```bash
gcloud builds submit --config cloudbuild.yaml
```

### Option 3: AWS ECS/Fargate

Create deployment script for AWS:
```bash
# Build and push to ECR
aws ecr create-repository --repository-name ml-house-prediction
docker build -f docker/api.Dockerfile -t ml-house-prediction .
docker tag ml-house-prediction:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/ml-house-prediction:latest
docker