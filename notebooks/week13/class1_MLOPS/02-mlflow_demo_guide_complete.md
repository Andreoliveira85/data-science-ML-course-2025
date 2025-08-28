# MLflow with Docker & PostgreSQL - 30 Minute Demo

## 🎯 Learning Objectives (2 minutes)
By the end of this demo, students will:
- Understand what MLflow is and why it's useful
- Run MLflow with PostgreSQL using Docker
- Track a simple machine learning experiment
- View and compare experiments in the MLflow UI

## 📋 Prerequisites
- Docker installed on your machine
- Basic Python knowledge
- Jupyter notebook environment

---

## 🚀 Part 1: Quick Setup (8 minutes)

### What is MLflow?
MLflow is an open-source platform for managing the ML lifecycle, including:
- **Tracking**: Record experiments, parameters, metrics, and artifacts
- **Projects**: Package ML code in a reusable format
- **Models**: Deploy models to various platforms
- **Registry**: Store, annotate, and manage models centrally

### 🔐 Database Credentials Overview
Our Docker setup creates everything automatically with these credentials:
- **Database Name**: `mlflow_db`
- **Username**: `mlflow_user` 
- **Password**: `mlflow_pass`
- **Host**: `localhost` (or `postgres` from inside Docker)
- **Port**: `5432`

**Important**: These are demo credentials! In production, use strong passwords and environment variables.

### Step 1: Create Project Structure (1 minute)
```bash
mkdir mlflow-demo
cd mlflow-demo
mkdir notebooks
```

### Step 2: Create Docker Compose File (2 minutes)
Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: mlflow_db
      POSTGRES_USER: mlflow_user
      POSTGRES_PASSWORD: mlflow_pass
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlflow_user -d mlflow_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  mlflow:
    image: python:3.9-slim
    depends_on:
      postgres:
        condition: service_healthy
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://mlflow_user:mlflow_pass@postgres:5432/mlflow_db
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
      - ./notebooks:/notebooks
    working_dir: /notebooks
    command: >
      bash -c "pip install mlflow psycopg2-binary &&
               mlflow server --host 0.0.0.0 --port 5000 
               --backend-store-uri postgresql://mlflow_user:mlflow_pass@postgres:5432/mlflow_db
               --default-artifact-root /mlflow/artifacts"

volumes:
  postgres_data:
  mlflow_artifacts:
```

### Step 3: Start Services (3 minutes)
```bash
# Start the services
docker-compose up -d

# Wait for services to be ready (about 30 seconds)
# Check if MLflow is running
curl http://localhost:5000
```

### Step 4: Verify Setup (2 minutes)
- Open browser: http://localhost:5000
- You should see the MLflow UI with no experiments yet
- PostgreSQL should be running on port 5432

---

## 🔬 Part 2: Simple ML Experiment (15 minutes)

### Step 1: Install Python Dependencies
```bash
pip install mlflow psycopg2-binary scikit-learn pandas matplotlib jupyter
```

### Step 2: Create the Demo Notebook
Save this as `notebooks/mlflow_demo.ipynb`:

```python
# Cell 1: Import Libraries and Setup
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Connect to our MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

print("✅ MLflow tracking URI:", mlflow.get_tracking_uri())
```

```python
# Cell 2: Create Sample Data
# Generate a simple classification dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_informative=5, 
    n_redundant=2, 
    random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("📊 Dataset created:")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")
```

```python
# Cell 3: First Experiment - Random Forest
# Set experiment name
mlflow.set_experiment("Student Demo - Binary Classification")

# Start MLflow run
with mlflow.start_run(run_name="Random Forest Baseline"):
    # Log parameters
    n_estimators = 100
    max_depth = 5
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Train model
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # Log model
    mlflow.sklearn.log_model(rf_model, "model")
    
    print(f"🌟 Random Forest Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
```

```python
# Cell 4: Second Experiment - Logistic Regression
with mlflow.start_run(run_name="Logistic Regression"):
    # Log parameters
    C = 1.0
    max_iter = 1000
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("C", C)
    mlflow.log_param("max_iter", max_iter)
    
    # Train model
    lr_model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # Create and log a simple plot
    plt.figure(figsize=(8, 6))
    feature_importance = np.abs(lr_model.coef_[0])
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title("Logistic Regression - Feature Importance")
    plt.xlabel("Feature Index")
    plt.ylabel("Coefficient Magnitude")
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.show()
    
    # Log model
    mlflow.sklearn.log_model(lr_model, "model")
    
    print(f"📈 Logistic Regression Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
```

```python
# Cell 5: Parameter Tuning Experiment
# Try different Random Forest parameters
param_combinations = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 7},
    {"n_estimators": 200, "max_depth": 10}
]

print("🔧 Running parameter tuning experiments...")

for i, params in enumerate(param_combinations):
    with mlflow.start_run(run_name=f"RF_Tuning_{i+1}"):
        # Log parameters
        mlflow.log_param("model_type", "RandomForest_Tuned")
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        
        # Train model
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        print(f"Run {i+1} - Accuracy: {accuracy:.3f}")

print("✅ All experiments completed!")
```

```python
# Cell 6: Query Experiment Results
# Get experiment data
experiment = mlflow.get_experiment_by_name("Student Demo - Binary Classification")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

print("📊 Experiment Summary:")
print("=" * 50)
print(runs[['run_name', 'metrics.accuracy', 'metrics.precision', 'metrics.recall', 
           'params.model_type']].to_string(index=False))

# Find best run
best_run = runs.loc[runs['metrics.accuracy'].idxmax()]
print(f"\n🏆 Best performing model:")
print(f"Run: {best_run['run_name']}")
print(f"Model: {best_run['params.model_type']}")
print(f"Accuracy: {best_run['metrics.accuracy']:.3f}")
```

---

## 🎓 Part 3: Explore MLflow UI (5 minutes)

### Navigate the MLflow Interface:

1. **Go to http://localhost:5000**
2. **Experiments Tab**: See your "Student Demo - Binary Classification" experiment
3. **Click on the experiment**: View all runs with their parameters and metrics
4. **Compare runs**: Select multiple runs and click "Compare"
5. **View artifacts**: Click on individual runs to see logged models and plots
6. **Metrics visualization**: See how metrics compare across runs

### Key Features to Highlight:
- **Run comparison**: Side-by-side parameter and metric comparison
- **Model versioning**: Each model is automatically versioned
- **Artifact storage**: Models, plots, and files are stored centrally
- **Search and filter**: Find runs based on parameters or metrics
- **Notes**: Add descriptions and tags to runs

---

## 🛠 Part 4: Database Exploration (Bonus - 3 minutes)

## 🛠 Part 4: Database Connection Details (Bonus - 5 minutes)

### 🔐 Getting Database Credentials

The Docker Compose setup automatically creates PostgreSQL with these credentials:

```bash
# Database Connection Details
Host: localhost
Port: 5432
Database: mlflow_db
Username: mlflow_user
Password: mlflow_pass
```

### 📋 Alternative Ways to Get Credentials

#### Option 1: From Docker Compose File
Check the `docker-compose.yml` file - credentials are defined in the environment section:
```yaml
environment:
  POSTGRES_DB: mlflow_db
  POSTGRES_USER: mlflow_user
  POSTGRES_PASSWORD: mlflow_pass
```

#### Option 2: Check Running Container
```bash
# See environment variables of running PostgreSQL container
docker exec mlflow-demo_postgres_1 env | grep POSTGRES
```

#### Option 3: Connect with Different Tools

**Using pgAdmin (GUI):**
1. Install pgAdmin
2. Create new server connection:
   - Host: `localhost`
   - Port: `5432`
   - Database: `mlflow_db`
   - Username: `mlflow_user`
   - Password: `mlflow_pass`

**Using psql command line:**
```bash
# Connect to PostgreSQL directly
docker exec -it mlflow-demo_postgres_1 psql -U mlflow_user -d mlflow_db

# Or from your local machine (if psql is installed)
psql -h localhost -p 5432 -U mlflow_user -d mlflow_db
```

**Using Python connection:**
```python
import psycopg2
import pandas as pd

# Connection details
connection_params = {
    'host': 'localhost',
    'port': 5432,
    'database': 'mlflow_db',
    'user': 'mlflow_user',
    'password': 'mlflow_pass'
}

# Connect and query
conn = psycopg2.connect(**connection_params)
query = "SELECT * FROM experiments;"
df = pd.read_sql(query, conn)
print(df)
conn.close()
```

### 🔍 Explore MLflow Database Tables
```bash
# Connect to PostgreSQL
docker exec -it mlflow-demo_postgres_1 psql -U mlflow_user -d mlflow_db

# List tables
\dt

# View experiments
SELECT * FROM experiments;

# View runs
SELECT * FROM runs LIMIT 5;

# Exit
\q
```

### What's Stored:
- **experiments**: Experiment metadata
- **runs**: Individual run information
- **metrics**: All logged metrics with timestamps
- **params**: All logged parameters
- **tags**: Run tags and metadata

---

## 🧹 Cleanup

```bash
# Stop services
docker-compose down

# Remove volumes (optional - removes all data)
docker-compose down -v
```

---

## 🔑 Key Takeaways

1. **MLflow simplifies ML experiment tracking** - No more spreadsheets!
2. **Everything is logged automatically** - Parameters, metrics, models, artifacts
3. **Easy comparison** - Compare multiple experiments side-by-side
4. **Reproducibility** - Every experiment is tracked with its exact parameters
5. **Collaboration** - Team members can see and compare each other's experiments
6. **Production ready** - Models can be deployed directly from MLflow

---

## 🚀 Next Steps

- Try different datasets (iris, wine, breast cancer from sklearn)
- Experiment with hyperparameter tuning using MLflow
- Add model registry functionality
- Deploy models using MLflow serving
- Integrate with CI/CD pipelines

---

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Examples](https://github.com/mlflow/mlflow/tree/master/examples)
- [MLflow Best Practices](https://mlflow.org/docs/latest/best-practices.html)