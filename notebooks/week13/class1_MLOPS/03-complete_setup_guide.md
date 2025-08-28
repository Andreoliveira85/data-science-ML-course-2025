# 🚀 Complete MLflow Demo Setup - Start from Zero

## ⏱ Total Time: 10 minutes setup + 30 minutes demo

---

## 📋 BEFORE WE START - What You Need

### Required Software (Install These First!)

#### 1. Docker Desktop 🐳
**Everyone needs this!**

**Windows:**
- Go to https://www.docker.com/products/docker-desktop
- Download "Docker Desktop for Windows"
- Run installer and restart computer
- Open Docker Desktop and wait for it to start

**Mac:**
- Go to https://www.docker.com/products/docker-desktop
- Download "Docker Desktop for Mac"
- Drag to Applications folder
- Open Docker Desktop and wait for it to start

**Linux:**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose
sudo systemctl start docker
sudo usermod -aG docker $USER
# Log out and log back in
```

**✅ Test Docker:**
```bash
docker --version
docker-compose --version
```
You should see version numbers!

#### 2. Python & Jupyter 🐍
**Option A - Anaconda (Recommended for beginners):**
- Download from https://www.anaconda.com/download
- Install with default settings
- This gives you Python + Jupyter + many packages

**Option B - Python + pip:**
```bash
# Check if Python is installed
python --version
# or
python3 --version

# Install Jupyter
pip install jupyter
# or
pip3 install jupyter
```

---

## 🎯 STEP-BY-STEP SETUP (10 minutes)

### Step 1: Create Project Folder (1 minute)
```bash
# Create a folder for our demo
mkdir mlflow-demo
cd mlflow-demo

# Create subfolder for notebooks
mkdir notebooks
```

### Step 2: Create Docker Configuration (2 minutes)

Create a file called `docker-compose.yml` in the `mlflow-demo` folder:

**Copy and paste this EXACTLY:**

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

**💡 Where are the credentials?**
Look at these lines in the file above:
```yaml
POSTGRES_DB: mlflow_db          # Database name
POSTGRES_USER: mlflow_user      # Username  
POSTGRES_PASSWORD: mlflow_pass  # Password
```

### Step 3: Start Docker Services (3 minutes)
```bash
# Make sure you're in the mlflow-demo folder
cd mlflow-demo

# Start everything (this downloads images first time - takes 2-3 minutes)
docker-compose up -d

# Wait for services to start (about 30 seconds)
# You'll see output like:
# Creating mlflow-demo_postgres_1 ... done
# Creating mlflow-demo_mlflow_1   ... done
```

**✅ Check if it's working:**
```bash
# Check running containers
docker ps

# Test MLflow UI (should show HTML)
curl http://localhost:5000

# Or just open browser: http://localhost:5000
```

### Step 4: Install Python Packages (2 minutes)
```bash
## virtual env 
python -m venv mlflow-demo-env
## activate
source mlflow-demo-env/bin/activate
# Install required packages for the notebook
pip install mlflow psycopg2-binary scikit-learn pandas matplotlib jupyter

# For Anaconda users:
conda install -c conda-forge mlflow psycopg2 scikit-learn pandas matplotlib
```

### Step 5: Download the Notebook (2 minutes)

Save this as `notebooks/mlflow_demo.ipynb`:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🚀 MLflow Demo - 30 Minute Tutorial\\n\\n## 🔐 Connection Details (Auto-configured by Docker)\\n- **Database**: mlflow_db\\n- **Username**: mlflow_user\\n- **Password**: mlflow_pass\\n- **MLflow UI**: http://localhost:5000\\n\\n**✅ Make sure Docker is running before starting!**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install packages if needed (uncomment next line if packages missing)\\n# !pip install mlflow psycopg2-binary scikit-learn pandas matplotlib\\n\\nimport mlflow\\nimport mlflow.sklearn\\nimport pandas as pd\\nimport numpy as np\\nfrom sklearn.model_selection import train_test_split\\nfrom sklearn.ensemble import RandomForestClassifier\\nfrom sklearn.linear_model import LogisticRegression\\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score\\nfrom sklearn.datasets import make_classification\\nimport matplotlib.pyplot as plt\\nimport warnings\\nwarnings.filterwarnings('ignore')\\n\\n# 🔗 Connect to MLflow server (running in Docker)\\nmlflow.set_tracking_uri(\\"http://localhost:5000\\")\\n\\nprint(\\"✅ MLflow tracking URI:\\", mlflow.get_tracking_uri())\\nprint(\\"🐳 Docker services should be running!\\")\\nprint(\\"📊 MLflow UI: http://localhost:5000\\")\\nprint(\\"🔐 Database credentials: mlflow_user / mlflow_pass\\")\\nprint(\\"🎯 Ready to start experiments!\\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 📊 Create sample dataset\\nprint(\\"🔬 Creating synthetic dataset...\\")\\n\\nX, y = make_classification(\\n    n_samples=1000, n_features=10, n_informative=5, \\n    n_redundant=2, random_state=42\\n)\\n\\nX_train, X_test, y_train, y_test = train_test_split(\\n    X, y, test_size=0.2, random_state=42\\n)\\n\\nprint(f\\"✅ Dataset ready: {X_train.shape[0]} train, {X_test.shape[0]} test samples\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 🌲 Experiment 1: Random Forest\\nmlflow.set_experiment(\\"Student Demo - Binary Classification\\")\\n\\nwith mlflow.start_run(run_name=\\"Random Forest - Baseline\\"):\\n    # Parameters\\n    n_estimators, max_depth = 100, 5\\n    \\n    # 📝 Log parameters (saved to PostgreSQL)\\n    mlflow.log_param(\\"model_type\\", \\"RandomForest\\")\\n    mlflow.log_param(\\"n_estimators\\", n_estimators)\\n    mlflow.log_param(\\"max_depth\\", max_depth)\\n    \\n    # 🏋️ Train model\\n    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)\\n    model.fit(X_train, y_train)\\n    \\n    # 📊 Evaluate and log metrics (saved to PostgreSQL)\\n    y_pred = model.predict(X_test)\\n    accuracy = accuracy_score(y_test, y_pred)\\n    mlflow.log_metric(\\"accuracy\\", accuracy)\\n    mlflow.log_metric(\\"precision\\", precision_score(y_test, y_pred))\\n    mlflow.log_metric(\\"recall\\", recall_score(y_test, y_pred))\\n    \\n    # 🤖 Save model\\n    mlflow.sklearn.log_model(model, \\"model\\")\\n    \\n    print(f\\"🎯 Random Forest Accuracy: {accuracy:.3f}\\")\\n    print(\\"✅ Experiment logged to database!\\")\\n    print(\\"👀 Check MLflow UI: http://localhost:5000\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 📈 Experiment 2: Logistic Regression\\nwith mlflow.start_run(run_name=\\"Logistic Regression\\"):\\n    # Parameters\\n    C = 1.0\\n    \\n    # Log parameters\\n    mlflow.log_param(\\"model_type\\", \\"LogisticRegression\\")\\n    mlflow.log_param(\\"C\\", C)\\n    \\n    # Train model\\n    model = LogisticRegression(C=C, max_iter=1000, random_state=42)\\n    model.fit(X_train, y_train)\\n    \\n    # Evaluate\\n    y_pred = model.predict(X_test)\\n    accuracy = accuracy_score(y_test, y_pred)\\n    mlflow.log_metric(\\"accuracy\\", accuracy)\\n    mlflow.log_metric(\\"precision\\", precision_score(y_test, y_pred))\\n    mlflow.log_metric(\\"recall\\", recall_score(y_test, y_pred))\\n    \\n    # Save model\\n    mlflow.sklearn.log_model(model, \\"model\\")\\n    \\n    print(f\\"🎯 Logistic Regression Accuracy: {accuracy:.3f}\\")\\n    print(\\"✅ Second experiment logged!\\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 🔧 Experiment 3: Hyperparameter Tuning\\nconfigs = [\\n    {\\"n_estimators\\": 50, \\"max_depth\\": 3, \\"name\\": \\"Small\\"},\\n    {\\"n_estimators\\": 200, \\"max_depth\\": 10, \\"name\\": \\"Large\\"}\\n]\\n\\nprint(\\"🔧 Testing different configurations...\\")\\n\\nfor config in configs:\\n    with mlflow.start_run(run_name=f\\"RF Tuning - {config['name']}\\"):\\n        # Extract params\\n        params = {k: v for k, v in config.items() if k != 'name'}\\n        \\n        # Log params\\n        mlflow.log_param(\\"model_type\\", \\"RandomForest_Tuned\\")\\n        for k, v in params.items():\\n            mlflow.log_param(k, v)\\n        \\n        # Train and evaluate\\n        model = RandomForestClassifier(**params, random_state=42)\\n        model.fit(X_train, y_train)\\n        accuracy = accuracy_score(y_test, model.predict(X_test))\\n        \\n        mlflow.log_metric(\\"accuracy\\", accuracy)\\n        \\n        print(f\\"🎯 {config['name']} model accuracy: {accuracy:.3f}\\")\\n\\nprint(\\"✅ All experiments completed!\\")\\nprint(\\"📊 View results at: http://localhost:5000\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 📊 Query all experiments from database\\nexperiment = mlflow.get_experiment_by_name(\\"Student Demo - Binary Classification\\")\\nruns_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])\\n\\nprint(\\"📈 All Experiment Results:\\")\\nprint(\\"=\\" * 60)\\nresult_cols = ['run_name', 'metrics.accuracy', 'params.model_type']\\nif not runs_df.empty:\\n    display_df = runs_df[result_cols].round(3)\\n    print(display_df.to_string(index=False))\\n    \\n    best_run = runs_df.loc[runs_df['metrics.accuracy'].idxmax()]\\n    print(f\\"\\\\n🏆 Best Model: {best_run['params.model_type']} (Accuracy: {best_run['metrics.accuracy']:.3f})\\")\\nelse:\\n    print(\\"No experiments found. Make sure to run the cells above first!\\")\\n\\nprint(\\"\\\\n🎯 Demo completed! Check the MLflow UI for detailed comparisons.\\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
```

---

## 🚀 START THE DEMO (30 minutes)

### Step 1: Start Jupyter (2 minutes)
```bash
# Navigate to your project folder
cd mlflow-demo

# Start Jupyter notebook
jupyter notebook

# This opens browser at http://localhost:8888
# Navigate to notebooks/ folder
# Open mlflow_demo.ipynb
```

### Step 2: Run the Demo (25 minutes)
1. **Open the notebook** in Jupyter
2. **Run each cell** one by one (Shift+Enter)
3. **Watch the magic happen!** 🎉

### Step 3: Explore Results (3 minutes)
- **MLflow UI**: http://localhost:5000
- **Compare experiments** side by side
- **View logged models** and metrics

---

## 🔐 CREDENTIALS SUMMARY

**No setup needed! Docker creates everything automatically:**

```
🗄️  PostgreSQL Database:
   Host:     localhost
   Port:     5432
   Database: mlflow_db
   Username: mlflow_user
   Password: mlflow_pass

🎯 MLflow UI:
   URL: http://localhost:5000

📓 Jupyter:
   URL: http://localhost:8888
```

---

## ❌ TROUBLESHOOTING

### Problem: "Docker command not found"
**Solution**: Install Docker Desktop first (see top of guide)

### Problem: "Port 5000 already in use"
**Solution**:
```bash
# Find what's using port 5000
lsof -i :5000

# Stop conflicting service or change port in docker-compose.yml
```

### Problem: "MLflow UI shows empty"
**Solution**:
```bash
# Check if containers are running
docker ps

# Restart if needed
docker-compose down
docker-compose up -d

# Wait 30 seconds, then try again
```

### Problem: "Can't connect to database"
**Solution**:
```bash
# Check container logs
docker-compose logs postgres
docker-compose logs mlflow

# Restart everything
docker-compose down -v  # This deletes data!
docker-compose up -d
```

### Problem: "Jupyter won't start"
**Solution**:
```bash
# Install Jupyter
pip install jupyter

# Or use Anaconda Navigator (GUI)
```

---

## 🎯 WHAT STUDENTS WILL LEARN

✅ **MLflow basics**: What it is and why use it  
✅ **Real database**: PostgreSQL backend storage  
✅ **Experiment tracking**: Parameters, metrics, models  
✅ **Model comparison**: Side-by-side analysis  
✅ **Production setup**: Docker containerization  
✅ **Best practices**: Organized ML workflows  

---

## 🧹 CLEANUP (After Demo)

```bash
# Stop services
docker-compose down

# Remove all data (optional)
docker-compose down -v

# Remove Docker images (optional)
docker rmi postgres:13 python:3.9-slim
```

---

**🎉 You're all set! The credentials are automatically created by Docker - no manual setup needed!**