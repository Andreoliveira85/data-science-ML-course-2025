# 🔐 PostgreSQL Database Access - Quick Reference

## 📋 Default Credentials

The Docker Compose setup creates PostgreSQL with these **demo credentials**:

```
Host:     localhost
Port:     5432
Database: mlflow_db
Username: mlflow_user
Password: mlflow_pass
```

> ⚠️ **Security Note**: These are demo credentials! In production:
> - Use strong, unique passwords
> - Store credentials in environment variables
> - Use secrets management tools
> - Restrict database access

---

## 🚀 Quick Start - Get Connected

### Method 1: Docker Command Line Access
```bash
# Connect directly to PostgreSQL inside Docker container
docker exec -it mlflow-demo_postgres_1 psql -U mlflow_user -d mlflow_db

# Alternative if container name is different
docker ps  # Find your postgres container name
docker exec -it [POSTGRES_CONTAINER_NAME] psql -U mlflow_user -d mlflow_db
```

### Method 2: Local psql (if installed)
```bash
# Connect from your local machine
psql -h localhost -p 5432 -U mlflow_user -d mlflow_db
# When prompted, enter password: mlflow_pass
```

### Method 3: Python Connection
```python
import psycopg2
import pandas as pd

# Connection string
conn_string = "postgresql://mlflow_user:mlflow_pass@localhost:5432/mlflow_db"

# Or using parameters
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="mlflow_db",
    user="mlflow_user",
    password="mlflow_pass"
)

# Test connection
cursor = conn.cursor()
cursor.execute("SELECT version();")
print("PostgreSQL version:", cursor.fetchone())
conn.close()
```

---

## 🔍 Useful Database Queries

### Basic Exploration
```sql
-- List all tables
\dt

-- See table structure
\d experiments
\d runs
\d metrics

-- Count records
SELECT COUNT(*) FROM experiments;
SELECT COUNT(*) FROM runs;
SELECT COUNT(*) FROM metrics;
```

### MLflow-Specific Queries
```sql
-- View all experiments
SELECT experiment_id, name, lifecycle_stage 
FROM experiments;

-- View recent runs
SELECT run_uuid, experiment_id, status, start_time 
FROM runs 
ORDER BY start_time DESC 
LIMIT 10;

-- View all metrics for a specific run
SELECT key, value, timestamp 
FROM metrics 
WHERE run_uuid = 'YOUR_RUN_ID'
ORDER BY timestamp;

-- Compare accuracy across all runs
SELECT r.run_uuid, r.experiment_id, m.value as accuracy
FROM runs r
JOIN metrics m ON r.run_uuid = m.run_uuid
WHERE m.key = 'accuracy'
ORDER BY m.value DESC;
```

---

## 🛠 GUI Tools Setup

### pgAdmin (Web-based)
1. **Install pgAdmin**: Download from https://www.pgadmin.org/
2. **Create Server Connection**:
   - Right-click "Servers" → "Create" → "Server"
   - **General Tab**: Name = "MLflow Demo"
   - **Connection Tab**:
     - Host: `localhost`
     - Port: `5432`
     - Database: `mlflow_db`
     - Username: `mlflow_user`
     - Password: `mlflow_pass`

### DBeaver (Desktop)
1. **Install DBeaver**: Download from https://dbeaver.io/
2. **New Connection**:
   - Select PostgreSQL
   - Host: `localhost`
   - Port: `5432`
   - Database: `mlflow_db`
   - Username: `mlflow_user`
   - Password: `mlflow_pass`

### VS Code Extension
1. **Install**: PostgreSQL extension by Chris Kolkman
2. **Connect**: Use connection string:
   ```
   postgresql://mlflow_user:mlflow_pass@localhost:5432/mlflow_db
   ```

---

## 🔧 Troubleshooting

### Connection Issues

**Problem**: `psql: error: connection to server at "localhost" (127.0.0.1), port 5432 failed`

**Solutions**:
```bash
# 1. Check if PostgreSQL container is running
docker ps | grep postgres

# 2. Check if port 5432 is accessible
docker port [POSTGRES_CONTAINER_NAME]

# 3. Restart the services
docker-compose down
docker-compose up -d

# 4. Check container logs
docker logs [POSTGRES_CONTAINER_NAME]
```

**Problem**: `password authentication failed for user "mlflow_user"`

**Solutions**:
```bash
# 1. Verify credentials in docker-compose.yml
cat docker-compose.yml | grep -A 5 POSTGRES

# 2. Reset the database (WARNING: This deletes all data)
docker-compose down -v
docker-compose up -d
```

**Problem**: Container name not found

**Solutions**:
```bash
# Find the actual container name
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"

# Use the correct container name in commands
docker exec -it [ACTUAL_CONTAINER_NAME] psql -U mlflow_user -d mlflow_db
```

---

## 📊 Verify MLflow Integration

### Check MLflow Tables
```sql
-- These tables should exist after running MLflow experiments
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name LIKE '%mlflow%' OR table_name IN 
('experiments', 'runs', 'metrics', 'params', 'tags');
```

### Sample Data Check
```sql
-- After running the demo notebook, you should see:
SELECT 
    e.name as experiment_name,
    COUNT(r.run_uuid) as num_runs
FROM experiments e
LEFT JOIN runs r ON e.experiment_id = r.experiment_id
GROUP BY e.name;
```

---

## 🔐 Production Security Tips

### Environment Variables Approach
```bash
# Create .env file (never commit to git!)
echo "POSTGRES_DB=mlflow_db" > .env
echo "POSTGRES_USER=mlflow_user" >> .env
echo "POSTGRES_PASSWORD=$(openssl rand -base64 32)" >> .env

# Update docker-compose.yml to use .env
env_file:
  - .env
```

### Secrets Management
```yaml
# For production, use Docker secrets
services:
  postgres:
    secrets:
      - postgres_password
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password

secrets:
  postgres_password:
    file: ./postgres_password.txt
```

---

## 📞 Quick Help Commands

```bash
# Get help inside psql
\?

# Exit psql
\q

# Show current connection info
\conninfo

# List databases
\l

# Switch database
\c mlflow_db

# Show table contents (limit 5 rows)
SELECT * FROM experiments LIMIT 5;
```

---

## 🎯 Next Steps

Once connected to PostgreSQL:
1. Run the MLflow demo notebook
2. Check that data appears in the database
3. Explore the MLflow UI at http://localhost:5000
4. Compare database content with UI content
5. Try your own experiments!

---

**Remember**: These credentials are for local development only. Always use proper security practices in production environments!