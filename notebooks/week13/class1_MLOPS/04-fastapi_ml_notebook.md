# FastAPI Machine Learning Demo - 30 Minute Tutorial

## 🎯 Learning Objectives
By the end of this tutorial, you will:
- Create a simple machine learning model
- Build a REST API using FastAPI
- Deploy and test your ML model via API endpoints
- Understand the basics of serving ML models in production

## 📋 Prerequisites
- Python 3.7+ installed
- Basic understanding of Python
- 15 minutes for setup, 15 minutes for coding!

---

## 🚀 Step 1: Setup (5 minutes)

### Install Required Packages
```bash
python3 -m venv fastapi-env    
source fastapi-env/bin/activate
pip install fastapi uvicorn scikit-learn pandas numpy
```

### Create Project Structure
```
ml_fastapi_demo/
├── main.py          # Our FastAPI application
├── model.py         # Machine learning model
└── requirements.txt # Dependencies
```

---

1.1 Introduction to FastAPI (15 minutes)

What is FastAPI?

Modern, fast web framework for building APIs with Python
Created by Sebastian Ramirez in December 2018
Built on top of Starlette for web parts and Pydantic for data parts
Key features: High performance, easy to use, automatic documentation

Why FastAPI?

Fast: Very high performance, on par with NodeJS and Go
Fast to code: Increase development speed by about 200% to 300%
Fewer bugs: Reduce about 40% of human errors
Intuitive: Great editor support with autocompletion
Easy: Designed to be easy to use and learn
Short: Minimize code duplication

## 🧠 Step 2: Create the Machine Learning Model (10 minutes)

Create `model.py`:

```python
# model.py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

class HousePricePredictor:
    """Simple house price prediction model based on size and rooms"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
    
    def create_sample_data(self):
        """Generate sample house data for training"""
        print("🏠 Creating sample house data...")
        
        # Generate synthetic data
        np.random.seed(42)  # For reproducible results
        n_samples = 1000
        
        # Features: house_size (sqft), num_rooms
        house_sizes = np.random.normal(2000, 500, n_samples)
        num_rooms = np.random.randint(1, 6, n_samples)
        
        # Create realistic price relationship
        # Price = base_price + size_factor * sqft + room_factor * rooms + noise
        prices = (
            100000 +  # Base price
            150 * house_sizes +  # $150 per sqft
            10000 * num_rooms +  # $10k per room
            np.random.normal(0, 20000, n_samples)  # Random noise
        )
        
        # Combine features
        X = np.column_stack([house_sizes, num_rooms])
        y = prices
        
        return X, y
    
    def train(self):
        """Train the model with sample data"""
        print("🎯 Training the model...")
        
        # Get training data
        X, y = self.create_sample_data()
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        score = self.model.score(X_test, y_test)
        print(f"✅ Model trained! R² Score: {score:.3f}")
        
        self.is_trained = True
        return score
    
    def predict(self, house_size: float, num_rooms: int):
        """Make a prediction for house price"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Prepare input data
        features = np.array([[house_size, num_rooms]])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        return max(0, prediction)  # Ensure non-negative price
    
    def save_model(self, filename="house_model.pkl"):
        """Save the trained model"""
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"💾 Model saved as {filename}")
    
    def load_model(self, filename="house_model.pkl"):
        """Load a pre-trained model"""
        try:
            with open(filename, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True
            print(f"📂 Model loaded from {filename}")
        except FileNotFoundError:
            print(f"❌ Model file {filename} not found")

# Test the model (run this if executing model.py directly)
if __name__ == "__main__":
    predictor = HousePricePredictor()
    predictor.train()
    
    # Test prediction
    price = predictor.predict(house_size=2500, num_rooms=4)
    print(f"🏡 Predicted price for 2500 sqft, 4 rooms: ${price:,.2f}")
```

---

## 🌐 Step 3: Create the FastAPI Application (10 minutes)

Create `main.py`:

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from model import HousePricePredictor
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="A simple ML API that predicts house prices based on size and number of rooms",
    version="1.0.0"
)

# Initialize our ML model
predictor = HousePricePredictor()

# Pydantic models for request/response validation
class HouseFeatures(BaseModel):
    """Input features for house price prediction"""
    house_size: float = Field(..., gt=0, description="House size in square feet")
    num_rooms: int = Field(..., ge=1, le=10, description="Number of rooms (1-10)")
    
    class Config:
        schema_extra = {
            "example": {
                "house_size": 2500.0,
                "num_rooms": 4
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_price: float
    house_size: float
    num_rooms: int
    message: str

# API Routes
@app.on_event("startup")
async def startup_event():
    """Train model on startup"""
    print("🚀 Starting up the API...")
    try:
        # Try to load existing model, or train new one
        predictor.load_model()
    except:
        print("No existing model found, training new one...")
        predictor.train()
        predictor.save_model()

@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "🏠 Welcome to the House Price Prediction API!",
        "docs": "Visit /docs for API documentation",
        "health": "Visit /health for system status"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_trained": predictor.is_trained,
        "message": "API is running smoothly! 🎯"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(house: HouseFeatures):
    """
    Predict house price based on features
    
    - **house_size**: Size of the house in square feet (must be > 0)
    - **num_rooms**: Number of rooms (must be between 1-10)
    
    Returns predicted price in USD.
    """
    try:
        # Make prediction
        predicted_price = predictor.predict(
            house_size=house.house_size,
            num_rooms=house.num_rooms
        )
        
        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            house_size=house.house_size,
            num_rooms=house.num_rooms,
            message=f"Prediction successful! 🎯"
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the current model"""
    if not predictor.is_trained:
        return {"error": "Model not trained yet"}
    
    # Get model coefficients for interpretation
    coefficients = predictor.model.coef_
    intercept = predictor.model.intercept_
    
    return {
        "model_type": "Linear Regression",
        "features": ["house_size", "num_rooms"],
        "coefficients": {
            "house_size": round(coefficients[0], 2),
            "num_rooms": round(coefficients[1], 2)
        },
        "intercept": round(intercept, 2),
        "interpretation": {
            "house_size": f"Each additional sqft adds ${coefficients[0]:.2f} to price",
            "num_rooms": f"Each additional room adds ${coefficients[1]:,.2f} to price"
        }
    }

# Run the app
if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True  # Auto-reload on code changes
    )
```

---

## 🔧 Step 4: Create Requirements File (1 minute)

Create `requirements.txt`:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
pydantic==2.4.2
```

---

## 🏃‍♂️ Step 5: Run and Test the API (4 minutes)

### 1. Start the Server
```bash
cd ml_fastapi_demo
python main.py
```

You should see:
```
🚀 Starting FastAPI server...
🏠 Creating sample house data...
🎯 Training the model...
✅ Model trained! R² Score: 0.xxx
💾 Model saved as house_model.pkl
INFO: Started server process
INFO: Uvicorn running on http://0.0.0.0:8000
```

### 2. Test the API

**Option A: Interactive Documentation (Recommended)**
- Visit: `http://localhost:8000/docs`
- Try the `/predict` endpoint with sample data

**Option B: Command Line (using curl)**
```bash
# Test prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"house_size": 2500, "num_rooms": 4}'

# Check health
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model-info
```

**Option C: Python requests**
```python
import requests

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"house_size": 2500, "num_rooms": 4}
)
print(response.json())
```

---

## 🎓 Teaching Points & Discussion (bonus content)

### Key Concepts Covered:
1. **Model Training**: Simple linear regression with synthetic data
2. **API Design**: RESTful endpoints with proper HTTP methods
3. **Data Validation**: Using Pydantic for input/output validation
4. **Error Handling**: Proper HTTP status codes and error messages
5. **Documentation**: Automatic API docs with FastAPI

### Extensions for Advanced Students:
- Add more features (location, age of house, etc.)
- Implement model versioning
- Add authentication
- Deploy to cloud (Heroku, AWS, etc.)
- Add logging and monitoring
- Use a real dataset (California housing, Boston housing)

### Production Considerations:
- Model persistence and versioning
- Input validation and sanitization
- Rate limiting and security
- Monitoring and logging
- Database integration
- Containerization with Docker

---

## 🏆 Congratulations!

You've successfully created and deployed a machine learning API! Your students now understand:
- How to wrap ML models in web APIs
- Basic FastAPI concepts
- API testing and documentation
- The bridge between data science and software engineering

**Next Steps**: Try modifying the model, adding new endpoints, or using real datasets!

---

## 📝 Quick Reference

### Useful Commands:
```bash
# Start development server
uvicorn main:app --reload

# Install dependencies
pip install -r requirements.txt

# Test API endpoint
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"house_size": 2000, "num_rooms": 3}'
```

### Important URLs:
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`