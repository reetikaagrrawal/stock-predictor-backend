import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import yfinance as yf
from keras.models import load_model # type: ignore
import joblib
import os
from datetime import datetime, timedelta

app = FastAPI(title="Stock Predictor API")

# Get port from environment variable (for Render)
port = int(os.getenv("PORT", 8000))

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler with error handling
try:
    model = load_model('model/lstm_model.h5')
    scaler = joblib.load('model/scaler.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

class StockRequest(BaseModel):
    stock_symbol: str

class PredictionResponse(BaseModel):
    stock: str
    predicted_prices: list[float]
    prediction_dates: list[str]

@app.post("/predict", response_model=PredictionResponse)
async def predict_stock(stock: StockRequest):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Download stock data
        df = yf.download(stock.stock_symbol, period="90d")
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {stock.stock_symbol}")
        
        # Prepare data
        data = df[['Close']]
        scaled_data = scaler.transform(data)
        last_60 = scaled_data[-60:]
        X_input = np.reshape(last_60, (1, 60, 1))
        
        # Make prediction
        prediction = model.predict(X_input)
        predicted_prices = scaler.inverse_transform(prediction).flatten().tolist()
        
        # Generate prediction dates (excluding weekends)
        dates = []
        current_date = datetime.now()
        for i in range(len(predicted_prices)):
            next_date = current_date + timedelta(days=i+1)
            while next_date.weekday() > 4:  # Skip weekends
                next_date += timedelta(days=1)
            dates.append(next_date.strftime("%Y-%m-%d"))
        
        return {
            "stock": stock.stock_symbol,
            "predicted_prices": predicted_prices,
            "prediction_dates": dates
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}