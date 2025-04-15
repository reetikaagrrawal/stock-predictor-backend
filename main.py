from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import yfinance as yf
from keras.models import load_model # type: ignore
import joblib

app = FastAPI()

# Get port from environment variable (for Render)
port = int(os.getenv("PORT", 8000))

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
model = load_model('model/lstm_model.h5')
scaler = joblib.load('model/scaler.pkl')

class StockRequest(BaseModel):
    stock_symbol: str

@app.post("/predict")
def predict_stock(stock: StockRequest):
    try:
        df = yf.download(stock.stock_symbol, period="90d")
        data = df[['Close']]
        scaled_data = scaler.transform(data)

        last_60 = scaled_data[-60:]
        X_input = np.reshape(last_60, (1, 60, 1))
        prediction = model.predict(X_input)
        predicted_prices = scaler.inverse_transform(prediction).flatten().tolist()

        return {
            "stock": stock.stock_symbol,
            "predicted_prices": predicted_prices
        }
    except Exception as e:
        return {"error": str(e)}
