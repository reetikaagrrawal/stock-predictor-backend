# Stock Predictor Backend

A FastAPI backend service for stock price prediction using LSTM neural networks.

## Features
- Stock price prediction using LSTM
- Historical data analysis
- 5-day price forecasting
- RESTful API endpoints

## Tech Stack
- Python 3.10
- FastAPI
- TensorFlow
- scikit-learn
- pandas
- yfinance

## Setup
1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
uvicorn main:app --reload
```
