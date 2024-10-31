# requirements.txt
pandas==2.1.0
scikit-learn==1.3.0
flask==2.3.3
joblib==1.3.2
pytest==7.4.2
numpy==1.24.3

# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic trade data for testing"""
    np.random.seed(42)
    
    data = {
        'trade_size': np.random.uniform(1000, 1000000, n_samples),
        'asset_type': np.random.choice([1, 2], n_samples),  # 1: Stock, 2: Bond
        'counterparty': np.random.choice([1, 2, 3], n_samples),  # Different banks
        'time_to_settle': np.random.choice([1, 2, 3], n_samples),  # Days to settle
    }
    
    # Create a more realistic failed_trade column based on certain conditions
    data['failed_trade'] = np.where(
        (data['trade_size'] > 800000) |  # Large trades more likely to fail
        ((data['time_to_settle'] == 3) & (data['asset_type'] == 2)) |  # Bonds with longer settlement
        ((data['counterparty'] == 3) & (data['trade_size'] > 500000)),  # Specific counterparty with large trades
        1, 0
    )
    
    return pd.DataFrame(data)

def train_and_evaluate_model():
    """Train the model and print evaluation metrics"""
    # Generate synthetic data
    df = generate_synthetic_data()
    
    # Split features and target
    X = df[['trade_size', 'asset_type', 'counterparty', 'time_to_settle']]
    y = df['failed_trade']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print metrics
    print("\nModel Evaluation Metrics:")
    print("-------------------------")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, 'trade_reconciliation_model.pkl')
    
    return model

# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
try:
    model = joblib.load('trade_reconciliation_model.pkl')
except:
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.get_json()
        input_data = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0]
        
        result = {
            'failed_trade': bool(prediction[0]),
            'probability': {
                'success': float(probability[0]),
                'failure': float(probability[1])
            }
        }
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# test_model.py
import pytest
import requests
import numpy as np

def test_model_prediction():
    """Test the model's prediction endpoint"""
    # Test data
    test_data = {
        'features': [500000, 2, 1, 2]  # [trade_size, asset_type, counterparty, time_to_settle]
    }
    
    # Make prediction request
    response = requests.post('http://localhost:5000/predict', json=test_data)
    
    # Verify response
    assert response.status_code == 200
    result = response.json()
    assert 'failed_trade' in result
    assert 'probability' in result

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
