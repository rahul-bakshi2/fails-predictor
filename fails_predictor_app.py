import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Trade Failure Predictor",
    page_icon="🏦",
    layout="wide"
)

# Generate sample data
def generate_trade_data(n_samples=1000):
    np.random.seed(42)
    
    data = {
        'trade_size': np.random.uniform(1000, 1000000, n_samples),
        'asset_type': np.random.choice([1, 2], n_samples),  # 1: Stock, 2: Bond
        'counterparty': np.random.choice([1, 2, 3], n_samples),  # Banks
        'time_to_settle': np.random.choice([1, 2, 3], n_samples)  # Days
    }
    
    # Create failed_trade column based on realistic conditions
    data['failed_trade'] = np.where(
        (data['trade_size'] > 800000) |  # Large trades more likely to fail
        ((data['time_to_settle'] == 3) & (data['asset_type'] == 2)) |  # Bonds with long settlement
        ((data['counterparty'] == 3) & (data['trade_size'] > 500000)),  # High-risk counterparty
        1, 0
    )
    
    return pd.DataFrame(data)

# Train model function
def train_trade_model():
    df = generate_trade_data()
    X = df[['trade_size', 'asset_type', 'counterparty', 'time_to_settle']]
    y = df['failed_trade']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    return model, df

# Main app
def main():
    st.title("🏦 Trade Failure Prediction System")
    st.markdown("""
    This system predicts the likelihood of trade failures based on key trade attributes.
    Enter your trade details below to get a prediction.
    """)
    
    # Initialize model
    if 'model' not in st.session_state:
        st.session_state.model, st.session_state.df = train_trade_model()
    
    # Input form
    st.subheader("Trade Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        trade_size = st.number_input(
            "Trade Size ($)",
            min_value=1000,
            max_value=1000000,
            value=500000,
            step=10000
        )
        
        asset_type = st.selectbox(
            "Asset Type",
            options=[1, 2],
            format_func=lambda x: "Stock" if x == 1 else "Bond"
        )
    
    with col2:
        counterparty = st.selectbox(
            "Counterparty Bank",
            options=[1, 2, 3],
            format_func=lambda x: f"Bank {x}"
        )
        
        time_to_settle = st.selectbox(
            "Settlement Time (Days)",
            options=[1, 2, 3]
        )
    
    # Make prediction
    if st.button("Predict Trade Risk"):
        features = np.array([trade_size, asset_type, counterparty, time_to_settle]).reshape(1, -1)
        prediction = st.session_state.model.predict(features)[0]
        probability = st.session_state.model.predict_proba(features)[0][1]
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Trade Status",
                "High Risk" if prediction == 1 else "Low Risk",
                f"{probability:.1%} Failure Probability"
            )
            
            # Risk factors
            st.subheader("Risk Factors:")
            if trade_size > 800000:
                st.warning("⚠️ Large trade size increases risk")
            if time_to_settle == 3 and asset_type == 2:
                st.warning("⚠️ Extended settlement time for bonds")
            if counterparty == 3 and trade_size > 500000:
                st.warning("⚠️ High-risk counterparty with large trade")
        
        with col2:
            # Gauge chart for failure probability
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Failure Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if probability > 0.7 else "orange" if probability > 0.3 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ]
                }
            ))
            st.plotly_chart(fig)
    
    # Show historical data visualization
    if st.checkbox("Show Historical Trade Analysis"):
        st.subheader("Historical Trade Size vs Failure Rate")
        
        fig = px.scatter(
            st.session_state.df,
            x='trade_size',
            y='failed_trade',
            color='asset_type',
            color_discrete_map={1: 'blue', 2: 'red'},
            labels={
                'trade_size': 'Trade Size ($)',
                'failed_trade': 'Failed Trade (1=Yes, 0=No)',
                'asset_type': 'Asset Type'
            },
            title="Historical Trade Failures"
        )
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
