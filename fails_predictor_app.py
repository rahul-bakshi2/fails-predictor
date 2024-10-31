import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import ta

class MarketDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def fetch_intraday_data(self, symbol):
        """Fetch real-time market data from Alpha Vantage"""
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": "5min",
            "apikey": self.api_key,
            "outputsize": "full"
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if "Time Series (5min)" not in data:
                st.error(f"Error fetching data: {data.get('Note', 'Unknown error')}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data["Time Series (5min)"], orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume"
            })
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            # Calculate additional metrics
            self._calculate_market_metrics(df)
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching market data: {str(e)}")
            return None
    
    def _calculate_market_metrics(self, df):
        """Calculate market metrics for risk assessment"""
        # Volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=12).std() * np.sqrt(12)  # Annualized
        
        # Volume metrics
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['relative_volume'] = df['volume'] / df['volume_ma']
        
        # Price metrics
        df['price_range'] = (df['high'] - df['low']) / df['low']
        df['average_trade_size'] = df['volume'] * df['close'] / df['volume'].count()
        
        # Liquidity indicator
        df['spread'] = (df['high'] - df['low']) / df['close']
        
        return df

class TradeFailurePredictor:
    def __init__(self):
        self.failure_thresholds = {
            'large_trade': 800000,
            'high_volatility': 0.02,
            'low_liquidity': 0.3,
            'late_day_cutoff': 15
        }
    
    def predict_failure_risk(self, trade_data, market_data, client_history=None):
        """Calculate trade failure risk using real market data"""
        if market_data is None:
            return 1.0, [{'factor': 'Market Data', 
                         'risk_level': 'High', 
                         'description': 'Unable to fetch market data for risk assessment',
                         'contribution': 1.0}]
        
        current_market = market_data.iloc[-1]
        risk_factors = []
        total_risk_score = 0
        
        # 1. Trade Size Risk (25% weight)
        size_risk = self._calculate_size_risk(
            trade_data['trade_size'], 
            current_market['average_trade_size'],
            client_history
        )
        if size_risk > 0:
            risk_factors.append({
                'factor': 'Trade Size',
                'risk_level': 'High' if size_risk > 0.7 else 'Medium',
                'description': f'Trade size (${trade_data["trade_size"]:,.0f}) is significantly larger than average market trade size (${current_market["average_trade_size"]:,.0f})',
                'contribution': size_risk * 0.25
            })
            total_risk_score += size_risk * 0.25
        
        # 2. Market Volatility Risk (20% weight)
        volatility_risk = self._calculate_volatility_risk(current_market['volatility'])
        if volatility_risk > 0:
            risk_factors.append({
                'factor': 'Market Volatility',
                'risk_level': 'High' if volatility_risk > 0.7 else 'Medium',
                'description': f'Current volatility ({current_market["volatility"]:.1%}) exceeds normal levels',
                'contribution': volatility_risk * 0.20
            })
            total_risk_score += volatility_risk * 0.20
        
        # 3. Liquidity Risk (20% weight)
        liquidity_risk = self._calculate_liquidity_risk(current_market['relative_volume'])
        if liquidity_risk > 0:
            risk_factors.append({
                'factor': 'Liquidity',
                'risk_level': 'High' if liquidity_risk > 0.7 else 'Medium',
                'description': f'Market liquidity ({current_market["relative_volume"]:.2f}x normal) is below threshold',
                'contribution': liquidity_risk * 0.20
            })
            total_risk_score += liquidity_risk * 0.20
        
        # 4. Market Timing Risk (20% weight)
        timing_risk = self._calculate_timing_risk(trade_data['time_of_day'])
        if timing_risk > 0:
            risk_factors.append({
                'factor': 'Market Timing',
                'risk_level': 'High' if timing_risk > 0.7 else 'Medium',
                'description': f'Late-day trading (after {self.failure_thresholds["late_day_cutoff"]:02d}:00) increases settlement risk',
                'contribution': timing_risk * 0.20
            })
            total_risk_score += timing_risk * 0.20
        
        return total_risk_score, risk_factors
    
    def _calculate_size_risk(self, trade_size, market_avg_trade_size, client_history=None):
        relative_size = trade_size / market_avg_trade_size if market_avg_trade_size > 0 else float('inf')
        base_risk = min(1.0, relative_size / 10)  # Scale based on market average
        
        if client_history and 'average_trade_size' in client_history:
            if trade_size <= client_history['average_trade_size'] * 1.5:
                base_risk *= 0.5  # Reduce risk if size is normal for client
        
        return base_risk
    
    def _calculate_volatility_risk(self, volatility):
        return min(1.0, volatility / self.failure_thresholds['high_volatility'])
    
    def _calculate_liquidity_risk(self, relative_volume):
        if relative_volume < self.failure_thresholds['low_liquidity']:
            return min(1.0, (self.failure_thresholds['low_liquidity'] - relative_volume) / 
                      self.failure_thresholds['low_liquidity'])
        return 0.0
    
    def _calculate_timing_risk(self, time_of_day):
        if time_of_day >= self.failure_thresholds['late_day_cutoff']:
            return min(1.0, (time_of_day - self.failure_thresholds['late_day_cutoff']) / 
                      (24 - self.failure_thresholds['late_day_cutoff']))
        return 0.0

def main():
    st.title("🏦 Trade Failure Risk Assessment")
    st.markdown("""
    Predict trade failure risk using real-time market data and trade characteristics.
    """)
    
    # Initialize market data fetcher with API key
    api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
    market_fetcher = MarketDataFetcher(api_key)
    predictor = TradeFailurePredictor()
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.text_input("Stock Symbol", "IBM").upper()
        trade_size = st.number_input(
            "Trade Size ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000
        )
    
    with col2:
        client_id = st.text_input("Client ID (optional)")
        time_of_day = st.selectbox(
            "Time of Trade",
            list(range(9, 17)),
            format_func=lambda x: f"{x:02d}:00"
        )
    
    # Assess Risk button
    if st.button("Assess Trade Failure Risk"):
        with st.spinner("Fetching market data..."):
            # Fetch real market data
            market_data = market_fetcher.fetch_intraday_data(symbol)
            
            if market_data is not None:
                # Prepare trade data
                trade_data = {
                    'symbol': symbol,
                    'trade_size': trade_size,
                    'time_of_day': time_of_day
                }
                
                # Optional client history
                client_history = None
                if client_id:
                    client_history = {
                        'average_trade_size': 50000,  # Example value
                        'failure_rate': 0.05,         # Example value
                    }
                
                # Calculate risk
                risk_score, risk_factors = predictor.predict_failure_risk(
                    trade_data, market_data, client_history
                )
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Failure Risk Score",
                        f"{risk_score:.1%}",
                        delta="High Risk" if risk_score > 0.7 else 
                              "Medium Risk" if risk_score > 0.3 else 
                              "Low Risk",
                        delta_color="inverse"
                    )
                
                with col2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_score * 100,
                        title={'text': "Risk Level"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "red" if risk_score > 0.7 else 
                                           "orange" if risk_score > 0.3 else 
                                           "green"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "lightyellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig)
                
                # Risk Breakdown
                st.subheader("Risk Factor Analysis")
                for factor in risk_factors:
                    with st.expander(f"{factor['factor']} - {factor['risk_level']} Risk"):
                        st.write(factor['description'])
                        st.progress(factor['contribution'])
                
                # Market Data Visualization
                if st.checkbox("Show Market Data Analysis"):
                    st.subheader("Market Conditions")
                    
                    fig = px.line(market_data, 
                                y=['close', 'volatility', 'relative_volume'],
                                title=f"{symbol} Market Metrics")
                    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
