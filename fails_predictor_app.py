import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from io import StringIO, BytesIO
import xlsxwriter
from datetime import datetime, timedelta
import ta

class MarketDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.cache = {}
    
    def fetch_intraday_data(self, symbol):
        """Fetch real-time market data from Polygon.io"""
        if symbol in self.cache:
            return self.cache[symbol]
            
        # Get today's date and previous trading day
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/5/minute/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        headers = {'Authorization': f'Bearer {self.api_key}'}
        
        try:
            response = requests.get(url, headers=headers)
            data = response.json()
            
            if 'results' not in data:
                st.warning(f"No data available for {symbol}, using simulated data")
                return self._generate_mock_data(symbol)
                
            # Convert to DataFrame
            df = pd.DataFrame(data['results'])
            df.index = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })
            
            # Keep only necessary columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Calculate additional metrics
            self._calculate_market_metrics(df)
            
            # Cache the result
            self.cache[symbol] = df
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching market data for {symbol}: {str(e)}")
            return self._generate_mock_data(symbol)
    
    def _calculate_market_metrics(self, df):
        """Calculate market metrics for risk assessment"""
        # Volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=12).std() * np.sqrt(12)
        
        # Volume metrics
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['relative_volume'] = df['volume'] / df['volume_ma']
        
        # Price metrics
        df['price_range'] = (df['high'] - df['low']) / df['low']
        df['average_trade_size'] = df['volume'] * df['close'] / df['volume'].count()
        
        # Liquidity indicator
        df['spread'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def _generate_mock_data(self, symbol):
        """Generate mock data when API fails"""
        st.warning(f"Using simulated data for {symbol}")
        dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
        df = pd.DataFrame(index=dates)
        df['close'] = np.random.uniform(100, 200, len(dates))
        df['volume'] = np.random.uniform(1000, 10000, len(dates))
        df['high'] = df['close'] * (1 + np.random.uniform(0, 0.02, len(dates)))
        df['low'] = df['close'] * (1 - np.random.uniform(0, 0.02, len(dates)))
        df['open'] = df['close'] * (1 + np.random.uniform(-0.01, 0.01, len(dates)))
        self._calculate_market_metrics(df)
        return df

class PCFValidator:
    REQUIRED_COLUMNS = {
        'Ticker': str,
        'Cusip': str,
        'Asset Class': str,
        'Securities Description': str,
        'Weight of Holdings': float,
        'Shares': float,
        'Market Value': float
    }
    
    @staticmethod
    def validate_pcf(df):
        """Validate PCF format and data types"""
        # Check for required columns
        missing_cols = [col for col in PCFValidator.REQUIRED_COLUMNS.keys() 
                       if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Validate data types and handle conversions
        for col, dtype in PCFValidator.REQUIRED_COLUMNS.items():
            try:
                if dtype == float:
                    # Handle if already numeric
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].astype(float)
                    else:
                        # Handle string format with currency and percentage symbols
                        df[col] = df[col].replace('[\$,%]', '', regex=True)
                        df[col] = pd.to_numeric(df[col])
                elif dtype == str:
                    df[col] = df[col].astype(str)
            except Exception as e:
                raise ValueError(f"Invalid data in column {col}: {str(e)}")
        
        return df

class RiskAnalyzer:
    def __init__(self):
        self.risk_thresholds = {
            'high_market_value': 1000000,  # $1M
            'high_weight': 0.05,  # 5%
            'low_liquidity': 0.3,
            'high_volatility': 0.02
        }
    
    def analyze_security(self, security_data, market_data):
        """Calculate risk metrics for a single security"""
        risk_score = 0
        risk_factors = []
        
        # 1. Market Value Risk
        if security_data['Market Value'] > self.risk_thresholds['high_market_value']:
            risk_score += 0.3
            risk_factors.append({
                'factor': 'Market Value',
                'level': 'High',
                'description': f"Large position size: ${security_data['Market Value']:,.2f}"
            })
        
        # 2. Weight Risk
        if security_data['Weight of Holdings'] > self.risk_thresholds['high_weight']:
            risk_score += 0.2
            risk_factors.append({
                'factor': 'Position Weight',
                'level': 'High',
                'description': f"High concentration: {security_data['Weight of Holdings']:.1%}"
            })
        
        # 3. Liquidity Risk
        if market_data is not None:
            avg_volume = market_data['volume'].mean()
            days_to_liquidate = security_data['Shares'] / avg_volume
            if days_to_liquidate > 1:
                risk_score += 0.3
                risk_factors.append({
                    'factor': 'Liquidity',
                    'level': 'High',
                    'description': f"Would take {days_to_liquidate:.1f} days to liquidate"
                })
        
        # 4. Asset Class Risk
        asset_class_risk = self.get_asset_class_risk(security_data['Asset Class'])
        risk_score += asset_class_risk['score']
        risk_factors.append({
            'factor': 'Asset Class',
            'level': asset_class_risk['level'],
            'description': asset_class_risk['description']
        })
        
        return {
            'risk_score': min(risk_score, 1.0),
            'risk_factors': risk_factors
        }
    
    @staticmethod
    def get_asset_class_risk(asset_class):
        """Get risk metrics based on asset class"""
        risk_metrics = {
            'Equity': {
                'score': 0.2,
                'level': 'Medium',
                'description': 'Standard equity settlement risk'
            },
            'Fixed Income': {
                'score': 0.3,
                'level': 'Medium-High',
                'description': 'Extended settlement cycle risk'
            },
            'Options': {
                'score': 0.4,
                'level': 'High',
                'description': 'Complex settlement requirements'
            }
        }
        return risk_metrics.get(asset_class, {
            'score': 0.25,
            'level': 'Medium',
            'description': 'Standard settlement risk'
        })

class ReportGenerator:
    @staticmethod
    def generate_excel_report(analysis_results):
        """Generate detailed Excel report of analysis"""
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_data = pd.DataFrame([{
                'Metric': 'Total Securities',
                'Value': analysis_results['summary']['total_securities']
            }, {
                'Metric': 'High Risk Securities',
                'Value': analysis_results['summary']['high_risk_securities']
            }, {
                'Metric': 'Total Market Value',
                'Value': f"${analysis_results['summary']['total_market_value']:,.2f}"
            }])
            
            summary_data.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed Analysis
            details_data = pd.DataFrame([{
                'Ticker': r['Ticker'],
                'Description': r['Description'],
                'Asset Class': r['Asset Class'],
                'Market Value': r['Market Value'],
                'Risk Score': r['Risk Score'],
                'Risk Level': 'High' if r['Risk Score'] >= 0.7 else 'Medium' if r['Risk Score'] >= 0.4 else 'Low'
            } for r in analysis_results['details']])
            
            details_data.to_excel(writer, sheet_name='Security Details', index=False)
            
            # Formatting
            workbook = writer.book
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D3D3D3',
                'border': 1
            })
            
            risk_format = workbook.add_format({
                'bg_color': '#FFB6C1',
                'border': 1
            })
            
            # Apply formats
            for sheet in writer.sheets.values():
                for col_num, value in enumerate(details_data.columns.values):
                    sheet.write(0, col_num, value, header_format)
                sheet.set_column('A:Z', 15)
            
            worksheet = writer.sheets['Security Details']
            worksheet.conditional_format('E2:E1000', {
                'type': 'cell',
                'criteria': '>=',
                'value': 0.7,
                'format': risk_format
            })
        
        output.seek(0)
        return output

def main():
    st.title("🏦 ETF Basket Trade Failure Risk Assessment")
    st.markdown("""
    Upload your PCF file to analyze potential trade failures and risk factors.
    Required columns: Ticker, Cusip, Asset Class, Securities Description, Weight of Holdings, Shares, and Market Value
    """)
    
    try:
        # Debug: Print available secrets
        st.write("Available secrets:", st.secrets.keys())
    
    # Initialize components
    api_key = st.secrets.get("POLYGON_API_KEY")
    if not api_key:
        st.error("POLYGON_API_KEY not found in secrets. Please check your configuration.")
        st.info("Using demo mode with limited functionality.")
        api_key = "demo_key"  # Fallback for testing
    
    market_fetcher = MarketDataFetcher(api_key)
    risk_analyzer = RiskAnalyzer()   
        
        # File upload
        uploaded_file = st.file_uploader("Upload PCF File (CSV format)", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read and validate PCF
                df = pd.read_csv(uploaded_file)
                df = PCFValidator.validate_pcf(df)
                
                st.write("PCF File Preview:")
                st.dataframe(df.head())
                
                if st.button("Analyze Risks"):
                    analysis_results = {'details': [], 'summary': {}}
                    
                    # Progress bar
                    progress_text = "Analyzing securities..."
                    progress_bar = st.progress(0)
                    
                    # Analyze each security
                    total_securities = len(df)
                    high_risk_count = 0
                    
                    for idx, row in df.iterrows():
                        # Update progress
                        progress = (idx + 1) / total_securities
                        progress_bar.progress(progress)
                        
                        # Fetch market data
                        market_data = market_fetcher.fetch_intraday_data(row['Ticker'])
                        
                        # Analyze risks
                        risk_analysis = risk_analyzer.analyze_security(row, market_data)
                        
                        if risk_analysis['risk_score'] >= 0.7:
                            high_risk_count += 1
                        
                        analysis_results['details'].append({
                            'Ticker': row['Ticker'],
                            'Description': row['Securities Description'],
                            'Asset Class': row['Asset Class'],
                            'Market Value': row['Market Value'],
                            'Weight': row['Weight of Holdings'],
                            'Risk Score': risk_analysis['risk_score'],
                            'Risk Factors': risk_analysis['risk_factors']
                        })
                    
                    # Clear progress bar
                    progress_bar.empty()
                    
                    # Summary metrics
                    analysis_results['summary'] = {
                        'total_securities': total_securities,
                        'high_risk_securities': high_risk_count,
                        'total_market_value': df['Market Value'].sum()
                    }
                    
                    # Display results
                    st.header("Analysis Results")
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Securities", total_securities)
                    with col2:
                        st.metric("High Risk Securities", high_risk_count)
                    with col3:
                        st.metric("High Risk Percentage", 
                                f"{(high_risk_count/total_securities)*100:.1f}%")
                    
                    # Risk table
                    st.subheader("Security Risk Analysis")
                    risk_df = pd.DataFrame([{
                        'Ticker': r['Ticker'],
                        'Description': r['Description'],
                        'Asset Class': r['Asset Class'],
                        'Market Value': f"${r['Market Value']:,.2f}",
                        'Risk Score': f"{r['Risk Score']:.2f}",
                        'Risk Level': 'High' if r['Risk Score'] >= 0.7 else 
                                    'Medium' if r['Risk Score'] >= 0.4 else 'Low'
                    } for r in analysis_results['details']])
                    
                    # Color code the risk levels
                    def color_risk(val):
                        if 'High' in str(val):
                            return 'background-color: #FFB6C1'
                        elif 'Medium' in str(val):
                            return 'background-color: #FFE4B5'
                        return ''
                    
                    st.dataframe(risk_df.style.applymap(color_risk, subset=['Risk Level']))
                     # Export functionality
                    if st.button("Export Analysis"):
                        excel_file = ReportGenerator.generate_excel_report(analysis_results)
                        st.download_button(
                            label="Download Detailed Report",
                            data=excel_file,
                            file_name=f"risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    # Individual security details
                    st.subheader("Detailed Risk Analysis")
                    for security in analysis_results['details']:
                        # Color code based on risk score
                        risk_color = (
                            "🔴" if float(security['Risk Score']) >= 0.7 else 
                            "🟡" if float(security['Risk Score']) >= 0.4 else 
                            "🟢"
                        )
                        
                        if float(security['Risk Score']) >= 0.4:  # Show only medium and high risk securities
                            with st.expander(f"{risk_color} {security['Ticker']} - {security['Description']}"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric(
                                        "Risk Score", 
                                        f"{security['Risk Score']:.2f}",
                                        delta="High Risk" if float(security['Risk Score']) >= 0.7 else "Medium Risk",
                                        delta_color="inverse"
                                    )
                                    st.metric(
                                        "Market Value", 
                                        f"${security['Market Value']:,.2f}"
                                    )
                                
                                with col2:
                                    st.metric("Asset Class", security['Asset Class'])
                                    st.metric(
                                        "Weight", 
                                        f"{security['Weight']:.2%}"
                                    )
                                
                                st.subheader("Risk Factors")
                                for factor in security['Risk Factors']:
                                    with st.container():
                                        if factor['level'] == 'High':
                                            st.error(f"**{factor['factor']}**: {factor['description']}")
                                        elif factor['level'] == 'Medium':
                                            st.warning(f"**{factor['factor']}**: {factor['description']}")
                                        else:
                                            st.info(f"**{factor['factor']}**: {factor['description']}")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.write("Please ensure your PCF file has all required columns:")
                st.write(", ".join(PCFValidator.REQUIRED_COLUMNS.keys()))
    
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.write("Please check your API key configuration and try again.")

if __name__ == "__main__":
    main()
