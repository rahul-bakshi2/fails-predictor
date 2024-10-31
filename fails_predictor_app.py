import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import yfinance as yf
from io import StringIO, BytesIO
import xlsxwriter
from datetime import datetime, timedelta
import ta

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
                    df[col] = pd.to_numeric(df[col].str.replace('$', '').str.replace(',', ''))
                elif dtype == str:
                    df[col] = df[col].astype(str)
            except Exception as e:
                raise ValueError(f"Invalid data in column {col}: {str(e)}")
        
        return df

class PCFProcessor:
    def __init__(self, market_fetcher):
        self.market_fetcher = market_fetcher
    
    def process_pcf(self, uploaded_file):
        """Process PCF file with enhanced validation and processing"""
        try:
            # Read PCF file
            content = uploaded_file.getvalue().decode('utf-8')
            df = pd.read_csv(StringIO(content))
            
            # Validate PCF format
            df = PCFValidator.validate_pcf(df)
            
            # Add additional analysis columns
            df['Risk_Category'] = 'Pending'
            df['Liquidity_Score'] = 0.0
            df['Market_Impact_Score'] = 0.0
            df['Settlement_Risk_Score'] = 0.0
            
            return df
            
        except Exception as e:
            st.error(f"Error processing PCF file: {str(e)}")
            return None

class BasketAnalyzer:
    def __init__(self, market_fetcher, predictor):
        self.market_fetcher = market_fetcher
        self.predictor = predictor
        
    def analyze_basket(self, basket_df):
        """Enhanced basket analysis with detailed metrics"""
        results = []
        summary_metrics = {
            'total_market_value': basket_df['Market Value'].sum(),
            'total_securities': len(basket_df),
            'asset_class_breakdown': basket_df.groupby('Asset Class').size().to_dict(),
            'high_risk_securities': 0,
            'risk_by_asset_class': {},
            'total_weight': basket_df['Weight of Holdings'].sum()
        }
        
        # Progress bar
        progress_bar = st.progress(0)
        
        for idx, row in basket_df.iterrows():
            progress = (idx + 1) / len(basket_df)
            progress_bar.progress(progress)
            
            # Get market data
            market_data = self.market_fetcher.fetch_intraday_data(row['Ticker'])
            if market_data is None:
                continue
            
            # Calculate risk metrics
            risk_score, risk_factors = self.calculate_security_risk(row, market_data)
            
            # Update high risk count
            if risk_score > 0.7:
                summary_metrics['high_risk_securities'] += 1
            
            # Track risk by asset class
            asset_class = row['Asset Class']
            if asset_class not in summary_metrics['risk_by_asset_class']:
                summary_metrics['risk_by_asset_class'][asset_class] = {
                    'count': 0,
                    'high_risk_count': 0,
                    'total_value': 0
                }
            
            summary_metrics['risk_by_asset_class'][asset_class]['count'] += 1
            if risk_score > 0.7:
                summary_metrics['risk_by_asset_class'][asset_class]['high_risk_count'] += 1
            summary_metrics['risk_by_asset_class'][asset_class]['total_value'] += row['Market Value']
            
            results.append({
                'Ticker': row['Ticker'],
                'Cusip': row['Cusip'],
                'Securities Description': row['Securities Description'],
                'Asset Class': row['Asset Class'],
                'Weight': row['Weight of Holdings'],
                'Shares': row['Shares'],
                'Market Value': row['Market Value'],
                'Risk_Score': risk_score,
                'Risk_Factors': risk_factors,
                'Market_Data': market_data
            })
        
        progress_bar.empty()
        
        return {
            'details': results,
            'summary': summary_metrics
        }
    
    def calculate_security_risk(self, security, market_data):
        """Calculate comprehensive risk metrics for a security"""
        risk_score = 0
        risk_factors = []
        
        # Market impact risk
        market_impact = (security['Market Value'] / 
                        (market_data['volume'].mean() * market_data['close'].mean()))
        
        # Liquidity risk
        avg_daily_volume = market_data['volume'].mean() * market_data['close'].mean()
        liquidity_risk = security['Market Value'] / avg_daily_volume
        
        # Asset class specific risk
        asset_class_risk = self.get_asset_class_risk(security['Asset Class'])
        
        # Weight concentration risk
        weight_risk = security['Weight of Holdings'] > 0.05
        
        # Compile risk factors
        if market_impact > 0.1:
            risk_factors.append({
                'factor': 'Market Impact',
                'risk_level': 'High' if market_impact > 0.2 else 'Medium',
                'description': f'Trade size is {market_impact:.1%} of average daily volume',
                'contribution': min(market_impact, 1.0)
            })
            
        if liquidity_risk > 0.2:
            risk_factors.append({
                'factor': 'Liquidity',
                'risk_level': 'High' if liquidity_risk > 0.4 else 'Medium',
                'description': f'Position requires {liquidity_risk:.1f} days to liquidate',
                'contribution': min(liquidity_risk/2, 1.0)
            })
        
        return sum(f['contribution'] for f in risk_factors)/len(risk_factors), risk_factors
    
    @staticmethod
    def get_asset_class_risk(asset_class):
        """Get risk factor based on asset class"""
        risk_factors = {
            'Equity': 0.3,
            'Fixed Income': 0.5,
            'ETF': 0.4,
            'Option': 0.7,
            'Default': 0.5
        }
        return risk_factors.get(asset_class, risk_factors['Default'])

class ReportGenerator:
    @staticmethod
    def generate_excel_report(analysis_results):
        """Generate detailed Excel report of analysis"""
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Total Market Value',
                    'Total Securities',
                    'High Risk Securities',
                    'Total Weight'
                ],
                'Value': [
                    f"${analysis_results['summary']['total_market_value']:,.2f}",
                    analysis_results['summary']['total_securities'],
                    analysis_results['summary']['high_risk_securities'],
                    f"{analysis_results['summary']['total_weight']:.2%}"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Asset Class Breakdown
            asset_class_data = pd.DataFrame(analysis_results['summary']['risk_by_asset_class']).T
            asset_class_data.to_excel(writer, sheet_name='Asset Class Analysis')
            
            # Detailed Security Analysis
            security_details = pd.DataFrame([
                {
                    'Ticker': r['Ticker'],
                    'Description': r['Securities Description'],
                    'Asset Class': r['Asset Class'],
                    'Market Value': r['Market Value'],
                    'Risk Score': r['Risk_Score'],
                    'Risk Factors': '; '.join([f"{rf['factor']}: {rf['risk_level']}" 
                                             for rf in r['Risk_Factors']])
                }
                for r in analysis_results['details']
            ])
            security_details.to_excel(writer, sheet_name='Security Details', index=False)
            
            # Format worksheets
            workbook = writer.book
            
            # Add formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D3D3D3',
                'border': 1
            })
            
            risk_format_high = workbook.add_format({
                'bg_color': '#FFB6C1',
                'border': 1
            })
            
            # Apply formats
            for sheet in writer.sheets.values():
                sheet.set_column('A:Z', 15)  # Set column width
                sheet.conditional_format('A2:Z1000', {
                    'type': 'formula',
                    'criteria': '=$E2>0.7',
                    'format': risk_format_high
                })
        
        output.seek(0)
        return output

def main():
    st.title("🏦 ETF Basket Trade Failure Risk Assessment")
    
    # Initialize components
    api_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
    market_fetcher = MarketDataFetcher(api_key)
    predictor = TradeFailurePredictor()
    pcf_processor = PCFProcessor(market_fetcher)
    basket_analyzer = BasketAnalyzer(market_fetcher, predictor)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload PCF File (CSV format)", 
        type=['csv']
    )
    
    if uploaded_file is not None:
        # Process PCF
        basket_df = pcf_processor.process_pcf(uploaded_file)
        
        if basket_df is not None:
            st.write("PCF File Preview:")
            st.dataframe(basket_df.head())
            
            if st.button("Analyze Basket Risk"):
                with st.spinner("Analyzing basket..."):
                    # Analyze basket
                    analysis_results = basket_analyzer.analyze_basket(basket_df)
                    
                    # Display summary metrics
                    st.header("Basket Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total Securities",
                            analysis_results['summary']['total_securities']
                        )
                    with col2:
                        st.metric(
                            "Total Value",
                            f"${analysis_results['summary']['total_market_value']:,.2f}"
                        )
                    with col3:
                        st.metric(
                            "High Risk Securities",
                            analysis_results['summary']['high_risk_securities']
                        )
                    with col4:
                        st.metric(
                            "Total Weight",
                            f"{analysis_results['summary']['total_weight']:.2%}"
                        )
                    
                    # Asset Class Breakdown
                    st.subheader("Asset Class Analysis")
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(analysis_results['summary']['asset_class_breakdown'].keys()),
                            y=list(analysis_results['summary']['asset_class_breakdown'].values())
                        )
                    ])
                    st.plotly_chart(fig)
                    
                    # Risk Analysis Table
                    st.subheader("Security Risk Analysis")
                    risk_df = pd.DataFrame([
                        {
                            'Ticker': r['Ticker'],
                            'Description': r['Securities Description'],
                            'Asset Class': r['Asset Class'],
                            'Market Value': r['Market Value'],
                            'Risk Score': r['Risk_Score']
                        }
                        for r in analysis_results['details']
                    ])
                    
                    # Color-coded risk display
                    def color_risk(val):
                        if val > 0.7:
                            return 'background-color: #FFB6C1'
                        elif val > 0.4:
                            return 'background-color: #FFE4B5'
                        return ''
                    
                    st.dataframe(risk_df.style.applymap(color_risk, subset=['Risk Score']))
                    
                    # Export functionality
                    if st.button("Export Analysis"):
                        excel_file = ReportGenerator.generate_excel_report(analysis_results)
                        st.download_button(
                            label="Download Detailed Report",
                            data=excel_file,
                            file_name=f"basket_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

if __name__ == "__main__":
    main()
