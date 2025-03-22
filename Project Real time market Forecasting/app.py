import streamlit as st
# st.set_page_config() must be the first Streamlit call.
st.set_page_config(layout="wide", page_title="Real Time Stock Dashboard")

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import ta
from streamlit_autorefresh import st_autorefresh

# ------------------------------
# Nifty 50 Company Name to Ticker Mapping
# ------------------------------
nifty_50_dict = {
    "Apple": "AAPL",
    "Adani Ports and SEZ": "ADANIPORTS.NS",
    "Axis Bank": "AXISBANK.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Bandhan Bank": "BANDHANBNK.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "BPCL": "BPCL.NS",
    "Cipla": "CIPLA.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Dr. Reddy's Laboratories": "DRREDDY.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Grasim Industries": "GRASIM.NS",
    "HCL Technologies": "HCLTECH.NS",
    "HDFC": "HDFC.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "HDFC Life Insurance": "HDFCLIFE.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Indian Oil Corporation": "IOC.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "ITC": "ITC.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Larsen & Toubro": "LT.NS",
    "Lupin": "LUPIN.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "M&M": "M&M.NS",
    "Muthoot Finance": "MUTHOOTFIN.NS",
    "Nestlé India": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "Power Grid Corporation": "POWERGRID.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Shree Cement": "SHREECEM.NS",
    "SBI Life Insurance": "SBILIFE.NS",
    "State Bank of India": "SBIN.NS",
    "Sun Pharmaceutical": "SUNPHARMA.NS",
    "Tata Consumer Products": "TATACONSUM.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Tech Mahindra": "TECHM.NS",
    "Titan": "TITAN.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Wipro": "WIPRO.NS",
    "Zee Entertainment": "ZEEL.NS",
    "Zydus Lifesciences": "ZYDUSLIFE.NS"
}


# ------------------------------
# Initialize Session State
# ------------------------------
if 'update_chart' not in st.session_state:
    st.session_state.update_chart = False

###########################################
## PART 1: Functions for Data Processing ##
###########################################

@st.cache_data(ttl=60)
def fetch_stock_data(ticker, period, interval):
    """
    Fetch historical stock data from yfinance.
    """
    end_date = datetime.now()
    if period == '1wk':
        start_date = end_date - timedelta(days=7)
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    else:
        data = yf.download(ticker, period=period, interval=interval)
    return data

def process_data(data):
    """
    Process the DataFrame:
      - Flatten MultiIndex columns (or remove ticker suffixes if present).
      - Convert the index to a timezone-aware Datetime in Asia/Kolkata.
      - Reset the index and rename 'Date' to 'Datetime' if needed.
      - Remove rows with missing 'Close' values.
    """
    if data.empty:
        st.error("No data fetched for the given ticker.")
        return data

    # Flatten MultiIndex columns if needed.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    else:
        # If column names include a comma (e.g., "Close,AAPL"), remove the ticker part.
        new_columns = {col: col.split(',')[0] for col in data.columns if isinstance(col, str)}
        data.rename(columns=new_columns, inplace=True)

    # Ensure the index is timezone-aware.
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('Asia/Kolkata')
    data.reset_index(inplace=True)

    # Rename 'Date' to 'Datetime' if necessary.
    if 'Date' in data.columns and 'Datetime' not in data.columns:
        data.rename(columns={'Date': 'Datetime'}, inplace=True)

    # Drop rows with missing 'Close' values.
    data = data.dropna(subset=['Close'])
    return data

def calculate_metrics(data):
    """
    Calculate key metrics from the data.
    """
    if data.empty or 'Close' not in data.columns:
        return None, None, None, None, None, None
    last_close = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[0]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100 if prev_close != 0 else 0
    high = data['High'].max() if 'High' in data.columns else None
    low = data['Low'].min() if 'Low' in data.columns else None
    volume = data['Volume'].sum() if 'Volume' in data.columns else None
    return last_close, change, pct_change, high, low, volume

def add_technical_indicators(data):
    """
    Add 20-period SMA and EMA to the data.
    """
    if data.empty or 'Close' not in data.columns:
        st.warning("Data is empty or missing 'Close' column for technical indicators.")
        return data

    # Drop any rows with missing 'Close' values.
    data = data.dropna(subset=['Close'])

    if len(data) < 20:
        st.warning("Not enough data to calculate SMA or EMA (need at least 20 data points).")
        return data

    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    return data

#############################################
## PART 2: Dashboard App Layout & Content  ##
#############################################

st.title('Real Time Stock Dashboard')

# ------------------------------
# Sidebar: Chart Parameters
# ------------------------------
st.sidebar.header('Chart Parameters')
company_name = st.sidebar.selectbox('Select Company', list(nifty_50_dict.keys()))
ticker = nifty_50_dict[company_name]
time_period = st.sidebar.selectbox('Time Period', ['1d', '1wk', '1mo', '1y', 'max'])
chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick', 'Line'])
indicators = st.sidebar.multiselect('Technical Indicators', ['SMA 20', 'EMA 20'])

# Mapping of time periods to data intervals.
interval_mapping = {
    '1d': '1m',
    '1wk': '30m',
    '1mo': '1d',
    '1y': '1wk',
    'max': '1wk'
}

# When the "Update Chart" button is clicked, set session state.
if st.sidebar.button('Update Chart'):
    st.session_state.update_chart = True

# ------------------------------
# Auto-Refresh Every 60 Seconds
# ------------------------------
st_autorefresh(interval=60000, key="real_time_data_refresh")

# ------------------------------
# Main Content: Chart & Data Display
# ------------------------------
if st.session_state.update_chart:
    data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
    data = process_data(data)
    if data.empty:
        st.warning(f"No data available for ticker: {ticker}")
    else:
        data = add_technical_indicators(data)
        last_close, change, pct_change, high, low, volume = calculate_metrics(data)
        if last_close is None:
            st.error("Insufficient data to display metrics.")
        else:
            # Display key metrics.
            st.metric(label=f"{company_name} Last Price", value=f"{last_close:.2f} INR",
                      delta=f"{change:.2f} ({pct_change:.2f}%)")
            col1, col2, col3 = st.columns(3)
            col1.metric("High", f"{high:.2f} INR" if high is not None else "N/A")
            col2.metric("Low", f"{low:.2f} INR" if low is not None else "N/A")
            col3.metric("Volume", f"{volume:,}" if volume is not None else "N/A")
            
            # Create the stock price chart.
            fig = go.Figure()
            if chart_type == 'Candlestick':
                if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                    fig.add_trace(go.Candlestick(x=data['Datetime'],
                                                 open=data['Open'],
                                                 high=data['High'],
                                                 low=data['Low'],
                                                 close=data['Close'],
                                                 name=ticker))
                else:
                    st.error("Required columns for Candlestick chart are missing.")
            else:
                if 'Close' in data.columns:
                    fig = px.line(data, x='Datetime', y='Close', title=f'{ticker} Price')
                else:
                    st.error("'Close' column is missing in data.")
            
            # Add selected technical indicators.
            for indicator in indicators:
                if indicator == 'SMA 20' and 'SMA_20' in data.columns:
                    fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA_20'],
                                             mode='lines', name='SMA 20'))
                elif indicator == 'EMA 20' and 'EMA_20' in data.columns:
                    fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA_20'],
                                             mode='lines', name='EMA 20'))
            fig.update_layout(
                title=f'{ticker} {time_period.upper()} Chart',
                xaxis_title='Time',
                yaxis_title='Price (INR)',
                xaxis_rangeslider_visible=True,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display historical data.
            st.subheader('Historical Data')
            display_cols = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [col for col in display_cols if col in data.columns]
            st.dataframe(data[available_cols])
            
            # Display technical indicators data.
            st.subheader('Technical Indicators')
            tech_cols = ['Datetime', 'SMA_20', 'EMA_20']
            available_tech = [col for col in tech_cols if col in data.columns]
            st.dataframe(data[available_tech][40:])

# ------------------------------
# Sidebar: Real-Time Stock Prices
# ------------------------------
st.sidebar.header('Real-Time Stock Prices')
stock_symbols = ["HDFC Bank", "HDFC Bank", "State Bank of India"]
for symbol in stock_symbols:
    tick = nifty_50_dict[symbol]
    rt_data = yf.download(tick, period='1d', interval='1m')
    rt_data = process_data(rt_data)
    if rt_data.empty or 'Open' not in rt_data.columns:
        st.sidebar.write(f"Data for {symbol} is not available.")
    else:
        try:
            first_open = float(rt_data['Open'].iloc[0])
        except Exception as e:
            st.sidebar.write(f"Error processing open price for {symbol}: {e}")
            continue

        if pd.notna(first_open) and first_open != 0:
            last_price = float(rt_data['Close'].iloc[-1])
            change = last_price - first_open
            pct_change = (change / first_open) * 100
        else:
            last_price = float(rt_data['Close'].iloc[-1])
            change = 0
            pct_change = 0

        st.sidebar.metric(f"{symbol}", f"{last_price:.2f} INR",
                          f"{change:.2f} ({pct_change:.2f}%)")

# ------------------------------
# Sidebar: About Section
# ------------------------------
st.sidebar.subheader('About')
st.sidebar.info(
    'This dashboard provides real-time and historical stock data with technical indicators. '
    'Use the sidebar to customize your view. Data refreshes automatically every minute.'
)
